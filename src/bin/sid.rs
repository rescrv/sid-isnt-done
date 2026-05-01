use std::io::{self, Write};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrrg::CommandLine;
use handled::SError;
use rc_conf::SwitchPosition;
use rustyline::DefaultEditor;
use rustyline::config::{Config, EditMode};
use rustyline::error::ReadlineError;
use utf8path::Path;

use claudius::chat::{
    ChatAgent, ChatArgs, ChatCommand, ChatConfig, ChatSession, PlainTextRenderer, SessionStats,
    help_text, parse_command,
};
use claudius::{Anthropic, Model};
use claudius::{OperatorLine, Renderer, StopReason, StreamContext};

use sid_isnt_done::config::{AGENTS_CONF_FILE, Config as SidConfig, TOOLS_CONF_FILE};
use sid_isnt_done::{
    COMPACTION_REQUEST_PROMPT, SidAgent, compacted_transcript, extract_last_assistant_text,
    seatbelt, session, session::SidSession,
};

const DEFAULT_SYSTEM_PROMPT: &str = concat!(
    "You are sid, a concise coding agent with access to the current workspace mounted at /.\n",
    "Use configured tools when they help accomplish the user's request.\n",
    "Ground your answers in the files and tool results you can access, explain changes you make, and do not claim to have run commands or changed files unless you actually did."
);
const BUILTIN_AGENT_ID: &str = "sid";
const SANDBOX_UNAVAILABLE_WARNING: &str = concat!(
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
    "!! WARNING: /usr/bin/sandbox-exec is unavailable.\n",
    "!! WARNING: sid will run bash and external tools UNSANDBOXED.\n",
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
);

struct SidTerminal {
    editor: DefaultEditor,
    renderer: PlainTextRenderer,
}

impl SidTerminal {
    fn new(use_color: bool, interrupted: Arc<AtomicBool>) -> Result<Self, SError> {
        let config = Config::builder().edit_mode(EditMode::Vi).build();
        let editor = DefaultEditor::with_config(config).map_err(|err| {
            cli_error(
                "readline_init_failed",
                "failed to initialize the terminal editor",
            )
            .with_string_field("cause", &err.to_string())
        })?;
        Ok(Self {
            editor,
            renderer: PlainTextRenderer::with_color_and_interrupt(use_color, interrupted),
        })
    }

    fn read_line(&mut self, prompt: &str) -> io::Result<OperatorLine> {
        match self.editor.readline(prompt) {
            Ok(line) => Ok(OperatorLine::Line(line)),
            Err(ReadlineError::Interrupted) => Ok(OperatorLine::Interrupted),
            Err(ReadlineError::Eof) => Ok(OperatorLine::Eof),
            Err(err) => Err(io::Error::other(err.to_string())),
        }
    }

    fn add_history_entry(&mut self, line: &str) {
        let _ = self.editor.add_history_entry(line);
    }
}

impl Renderer for SidTerminal {
    fn start_agent(&mut self, context: &dyn StreamContext) {
        self.renderer.start_agent(context);
    }

    fn finish_agent(&mut self, context: &dyn StreamContext, stop_reason: Option<&StopReason>) {
        self.renderer.finish_agent(context, stop_reason);
    }

    fn print_text(&mut self, context: &dyn StreamContext, text: &str) {
        self.renderer.print_text(context, text);
    }

    fn print_thinking(&mut self, context: &dyn StreamContext, text: &str) {
        self.renderer.print_thinking(context, text);
    }

    fn print_error(&mut self, context: &dyn StreamContext, error: &str) {
        self.renderer.print_error(context, error);
    }

    fn print_info(&mut self, context: &dyn StreamContext, info: &str) {
        self.renderer.print_info(context, info);
    }

    fn start_tool_use(&mut self, context: &dyn StreamContext, name: &str, id: &str) {
        self.renderer.start_tool_use(context, name, id);
    }

    fn print_tool_input(&mut self, context: &dyn StreamContext, partial_json: &str) {
        self.renderer.print_tool_input(context, partial_json);
    }

    fn finish_tool_use(&mut self, context: &dyn StreamContext) {
        self.renderer.finish_tool_use(context);
    }

    fn start_tool_result(
        &mut self,
        context: &dyn StreamContext,
        tool_use_id: &str,
        is_error: bool,
    ) {
        self.renderer
            .start_tool_result(context, tool_use_id, is_error);
    }

    fn print_tool_result_text(&mut self, context: &dyn StreamContext, text: &str) {
        self.renderer.print_tool_result_text(context, text);
    }

    fn finish_tool_result(&mut self, context: &dyn StreamContext) {
        self.renderer.finish_tool_result(context);
    }

    fn finish_response(&mut self, context: &dyn StreamContext) {
        self.renderer.finish_response(context);
    }

    fn print_interrupted(&mut self, context: &dyn StreamContext) {
        self.renderer.print_interrupted(context);
    }

    fn should_interrupt(&self) -> bool {
        self.renderer.should_interrupt()
    }

    fn read_operator_line(&mut self, prompt: &str) -> io::Result<Option<OperatorLine>> {
        self.read_line(prompt).map(Some)
    }
}

#[derive(arrrg_derive::CommandLine, Debug, Default, PartialEq, Eq)]
struct SidArgs {
    #[arrrg(nested)]
    param: ChatArgs,

    #[arrrg(
        optional,
        "Run COMMAND through the builtin bash tool and exit",
        "COMMAND"
    )]
    bash_debug: Option<String>,

    #[arrrg(optional, "Resume an existing session by ID or directory", "SESSION")]
    resume: Option<String>,
}

/// Data computed synchronously before the tokio runtime starts.
struct PreRuntimeSetup {
    config: ChatConfig,
    workspace_root: Path<'static>,
    config_root: Path<'static>,
    sid_session: Arc<SidSession>,
    workspace_display: String,
    session_display: String,
    bash_debug: Option<String>,
    resumed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AgentSummary {
    id: String,
    display_name: Option<String>,
    description: Option<String>,
    enabled: SwitchPosition,
    current: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SidCommand {
    ShowAgent,
    AgentList,
    SwitchAgent(String),
    Compact,
    Invalid(String),
}

struct QuietRenderer;

impl Renderer for QuietRenderer {
    fn print_text(&mut self, _context: &dyn StreamContext, _text: &str) {}

    fn print_thinking(&mut self, _context: &dyn StreamContext, _text: &str) {}

    fn print_error(&mut self, _context: &dyn StreamContext, _error: &str) {}

    fn print_info(&mut self, _context: &dyn StreamContext, _info: &str) {}

    fn start_tool_use(&mut self, _context: &dyn StreamContext, _name: &str, _id: &str) {}

    fn print_tool_input(&mut self, _context: &dyn StreamContext, _partial_json: &str) {}

    fn finish_tool_use(&mut self, _context: &dyn StreamContext) {}

    fn start_tool_result(
        &mut self,
        _context: &dyn StreamContext,
        _tool_use_id: &str,
        _is_error: bool,
    ) {
    }

    fn print_tool_result_text(&mut self, _context: &dyn StreamContext, _text: &str) {}

    fn finish_tool_result(&mut self, _context: &dyn StreamContext) {}

    fn finish_response(&mut self, _context: &dyn StreamContext) {}
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SessionStatsRollup {
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_requests: u64,
    total_cache_creation_tokens: u64,
    total_cache_read_tokens: u64,
}

impl SessionStatsRollup {
    fn add(&mut self, stats: &SessionStats) {
        self.total_input_tokens = self
            .total_input_tokens
            .saturating_add(stats.total_input_tokens);
        self.total_output_tokens = self
            .total_output_tokens
            .saturating_add(stats.total_output_tokens);
        self.total_requests = self.total_requests.saturating_add(stats.total_requests);
        self.total_cache_creation_tokens = self
            .total_cache_creation_tokens
            .saturating_add(stats.total_cache_creation_tokens);
        self.total_cache_read_tokens = self
            .total_cache_read_tokens
            .saturating_add(stats.total_cache_read_tokens);
    }

    fn apply(&self, stats: &mut SessionStats) {
        stats.total_input_tokens = stats
            .total_input_tokens
            .saturating_add(self.total_input_tokens);
        stats.total_output_tokens = stats
            .total_output_tokens
            .saturating_add(self.total_output_tokens);
        stats.total_requests = stats.total_requests.saturating_add(self.total_requests);
        stats.total_cache_creation_tokens = stats
            .total_cache_creation_tokens
            .saturating_add(self.total_cache_creation_tokens);
        stats.total_cache_read_tokens = stats
            .total_cache_read_tokens
            .saturating_add(self.total_cache_read_tokens);
    }
}

#[derive(Clone, Debug, Default)]
struct SessionOverrides {
    model: Option<Model>,
    system_prompt: Option<Option<String>>,
    max_tokens: Option<u32>,
    temperature: Option<Option<f32>>,
    top_p: Option<Option<f32>>,
    top_k: Option<Option<u32>>,
    stop_sequences: Option<Option<Vec<String>>>,
    thinking_budget: Option<Option<u32>>,
    session_budget: Option<Option<u64>>,
    caching_enabled: Option<bool>,
}

impl SessionOverrides {
    fn apply_to(&self, config: &mut ChatConfig) {
        if let Some(model) = self.model.as_ref() {
            config.set_model(model.clone());
        }
        if let Some(prompt) = self.system_prompt.as_ref() {
            config.set_system_prompt(prompt.clone());
        }
        if let Some(max_tokens) = self.max_tokens {
            config.set_max_tokens(max_tokens);
        }
        if let Some(temperature) = self.temperature {
            config.set_temperature(temperature);
        }
        if let Some(top_p) = self.top_p {
            config.set_top_p(top_p);
        }
        if let Some(top_k) = self.top_k {
            config.set_top_k(top_k);
        }
        if let Some(stop_sequences) = self.stop_sequences.as_ref() {
            config.template.stop_sequences = stop_sequences.clone();
        }
        if let Some(thinking_budget) = self.thinking_budget {
            config.set_thinking_budget(thinking_budget);
        }
        if let Some(session_budget) = self.session_budget {
            config.set_session_budget(session_budget);
        }
        if let Some(caching_enabled) = self.caching_enabled {
            config.caching_enabled = caching_enabled;
        }
    }

    fn apply_to_without_system_prompt(&self, config: &mut ChatConfig) {
        let mut overrides = self.clone();
        overrides.system_prompt = None;
        overrides.apply_to(config);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AgentSwitchResult {
    NoChange,
    Switched(AgentSummary),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompactResult {
    parent_session_id: String,
    new_session_id: String,
    new_session_root: String,
}

struct SidRuntimeSession {
    client: Anthropic,
    chat: ChatSession<SidAgent>,
    fallback_config: ChatConfig,
    workspace_root: Path<'static>,
    config_root: Path<'static>,
    sid_session: Arc<SidSession>,
    current_agent_id: String,
    overrides: SessionOverrides,
    rolled_up_stats: SessionStatsRollup,
}

impl SidRuntimeSession {
    #[allow(clippy::too_many_arguments)]
    fn new(
        client: Anthropic,
        chat: ChatSession<SidAgent>,
        fallback_config: ChatConfig,
        workspace_root: Path<'static>,
        config_root: Path<'static>,
        sid_session: Arc<SidSession>,
        current_agent_id: String,
    ) -> Self {
        Self {
            client,
            chat,
            fallback_config,
            workspace_root,
            config_root,
            sid_session,
            current_agent_id,
            overrides: SessionOverrides::default(),
            rolled_up_stats: SessionStatsRollup::default(),
        }
    }

    fn current_agent_id(&self) -> &str {
        &self.current_agent_id
    }

    fn config(&self) -> &ChatConfig {
        self.chat.config()
    }

    fn stats(&self) -> SessionStats {
        let mut stats = self.chat.stats();
        self.rolled_up_stats.apply(&mut stats);
        stats
    }

    async fn send_message(
        &mut self,
        message: claudius::MessageParam,
        renderer: &mut dyn Renderer,
    ) -> Result<(), claudius::Error> {
        self.chat.send_message(message, renderer).await
    }

    fn clear(&mut self) -> Result<(), SError> {
        self.chat.clear();
        self.persist_transcript()
    }

    fn save_transcript_to(&self, path: &str) -> Result<(), SError> {
        self.chat
            .save_transcript_to(path)
            .map_err(|err| transcript_error("transcript_save_failed", path, &err.to_string()))
    }

    fn load_transcript_from(&mut self, path: &str) -> Result<(), SError> {
        self.chat
            .load_transcript_from(path)
            .map_err(|err| transcript_error("transcript_load_failed", path, &err.to_string()))?;
        self.persist_transcript()
    }

    fn load_resumed_transcript(&mut self, resumed: bool) -> Result<(), SError> {
        if !resumed {
            return Ok(());
        }

        let transcript_path = self.sid_session.transcript_path();
        if !transcript_path.is_file() {
            return Ok(());
        }

        self.chat
            .load_transcript_from(&transcript_path)
            .map_err(|err| {
                cli_error(
                    "resume_transcript_failed",
                    "failed to load resumed transcript",
                )
                .with_string_field("path", transcript_path.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())
            })
    }

    fn set_model(&mut self, model_name: &str) {
        let model = model_name
            .parse()
            .unwrap_or_else(|_| Model::Custom(model_name.to_string()));
        self.chat.config_mut().set_model(model.clone());
        self.overrides.model = Some(model);
    }

    fn set_system_prompt(&mut self, prompt: Option<String>) {
        self.chat.config_mut().set_system_prompt(prompt.clone());
        self.overrides.system_prompt = Some(prompt);
    }

    fn set_max_tokens(&mut self, max_tokens: u32) {
        self.chat.config_mut().set_max_tokens(max_tokens);
        self.overrides.max_tokens = Some(max_tokens);
    }

    fn set_temperature(&mut self, temperature: Option<f32>) {
        self.chat.config_mut().set_temperature(temperature);
        self.overrides.temperature = Some(temperature);
    }

    fn set_top_p(&mut self, top_p: Option<f32>) {
        self.chat.config_mut().set_top_p(top_p);
        self.overrides.top_p = Some(top_p);
    }

    fn set_top_k(&mut self, top_k: Option<u32>) {
        self.chat.config_mut().set_top_k(top_k);
        self.overrides.top_k = Some(top_k);
    }

    fn add_stop_sequence(&mut self, sequence: String) {
        let stop_sequences = self
            .chat
            .template_mut()
            .stop_sequences
            .get_or_insert_with(Vec::new);
        if !stop_sequences.iter().any(|existing| existing == &sequence) {
            stop_sequences.push(sequence);
        }
        self.overrides.stop_sequences = Some(self.chat.template().stop_sequences.clone());
    }

    fn clear_stop_sequences(&mut self) {
        self.chat.template_mut().stop_sequences = None;
        self.overrides.stop_sequences = Some(None);
    }

    fn set_thinking_budget(&mut self, thinking_budget: Option<u32>) {
        self.chat.config_mut().set_thinking_budget(thinking_budget);
        self.overrides.thinking_budget = Some(thinking_budget);
    }

    fn set_session_budget(&mut self, session_budget: Option<u64>) {
        self.chat.config_mut().set_session_budget(session_budget);
        self.overrides.session_budget = Some(session_budget);
    }

    fn set_caching_enabled(&mut self, enabled: bool) {
        self.chat.config_mut().caching_enabled = enabled;
        self.overrides.caching_enabled = Some(enabled);
    }

    fn current_agent_summary(&self) -> Result<AgentSummary, SError> {
        self.agent_summary(self.current_agent_id())?.ok_or_else(|| {
            cli_error("missing_agent", "current agent is unavailable")
                .with_string_field("agent", self.current_agent_id())
        })
    }

    fn agent_summary(&self, agent_id: &str) -> Result<Option<AgentSummary>, SError> {
        Ok(self
            .list_agents()?
            .into_iter()
            .find(|summary| summary.id == agent_id))
    }

    fn list_agents(&self) -> Result<Vec<AgentSummary>, SError> {
        load_agent_summaries(&self.config_root, self.current_agent_id())
    }

    fn switch_agent(&mut self, agent_id: &str) -> Result<AgentSwitchResult, SError> {
        if agent_id == self.current_agent_id() {
            return Ok(AgentSwitchResult::NoChange);
        }

        let messages = self.chat.clone_messages();
        let mut next_agent = SidAgent::from_workspace_agent_with_config_root(
            &self.workspace_root,
            &self.config_root,
            agent_id,
            self.fallback_config.clone(),
        )?
        .with_session(self.sid_session.clone());
        self.overrides.apply_to(next_agent.config_mut());

        self.roll_up_current_stats();

        let mut next_chat = ChatSession::with_agent(self.client.clone(), next_agent);
        next_chat.replace_messages(messages);

        self.chat = next_chat;
        self.current_agent_id = agent_id.to_string();
        self.persist_transcript()?;

        let summary = self.current_agent_summary()?;
        Ok(AgentSwitchResult::Switched(summary))
    }

    async fn compact(&mut self) -> Result<CompactResult, SError> {
        let parent_session_id = self.sid_session.id().to_string();
        let parent_session_dir = self.sid_session.root().to_string_lossy().into_owned();
        let messages = self.chat.clone_messages();
        let mut compactor_fallback = self.fallback_config.clone();
        compactor_fallback.set_system_prompt(None);
        compactor_fallback.transcript_path = None;

        let mut compactor = SidAgent::from_workspace_compactor_with_config_root(
            &self.workspace_root,
            &self.config_root,
            compactor_fallback,
        )?
        .with_memory_source(self.sid_session.compaction_provenance().cloned());
        self.overrides
            .apply_to_without_system_prompt(compactor.config_mut());

        let expert = compactor.compaction_snapshot();
        let mut compactor_chat = ChatSession::with_agent(self.client.clone(), compactor);
        compactor_chat.replace_messages(messages);

        let mut renderer = QuietRenderer;
        compactor_chat
            .send_message(
                claudius::MessageParam::user(COMPACTION_REQUEST_PROMPT),
                &mut renderer,
            )
            .await
            .map_err(|err| {
                cli_error(
                    "compaction_failed",
                    "failed to generate conversation summary",
                )
                .with_string_field("cause", &err.to_string())
            })?;

        let summary =
            extract_last_assistant_text(&compactor_chat.clone_messages()).ok_or_else(|| {
                cli_error(
                    "compaction_empty",
                    "compaction agent produced no assistant summary",
                )
            })?;

        let next_sid_session = Arc::new(SidSession::create_compacted(
            &self.config_root,
            session::CompactionProvenance {
                session_id: parent_session_id.clone(),
                session_dir: parent_session_dir,
                expert,
            },
        )?);

        if let Some(state) = self.sid_session.read_bash_state()? {
            next_sid_session.write_bash_state(&state)?;
        }

        let transcript_path = next_sid_session.transcript_path();
        self.fallback_config.transcript_path = Some(transcript_path.clone());

        let mut next_agent = SidAgent::from_workspace_agent_with_config_root(
            &self.workspace_root,
            &self.config_root,
            &self.current_agent_id,
            self.fallback_config.clone(),
        )?
        .with_session(next_sid_session.clone());
        self.overrides.apply_to(next_agent.config_mut());
        next_agent.config_mut().transcript_path = Some(transcript_path);

        let mut next_chat = ChatSession::with_agent(self.client.clone(), next_agent);
        next_chat.replace_messages(compacted_transcript(&parent_session_id, &summary));

        self.chat = next_chat;
        self.sid_session = next_sid_session.clone();
        self.rolled_up_stats = SessionStatsRollup::default();
        self.persist_transcript()?;

        Ok(CompactResult {
            parent_session_id,
            new_session_id: next_sid_session.id().to_string(),
            new_session_root: next_sid_session.root().display().to_string(),
        })
    }

    fn roll_up_current_stats(&mut self) {
        let stats = self.chat.stats();
        self.rolled_up_stats.add(&stats);
    }

    fn persist_transcript(&self) -> Result<(), SError> {
        let Some(path) = self.chat.config().transcript_path.as_ref() else {
            return Ok(());
        };
        self.chat
            .save_transcript_to(path)
            .map_err(|err| transcript_error_path("transcript_save_failed", path, &err.to_string()))
    }

    #[cfg(test)]
    fn clone_messages(&self) -> Vec<claudius::MessageParam> {
        self.chat.clone_messages()
    }

    #[cfg(test)]
    fn replace_messages(&mut self, messages: Vec<claudius::MessageParam>) -> Result<(), SError> {
        self.chat.replace_messages(messages);
        self.persist_transcript()
    }
}

/// Parse arguments, resolve paths, create the session directory, and set
/// environment variables for child processes.  Runs before the tokio runtime
/// so that process-global `set_var` calls are single-threaded and safe.
fn pre_runtime_setup() -> Result<PreRuntimeSetup, SError> {
    let SidArgs {
        param,
        bash_debug,
        resume,
    } = parse_sid_args()?;
    let mut config = ChatConfig::try_from(param).map_err(|err| {
        cli_error("invalid_cli_args", "failed to parse command line arguments")
            .with_string_field("cause", &err.to_string())
    })?;
    if config.template.system.is_none() {
        config = config.with_system_prompt(DEFAULT_SYSTEM_PROMPT.to_string());
    }

    let workspace_root = Path::try_from(std::env::current_dir().map_err(|err| {
        cli_error(
            "io_error",
            "failed to determine the current working directory",
        )
        .with_string_field("cause", &err.to_string())
    })?)
    .map_err(|err| {
        cli_error(
            "invalid_workspace_path",
            "current working directory is not valid UTF-8",
        )
        .with_string_field("cause", &err.to_string())
    })?
    .into_owned();
    let config_root = resolve_sid_home(&workspace_root)?;
    let resumed = resume.is_some();
    let sid_session = Arc::new(match resume {
        Some(session) => SidSession::resume(&config_root, &session)?,
        None => SidSession::create(&config_root)?,
    });
    config.transcript_path = Some(sid_session.transcript_path());
    // Safe: no other threads are running yet.
    unsafe {
        std::env::set_var("SID_WORKSPACE_ROOT", workspace_root.as_str());
        std::env::set_var(session::SID_SESSION_ID_ENV, sid_session.id());
        std::env::set_var(session::SID_SESSION_DIR_ENV, sid_session.root());
        std::env::set_var(session::SID_SESSIONS_ENV, sid_session.sessions_root());
    }
    let workspace_display = workspace_root.as_str().to_string();
    let session_display = sid_session.root().display().to_string();

    Ok(PreRuntimeSetup {
        config,
        workspace_root,
        config_root,
        sid_session,
        workspace_display,
        session_display: if resumed {
            format!("{session_display} (resumed)")
        } else {
            session_display
        },
        bash_debug,
        resumed,
    })
}

fn main() {
    let setup = match pre_runtime_setup() {
        Ok(s) => s,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };
    let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    if let Err(err) = runtime.block_on(try_main(setup)) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

async fn try_main(setup: PreRuntimeSetup) -> Result<(), SError> {
    let PreRuntimeSetup {
        config,
        workspace_root,
        config_root,
        sid_session,
        workspace_display,
        session_display,
        bash_debug,
        resumed,
    } = setup;

    warn_if_sandbox_unavailable();

    let agent =
        SidAgent::from_workspace_with_config_root(&workspace_root, &config_root, config.clone())?
            .with_session(sid_session.clone());
    let agent_id = agent.id().to_string();
    let use_color = agent.config().use_color;

    let interrupted = Arc::new(AtomicBool::new(false));
    let mut terminal = SidTerminal::new(use_color, interrupted.clone())?;
    let context = ();

    if agent.requires_confirmation() && !confirm_manual_agent(&mut terminal, &agent_id)? {
        println!("Aborted.");
        return Ok(());
    }

    if let Some(command) = bash_debug {
        return run_bash_debug(
            &agent,
            &agent_id,
            &workspace_display,
            &command,
            &mut terminal,
        )
        .await;
    }

    let client = Anthropic::new(None).map_err(|err| {
        cli_error(
            "client_init_failed",
            "failed to initialize the Anthropic client",
        )
        .with_string_field("cause", &err.to_string())
    })?;
    let chat = ChatSession::with_agent(client.clone(), agent);
    let mut session = SidRuntimeSession::new(
        client,
        chat,
        config,
        workspace_root.clone(),
        config_root.clone(),
        sid_session.clone(),
        agent_id.clone(),
    );
    session.load_resumed_transcript(resumed)?;

    let interrupted_clone = interrupted.clone();
    ctrlc::set_handler(move || {
        interrupted_clone.store(true, Ordering::Relaxed);
    })
    .map_err(|err| {
        cli_error(
            "signal_handler_failed",
            "failed to install the Ctrl-C handler",
        )
        .with_string_field("cause", &err.to_string())
    })?;

    println!(
        "sid (agent: {agent_id}, model: {})",
        session.config().model()
    );
    println!("workspace: {workspace_display}");
    println!("session: {session_display}");
    println!("Type /help for commands, /quit to exit\n");

    loop {
        interrupted.store(false, Ordering::Relaxed);

        match terminal.read_line("You: ") {
            Ok(OperatorLine::Line(line)) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                terminal.add_history_entry(line);

                if line == "/edit" {
                    match invoke_external_editor() {
                        Ok(Some(content)) => {
                            let content = content.trim();
                            terminal.add_history_entry(content);
                            let message = claudius::MessageParam::user(content);
                            if let Err(err) = session.send_message(message, &mut terminal).await {
                                terminal.print_error(&context, &err.to_string());
                            }
                        }
                        Ok(None) => {
                            terminal.print_info(&context, "Editor returned empty; nothing sent.");
                        }
                        Err(err) => {
                            terminal
                                .print_error(&context, &format!("Failed to invoke editor: {err}"));
                        }
                    }
                    continue;
                }

                if let Some(cmd) = parse_sid_command(line) {
                    match cmd {
                        SidCommand::ShowAgent => match session.current_agent_summary() {
                            Ok(summary) => print_agent_summary(&summary, session.config()),
                            Err(err) => terminal.print_error(&context, &err.to_string()),
                        },
                        SidCommand::AgentList => match session.list_agents() {
                            Ok(agents) => print_agent_list(&agents),
                            Err(err) => terminal.print_error(&context, &err.to_string()),
                        },
                        SidCommand::SwitchAgent(agent_name) => {
                            match session.agent_summary(&agent_name) {
                                Ok(Some(summary)) if summary.enabled == SwitchPosition::Manual => {
                                    if !confirm_manual_agent(&mut terminal, &summary.id)? {
                                        terminal.print_info(&context, "Agent switch cancelled.");
                                        continue;
                                    }
                                }
                                Ok(_) => {}
                                Err(err) => {
                                    terminal.print_error(&context, &err.to_string());
                                    continue;
                                }
                            }

                            match session.switch_agent(&agent_name) {
                                Ok(AgentSwitchResult::NoChange) => terminal.print_info(
                                    &context,
                                    &format!("Already using agent: {agent_name}"),
                                ),
                                Ok(AgentSwitchResult::Switched(summary)) => {
                                    terminal.print_info(
                                        &context,
                                        &format!(
                                            "Switched to agent: {}",
                                            format_agent_label(&summary)
                                        ),
                                    );
                                }
                                Err(err) => terminal.print_error(&context, &err.to_string()),
                            }
                        }
                        SidCommand::Compact => match session.compact().await {
                            Ok(result) => terminal.print_info(
                                &context,
                                &format!(
                                    "Compacted session {} into {} ({})",
                                    result.parent_session_id,
                                    result.new_session_id,
                                    result.new_session_root
                                ),
                            ),
                            Err(err) => terminal.print_error(&context, &err.to_string()),
                        },
                        SidCommand::Invalid(message) => {
                            terminal.print_error(&context, &message);
                        }
                    }
                    continue;
                }

                if let Some(cmd) = parse_command(line) {
                    match cmd {
                        ChatCommand::Quit => {
                            println!("Goodbye!");
                            break;
                        }
                        ChatCommand::Clear => match session.clear() {
                            Ok(()) => terminal.print_info(&context, "Conversation cleared."),
                            Err(err) => terminal.print_error(&context, &err.to_string()),
                        },
                        ChatCommand::Help => {
                            print_help();
                        }
                        ChatCommand::Model(model_name) => {
                            session.set_model(&model_name);
                            terminal
                                .print_info(&context, &format!("Model changed to: {model_name}"));
                        }
                        ChatCommand::System(prompt) => {
                            session.set_system_prompt(prompt.clone());
                            match prompt {
                                Some(prompt) => terminal.print_info(
                                    &context,
                                    &format!("System prompt set to: {prompt}"),
                                ),
                                None => terminal.print_info(&context, "System prompt cleared."),
                            }
                        }
                        ChatCommand::MaxTokens(value) => {
                            session.set_max_tokens(value);
                            terminal.print_info(&context, &format!("max_tokens set to {value}"));
                        }
                        ChatCommand::Temperature(value) => {
                            session.set_temperature(Some(value));
                            terminal
                                .print_info(&context, &format!("temperature set to {value:.2}"));
                        }
                        ChatCommand::ClearTemperature => {
                            session.set_temperature(None);
                            terminal.print_info(&context, "temperature reset to model default");
                        }
                        ChatCommand::TopP(value) => {
                            session.set_top_p(Some(value));
                            terminal.print_info(&context, &format!("top_p set to {value:.2}"));
                        }
                        ChatCommand::ClearTopP => {
                            session.set_top_p(None);
                            terminal.print_info(&context, "top_p reset to model default");
                        }
                        ChatCommand::TopK(value) => {
                            session.set_top_k(Some(value));
                            terminal.print_info(&context, &format!("top_k set to {value}"));
                        }
                        ChatCommand::ClearTopK => {
                            session.set_top_k(None);
                            terminal.print_info(&context, "top_k reset to model default");
                        }
                        ChatCommand::AddStopSequence(sequence) => {
                            session.add_stop_sequence(sequence.clone());
                            terminal
                                .print_info(&context, &format!("Added stop sequence: {sequence}"));
                        }
                        ChatCommand::ClearStopSequences => {
                            session.clear_stop_sequences();
                            terminal.print_info(&context, "Stop sequences cleared.");
                        }
                        ChatCommand::ListStopSequences => {
                            print_stop_sequences(session.config().stop_sequences());
                        }
                        ChatCommand::Thinking(budget) => {
                            session.set_thinking_budget(budget);
                            match budget {
                                Some(tokens) => terminal.print_info(
                                    &context,
                                    &format!(
                                        "Extended thinking enabled with {} token budget.",
                                        tokens
                                    ),
                                ),
                                None => {
                                    terminal.print_info(&context, "Extended thinking disabled.");
                                }
                            }
                        }
                        ChatCommand::Budget(tokens) => {
                            session.set_session_budget(Some(tokens));
                            terminal.print_info(
                                &context,
                                &format!("Session budget set to {tokens} tokens."),
                            );
                        }
                        ChatCommand::ClearBudget => {
                            session.set_session_budget(None);
                            terminal.print_info(&context, "Session budget cleared.");
                        }
                        ChatCommand::Caching(enabled) => {
                            session.set_caching_enabled(enabled);
                            if enabled {
                                terminal.print_info(&context, "Prompt caching enabled.");
                            } else {
                                terminal.print_info(&context, "Prompt caching disabled.");
                            }
                        }
                        ChatCommand::TranscriptPath(_) | ChatCommand::ClearTranscriptPath => {
                            terminal.print_info(
                                &context,
                                "Transcript auto-save is managed by the session system.",
                            );
                        }
                        ChatCommand::SaveTranscript(path) => {
                            match session.save_transcript_to(&path) {
                                Ok(()) => terminal
                                    .print_info(&context, &format!("Transcript saved to {path}")),
                                Err(err) => terminal.print_error(
                                    &context,
                                    &format!("Failed to save transcript: {err}"),
                                ),
                            }
                        }
                        ChatCommand::LoadTranscript(path) => {
                            match session.load_transcript_from(&path) {
                                Ok(()) => terminal.print_info(
                                    &context,
                                    &format!("Transcript loaded from {path}"),
                                ),
                                Err(err) => terminal.print_error(
                                    &context,
                                    &format!("Failed to load transcript: {err}"),
                                ),
                            }
                        }
                        ChatCommand::Stats => {
                            print_stats(&session.stats());
                        }
                        ChatCommand::ShowConfig => {
                            print_config(&session.stats());
                        }
                        ChatCommand::Invalid(message) => {
                            terminal.print_error(&context, &message);
                        }
                    }
                    continue;
                }

                let message = claudius::MessageParam::user(line);
                if let Err(err) = session.send_message(message, &mut terminal).await {
                    terminal.print_error(&context, &err.to_string());
                }
            }
            Ok(OperatorLine::Interrupted) => {
                println!();
                continue;
            }
            Ok(OperatorLine::Eof) => {
                println!("\nGoodbye!");
                break;
            }
            Err(err) => {
                terminal.print_error(&context, &format!("Input error: {err}"));
                break;
            }
        }
    }

    Ok(())
}

fn parse_sid_command(input: &str) -> Option<SidCommand> {
    let input = input.trim();
    if !input.starts_with('/') {
        return None;
    }

    let mut parts = input[1..].splitn(2, ' ');
    let command = parts.next()?.to_ascii_lowercase();
    let argument = parts.next().map(str::trim).filter(|s| !s.is_empty());
    if command == "compact" {
        return if argument.is_some() {
            Some(SidCommand::Invalid(
                "/compact does not take any arguments".to_string(),
            ))
        } else {
            Some(SidCommand::Compact)
        };
    }
    if command != "agent" && command != "agents" {
        return None;
    }

    let Some(argument) = argument else {
        return Some(match command.as_str() {
            "agents" => SidCommand::AgentList,
            _ => SidCommand::ShowAgent,
        });
    };

    let mut parts = argument.splitn(2, ' ');
    let action = parts.next().unwrap_or_default().to_ascii_lowercase();
    match action.as_str() {
        "show" | "current" => Some(SidCommand::ShowAgent),
        "list" => Some(SidCommand::AgentList),
        "switch" => {
            let Some(agent) = parts.next().map(str::trim).filter(|s| !s.is_empty()) else {
                return Some(SidCommand::Invalid(
                    "/agent switch requires an agent name".to_string(),
                ));
            };
            Some(SidCommand::SwitchAgent(agent.to_string()))
        }
        _ => Some(SidCommand::Invalid(
            "Use /agent, /agent list, or /agent switch <name>.".to_string(),
        )),
    }
}

fn print_help() {
    for line in help_text().lines() {
        println!("    {}", line);
    }
    println!(
        "      /compact              Summarize the session and continue in a new child session"
    );
    println!("      /agent                Show the current agent");
    println!("      /agent list           List configured agents");
    println!("      /agent switch <name>  Switch to another agent in this session");
}

fn load_agent_summaries(
    config_root: &Path,
    current_agent_id: &str,
) -> Result<Vec<AgentSummary>, SError> {
    if !config_root.join(AGENTS_CONF_FILE).is_file() && !config_root.join(TOOLS_CONF_FILE).is_file()
    {
        return Ok(vec![AgentSummary {
            id: BUILTIN_AGENT_ID.to_string(),
            display_name: None,
            description: None,
            enabled: SwitchPosition::Yes,
            current: current_agent_id == BUILTIN_AGENT_ID,
        }]);
    }

    let config = SidConfig::load(config_root)?;
    Ok(config
        .agents
        .values()
        .map(|agent| AgentSummary {
            id: agent.id.clone(),
            display_name: agent.display_name.clone(),
            description: agent.description.clone(),
            enabled: agent.enabled,
            current: agent.id == current_agent_id,
        })
        .collect())
}

fn format_agent_label(summary: &AgentSummary) -> String {
    match summary.display_name.as_deref() {
        Some(name) => format!("{} ({name})", summary.id),
        None => summary.id.clone(),
    }
}

fn print_agent_summary(summary: &AgentSummary, config: &ChatConfig) {
    println!("    Agent: {}", format_agent_label(summary));
    println!(
        "      Status: {}{}",
        describe_agent_enabled(summary.enabled),
        if summary.current { " (current)" } else { "" }
    );
    println!("      Model: {}", config.model());
    if let Some(description) = summary.description.as_deref() {
        println!("      Description: {description}");
    }
}

fn print_agent_list(agents: &[AgentSummary]) {
    println!("    Agents:");
    for agent in agents {
        let marker = if agent.current { "*" } else { " " };
        let mut line = format!(
            "      {marker} {} [{}]",
            format_agent_label(agent),
            describe_agent_enabled(agent.enabled)
        );
        if let Some(description) = agent.description.as_deref() {
            line.push_str(&format!(" - {description}"));
        }
        println!("{line}");
    }
}

fn describe_agent_enabled(enabled: SwitchPosition) -> &'static str {
    match enabled {
        SwitchPosition::Yes => "YES",
        SwitchPosition::Manual => "MANUAL",
        SwitchPosition::No => "NO",
    }
}

fn transcript_error(code: &str, path: &str, cause: &str) -> SError {
    cli_error(code, "transcript operation failed")
        .with_string_field("path", path)
        .with_string_field("cause", cause)
}

fn transcript_error_path(code: &str, path: &std::path::Path, cause: &str) -> SError {
    cli_error(code, "transcript operation failed")
        .with_string_field("path", &path.display().to_string())
        .with_string_field("cause", cause)
}

fn parse_sid_args() -> Result<SidArgs, SError> {
    let (args, free) = SidArgs::from_command_line_relaxed("sid [OPTIONS]");
    validate_no_free_args(&free)?;
    Ok(args)
}

fn validate_no_free_args(free: &[String]) -> Result<(), SError> {
    if free.is_empty() {
        return Ok(());
    }
    Err(
        cli_error("invalid_cli_args", "unexpected positional arguments")
            .with_string_field("args", &free.join(" ")),
    )
}

async fn run_bash_debug(
    agent: &SidAgent,
    agent_id: &str,
    workspace_display: &str,
    command: &str,
    terminal: &mut SidTerminal,
) -> Result<(), SError> {
    eprintln!("sid bash debug (agent: {agent_id})");
    eprintln!("workspace: {workspace_display}");

    match agent.bash_with_renderer(command, false, terminal).await {
        Ok(output) => {
            print!("{output}");
            if !output.ends_with('\n') {
                println!();
            }
            io::stdout().flush().map_err(|err| {
                cli_error("io_error", "failed to flush bash debug output")
                    .with_string_field("cause", &err.to_string())
            })?;
            Ok(())
        }
        Err(err) => {
            let mut message = err.to_string();
            if err.kind() == io::ErrorKind::Unsupported {
                message.push_str("\nhint: configure bash in tools.conf or set SID_HOME=init");
            }
            Err(cli_error("bash_debug_failed", &message))
        }
    }
}

fn cli_error(code: &str, message: &str) -> SError {
    SError::new("sid-cli").with_code(code).with_message(message)
}

fn warn_if_sandbox_unavailable() {
    if seatbelt::sandbox_available() {
        return;
    }

    eprint!("{SANDBOX_UNAVAILABLE_WARNING}");
}

fn resolve_sid_home(workspace_root: &Path) -> Result<Path<'static>, SError> {
    match std::env::var("SID_HOME") {
        Ok(path) if !path.is_empty() => Ok(Path::new(&path).into_owned()),
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(workspace_root.clone().into_owned()),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(cli_error("invalid_sid_home", "SID_HOME is not valid UTF-8"))
        }
    }
}

fn confirm_manual_agent(terminal: &mut SidTerminal, agent_id: &str) -> Result<bool, SError> {
    loop {
        let prompt = format!("Agent '{agent_id}' is MANUAL. Continue? [yes/no]: ");
        let input = match terminal.read_line(&prompt).map_err(|err| {
            cli_error("io_error", "failed to read manual-agent confirmation input")
                .with_string_field("agent", agent_id)
                .with_string_field("cause", &err.to_string())
        })? {
            OperatorLine::Line(input) => input,
            OperatorLine::Eof | OperatorLine::Interrupted => {
                println!();
                return Ok(false);
            }
        };

        match parse_confirmation(&input) {
            Some(answer) => return Ok(answer),
            None => println!("Please answer yes or no."),
        }
    }
}

fn parse_confirmation(input: &str) -> Option<bool> {
    match input.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Some(true),
        "n" | "no" => Some(false),
        _ => None,
    }
}

fn print_stats(stats: &SessionStats) {
    println!("    Session Statistics:");
    println!("      Model: {}", stats.model);
    println!("      Messages: {}", stats.message_count);
    println!("      Max tokens: {}", stats.max_tokens);
    println!("      Temperature: {}", describe_float(stats.temperature));
    println!("      Top-p: {}", describe_float(stats.top_p));
    println!("      Top-k: {}", describe_top_k(stats.top_k));
    if let Some(prompt) = stats.system_prompt.as_deref() {
        println!("      System prompt: {}", prompt);
    } else {
        println!("      System prompt: (none)");
    }
    println!(
        "      Thinking: {}",
        match stats.thinking_budget {
            Some(budget) => format!("enabled ({budget} tokens)"),
            None => "disabled".to_string(),
        }
    );
    print_stop_sequences(&stats.stop_sequences);
    println!(
        "      Total tokens: {} in / {} out ({} requests)",
        stats.total_input_tokens, stats.total_output_tokens, stats.total_requests
    );
    if stats.caching_enabled {
        println!(
            "      Cache tokens: {} created / {} read",
            stats.total_cache_creation_tokens, stats.total_cache_read_tokens
        );
    }
    if let Some(input) = stats.last_turn_input_tokens {
        let output = stats.last_turn_output_tokens.unwrap_or(0);
        println!("      Last turn tokens: {input} in / {output} out");
    }
    if let Some(limit) = stats.session_budget_tokens {
        let remaining = limit.saturating_sub(stats.budget_spent_tokens);
        println!(
            "      Budget: {}/{} tokens ({} remaining)",
            stats.budget_spent_tokens, limit, remaining
        );
    } else {
        println!("      Budget: (not set)");
    }
}

fn print_config(stats: &SessionStats) {
    println!("    Current Configuration:");
    println!("      Model: {}", stats.model);
    println!("      Max tokens: {}", stats.max_tokens);
    println!("      Temperature: {}", describe_float(stats.temperature));
    println!("      Top-p: {}", describe_float(stats.top_p));
    println!("      Top-k: {}", describe_top_k(stats.top_k));
    println!(
        "      Thinking: {}",
        match stats.thinking_budget {
            Some(budget) => format!("enabled ({budget} tokens)"),
            None => "disabled".to_string(),
        }
    );
    println!(
        "      Caching: {}",
        if stats.caching_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    if let Some(prompt) = stats.system_prompt.as_deref() {
        println!("      System prompt: {}", prompt);
    } else {
        println!("      System prompt: (none)");
    }
    print_stop_sequences(&stats.stop_sequences);
}

fn print_stop_sequences(stop_sequences: &[String]) {
    if stop_sequences.is_empty() {
        println!("      Stop sequences: (none)");
    } else {
        println!("      Stop sequences:");
        for stop_sequence in stop_sequences {
            println!("        - {}", stop_sequence);
        }
    }
}

fn describe_float(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "default".to_string())
}

fn describe_top_k(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "default".to_string())
}

fn invoke_external_editor() -> io::Result<Option<String>> {
    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .unwrap_or_else(|_| "vi".to_string());

    let tmp_dir = std::env::temp_dir();
    let filename = format!(
        "sid-{}-{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros()
    );
    let tmp_path = tmp_dir.join(filename);

    let status = Command::new(&editor)
        .arg(&tmp_path)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|err| io::Error::other(format!("{editor}: {err}")))?;

    let result = if status.success() {
        match std::fs::read_to_string(&tmp_path) {
            Ok(content) if content.trim().is_empty() => Ok(None),
            Ok(content) => Ok(Some(content)),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    } else {
        Ok(None)
    };

    let _ = std::fs::remove_file(&tmp_path);
    result
}

#[cfg(test)]
mod tests {
    use super::{
        AgentSummary, AgentSwitchResult, DEFAULT_SYSTEM_PROMPT, SANDBOX_UNAVAILABLE_WARNING,
        SidArgs, SidCommand, SidRuntimeSession, SwitchPosition, parse_confirmation,
        parse_sid_command, validate_no_free_args,
    };
    use arrrg::{CommandLine, NoExitCommandLine};
    use claudius::Anthropic;
    use claudius::MessageParam;
    use claudius::chat::{ChatConfig, ChatSession};
    use serde::Deserialize;
    use sid_isnt_done::{SidAgent, session::SidSession};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    use utf8path::Path;

    #[derive(Debug, PartialEq)]
    struct SessionSnapshot {
        agent_id: String,
        messages: Vec<MessageParam>,
        model: claudius::Model,
        temperature: Option<f32>,
        stop_sequences: Vec<String>,
        session_budget_tokens: Option<u64>,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct TranscriptSnapshot {
        version: u8,
        messages: Vec<MessageParam>,
    }

    fn parse_args(argv: &[&str]) -> (SidArgs, Vec<String>, i32, Vec<String>) {
        let (parsed, free) =
            NoExitCommandLine::<SidArgs>::from_arguments_relaxed("sid [OPTIONS]", argv);
        let (args, messages, status) = parsed.into_parts();
        (args, free, status, messages)
    }

    #[test]
    fn parse_confirmation_accepts_yes_values() {
        assert_eq!(parse_confirmation("yes"), Some(true));
        assert_eq!(parse_confirmation("Y"), Some(true));
    }

    #[test]
    fn parse_confirmation_accepts_no_values() {
        assert_eq!(parse_confirmation("no"), Some(false));
        assert_eq!(parse_confirmation("N"), Some(false));
    }

    #[test]
    fn parse_confirmation_rejects_other_values() {
        assert_eq!(parse_confirmation(""), None);
        assert_eq!(parse_confirmation("maybe"), None);
    }

    #[test]
    fn parse_args_accept_resume_option() {
        let (args, free, status, messages) =
            parse_args(&["--resume", "2026-04-20T18-42-13.123456-0700"]);
        assert_eq!(status, 0, "unexpected parser status: {messages:?}");
        assert!(free.is_empty());
        assert_eq!(
            args.resume.as_deref(),
            Some("2026-04-20T18-42-13.123456-0700")
        );
    }

    #[test]
    fn load_resumed_transcript_restores_saved_history() {
        let workspace_root = unique_workspace_root("resume-transcript");
        let sid_session = Arc::new(SidSession::create(&workspace_root).unwrap());

        fs::write(
            sid_session.transcript_path(),
            serde_json::to_vec_pretty(&serde_json::json!({
                "version": 1,
                "messages": [MessageParam::user("resume me")]
            }))
            .unwrap(),
        )
        .unwrap();

        let mut session =
            new_runtime_session(&workspace_root, &workspace_root, sid_session.clone(), None);

        session.load_resumed_transcript(true).unwrap();
        assert_eq!(
            session.clone_messages(),
            vec![MessageParam::user("resume me")]
        );

        fs::remove_dir_all(PathBuf::from(workspace_root.as_str())).unwrap();
    }

    #[test]
    fn parse_agent_commands() {
        assert_eq!(parse_sid_command("/agent"), Some(SidCommand::ShowAgent));
        assert_eq!(parse_sid_command("/agents"), Some(SidCommand::AgentList));
        assert_eq!(parse_sid_command("/compact"), Some(SidCommand::Compact));
        assert_eq!(
            parse_sid_command("/agent list"),
            Some(SidCommand::AgentList)
        );
        assert_eq!(
            parse_sid_command("/agent switch review"),
            Some(SidCommand::SwitchAgent("review".to_string()))
        );
        assert_eq!(
            parse_sid_command("/agent switch"),
            Some(SidCommand::Invalid(
                "/agent switch requires an agent name".to_string()
            ))
        );
        assert_eq!(
            parse_sid_command("/compact now"),
            Some(SidCommand::Invalid(
                "/compact does not take any arguments".to_string()
            ))
        );
    }

    #[test]
    fn list_agents_reports_current_manual_and_disabled_agents() {
        let root = unique_workspace_root("agent-list");
        write_multi_agent_config(&root);
        let session = configured_runtime_session(&root, "build");

        assert_eq!(
            session.list_agents().unwrap(),
            vec![
                AgentSummary {
                    id: "build".to_string(),
                    display_name: Some("Builder".to_string()),
                    description: Some("Build changes".to_string()),
                    enabled: SwitchPosition::Yes,
                    current: true,
                },
                AgentSummary {
                    id: "off".to_string(),
                    display_name: None,
                    description: Some("Disabled agent".to_string()),
                    enabled: SwitchPosition::No,
                    current: false,
                },
                AgentSummary {
                    id: "review".to_string(),
                    display_name: Some("Reviewer".to_string()),
                    description: Some("Review changes".to_string()),
                    enabled: SwitchPosition::Manual,
                    current: false,
                },
            ]
        );

        fs::remove_dir_all(PathBuf::from(root.as_str())).unwrap();
    }

    #[test]
    fn switch_agent_preserves_transcript_overrides_and_saved_transcript() {
        let root = unique_workspace_root("agent-switch");
        write_multi_agent_config(&root);
        let mut session = configured_runtime_session(&root, "build");

        session
            .replace_messages(vec![MessageParam::user("resume me")])
            .unwrap();
        session.set_model("claude-sonnet-4-0");
        session.set_temperature(Some(0.5));
        session.add_stop_sequence("END".to_string());
        session.set_session_budget(Some(1234));

        assert_eq!(
            session.switch_agent("review").unwrap(),
            AgentSwitchResult::Switched(AgentSummary {
                id: "review".to_string(),
                display_name: Some("Reviewer".to_string()),
                description: Some("Review changes".to_string()),
                enabled: SwitchPosition::Manual,
                current: true,
            })
        );
        assert_eq!(
            snapshot(&session),
            SessionSnapshot {
                agent_id: "review".to_string(),
                messages: vec![MessageParam::user("resume me")],
                model: "claude-sonnet-4-0".parse().unwrap(),
                temperature: Some(0.5),
                stop_sequences: vec!["END".to_string()],
                session_budget_tokens: Some(1234),
            }
        );
        assert!(
            session
                .config()
                .system_prompt_text()
                .unwrap()
                .contains("# Review Agent"),
            "expected review prompt in system prompt"
        );
        assert_eq!(
            read_saved_transcript(&session.sid_session.transcript_path()),
            TranscriptSnapshot {
                version: 1,
                messages: vec![MessageParam::user("resume me")],
            }
        );

        fs::remove_dir_all(PathBuf::from(root.as_str())).unwrap();
    }

    #[test]
    fn switch_agent_rejects_disabled_agents() {
        let root = unique_workspace_root("agent-disabled");
        write_multi_agent_config(&root);
        let mut session = configured_runtime_session(&root, "build");

        let err = session.switch_agent("off").unwrap_err().to_string();
        assert!(err.contains("disabled_agent"), "error: {err}");

        fs::remove_dir_all(PathBuf::from(root.as_str())).unwrap();
    }

    fn unique_workspace_root(prefix: &str) -> Path<'static> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sid-isnt-done-{prefix}-{nanos}"));
        fs::create_dir_all(&root).unwrap();
        Path::try_from(root).unwrap().into_owned()
    }

    fn test_chat_config(transcript_path: PathBuf) -> ChatConfig {
        let mut config = ChatConfig::new().with_system_prompt(DEFAULT_SYSTEM_PROMPT.to_string());
        config.transcript_path = Some(transcript_path);
        config
    }

    fn new_runtime_session(
        workspace_root: &Path,
        config_root: &Path,
        sid_session: Arc<SidSession>,
        agent: Option<&str>,
    ) -> SidRuntimeSession {
        let config = test_chat_config(sid_session.transcript_path());
        let agent = match agent {
            Some(agent) => SidAgent::from_workspace_agent_with_config_root(
                workspace_root,
                config_root,
                agent,
                config.clone(),
            )
            .unwrap(),
            None => SidAgent::from_workspace_with_config_root(
                workspace_root,
                config_root,
                config.clone(),
            )
            .unwrap(),
        }
        .with_session(sid_session.clone());
        let current_agent_id = agent.id().to_string();
        let client = Anthropic::new(Some("test-api-key".to_string())).unwrap();
        let chat = ChatSession::with_agent(client.clone(), agent);
        SidRuntimeSession::new(
            client,
            chat,
            config,
            workspace_root.clone().into_owned(),
            config_root.clone().into_owned(),
            sid_session,
            current_agent_id,
        )
    }

    fn configured_runtime_session(root: &Path, agent: &str) -> SidRuntimeSession {
        let sid_session = Arc::new(SidSession::create(root).unwrap());
        new_runtime_session(root, root, sid_session, Some(agent))
    }

    fn write_multi_agent_config(root: &Path) {
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            concat!(
                "DEFAULT_AGENT=build\n",
                "build_ENABLED=YES\n",
                "build_NAME='Builder'\n",
                "build_DESC='Build changes'\n",
                "build_TOOLS='bash'\n",
                "review_ENABLED=MANUAL\n",
                "review_NAME='Reviewer'\n",
                "review_DESC='Review changes'\n",
                "review_TOOLS='bash'\n",
                "off_ENABLED=NO\n",
                "off_DESC='Disabled agent'\n",
                "off_TOOLS='bash'\n",
            ),
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        fs::write(root.join("agents/build.md").as_str(), "# Build Agent\n").unwrap();
        fs::write(root.join("agents/review.md").as_str(), "# Review Agent\n").unwrap();
        fs::write(root.join("agents/off.md").as_str(), "# Off Agent\n").unwrap();
    }

    fn snapshot(session: &SidRuntimeSession) -> SessionSnapshot {
        let stats = session.stats();
        SessionSnapshot {
            agent_id: session.current_agent_id().to_string(),
            messages: session.clone_messages(),
            model: session.config().model(),
            temperature: session.config().template.temperature,
            stop_sequences: session.config().stop_sequences().to_vec(),
            session_budget_tokens: stats.session_budget_tokens,
        }
    }

    fn read_saved_transcript(path: &std::path::Path) -> TranscriptSnapshot {
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }

    #[test]
    fn sid_args_parse_prefixed_temperature_param() {
        let (args, free, status, messages) = parse_args(&["--param-temperature", "0.5"]);

        assert_eq!(status, 0);
        assert!(
            messages.is_empty(),
            "unexpected parser messages: {messages:?}"
        );
        assert!(free.is_empty(), "unexpected free args: {free:?}");
        assert_eq!(args.param.temperature.as_deref(), Some("0.5"));
    }

    #[test]
    fn sid_args_parse_prefixed_param_and_bash_debug() {
        let (args, free, status, messages) =
            parse_args(&["--param-model", "claude-haiku-4-5", "--bash-debug", "pwd"]);

        assert_eq!(status, 0);
        assert!(
            messages.is_empty(),
            "unexpected parser messages: {messages:?}"
        );
        assert!(free.is_empty(), "unexpected free args: {free:?}");
        assert_eq!(args.param.model.as_deref(), Some("claude-haiku-4-5"));
        assert_eq!(args.bash_debug.as_deref(), Some("pwd"));
    }

    #[test]
    fn sid_args_reject_unprefixed_chat_params() {
        let (_args, _free, status, messages) = parse_args(&["--temperature", "0.5"]);

        assert_eq!(status, 64);
        assert!(
            messages
                .iter()
                .any(|message| message.contains("Unrecognized option: 'temperature'")),
            "expected unrecognized option message, got {messages:?}"
        );
    }

    #[test]
    fn sid_args_reject_debug_alias() {
        let (_args, _free, status, messages) = parse_args(&["--debug", "pwd"]);

        assert_eq!(status, 64);
        assert!(
            messages
                .iter()
                .any(|message| message.contains("Unrecognized option: 'debug'")),
            "expected unrecognized option message, got {messages:?}"
        );
    }

    #[test]
    fn sid_args_reject_free_positional_args() {
        let (_args, free, status, messages) = parse_args(&["hello"]);

        assert_eq!(status, 0);
        assert!(
            messages.is_empty(),
            "unexpected parser messages: {messages:?}"
        );
        assert_eq!(free, vec!["hello".to_string()]);
        assert!(validate_no_free_args(&free).is_err());
    }

    #[test]
    fn sid_args_reject_missing_bash_debug_command() {
        let (_args, _free, status, messages) = parse_args(&["--bash-debug"]);

        assert_eq!(status, 64);
        assert!(
            messages
                .iter()
                .any(|message| message.contains("Argument to option 'bash-debug' missing")),
            "expected missing argument message, got {messages:?}"
        );
    }

    #[test]
    fn sandbox_warning_is_loud_and_explicit() {
        assert!(SANDBOX_UNAVAILABLE_WARNING.contains("WARNING"));
        assert!(SANDBOX_UNAVAILABLE_WARNING.contains("/usr/bin/sandbox-exec is unavailable"));
        assert!(SANDBOX_UNAVAILABLE_WARNING.contains("UNSANDBOXED"));
    }
}
