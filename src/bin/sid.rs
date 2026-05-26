use std::io::{self, Write};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use arrrg::CommandLine;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use handled::SError;
use rc_conf::SwitchPosition;
use rustyline::DefaultEditor;
use rustyline::config::{Config, EditMode};
use rustyline::error::ReadlineError;
use serde_json::{Value, json};
use utf8path::Path;

use claudius::chat::{
    ChatAgent, ChatArgs, ChatCommand, ChatConfig, ChatSession, PlainTextRenderer, SessionStats,
    help_text, parse_command,
};
use claudius::{Anthropic, Effort, KnownModel, Model, TokenRates};
use claudius::{OperatorLine, Renderer, StopReason, StreamContext};

use sid_isnt_done::config::{
    AGENTS_CONF_FILE, COMPACTION_PROMPT_ID, Config as SidConfig, TOOLS_CONF_FILE,
};
use sid_isnt_done::raw_mode::{RawInput, RawServer, RawToolOutputObserver, RawUsageReportObserver};
use sid_isnt_done::raw_protocol::{
    RAW_PROTOCOL_VERSION, RawEvent, RawHello, RawPrompt, RawRequest, RawRequestEnvelope,
    RawResultEnvelope, RawServerMessage, install_tool_output_observer,
    install_usage_report_observer,
};
use sid_isnt_done::{
    COMPACTION_REQUEST_PROMPT, SidAgent, append_resumed_bash_reset_marker, compacted_transcript,
    extract_last_assistant_text, seatbelt, session, session::SidSession,
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
const MICRO_CENTS_PER_DOLLAR: f64 = 100_000_000.0;
const SESSION_SPEND_EXHAUSTED: &str = concat!(
    "Session spend limit exhausted. ",
    "Use /spend to increase or /spend clear to remove the limit."
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SidSpendState {
    limit_micro_cents: u64,
    used_micro_cents: u64,
}

impl SidSpendState {
    fn new(limit_micro_cents: u64) -> Self {
        Self {
            limit_micro_cents,
            used_micro_cents: 0,
        }
    }

    fn from_dollars(dollars: f64) -> Self {
        Self::new(dollars_to_micro_cents(dollars))
    }

    fn remaining_micro_cents(&self) -> u64 {
        self.limit_micro_cents.saturating_sub(self.used_micro_cents)
    }

    fn record_cost(&mut self, cost_micro_cents: u64) {
        self.used_micro_cents = self.used_micro_cents.saturating_add(cost_micro_cents);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SpendUsageTotals {
    input_tokens: u64,
    output_tokens: u64,
    cache_creation_tokens: u64,
    cache_read_tokens: u64,
}

impl SpendUsageTotals {
    fn from_stats_delta(before: &SessionStats, after: &SessionStats) -> Self {
        Self {
            input_tokens: after
                .total_input_tokens
                .saturating_sub(before.total_input_tokens),
            output_tokens: after
                .total_output_tokens
                .saturating_sub(before.total_output_tokens),
            cache_creation_tokens: after
                .total_cache_creation_tokens
                .saturating_sub(before.total_cache_creation_tokens),
            cache_read_tokens: after
                .total_cache_read_tokens
                .saturating_sub(before.total_cache_read_tokens),
        }
    }

    fn cost_micro_cents(&self, rates: TokenRates) -> u64 {
        self.input_tokens
            .saturating_mul(rates.input)
            .saturating_add(self.output_tokens.saturating_mul(rates.output))
            .saturating_add(
                self.cache_creation_tokens
                    .saturating_mul(rates.cache_creation),
            )
            .saturating_add(self.cache_read_tokens.saturating_mul(rates.cache_read))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SpendTurnClamp {
    original_max_tokens: Option<u32>,
    clamped_max_tokens: u32,
}

fn dollars_to_micro_cents(dollars: f64) -> u64 {
    let result = dollars * MICRO_CENTS_PER_DOLLAR;
    if result.is_finite() && result >= 0.0 {
        result as u64
    } else {
        u64::MAX
    }
}

fn model_token_rates(model: &Model) -> TokenRates {
    match model {
        Model::Known(KnownModel::ClaudeHaiku45 | KnownModel::ClaudeHaiku4520251001) => {
            KnownModel::ClaudeHaiku45.token_rates()
        }
        Model::Known(
            KnownModel::Claude37SonnetLatest
            | KnownModel::Claude37Sonnet20250219
            | KnownModel::ClaudeSonnet40
            | KnownModel::ClaudeSonnet420250514
            | KnownModel::Claude4Sonnet20250514
            | KnownModel::ClaudeSonnet45
            | KnownModel::ClaudeSonnet4520250929,
        ) => KnownModel::ClaudeSonnet45.token_rates(),
        Model::Known(
            KnownModel::ClaudeOpus4520251101
            | KnownModel::ClaudeOpus45
            | KnownModel::ClaudeOpus46
            | KnownModel::ClaudeOpus47,
        ) => KnownModel::ClaudeOpus45.token_rates(),
        Model::Known(_) | Model::Custom(_) => KnownModel::ClaudeSonnet45.token_rates(),
    }
}

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

struct PromptRenderer {
    renderer: PlainTextRenderer,
}

impl PromptRenderer {
    fn new(use_color: bool, interrupted: Arc<AtomicBool>) -> Self {
        Self {
            renderer: PlainTextRenderer::with_color_and_interrupt(use_color, interrupted),
        }
    }
}

impl Renderer for PromptRenderer {
    fn start_agent(&mut self, context: &dyn StreamContext) {
        self.renderer.start_agent(context);
    }

    fn finish_agent(&mut self, context: &dyn StreamContext, stop_reason: Option<&StopReason>) {
        self.renderer.finish_agent(context, stop_reason);
    }

    fn print_text(&mut self, _context: &dyn StreamContext, _text: &str) {}

    fn print_thinking(&mut self, _context: &dyn StreamContext, _text: &str) {}

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
        print!("{prompt}");
        io::stdout().flush()?;

        let mut line = String::new();
        match io::stdin().read_line(&mut line) {
            Ok(0) => {
                println!();
                Ok(Some(OperatorLine::Eof))
            }
            Ok(_) => Ok(Some(OperatorLine::Line(line))),
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                println!();
                Ok(Some(OperatorLine::Interrupted))
            }
            Err(err) => Err(err),
        }
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

    #[arrrg(optional, "Run one prompt non-interactively and exit", "PROMPT")]
    prompt: Option<String>,

    #[arrrg(flag, "Run a JSONL protocol server on stdin/stdout")]
    raw: bool,

    #[arrrg(
        optional,
        "Run a reconnectable JSONL protocol server on SPEC; implies --raw",
        "SPEC"
    )]
    listen: Option<String>,

    #[arrrg(
        optional,
        "Connect to a reconnectable JSONL protocol server on SPEC",
        "SPEC"
    )]
    connect: Option<String>,
}

enum StartupSetup {
    Local(Box<PreRuntimeSetup>),
    Connect(ConnectSetup),
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
    prompt: Option<String>,
    raw: bool,
    listen: Option<String>,
    resumed: bool,
}

/// Data needed to run `sid --connect` as a terminal frontend.
struct ConnectSetup {
    spec: String,
    use_color: bool,
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
    thinking_adaptive: Option<Option<Effort>>,
    effort: Option<Option<Effort>>,
    session_spend: Option<Option<f64>>,
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
        if let Some(effort) = self.thinking_adaptive {
            config.set_thinking_adaptive(effort);
        }
        if let Some(effort) = self.effort {
            config.set_effort(effort);
        }
        if let Some(session_spend) = self.session_spend {
            config.set_session_spend(session_spend);
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
    sid_spend: Option<SidSpendState>,
    auto_compact_tokens: Option<u64>,
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
        auto_compact_tokens: Option<u64>,
    ) -> Self {
        let sid_spend = chat
            .config()
            .session_spend
            .as_ref()
            .map(|spend| SidSpendState::new(spend.total_micro_cents()));
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
            sid_spend,
            auto_compact_tokens,
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
        match self.sid_spend {
            Some(spend) => {
                stats.session_spend_micro_cents = Some(spend.limit_micro_cents);
                stats.spend_used_micro_cents = spend.used_micro_cents;
            }
            None => {
                stats.session_spend_micro_cents = None;
                stats.spend_used_micro_cents = 0;
            }
        }
        stats
    }

    /// Return `true` when auto-compaction is configured and the session has
    /// generated at least that many output tokens.
    fn should_auto_compact(&self) -> bool {
        match self.auto_compact_tokens {
            Some(threshold) => self.stats().total_output_tokens >= threshold,
            None => false,
        }
    }

    async fn send_message(
        &mut self,
        message: claudius::MessageParam,
        renderer: &mut dyn Renderer,
    ) -> Result<(), claudius::Error> {
        let model = self.chat.config().model();
        let before = self.chat.stats();
        let clamp = match self.spend_turn_clamp() {
            Ok(clamp) => clamp,
            Err(err) => {
                renderer.print_error(&(), SESSION_SPEND_EXHAUSTED);
                return Err(err);
            }
        };
        if let Some(clamp) = clamp {
            self.chat
                .config_mut()
                .set_max_tokens(clamp.clamped_max_tokens);
        }

        let result = self.chat.send_message(message, renderer).await;
        let after = self.chat.stats();
        if let Some(clamp) = clamp {
            self.chat.config_mut().template.max_tokens = clamp.original_max_tokens;
        }
        if result.is_ok() {
            self.record_sid_spend_delta(&before, &after, &model);
        }
        result
    }

    fn spend_turn_clamp(&self) -> Result<Option<SpendTurnClamp>, claudius::Error> {
        let Some(spend) = self.sid_spend else {
            return Ok(None);
        };
        let remaining = spend.remaining_micro_cents();
        let rates = model_token_rates(&self.chat.config().model());
        let affordable_tokens = remaining / rates.output.max(1);
        if affordable_tokens == 0 {
            return Err(claudius::Error::bad_request(
                "session spend exhausted",
                Some("spend".to_string()),
            ));
        }

        let max_tokens = self.chat.config().max_tokens();
        let affordable_tokens = u32::try_from(affordable_tokens).unwrap_or(u32::MAX);
        if affordable_tokens < max_tokens {
            Ok(Some(SpendTurnClamp {
                original_max_tokens: self.chat.config().template.max_tokens,
                clamped_max_tokens: affordable_tokens,
            }))
        } else {
            Ok(None)
        }
    }

    fn record_sid_spend_delta(
        &mut self,
        before: &SessionStats,
        after: &SessionStats,
        model: &Model,
    ) {
        let Some(spend) = self.sid_spend.as_mut() else {
            return;
        };
        let usage = SpendUsageTotals::from_stats_delta(before, after);
        spend.record_cost(usage.cost_micro_cents(model_token_rates(model)));
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
        if transcript_path.is_file() {
            self.chat
                .load_transcript_from(&transcript_path)
                .map_err(|err| {
                    cli_error(
                        "resume_transcript_failed",
                        "failed to load resumed transcript",
                    )
                    .with_string_field("path", transcript_path.to_string_lossy().as_ref())
                    .with_string_field("cause", &err.to_string())
                })?;
        }
        let mut messages = self.chat.clone_messages();
        append_resumed_bash_reset_marker(&mut messages);
        self.chat.replace_messages(messages);
        self.persist_transcript()
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

    fn set_thinking_adaptive(&mut self, effort: Option<Effort>) {
        self.chat.config_mut().set_thinking_adaptive(effort);
        self.overrides.thinking_adaptive = Some(effort);
    }

    fn set_effort(&mut self, effort: Option<Effort>) {
        self.chat.config_mut().set_effort(effort);
        self.overrides.effort = Some(effort);
    }

    fn set_session_spend(&mut self, dollars: Option<f64>) {
        self.chat.config_mut().set_session_spend(dollars);
        self.overrides.session_spend = Some(dollars);
        self.sid_spend = dollars.map(SidSpendState::from_dollars);
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

        self.auto_compact_tokens = next_agent.auto_compact_tokens();
        self.roll_up_current_stats();

        let mut next_chat = ChatSession::with_agent(self.client.clone(), next_agent);
        next_chat.replace_messages(messages);

        self.chat = next_chat;
        self.current_agent_id = agent_id.to_string();
        self.sync_configured_sid_spend();
        self.persist_transcript()?;

        let summary = self.current_agent_summary()?;
        Ok(AgentSwitchResult::Switched(summary))
    }

    fn sync_configured_sid_spend(&mut self) {
        if self.sid_spend.is_some() || self.overrides.session_spend.is_some() {
            return;
        }
        self.sid_spend = self
            .chat
            .config()
            .session_spend
            .as_ref()
            .map(|spend| SidSpendState::new(spend.total_micro_cents()));
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
        let compaction_prompt = compactor
            .named_prompt_markdown(COMPACTION_PROMPT_ID)
            .unwrap_or(COMPACTION_REQUEST_PROMPT)
            .to_string();
        let mut compactor_chat = ChatSession::with_agent(self.client.clone(), compactor);
        compactor_chat.replace_messages(messages);

        let mut renderer = QuietRenderer;
        compactor_chat
            .send_message(
                claudius::MessageParam::user(compaction_prompt),
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

        let next_sid_session = Arc::new(SidSession::create_compacted_with_workspace(
            &self.config_root,
            &self.workspace_root,
            session::CompactionProvenance {
                session_id: parent_session_id.clone(),
                session_dir: parent_session_dir,
                expert,
            },
        )?);

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
        self.auto_compact_tokens = next_agent.auto_compact_tokens();

        let mut next_chat = ChatSession::with_agent(self.client.clone(), next_agent);
        next_chat.replace_messages(compacted_transcript(&parent_session_id, &summary));

        self.chat = next_chat;
        self.sid_session = next_sid_session.clone();
        self.rolled_up_stats = SessionStatsRollup::default();
        self.sync_configured_sid_spend();
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

    fn last_assistant_text(&self) -> Option<String> {
        extract_last_assistant_text(&self.chat.clone_messages())
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
fn pre_runtime_setup() -> Result<StartupSetup, SError> {
    let SidArgs {
        param,
        bash_debug,
        resume,
        prompt,
        raw,
        listen,
        connect,
    } = parse_sid_args()?;
    if let Some(connect) = connect {
        validate_connect_mode(
            raw,
            listen.as_deref(),
            resume.as_deref(),
            bash_debug.as_deref(),
            prompt.as_deref(),
        )?;
        let config = ChatConfig::try_from(param).map_err(|err| {
            cli_error("invalid_cli_args", "failed to parse command line arguments")
                .with_string_field("cause", &err.to_string())
        })?;
        return Ok(StartupSetup::Connect(ConnectSetup {
            spec: connect,
            use_color: config.use_color,
        }));
    }

    let raw = raw || listen.is_some();
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
    let config_root = resolve_sid_home()?;
    let (sid_session, resumed) = match (resume.as_deref(), prompt.as_ref()) {
        (Some(session), _) => (Arc::new(SidSession::resume(&config_root, session)?), true),
        (None, Some(_)) => {
            match SidSession::find_latest_for_workspace(&config_root, &workspace_root)? {
                Some(session) => (Arc::new(session), true),
                None => (
                    Arc::new(SidSession::create_with_workspace(
                        &config_root,
                        &workspace_root,
                    )?),
                    false,
                ),
            }
        }
        (None, None) => (
            Arc::new(SidSession::create_with_workspace(
                &config_root,
                &workspace_root,
            )?),
            false,
        ),
    };
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

    Ok(StartupSetup::Local(Box::new(PreRuntimeSetup {
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
        prompt,
        raw,
        listen,
        resumed,
    })))
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

async fn try_main(setup: StartupSetup) -> Result<(), SError> {
    let setup = match setup {
        StartupSetup::Local(setup) => *setup,
        StartupSetup::Connect(setup) => return run_connect_mode(setup),
    };

    let PreRuntimeSetup {
        config,
        workspace_root,
        config_root,
        sid_session,
        workspace_display,
        session_display,
        bash_debug,
        prompt,
        raw,
        listen,
        resumed,
    } = setup;

    validate_runtime_mode(raw, bash_debug.as_deref(), prompt.as_deref())?;

    warn_if_sandbox_unavailable();

    let agent =
        SidAgent::from_workspace_with_config_root(&workspace_root, &config_root, config.clone())?
            .with_session(sid_session.clone());
    let agent_id = agent.id().to_string();
    let startup_confirmation_required = agent.requires_confirmation();
    let use_color = agent.config().use_color;

    if raw {
        let client = Anthropic::new(None).map_err(|err| {
            cli_error(
                "client_init_failed",
                "failed to initialize the Anthropic client",
            )
            .with_string_field("cause", &err.to_string())
        })?;
        let auto_compact_tokens = agent.auto_compact_tokens();
        let chat = ChatSession::with_agent(client.clone(), agent);
        let mut session = SidRuntimeSession::new(
            client,
            chat,
            config,
            workspace_root.clone(),
            config_root.clone(),
            sid_session.clone(),
            agent_id,
            auto_compact_tokens,
        );
        session.load_resumed_transcript(resumed)?;
        if let Some(listen) = listen {
            return run_raw_session(
                session,
                sid_isnt_done::raw_listen::listen(&listen)
                    .map_err(|err| raw_io_error("failed to start raw listener", &err))?,
                workspace_display,
                session_display,
                resumed,
                startup_confirmation_required,
            )
            .await;
        }
        return run_raw_session(
            session,
            RawServer::stdio(),
            workspace_display,
            session_display,
            resumed,
            startup_confirmation_required,
        )
        .await;
    }

    if let Some(prompt) = prompt {
        let interrupted = Arc::new(AtomicBool::new(false));
        let mut renderer = PromptRenderer::new(use_color, interrupted.clone());
        install_ctrlc_handler(interrupted)?;

        if startup_confirmation_required && !confirm_manual_agent(&mut renderer, &agent_id)? {
            println!("Aborted.");
            return Ok(());
        }

        let client = Anthropic::new(None).map_err(|err| {
            cli_error(
                "client_init_failed",
                "failed to initialize the Anthropic client",
            )
            .with_string_field("cause", &err.to_string())
        })?;
        let auto_compact_tokens = agent.auto_compact_tokens();
        let chat = ChatSession::with_agent(client.clone(), agent);
        let mut session = SidRuntimeSession::new(
            client,
            chat,
            config,
            workspace_root.clone(),
            config_root.clone(),
            sid_session.clone(),
            agent_id.clone(),
            auto_compact_tokens,
        );
        session.load_resumed_transcript(resumed)?;
        return run_prompt_mode(session, prompt, &mut renderer).await;
    }

    let interrupted = Arc::new(AtomicBool::new(false));
    let mut terminal = SidTerminal::new(use_color, interrupted.clone())?;
    let context = ();

    if startup_confirmation_required && !confirm_manual_agent(&mut terminal, &agent_id)? {
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
    let auto_compact_tokens = agent.auto_compact_tokens();
    let chat = ChatSession::with_agent(client.clone(), agent);
    let mut session = SidRuntimeSession::new(
        client,
        chat,
        config,
        workspace_root.clone(),
        config_root.clone(),
        sid_session.clone(),
        agent_id.clone(),
        auto_compact_tokens,
    );
    session.load_resumed_transcript(resumed)?;

    let interrupted_clone = interrupted.clone();
    install_ctrlc_handler(interrupted_clone)?;

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
                            } else {
                                maybe_auto_compact(&mut session, &mut terminal, &context).await;
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
                                        "Extended thinking enabled with {tokens} token budget."
                                    ),
                                ),
                                None => {
                                    terminal.print_info(&context, "Extended thinking disabled.");
                                }
                            }
                        }
                        ChatCommand::ThinkingAdaptive => {
                            let effort = session.config().effort();
                            session.set_thinking_adaptive(effort);
                            terminal.print_info(&context, "Adaptive thinking enabled.");
                        }
                        ChatCommand::Effort(effort) => {
                            session.set_effort(Some(effort));
                            let label = match effort {
                                Effort::Low => "low",
                                Effort::Medium => "medium",
                                Effort::High => "high",
                            };
                            terminal.print_info(&context, &format!("Effort level set to {label}."));
                        }
                        ChatCommand::ClearEffort => {
                            session.set_effort(None);
                            terminal.print_info(&context, "Effort level cleared.");
                        }
                        ChatCommand::Spend(dollars) => {
                            session.set_session_spend(Some(dollars));
                            terminal.print_info(
                                &context,
                                &format!("Session spend limit set to ${dollars:.2}."),
                            );
                        }
                        ChatCommand::ClearSpend => {
                            session.set_session_spend(None);
                            terminal.print_info(&context, "Session spend limit cleared.");
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
                } else {
                    maybe_auto_compact(&mut session, &mut terminal, &context).await;
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

fn run_connect_mode(setup: ConnectSetup) -> Result<(), SError> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let mut terminal = SidTerminal::new(setup.use_color, interrupted.clone())?;
    install_ctrlc_handler(interrupted.clone())?;

    let mut client = RawTerminalClient::connect(&setup.spec, interrupted.clone())?;
    let hello = client.read_hello(&mut terminal)?;
    if !hello.sandbox_available {
        eprint!("{SANDBOX_UNAVAILABLE_WARNING}");
    }

    println!(
        "sid (agent: {}, model: {})",
        hello.current_agent, hello.model
    );
    println!("workspace: {}", hello.workspace_root);
    println!("session: {}", hello.session_dir);
    println!("connected: {}", setup.spec);
    println!("Type /help for commands, /quit to exit\n");

    if hello.startup_confirmation_required {
        let result = client.read_until_result("startup", &mut terminal)?;
        if !result.ok {
            terminal.print_error(&(), &raw_result_message(&result));
            return Ok(());
        }
    }
    client.drain_replay(&mut terminal)?;

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
                            client.send_user_turn(content, &mut terminal)?;
                        }
                        Ok(None) => {
                            terminal.print_info(&(), "Editor returned empty; nothing sent.");
                        }
                        Err(err) => {
                            terminal.print_error(&(), &format!("Failed to invoke editor: {err}"));
                        }
                    }
                    continue;
                }

                if let Some(cmd) = parse_sid_command(line) {
                    client.handle_sid_command(cmd, &mut terminal)?;
                    continue;
                }

                if let Some(cmd) = parse_command(line) {
                    if client.handle_chat_command(cmd, &mut terminal)? {
                        break;
                    }
                    continue;
                }

                client.send_user_turn(line, &mut terminal)?;
            }
            Ok(OperatorLine::Interrupted) => {
                println!();
                client.shutdown_server(&mut terminal)?;
                break;
            }
            Ok(OperatorLine::Eof) => {
                println!("\nGoodbye!");
                break;
            }
            Err(err) => {
                terminal.print_error(&(), &format!("Input error: {err}"));
                break;
            }
        }
    }

    Ok(())
}

async fn run_prompt_mode(
    mut session: SidRuntimeSession,
    prompt: String,
    renderer: &mut dyn Renderer,
) -> Result<(), SError> {
    session
        .send_message(claudius::MessageParam::user(prompt), renderer)
        .await
        .map_err(|err| cli_error("send_message_failed", &err.to_string()))?;
    let response = session.last_assistant_text().ok_or_else(|| {
        cli_error(
            "empty_response",
            "assistant produced no text response for --prompt",
        )
    })?;
    maybe_auto_compact(&mut session, renderer, &()).await;

    print!("{response}");
    if !response.ends_with('\n') {
        println!();
    }
    io::stdout().flush().map_err(|err| {
        cli_error("io_error", "failed to flush prompt output")
            .with_string_field("cause", &err.to_string())
    })
}

struct RawTerminalClient {
    connection: sid_isnt_done::raw_listen::RawConnection,
    interrupted: Arc<AtomicBool>,
    next_request_id: u64,
    current_agent: Option<String>,
    replay_complete: bool,
}

impl RawTerminalClient {
    fn connect(spec: &str, interrupted: Arc<AtomicBool>) -> Result<Self, SError> {
        let connection = sid_isnt_done::raw_listen::RawConnection::connect(spec)
            .map_err(|err| raw_io_error("failed to connect raw listener", &err))?;
        Ok(Self {
            connection,
            interrupted,
            next_request_id: 1,
            current_agent: None,
            replay_complete: false,
        })
    }

    fn read_hello(&mut self, terminal: &mut SidTerminal) -> Result<RawHello, SError> {
        loop {
            match self.read_message()? {
                RawServerMessage::Hello(hello) => {
                    self.current_agent = Some(hello.current_agent.clone());
                    return Ok(hello);
                }
                message => {
                    self.handle_server_message(message, terminal, None)?;
                }
            }
        }
    }

    fn drain_replay(&mut self, terminal: &mut SidTerminal) -> Result<(), SError> {
        let mut resumed_request_id = None;
        while !self.replay_complete {
            match self.read_message()? {
                RawServerMessage::ReplayComplete(_) => {
                    self.replay_complete = true;
                }
                RawServerMessage::Hello(hello) => {
                    self.current_agent = Some(hello.current_agent);
                }
                RawServerMessage::Prompt(prompt) => {
                    resumed_request_id.get_or_insert_with(|| prompt.request_id.clone());
                    self.answer_prompt(prompt, terminal)?;
                }
                RawServerMessage::Event(event) => {
                    render_raw_event(event.event, terminal)?;
                }
                RawServerMessage::PromptAck(_) | RawServerMessage::Result(_) => {}
            }
        }
        if let Some(request_id) = resumed_request_id {
            let result = self.read_until_result(&request_id, terminal)?;
            if !result.ok {
                terminal.print_error(&(), &raw_result_message(&result));
            }
        }
        Ok(())
    }

    fn send_user_turn(&mut self, text: &str, terminal: &mut SidTerminal) -> Result<(), SError> {
        let result = self.send_request(
            "turn",
            RawRequest::UserTurn {
                text: text.to_string(),
            },
            terminal,
        )?;
        if !result.ok {
            terminal.print_error(&(), &raw_result_message(&result));
        } else if let Some(data) = result.data.as_ref() {
            self.update_identity(data);
        }
        Ok(())
    }

    fn shutdown_server(&mut self, terminal: &mut SidTerminal) -> Result<(), SError> {
        let result = self.send_request("shutdown", RawRequest::Shutdown, terminal)?;
        if !result.ok {
            terminal.print_error(&(), &raw_result_message(&result));
        }
        Ok(())
    }

    fn handle_sid_command(
        &mut self,
        cmd: SidCommand,
        terminal: &mut SidTerminal,
    ) -> Result<(), SError> {
        match cmd {
            SidCommand::ShowAgent => {
                if let Some(data) =
                    self.send_request_data("agent", RawRequest::ShowAgent, terminal)?
                {
                    print_remote_agent_summary(&data);
                }
            }
            SidCommand::AgentList => {
                if let Some(data) =
                    self.send_request_data("agents", RawRequest::ListAgents, terminal)?
                {
                    print_remote_agent_list(&data);
                }
            }
            SidCommand::SwitchAgent(agent) => {
                let already_current = self.current_agent.as_deref() == Some(agent.as_str());
                if let Some(data) = self.send_request_data(
                    "switch-agent",
                    RawRequest::SwitchAgent {
                        agent: agent.clone(),
                    },
                    terminal,
                )? {
                    if already_current {
                        terminal.print_info(&(), &format!("Already using agent: {agent}"));
                    } else {
                        terminal.print_info(
                            &(),
                            &format!("Switched to agent: {}", format_remote_agent_label(&data)),
                        );
                    }
                }
            }
            SidCommand::Compact => {
                if let Some(data) =
                    self.send_request_data("compact", RawRequest::Compact, terminal)?
                {
                    terminal.print_info(
                        &(),
                        &format!(
                            "Compacted session {} into {} ({})",
                            json_str(&data, "parent_session_id").unwrap_or("?"),
                            json_str(&data, "new_session_id").unwrap_or("?"),
                            json_str(&data, "new_session_root").unwrap_or("?")
                        ),
                    );
                }
            }
            SidCommand::Invalid(message) => {
                terminal.print_error(&(), &message);
            }
        }
        Ok(())
    }

    fn handle_chat_command(
        &mut self,
        cmd: ChatCommand,
        terminal: &mut SidTerminal,
    ) -> Result<bool, SError> {
        match cmd {
            ChatCommand::Quit => {
                println!("Goodbye!");
                return Ok(true);
            }
            ChatCommand::Clear => {
                if self
                    .send_request_data("clear", RawRequest::Clear, terminal)?
                    .is_some()
                {
                    terminal.print_info(&(), "Conversation cleared.");
                }
            }
            ChatCommand::Help => {
                print_help();
            }
            ChatCommand::Model(model_name) => {
                if self
                    .send_request_data(
                        "model",
                        RawRequest::SetModel {
                            model: model_name.clone(),
                        },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(&(), &format!("Model changed to: {model_name}"));
                }
            }
            ChatCommand::System(prompt) => {
                if self
                    .send_request_data(
                        "system",
                        RawRequest::SetSystemPrompt {
                            prompt: prompt.clone(),
                        },
                        terminal,
                    )?
                    .is_some()
                {
                    match prompt {
                        Some(prompt) => {
                            terminal.print_info(&(), &format!("System prompt set to: {prompt}"))
                        }
                        None => terminal.print_info(&(), "System prompt cleared."),
                    }
                }
            }
            ChatCommand::MaxTokens(value) => {
                if self
                    .send_request_data(
                        "max-tokens",
                        RawRequest::SetMaxTokens { max_tokens: value },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(&(), &format!("max_tokens set to {value}"));
                }
            }
            ChatCommand::Temperature(value) => {
                if self
                    .send_request_data(
                        "temperature",
                        RawRequest::SetTemperature {
                            temperature: Some(value),
                        },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(&(), &format!("temperature set to {value:.2}"));
                }
            }
            ChatCommand::ClearTemperature => {
                if self
                    .send_request_data(
                        "temperature",
                        RawRequest::SetTemperature { temperature: None },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(&(), "temperature reset to model default");
                }
            }
            ChatCommand::TopP(value) => {
                if self
                    .send_request_data(
                        "top-p",
                        RawRequest::SetTopP { top_p: Some(value) },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(&(), &format!("top_p set to {value:.2}"));
                }
            }
            ChatCommand::ClearTopP => {
                if self
                    .send_request_data("top-p", RawRequest::SetTopP { top_p: None }, terminal)?
                    .is_some()
                {
                    terminal.print_info(&(), "top_p reset to model default");
                }
            }
            ChatCommand::TopK(value) => {
                if self
                    .send_request_data(
                        "top-k",
                        RawRequest::SetTopK { top_k: Some(value) },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(&(), &format!("top_k set to {value}"));
                }
            }
            ChatCommand::ClearTopK => {
                if self
                    .send_request_data("top-k", RawRequest::SetTopK { top_k: None }, terminal)?
                    .is_some()
                {
                    terminal.print_info(&(), "top_k reset to model default");
                }
            }
            ChatCommand::AddStopSequence(sequence) => {
                if self
                    .send_request_data(
                        "stop-sequence",
                        RawRequest::AddStopSequence {
                            sequence: sequence.clone(),
                        },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(&(), &format!("Added stop sequence: {sequence}"));
                }
            }
            ChatCommand::ClearStopSequences => {
                if self
                    .send_request_data("stop-sequences", RawRequest::ClearStopSequences, terminal)?
                    .is_some()
                {
                    terminal.print_info(&(), "Stop sequences cleared.");
                }
            }
            ChatCommand::ListStopSequences => {
                if let Some(data) = self.send_request_data(
                    "stop-sequences",
                    RawRequest::ListStopSequences,
                    terminal,
                )? {
                    print_stop_sequences(&json_string_array(&data, "stop_sequences"));
                }
            }
            ChatCommand::Thinking(budget) => {
                if self
                    .send_request_data(
                        "thinking",
                        RawRequest::SetThinkingBudget { tokens: budget },
                        terminal,
                    )?
                    .is_some()
                {
                    match budget {
                        Some(tokens) => terminal.print_info(
                            &(),
                            &format!("Extended thinking enabled with {tokens} token budget."),
                        ),
                        None => terminal.print_info(&(), "Extended thinking disabled."),
                    }
                }
            }
            ChatCommand::ThinkingAdaptive => {
                if self
                    .send_request_data("thinking", RawRequest::SetThinkingAdaptive, terminal)?
                    .is_some()
                {
                    terminal.print_info(&(), "Adaptive thinking enabled.");
                }
            }
            ChatCommand::Effort(effort) => {
                if self
                    .send_request_data(
                        "effort",
                        RawRequest::SetEffort {
                            effort: Some(effort),
                        },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(
                        &(),
                        &format!("Effort level set to {}.", effort_name(effort)),
                    );
                }
            }
            ChatCommand::ClearEffort => {
                if self
                    .send_request_data("effort", RawRequest::SetEffort { effort: None }, terminal)?
                    .is_some()
                {
                    terminal.print_info(&(), "Effort level cleared.");
                }
            }
            ChatCommand::Spend(dollars) => {
                if self
                    .send_request_data(
                        "spend",
                        RawRequest::SetSpend {
                            dollars: Some(dollars),
                        },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(&(), &format!("Session spend limit set to ${dollars:.2}."));
                }
            }
            ChatCommand::ClearSpend => {
                if self
                    .send_request_data("spend", RawRequest::SetSpend { dollars: None }, terminal)?
                    .is_some()
                {
                    terminal.print_info(&(), "Session spend limit cleared.");
                }
            }
            ChatCommand::Caching(enabled) => {
                if self
                    .send_request_data("caching", RawRequest::SetCaching { enabled }, terminal)?
                    .is_some()
                {
                    if enabled {
                        terminal.print_info(&(), "Prompt caching enabled.");
                    } else {
                        terminal.print_info(&(), "Prompt caching disabled.");
                    }
                }
            }
            ChatCommand::TranscriptPath(_) | ChatCommand::ClearTranscriptPath => {
                terminal.print_info(
                    &(),
                    "Transcript auto-save is managed by the session system.",
                );
            }
            ChatCommand::SaveTranscript(path) => {
                if self
                    .send_request_data(
                        "save-transcript",
                        RawRequest::SaveTranscript { path: path.clone() },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(&(), &format!("Transcript saved to {path}"));
                }
            }
            ChatCommand::LoadTranscript(path) => {
                if self
                    .send_request_data(
                        "load-transcript",
                        RawRequest::LoadTranscript { path: path.clone() },
                        terminal,
                    )?
                    .is_some()
                {
                    terminal.print_info(&(), &format!("Transcript loaded from {path}"));
                }
            }
            ChatCommand::Stats => {
                if let Some(data) = self.send_request_data("stats", RawRequest::Stats, terminal)? {
                    print_remote_stats(&data);
                }
            }
            ChatCommand::ShowConfig => {
                if let Some(data) =
                    self.send_request_data("config", RawRequest::ShowConfig, terminal)?
                {
                    print_remote_config(&data);
                }
            }
            ChatCommand::Invalid(message) => {
                terminal.print_error(&(), &message);
            }
        }
        Ok(false)
    }

    fn send_request_data(
        &mut self,
        prefix: &str,
        request: RawRequest,
        terminal: &mut SidTerminal,
    ) -> Result<Option<Value>, SError> {
        let result = self.send_request(prefix, request, terminal)?;
        if result.ok {
            if let Some(data) = result.data.as_ref() {
                self.update_identity(data);
            }
            Ok(result.data)
        } else {
            terminal.print_error(&(), &raw_result_message(&result));
            Ok(None)
        }
    }

    fn send_request(
        &mut self,
        prefix: &str,
        request: RawRequest,
        terminal: &mut SidTerminal,
    ) -> Result<RawResultEnvelope, SError> {
        let request_id = self.next_request_id(prefix);
        self.interrupted.store(false, Ordering::Relaxed);
        self.write_request(&request_id, request)?;
        let _interrupt_watcher = self.start_interrupt_watcher(&request_id);
        self.read_until_result(&request_id, terminal)
    }

    fn start_interrupt_watcher(&self, request_id: &str) -> RawInterruptWatcher {
        let writer = self.connection.writer_handle();
        let interrupted = self.interrupted.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let request_id = format!("{request_id}:interrupt");
        let handle = thread::spawn(move || {
            while !stop_thread.load(Ordering::Relaxed) {
                if interrupted.swap(false, Ordering::Relaxed) {
                    let _ = writer.write_request(&RawRequestEnvelope {
                        protocol_version: RAW_PROTOCOL_VERSION,
                        request_id,
                        request: RawRequest::Interrupt,
                    });
                    break;
                }
                thread::sleep(Duration::from_millis(25));
            }
        });
        RawInterruptWatcher {
            stop,
            handle: Some(handle),
        }
    }

    fn read_until_result(
        &mut self,
        request_id: &str,
        terminal: &mut SidTerminal,
    ) -> Result<RawResultEnvelope, SError> {
        loop {
            let message = self.read_message()?;
            if let Some(result) = self.handle_server_message(message, terminal, Some(request_id))? {
                return Ok(result);
            }
        }
    }

    fn handle_server_message(
        &mut self,
        message: RawServerMessage,
        terminal: &mut SidTerminal,
        awaited_request_id: Option<&str>,
    ) -> Result<Option<RawResultEnvelope>, SError> {
        match message {
            RawServerMessage::Hello(hello) => {
                self.current_agent = Some(hello.current_agent);
                Ok(None)
            }
            RawServerMessage::ReplayComplete(_) => {
                self.replay_complete = true;
                Ok(None)
            }
            RawServerMessage::Event(event) => {
                render_raw_event(event.event, terminal)?;
                Ok(None)
            }
            RawServerMessage::Prompt(prompt) => {
                self.answer_prompt(prompt, terminal)?;
                Ok(None)
            }
            RawServerMessage::PromptAck(_) => Ok(None),
            RawServerMessage::Result(result) => {
                if awaited_request_id == Some(result.request_id.as_str()) {
                    return Ok(Some(result));
                }
                if !result.ok {
                    terminal.print_error(&(), &raw_result_message(&result));
                }
                Ok(None)
            }
        }
    }

    fn answer_prompt(
        &mut self,
        prompt: RawPrompt,
        terminal: &mut SidTerminal,
    ) -> Result<(), SError> {
        loop {
            let input = terminal
                .read_operator_line(&prompt.message)
                .map_err(|err| {
                    cli_error("io_error", "failed to read raw prompt response")
                        .with_string_field("prompt_id", &prompt.prompt_id)
                        .with_string_field("cause", &err.to_string())
                })?;
            let response = match input {
                Some(OperatorLine::Line(input)) if prompt.kind == "confirmation" => {
                    match parse_confirmation(&input) {
                        Some(true) => "yes".to_string(),
                        Some(false) => "no".to_string(),
                        None => {
                            println!("Please answer yes or no.");
                            continue;
                        }
                    }
                }
                Some(OperatorLine::Line(input)) => input,
                Some(OperatorLine::Interrupted) | Some(OperatorLine::Eof) | None => prompt
                    .choices
                    .iter()
                    .find(|choice| choice.eq_ignore_ascii_case("no"))
                    .cloned()
                    .or_else(|| prompt.choices.last().cloned())
                    .unwrap_or_default(),
            };
            let request_id = self.next_request_id("prompt");
            self.write_request(
                &request_id,
                RawRequest::PromptResponse {
                    prompt_id: prompt.prompt_id,
                    response,
                },
            )?;
            return Ok(());
        }
    }

    fn next_request_id(&mut self, prefix: &str) -> String {
        let request_id = format!("sid-connect-{prefix}-{}", self.next_request_id);
        self.next_request_id = self.next_request_id.saturating_add(1);
        request_id
    }

    fn read_message(&mut self) -> Result<RawServerMessage, SError> {
        self.connection
            .read_message()
            .map_err(|err| raw_io_error("failed to read raw server message", &err))?
            .ok_or_else(|| cli_error("io_error", "raw connection closed"))
    }

    fn write_request(&mut self, request_id: &str, request: RawRequest) -> Result<(), SError> {
        self.connection
            .write_request(&RawRequestEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                request_id: request_id.to_string(),
                request,
            })
            .map_err(|err| raw_io_error("failed to write raw request", &err))
    }

    fn update_identity(&mut self, data: &Value) {
        if let Some(agent) = json_str(data, "current_agent").or_else(|| json_str(data, "id")) {
            self.current_agent = Some(agent.to_string());
        }
    }
}

struct RawInterruptWatcher {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Drop for RawInterruptWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct RemoteStreamContext {
    label: Option<String>,
    depth: usize,
}

impl StreamContext for RemoteStreamContext {
    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    fn depth(&self) -> usize {
        self.depth
    }
}

fn render_raw_event(event: RawEvent, terminal: &mut SidTerminal) -> Result<(), SError> {
    match event {
        RawEvent::AgentStart { label, depth } => {
            let context = RemoteStreamContext { label, depth };
            terminal.start_agent(&context);
        }
        RawEvent::AgentFinish {
            label,
            depth,
            stop_reason,
        } => {
            let context = RemoteStreamContext { label, depth };
            let stop_reason = stop_reason.as_deref().and_then(parse_raw_stop_reason);
            terminal.finish_agent(&context, stop_reason.as_ref());
        }
        RawEvent::AssistantTextDelta { label, depth, text } => {
            let context = RemoteStreamContext { label, depth };
            terminal.print_text(&context, &text);
        }
        RawEvent::ThinkingDelta { label, depth, text } => {
            let context = RemoteStreamContext { label, depth };
            terminal.print_thinking(&context, &text);
        }
        RawEvent::Info {
            label,
            depth,
            message,
        } => {
            let context = RemoteStreamContext { label, depth };
            terminal.print_info(&context, &message);
        }
        RawEvent::Error {
            label,
            depth,
            message,
        } => {
            let context = RemoteStreamContext { label, depth };
            terminal.print_error(&context, &message);
        }
        RawEvent::ToolUseStart {
            label,
            depth,
            name,
            tool_use_id,
        } => {
            let context = RemoteStreamContext { label, depth };
            terminal.start_tool_use(&context, &name, &tool_use_id);
        }
        RawEvent::ToolInputDelta {
            label,
            depth,
            partial_json,
        } => {
            let context = RemoteStreamContext { label, depth };
            terminal.print_tool_input(&context, &partial_json);
        }
        RawEvent::ToolUseEnd { label, depth } => {
            let context = RemoteStreamContext { label, depth };
            terminal.finish_tool_use(&context);
        }
        RawEvent::ToolResultStart {
            label,
            depth,
            tool_use_id,
            is_error,
        } => {
            let context = RemoteStreamContext { label, depth };
            terminal.start_tool_result(&context, &tool_use_id, is_error);
        }
        RawEvent::ToolResultTextDelta { label, depth, text } => {
            let context = RemoteStreamContext { label, depth };
            terminal.print_tool_result_text(&context, &text);
        }
        RawEvent::ToolResultEnd { label, depth } => {
            let context = RemoteStreamContext { label, depth };
            terminal.finish_tool_result(&context);
        }
        RawEvent::ResponseFinish { label, depth } => {
            let context = RemoteStreamContext { label, depth };
            terminal.finish_response(&context);
        }
        RawEvent::Interrupted { label, depth } => {
            let context = RemoteStreamContext { label, depth };
            terminal.print_interrupted(&context);
        }
        RawEvent::ToolOutput {
            stream,
            text,
            data_b64,
            ..
        } => {
            write_raw_tool_output(&stream, text.as_deref(), data_b64.as_deref())?;
        }
    }
    Ok(())
}

fn parse_raw_stop_reason(value: &str) -> Option<StopReason> {
    value.parse::<StopReason>().ok().or(match value {
        "EndTurn" => Some(StopReason::EndTurn),
        "MaxTokens" => Some(StopReason::MaxTokens),
        "StopSequence" => Some(StopReason::StopSequence),
        "ToolUse" => Some(StopReason::ToolUse),
        "PauseTurn" => Some(StopReason::PauseTurn),
        "Refusal" => Some(StopReason::Refusal),
        "ModelContextWindowExceeded" => Some(StopReason::ModelContextWindowExceeded),
        _ => None,
    })
}

fn write_raw_tool_output(
    stream: &str,
    text: Option<&str>,
    data_b64: Option<&str>,
) -> Result<(), SError> {
    let bytes = match (text, data_b64) {
        (Some(text), _) => text.as_bytes().to_vec(),
        (None, Some(data_b64)) => BASE64_STANDARD.decode(data_b64).map_err(|err| {
            cli_error(
                "invalid_raw_tool_output",
                "failed to decode raw tool output",
            )
            .with_string_field("cause", &err.to_string())
        })?,
        (None, None) => Vec::new(),
    };
    if stream == "stderr" {
        let mut stderr = io::stderr();
        stderr.write_all(&bytes).map_err(|err| {
            cli_error("io_error", "failed to write raw tool stderr")
                .with_string_field("cause", &err.to_string())
        })?;
        stderr.flush().map_err(|err| {
            cli_error("io_error", "failed to flush raw tool stderr")
                .with_string_field("cause", &err.to_string())
        })?;
    } else {
        let mut stdout = io::stdout();
        stdout.write_all(&bytes).map_err(|err| {
            cli_error("io_error", "failed to write raw tool stdout")
                .with_string_field("cause", &err.to_string())
        })?;
        stdout.flush().map_err(|err| {
            cli_error("io_error", "failed to flush raw tool stdout")
                .with_string_field("cause", &err.to_string())
        })?;
    }
    Ok(())
}

fn raw_result_message(result: &RawResultEnvelope) -> String {
    result
        .error
        .as_ref()
        .map(|error| error.message.clone())
        .unwrap_or_else(|| "raw request failed".to_string())
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
        println!("    {line}");
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

/// Trigger compaction automatically when the output token threshold is reached.
async fn maybe_auto_compact(
    session: &mut SidRuntimeSession,
    terminal: &mut dyn Renderer,
    context: &(),
) {
    if !session.should_auto_compact() {
        return;
    }
    terminal.print_info(
        context,
        &format!(
            "Auto-compacting: output token threshold ({}) reached.",
            session.auto_compact_tokens.unwrap_or(0),
        ),
    );
    match session.compact().await {
        Ok(result) => terminal.print_info(
            context,
            &format!(
                "Compacted session {} into {} ({})",
                result.parent_session_id, result.new_session_id, result.new_session_root
            ),
        ),
        Err(err) => terminal.print_error(context, &format!("Auto-compaction failed: {err}")),
    }
}

async fn run_raw_session(
    mut session: SidRuntimeSession,
    mut server: RawServer<impl RawInput, impl Write + Send + 'static>,
    workspace_display: String,
    session_display: String,
    resumed: bool,
    startup_confirmation_required: bool,
) -> Result<(), SError> {
    server
        .write_message(&RawServerMessage::Hello(RawHello {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
            session_id: session.sid_session.id().to_string(),
            session_dir: session_display,
            workspace_root: workspace_display,
            current_agent: session.current_agent_id().to_string(),
            model: session.config().model().to_string(),
            resumed,
            startup_confirmation_required,
            sandbox_available: sid_isnt_done::seatbelt::sandbox_available(),
        }))
        .map_err(|err| raw_io_error("failed to write raw hello", &err))?;

    if startup_confirmation_required {
        server.set_request_id("startup".to_string());
        let allowed = confirm_manual_agent(&mut server, session.current_agent_id())?;
        if allowed {
            server
                .write_ok_result("startup", Some(json!({ "started": true })))
                .map_err(|err| raw_io_error("failed to write raw startup result", &err))?;
        } else {
            server
                .write_error_result(
                    "startup",
                    Some("manual_agent_denied"),
                    "manual agent start denied",
                )
                .map_err(|err| raw_io_error("failed to write raw startup error", &err))?;
            server.clear_request_id();
            return Ok(());
        }
        server.clear_request_id();
    }

    while let Some(request) = server
        .read_request()
        .map_err(|err| raw_io_error("failed to read raw request", &err))?
    {
        let request_id = request.request_id.clone();
        if request.protocol_version != RAW_PROTOCOL_VERSION {
            server
                .write_error_result(
                    &request_id,
                    Some("unsupported_protocol_version"),
                    &format!(
                        "unsupported raw protocol version {}",
                        request.protocol_version
                    ),
                )
                .map_err(|err| raw_io_error("failed to write raw protocol error", &err))?;
            continue;
        }
        server.set_request_id(request_id.clone());
        let usage_observer = Arc::new(RawUsageReportObserver::new(server.output(), &request_id));
        let _usage_observer = install_usage_report_observer(Some(usage_observer));
        let result = handle_raw_request(&mut session, &mut server, request).await;
        server.clear_request_id();
        match result {
            Ok(RequestDisposition::Continue(data)) => {
                server
                    .write_ok_result(&request_id, data)
                    .map_err(|err| raw_io_error("failed to write raw result", &err))?;
            }
            Ok(RequestDisposition::Shutdown(data)) => {
                server
                    .write_ok_result(&request_id, data)
                    .map_err(|err| raw_io_error("failed to write raw shutdown result", &err))?;
                break;
            }
            Err(err) => {
                server
                    .write_error_result(&request_id, None, &err.to_string())
                    .map_err(|io_err| raw_io_error("failed to write raw error result", &io_err))?;
            }
        }
    }

    Ok(())
}

enum RequestDisposition {
    Continue(Option<Value>),
    Shutdown(Option<Value>),
}

async fn handle_raw_request<R, W>(
    session: &mut SidRuntimeSession,
    server: &mut RawServer<R, W>,
    request: RawRequestEnvelope,
) -> Result<RequestDisposition, SError>
where
    R: RawInput,
    W: std::io::Write + Send + 'static,
{
    match request.request {
        RawRequest::UserTurn { text } => {
            let observer = Arc::new(RawToolOutputObserver::new(server.output()));
            let _observer = install_tool_output_observer(Some(observer));
            session
                .send_message(claudius::MessageParam::user(text), server)
                .await
                .map_err(|err| cli_error("send_message_failed", &err.to_string()))?;
            maybe_auto_compact(session, server, &()).await;
            Ok(RequestDisposition::Continue(Some(session_identity_json(
                session,
            ))))
        }
        RawRequest::PromptResponse { .. } => Err(cli_error(
            "unexpected_prompt_response",
            "prompt_response is only valid while the server is waiting on a prompt",
        )),
        RawRequest::Interrupt => Ok(RequestDisposition::Continue(None)),
        RawRequest::ShowAgent => Ok(RequestDisposition::Continue(Some(agent_summary_json(
            &session.current_agent_summary()?,
            session,
        )))),
        RawRequest::ListAgents => Ok(RequestDisposition::Continue(Some(json!({
            "agents": session
                .list_agents()?
                .iter()
                .map(agent_json)
                .collect::<Vec<_>>()
        })))),
        RawRequest::SwitchAgent { agent } => {
            match session.agent_summary(&agent)? {
                Some(summary)
                    if summary.enabled == SwitchPosition::Manual
                        && !confirm_manual_agent(server, &summary.id)? =>
                {
                    return Err(cli_error(
                        "manual_agent_denied",
                        "manual agent switch denied",
                    ));
                }
                _ => {}
            }
            session.switch_agent(&agent)?;
            Ok(RequestDisposition::Continue(Some(agent_summary_json(
                &session.current_agent_summary()?,
                session,
            ))))
        }
        RawRequest::Compact => {
            let result = session.compact().await?;
            Ok(RequestDisposition::Continue(Some(json!({
                "parent_session_id": result.parent_session_id,
                "new_session_id": result.new_session_id,
                "new_session_root": result.new_session_root,
                "current_agent": session.current_agent_id(),
            }))))
        }
        RawRequest::Clear => {
            session.clear()?;
            Ok(RequestDisposition::Continue(Some(session_identity_json(
                session,
            ))))
        }
        RawRequest::SetModel { model } => {
            session.set_model(&model);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::SetSystemPrompt { prompt } => {
            session.set_system_prompt(prompt);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::SetMaxTokens { max_tokens } => {
            session.set_max_tokens(max_tokens);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::SetTemperature { temperature } => {
            session.set_temperature(temperature);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::SetTopP { top_p } => {
            session.set_top_p(top_p);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::SetTopK { top_k } => {
            session.set_top_k(top_k);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::AddStopSequence { sequence } => {
            session.add_stop_sequence(sequence);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::ClearStopSequences => {
            session.clear_stop_sequences();
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::ListStopSequences => Ok(RequestDisposition::Continue(Some(json!({
            "stop_sequences": session.config().stop_sequences(),
        })))),
        RawRequest::SetThinkingBudget { tokens } => {
            session.set_thinking_budget(tokens);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::SetThinkingAdaptive => {
            session.set_thinking_adaptive(session.config().effort());
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::SetEffort { effort } => {
            session.set_effort(effort);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::SetSpend { dollars } => {
            session.set_session_spend(dollars);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::SetCaching { enabled } => {
            session.set_caching_enabled(enabled);
            Ok(RequestDisposition::Continue(Some(config_json(
                &session.stats(),
            ))))
        }
        RawRequest::SaveTranscript { path } => {
            session.save_transcript_to(&path)?;
            Ok(RequestDisposition::Continue(Some(json!({ "path": path }))))
        }
        RawRequest::LoadTranscript { path } => {
            session.load_transcript_from(&path)?;
            Ok(RequestDisposition::Continue(Some(session_identity_json(
                session,
            ))))
        }
        RawRequest::Stats => Ok(RequestDisposition::Continue(Some(stats_json(
            &session.stats(),
        )))),
        RawRequest::ShowConfig => Ok(RequestDisposition::Continue(Some(config_json(
            &session.stats(),
        )))),
        RawRequest::Shutdown => Ok(RequestDisposition::Shutdown(Some(json!({
            "session_id": session.sid_session.id(),
        })))),
    }
}

fn agent_summary_json(summary: &AgentSummary, session: &SidRuntimeSession) -> Value {
    let mut agent = agent_json(summary);
    if let Some(object) = agent.as_object_mut() {
        object.insert(
            "model".to_string(),
            json!(session.config().model().to_string()),
        );
    }
    agent
}

fn agent_json(summary: &AgentSummary) -> Value {
    json!({
        "id": summary.id,
        "display_name": summary.display_name,
        "description": summary.description,
        "enabled": describe_agent_enabled(summary.enabled),
        "current": summary.current,
    })
}

fn session_identity_json(session: &SidRuntimeSession) -> Value {
    json!({
        "session_id": session.sid_session.id(),
        "session_dir": session.sid_session.root().display().to_string(),
        "current_agent": session.current_agent_id(),
        "model": session.config().model().to_string(),
    })
}

fn stats_json(stats: &SessionStats) -> Value {
    json!({
        "model": stats.model.to_string(),
        "message_count": stats.message_count,
        "max_tokens": stats.max_tokens,
        "system_prompt": stats.system_prompt,
        "temperature": stats.temperature,
        "top_p": stats.top_p,
        "top_k": stats.top_k,
        "stop_sequences": stats.stop_sequences,
        "thinking_budget": stats.thinking_budget,
        "thinking": describe_thinking(stats),
        "thinking_adaptive": stats.thinking_adaptive,
        "effort": stats.effort.map(effort_name),
        "session_spend_micro_cents": stats.session_spend_micro_cents,
        "spend_used_micro_cents": stats.spend_used_micro_cents,
        "transcript_path": stats.transcript_path.as_ref().map(|path| path.display().to_string()),
        "total_input_tokens": stats.total_input_tokens,
        "total_output_tokens": stats.total_output_tokens,
        "total_requests": stats.total_requests,
        "last_turn_input_tokens": stats.last_turn_input_tokens,
        "last_turn_output_tokens": stats.last_turn_output_tokens,
        "caching_enabled": stats.caching_enabled,
        "total_cache_creation_tokens": stats.total_cache_creation_tokens,
        "total_cache_read_tokens": stats.total_cache_read_tokens,
    })
}

fn config_json(stats: &SessionStats) -> Value {
    json!({
        "model": stats.model.to_string(),
        "max_tokens": stats.max_tokens,
        "system_prompt": stats.system_prompt,
        "temperature": stats.temperature,
        "top_p": stats.top_p,
        "top_k": stats.top_k,
        "stop_sequences": stats.stop_sequences,
        "thinking": describe_thinking(stats),
        "thinking_budget": stats.thinking_budget,
        "thinking_adaptive": stats.thinking_adaptive,
        "effort": stats.effort.map(effort_name),
        "caching_enabled": stats.caching_enabled,
        "transcript_path": stats.transcript_path.as_ref().map(|path| path.display().to_string()),
    })
}

fn effort_name(effort: Effort) -> &'static str {
    match effort {
        Effort::Low => "low",
        Effort::Medium => "medium",
        Effort::High => "high",
    }
}

fn raw_io_error(context: &str, err: &io::Error) -> SError {
    cli_error("io_error", context).with_string_field("cause", &err.to_string())
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

fn print_remote_agent_summary(agent: &Value) {
    println!("    Agent: {}", format_remote_agent_label(agent));
    println!(
        "      Status: {}{}",
        json_str(agent, "enabled").unwrap_or("?"),
        if json_bool(agent, "current").unwrap_or(false) {
            " (current)"
        } else {
            ""
        }
    );
    if let Some(model) = json_str(agent, "model") {
        println!("      Model: {model}");
    }
    if let Some(description) = json_str(agent, "description") {
        println!("      Description: {description}");
    }
}

fn print_remote_agent_list(data: &Value) {
    println!("    Agents:");
    let Some(agents) = data.get("agents").and_then(Value::as_array) else {
        println!("      (unavailable)");
        return;
    };
    for agent in agents {
        let marker = if json_bool(agent, "current").unwrap_or(false) {
            "*"
        } else {
            " "
        };
        let mut line = format!(
            "      {marker} {} [{}]",
            format_remote_agent_label(agent),
            json_str(agent, "enabled").unwrap_or("?")
        );
        if let Some(description) = json_str(agent, "description") {
            line.push_str(&format!(" - {description}"));
        }
        println!("{line}");
    }
}

fn format_remote_agent_label(agent: &Value) -> String {
    match (
        json_str(agent, "id"),
        json_str(agent, "display_name").filter(|name| !name.is_empty()),
    ) {
        (Some(id), Some(name)) => format!("{id} ({name})"),
        (Some(id), None) => id.to_string(),
        (None, Some(name)) => name.to_string(),
        (None, None) => "?".to_string(),
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

fn validate_runtime_mode(
    raw: bool,
    bash_debug: Option<&str>,
    prompt: Option<&str>,
) -> Result<(), SError> {
    if raw && bash_debug.is_some() {
        return Err(cli_error(
            "invalid_cli_args",
            "--raw cannot be combined with --bash-debug",
        ));
    }
    if raw && prompt.is_some() {
        return Err(cli_error(
            "invalid_cli_args",
            "--raw cannot be combined with --prompt",
        ));
    }
    if bash_debug.is_some() && prompt.is_some() {
        return Err(cli_error(
            "invalid_cli_args",
            "--bash-debug cannot be combined with --prompt",
        ));
    }
    Ok(())
}

fn validate_connect_mode(
    raw: bool,
    listen: Option<&str>,
    resume: Option<&str>,
    bash_debug: Option<&str>,
    prompt: Option<&str>,
) -> Result<(), SError> {
    if raw {
        return Err(cli_error(
            "invalid_cli_args",
            "--connect cannot be combined with --raw",
        ));
    }
    if listen.is_some() {
        return Err(cli_error(
            "invalid_cli_args",
            "--connect cannot be combined with --listen",
        ));
    }
    if resume.is_some() {
        return Err(cli_error(
            "invalid_cli_args",
            "--connect cannot be combined with --resume",
        ));
    }
    if bash_debug.is_some() {
        return Err(cli_error(
            "invalid_cli_args",
            "--connect cannot be combined with --bash-debug",
        ));
    }
    if prompt.is_some() {
        return Err(cli_error(
            "invalid_cli_args",
            "--connect cannot be combined with --prompt",
        ));
    }
    Ok(())
}

fn install_ctrlc_handler(interrupted: Arc<AtomicBool>) -> Result<(), SError> {
    ctrlc::set_handler(move || {
        interrupted.store(true, Ordering::Relaxed);
    })
    .map_err(|err| {
        cli_error(
            "signal_handler_failed",
            "failed to install the Ctrl-C handler",
        )
        .with_string_field("cause", &err.to_string())
    })
}

fn warn_if_sandbox_unavailable() {
    if seatbelt::sandbox_available() {
        return;
    }

    eprint!("{SANDBOX_UNAVAILABLE_WARNING}");
}

fn resolve_sid_home() -> Result<Path<'static>, SError> {
    resolve_sid_home_from_env(std::env::var("SID_HOME"))
}

fn resolve_sid_home_from_env(
    sid_home: Result<String, std::env::VarError>,
) -> Result<Path<'static>, SError> {
    match sid_home {
        Ok(path) if !path.is_empty() => Ok(Path::new(&path).into_owned()),
        Ok(_) | Err(std::env::VarError::NotPresent) => Err(cli_error(
            "missing_sid_home",
            "SID_HOME must be set and non-empty",
        )),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(cli_error("invalid_sid_home", "SID_HOME is not valid UTF-8"))
        }
    }
}

fn confirm_manual_agent(renderer: &mut dyn Renderer, agent_id: &str) -> Result<bool, SError> {
    loop {
        let prompt = format!("Agent '{agent_id}' is MANUAL. Continue? [yes/no]: ");
        let input = match renderer
            .read_operator_line(&prompt)
            .map_err(|err| {
                cli_error("io_error", "failed to read manual-agent confirmation input")
                    .with_string_field("agent", agent_id)
                    .with_string_field("cause", &err.to_string())
            })?
            .ok_or_else(|| {
                cli_error(
                    "io_error",
                    "manual-agent confirmation requires an interactive renderer",
                )
                .with_string_field("agent", agent_id)
            })? {
            OperatorLine::Line(input) => input,
            OperatorLine::Eof | OperatorLine::Interrupted => return Ok(false),
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

fn describe_thinking(stats: &SessionStats) -> String {
    if stats.thinking_adaptive {
        match stats.effort {
            Some(Effort::Low) => "adaptive (effort: low)".to_string(),
            Some(Effort::Medium) => "adaptive (effort: medium)".to_string(),
            Some(Effort::High) => "adaptive (effort: high)".to_string(),
            None => "adaptive".to_string(),
        }
    } else {
        match stats.thinking_budget {
            Some(budget) => format!("enabled ({budget} tokens)"),
            None => "disabled".to_string(),
        }
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
        println!("      System prompt: {prompt}");
    } else {
        println!("      System prompt: (none)");
    }
    println!("      Thinking: {}", describe_thinking(stats));
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
    if let Some(limit) = stats.session_spend_micro_cents {
        let spent = stats.spend_used_micro_cents as f64 / 100_000_000.0;
        let total = limit as f64 / 100_000_000.0;
        let remaining = limit.saturating_sub(stats.spend_used_micro_cents) as f64 / 100_000_000.0;
        println!("      Spend limit: ${spent:.4}/${total:.2} (${remaining:.4} remaining)",);
    } else {
        println!("      Spend limit: (not set)");
    }
}

fn print_config(stats: &SessionStats) {
    println!("    Current Configuration:");
    println!("      Model: {}", stats.model);
    println!("      Max tokens: {}", stats.max_tokens);
    println!("      Temperature: {}", describe_float(stats.temperature));
    println!("      Top-p: {}", describe_float(stats.top_p));
    println!("      Top-k: {}", describe_top_k(stats.top_k));
    println!("      Thinking: {}", describe_thinking(stats));
    println!(
        "      Caching: {}",
        if stats.caching_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    if let Some(prompt) = stats.system_prompt.as_deref() {
        println!("      System prompt: {prompt}");
    } else {
        println!("      System prompt: (none)");
    }
    print_stop_sequences(&stats.stop_sequences);
}

fn print_remote_stats(stats: &Value) {
    println!("    Session Statistics:");
    println!("      Model: {}", json_str(stats, "model").unwrap_or("?"));
    println!(
        "      Messages: {}",
        json_u64(stats, "message_count").unwrap_or(0)
    );
    println!(
        "      Max tokens: {}",
        json_u64(stats, "max_tokens").unwrap_or(0)
    );
    println!(
        "      Temperature: {}",
        describe_json_float(stats, "temperature")
    );
    println!("      Top-p: {}", describe_json_float(stats, "top_p"));
    println!("      Top-k: {}", describe_json_u64(stats, "top_k"));
    if let Some(prompt) = json_str(stats, "system_prompt") {
        println!("      System prompt: {prompt}");
    } else {
        println!("      System prompt: (none)");
    }
    println!(
        "      Thinking: {}",
        json_str(stats, "thinking").unwrap_or("disabled")
    );
    print_stop_sequences(&json_string_array(stats, "stop_sequences"));
    println!(
        "      Total tokens: {} in / {} out ({} requests)",
        json_u64(stats, "total_input_tokens").unwrap_or(0),
        json_u64(stats, "total_output_tokens").unwrap_or(0),
        json_u64(stats, "total_requests").unwrap_or(0)
    );
    if json_bool(stats, "caching_enabled").unwrap_or(false) {
        println!(
            "      Cache tokens: {} created / {} read",
            json_u64(stats, "total_cache_creation_tokens").unwrap_or(0),
            json_u64(stats, "total_cache_read_tokens").unwrap_or(0)
        );
    }
    if let Some(input) = json_u64(stats, "last_turn_input_tokens") {
        let output = json_u64(stats, "last_turn_output_tokens").unwrap_or(0);
        println!("      Last turn tokens: {input} in / {output} out");
    }
    if let Some(limit) = json_u64(stats, "session_spend_micro_cents") {
        let spent = json_u64(stats, "spend_used_micro_cents").unwrap_or(0);
        let spent_dollars = spent as f64 / 100_000_000.0;
        let total = limit as f64 / 100_000_000.0;
        let remaining = limit.saturating_sub(spent) as f64 / 100_000_000.0;
        println!("      Spend limit: ${spent_dollars:.4}/${total:.2} (${remaining:.4} remaining)");
    } else {
        println!("      Spend limit: (not set)");
    }
}

fn print_remote_config(config: &Value) {
    println!("    Current Configuration:");
    println!("      Model: {}", json_str(config, "model").unwrap_or("?"));
    println!(
        "      Max tokens: {}",
        json_u64(config, "max_tokens").unwrap_or(0)
    );
    println!(
        "      Temperature: {}",
        describe_json_float(config, "temperature")
    );
    println!("      Top-p: {}", describe_json_float(config, "top_p"));
    println!("      Top-k: {}", describe_json_u64(config, "top_k"));
    println!(
        "      Thinking: {}",
        json_str(config, "thinking").unwrap_or("disabled")
    );
    println!(
        "      Caching: {}",
        if json_bool(config, "caching_enabled").unwrap_or(false) {
            "enabled"
        } else {
            "disabled"
        }
    );
    if let Some(prompt) = json_str(config, "system_prompt") {
        println!("      System prompt: {prompt}");
    } else {
        println!("      System prompt: (none)");
    }
    print_stop_sequences(&json_string_array(config, "stop_sequences"));
}

fn print_stop_sequences(stop_sequences: &[String]) {
    if stop_sequences.is_empty() {
        println!("      Stop sequences: (none)");
    } else {
        println!("      Stop sequences:");
        for stop_sequence in stop_sequences {
            println!("        - {stop_sequence}");
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

fn describe_json_float(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "default".to_string())
}

fn describe_json_u64(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "default".to_string())
}

fn json_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn json_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn json_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn json_string_array(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
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
        AgentSummary, AgentSwitchResult, DEFAULT_SYSTEM_PROMPT, QuietRenderer,
        SANDBOX_UNAVAILABLE_WARNING, SidArgs, SidCommand, SidRuntimeSession, SpendTurnClamp,
        SwitchPosition, dollars_to_micro_cents, handle_raw_request, parse_confirmation,
        parse_sid_command, resolve_sid_home_from_env, validate_connect_mode, validate_no_free_args,
        validate_runtime_mode,
    };
    use arrrg::{CommandLine, NoExitCommandLine};
    use claudius::Anthropic;
    use claudius::MessageParam;
    use claudius::chat::{ChatConfig, ChatSession};
    use serde::Deserialize;
    use sid_isnt_done::raw_mode::RawServer;
    use sid_isnt_done::raw_protocol::{RAW_PROTOCOL_VERSION, RawRequest, RawRequestEnvelope};
    use sid_isnt_done::{SidAgent, append_resumed_bash_reset_marker, session::SidSession};
    use std::fs;
    use std::io::{BufReader, Cursor};
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
        has_session_spend: bool,
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
    fn resolve_sid_home_accepts_non_empty_sid_home() {
        let root = resolve_sid_home_from_env(Ok("init".to_string())).unwrap();
        assert_eq!(root.as_str(), "init");
    }

    #[test]
    fn resolve_sid_home_rejects_missing_sid_home() {
        let err = resolve_sid_home_from_env(Err(std::env::VarError::NotPresent))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("SID_HOME must be set and non-empty"),
            "error: {err}"
        );
    }

    #[test]
    fn resolve_sid_home_rejects_empty_sid_home() {
        let err = resolve_sid_home_from_env(Ok(String::new()))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("SID_HOME must be set and non-empty"),
            "error: {err}"
        );
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
    fn parse_args_accept_raw_option() {
        let (args, free, status, messages) = parse_args(&["--raw"]);
        assert_eq!(status, 0, "unexpected parser status: {messages:?}");
        assert!(free.is_empty());
        assert!(args.raw);
    }

    #[test]
    fn parse_args_accept_listen_option() {
        let (args, free, status, messages) = parse_args(&["--listen", "unix:///tmp/sid.sock"]);
        assert_eq!(status, 0, "unexpected parser status: {messages:?}");
        assert!(free.is_empty());
        assert_eq!(args.listen.as_deref(), Some("unix:///tmp/sid.sock"));
    }

    #[test]
    fn parse_args_accept_connect_option() {
        let (args, free, status, messages) = parse_args(&["--connect", "unix:///tmp/sid.sock"]);
        assert_eq!(status, 0, "unexpected parser status: {messages:?}");
        assert!(free.is_empty());
        assert_eq!(args.connect.as_deref(), Some("unix:///tmp/sid.sock"));
    }

    #[test]
    fn parse_args_accept_prompt_option() {
        let (args, free, status, messages) = parse_args(&["--prompt", "review this"]);
        assert_eq!(status, 0, "unexpected parser status: {messages:?}");
        assert!(free.is_empty());
        assert_eq!(args.prompt.as_deref(), Some("review this"));
    }

    #[test]
    fn load_resumed_transcript_restores_saved_history_and_appends_bash_reset_marker() {
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
        let mut expected = vec![MessageParam::user("resume me")];
        append_resumed_bash_reset_marker(&mut expected);
        assert_eq!(session.clone_messages(), expected);

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
        session.set_session_spend(Some(5.0));

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
                has_session_spend: true,
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

    #[tokio::test]
    async fn sid_spend_blocks_next_turn_after_cumulative_cost_exhausts_limit() {
        let root = unique_workspace_root("spend-block");
        let sid_session = Arc::new(SidSession::create(&root).unwrap());
        let mut session = new_runtime_session(&root, &root, sid_session, None);
        session.set_model("claude-sonnet-4-5");
        session.set_session_spend(Some(0.000015));
        session
            .sid_spend
            .as_mut()
            .unwrap()
            .record_cost(dollars_to_micro_cents(0.000015));

        let mut renderer = QuietRenderer;
        let err = session
            .send_message(
                MessageParam::user("should not reach the API"),
                &mut renderer,
            )
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("session spend exhausted"),
            "error: {err}"
        );
        assert!(session.clone_messages().is_empty());

        fs::remove_dir_all(PathBuf::from(root.as_str())).unwrap();
    }

    #[test]
    fn sid_spend_clamps_max_tokens_to_affordable_output_tokens() {
        let root = unique_workspace_root("spend-clamp");
        let sid_session = Arc::new(SidSession::create(&root).unwrap());
        let mut session = new_runtime_session(&root, &root, sid_session, None);
        session.set_model("claude-sonnet-4-5");
        session.set_max_tokens(100);
        session.set_session_spend(Some(0.000045));

        assert_eq!(
            session.spend_turn_clamp().unwrap(),
            Some(SpendTurnClamp {
                original_max_tokens: Some(100),
                clamped_max_tokens: 3,
            })
        );

        fs::remove_dir_all(PathBuf::from(root.as_str())).unwrap();
    }

    #[test]
    fn sid_spend_uses_sonnet_rates_for_custom_model_clamping() {
        let root = unique_workspace_root("spend-custom-model");
        let sid_session = Arc::new(SidSession::create(&root).unwrap());
        let mut session = new_runtime_session(&root, &root, sid_session, None);
        session.set_model("custom-model");
        session.set_max_tokens(100);
        session.set_session_spend(Some(0.000030));

        assert_eq!(
            session.spend_turn_clamp().unwrap(),
            Some(SpendTurnClamp {
                original_max_tokens: Some(100),
                clamped_max_tokens: 2,
            })
        );

        fs::remove_dir_all(PathBuf::from(root.as_str())).unwrap();
    }

    #[test]
    fn sid_spend_switch_agent_preserves_used_spend_and_effective_limit() {
        let root = unique_workspace_root("spend-agent-switch");
        write_multi_agent_config(&root);
        let mut session = configured_runtime_session(&root, "build");
        session.set_model("claude-sonnet-4-5");
        session.set_max_tokens(100);
        session.set_session_spend(Some(0.000030));
        session
            .sid_spend
            .as_mut()
            .unwrap()
            .record_cost(dollars_to_micro_cents(0.000015));

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
        let stats = session.stats();
        assert_eq!(
            stats.session_spend_micro_cents,
            Some(dollars_to_micro_cents(0.000030))
        );
        assert_eq!(
            stats.spend_used_micro_cents,
            dollars_to_micro_cents(0.000015)
        );
        assert_eq!(
            session.spend_turn_clamp().unwrap(),
            Some(SpendTurnClamp {
                original_max_tokens: Some(100),
                clamped_max_tokens: 1,
            })
        );

        fs::remove_dir_all(PathBuf::from(root.as_str())).unwrap();
    }

    #[test]
    fn sid_spend_clear_removes_local_guard() {
        let root = unique_workspace_root("spend-clear");
        let sid_session = Arc::new(SidSession::create(&root).unwrap());
        let mut session = new_runtime_session(&root, &root, sid_session, None);
        session.set_model("claude-sonnet-4-5");
        session.set_session_spend(Some(0.000015));
        session
            .sid_spend
            .as_mut()
            .unwrap()
            .record_cost(dollars_to_micro_cents(0.000015));

        session.set_session_spend(None);

        assert!(session.spend_turn_clamp().unwrap().is_none());
        let stats = session.stats();
        assert_eq!(stats.session_spend_micro_cents, None);
        assert_eq!(stats.spend_used_micro_cents, 0);

        fs::remove_dir_all(PathBuf::from(root.as_str())).unwrap();
    }

    #[test]
    fn sid_spend_set_after_prior_spend_starts_new_cap() {
        let root = unique_workspace_root("spend-reset");
        let sid_session = Arc::new(SidSession::create(&root).unwrap());
        let mut session = new_runtime_session(&root, &root, sid_session, None);
        session.set_model("claude-sonnet-4-5");
        session.set_session_spend(Some(0.000015));
        session
            .sid_spend
            .as_mut()
            .unwrap()
            .record_cost(dollars_to_micro_cents(0.000015));

        session.set_session_spend(Some(0.000030));

        let stats = session.stats();
        assert_eq!(
            stats.session_spend_micro_cents,
            Some(dollars_to_micro_cents(0.000030))
        );
        assert_eq!(stats.spend_used_micro_cents, 0);
        assert_eq!(
            session.spend_turn_clamp().unwrap(),
            Some(SpendTurnClamp {
                original_max_tokens: Some(4096),
                clamped_max_tokens: 2,
            })
        );

        fs::remove_dir_all(PathBuf::from(root.as_str())).unwrap();
    }

    #[tokio::test]
    async fn raw_set_spend_follows_interactive_reset_behavior() {
        let root = unique_workspace_root("raw-spend-reset");
        let sid_session = Arc::new(SidSession::create(&root).unwrap());
        let mut session = new_runtime_session(&root, &root, sid_session, None);
        session.set_model("claude-sonnet-4-5");
        session.set_session_spend(Some(0.000015));
        session
            .sid_spend
            .as_mut()
            .unwrap()
            .record_cost(dollars_to_micro_cents(0.000015));
        let mut server = RawServer::new(BufReader::new(Cursor::new(Vec::new())), Vec::new());

        handle_raw_request(
            &mut session,
            &mut server,
            RawRequestEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                request_id: "set-spend".to_string(),
                request: RawRequest::SetSpend {
                    dollars: Some(0.000030),
                },
            },
        )
        .await
        .unwrap();
        let stats = match handle_raw_request(
            &mut session,
            &mut server,
            RawRequestEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                request_id: "stats".to_string(),
                request: RawRequest::Stats,
            },
        )
        .await
        .unwrap()
        {
            super::RequestDisposition::Continue(Some(stats)) => stats,
            _ => panic!("unexpected raw disposition"),
        };

        assert_eq!(
            stats["session_spend_micro_cents"].as_u64(),
            Some(dollars_to_micro_cents(0.000030))
        );
        assert_eq!(stats["spend_used_micro_cents"].as_u64(), Some(0));

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
        let auto_compact_tokens = agent.auto_compact_tokens();
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
            auto_compact_tokens,
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
            has_session_spend: stats.session_spend_micro_cents.is_some(),
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
    fn sid_args_parse_prompt_and_resume() {
        let (args, free, status, messages) =
            parse_args(&["--resume", "sid-session", "--prompt", "continue"]);

        assert_eq!(status, 0);
        assert!(
            messages.is_empty(),
            "unexpected parser messages: {messages:?}"
        );
        assert!(free.is_empty(), "unexpected free args: {free:?}");
        assert_eq!(args.resume.as_deref(), Some("sid-session"));
        assert_eq!(args.prompt.as_deref(), Some("continue"));
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
    fn validate_runtime_mode_rejects_prompt_with_raw() {
        let err = validate_runtime_mode(true, None, Some("hello"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--raw cannot be combined with --prompt"));
    }

    #[test]
    fn validate_runtime_mode_rejects_prompt_with_bash_debug() {
        let err = validate_runtime_mode(false, Some("pwd"), Some("hello"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--bash-debug cannot be combined with --prompt"));
    }

    #[test]
    fn validate_connect_mode_rejects_raw() {
        let err = validate_connect_mode(true, None, None, None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--connect cannot be combined with --raw"));
    }

    #[test]
    fn sandbox_warning_is_loud_and_explicit() {
        assert!(SANDBOX_UNAVAILABLE_WARNING.contains("WARNING"));
        assert!(SANDBOX_UNAVAILABLE_WARNING.contains("/usr/bin/sandbox-exec is unavailable"));
        assert!(SANDBOX_UNAVAILABLE_WARNING.contains("UNSANDBOXED"));
    }
}
