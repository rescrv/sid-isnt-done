use std::io::{self, Write};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrrg::CommandLine;
use handled::SError;
use rustyline::DefaultEditor;
use rustyline::config::{Config, EditMode};
use rustyline::error::ReadlineError;
use utf8path::Path;

use claudius::chat::{
    ChatAgent, ChatArgs, ChatCommand, ChatConfig, ChatSession, PlainTextRenderer, help_text,
    parse_command,
};
use claudius::{Anthropic, Model, SystemPrompt, ThinkingConfig};
use claudius::{OperatorLine, Renderer, StopReason, StreamContext};

use sid_isnt_done::{SidAgent, seatbelt, session, session::SidSession};

const DEFAULT_SYSTEM_PROMPT: &str = concat!(
    "You are sid, a concise coding agent with access to the current workspace mounted at /.\n",
    "Use configured tools when they help accomplish the user's request.\n",
    "Ground your answers in the files and tool results you can access, explain changes you make, and do not claim to have run commands or changed files unless you actually did."
);
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
}

/// Parse arguments, resolve paths, create the session directory, and set
/// environment variables for child processes.  Runs before the tokio runtime
/// so that process-global `set_var` calls are single-threaded and safe.
fn pre_runtime_setup() -> Result<PreRuntimeSetup, SError> {
    let SidArgs { param, bash_debug } = parse_sid_args()?;
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
    let sid_session = Arc::new(SidSession::create(&config_root)?);
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
        session_display,
        bash_debug,
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
    } = setup;

    warn_if_sandbox_unavailable();

    let agent = SidAgent::from_workspace_with_config_root(&workspace_root, &config_root, config)?
        .with_session(sid_session);
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
    let mut session = ChatSession::with_agent(client, agent);

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
                            if let Err(err) =
                                session.send_message(message, &mut terminal).await
                            {
                                terminal.print_error(&context, &err.to_string());
                            }
                        }
                        Ok(None) => {
                            terminal.print_info(&context, "Editor returned empty; nothing sent.");
                        }
                        Err(err) => {
                            terminal.print_error(
                                &context,
                                &format!("Failed to invoke editor: {err}"),
                            );
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
                        ChatCommand::Clear => {
                            session.clear();
                            terminal.print_info(&context, "Conversation cleared.");
                        }
                        ChatCommand::Help => {
                            for line in help_text().lines() {
                                println!("    {}", line);
                            }
                        }
                        ChatCommand::Model(model_name) => {
                            let model = model_name
                                .parse()
                                .unwrap_or_else(|_| Model::Custom(model_name.clone()));
                            session.template_mut().model = Some(model);
                            terminal
                                .print_info(&context, &format!("Model changed to: {model_name}"));
                        }
                        ChatCommand::System(prompt) => {
                            session.template_mut().system = prompt.clone().map(SystemPrompt::from);
                            match prompt {
                                Some(prompt) => terminal.print_info(
                                    &context,
                                    &format!("System prompt set to: {prompt}"),
                                ),
                                None => terminal.print_info(&context, "System prompt cleared."),
                            }
                        }
                        ChatCommand::MaxTokens(value) => {
                            session.template_mut().max_tokens = Some(value);
                            terminal.print_info(&context, &format!("max_tokens set to {value}"));
                        }
                        ChatCommand::Temperature(value) => {
                            session.template_mut().temperature = Some(value);
                            terminal
                                .print_info(&context, &format!("temperature set to {value:.2}"));
                        }
                        ChatCommand::ClearTemperature => {
                            session.template_mut().temperature = None;
                            terminal.print_info(&context, "temperature reset to model default");
                        }
                        ChatCommand::TopP(value) => {
                            session.template_mut().top_p = Some(value);
                            terminal.print_info(&context, &format!("top_p set to {value:.2}"));
                        }
                        ChatCommand::ClearTopP => {
                            session.template_mut().top_p = None;
                            terminal.print_info(&context, "top_p reset to model default");
                        }
                        ChatCommand::TopK(value) => {
                            session.template_mut().top_k = Some(value);
                            terminal.print_info(&context, &format!("top_k set to {value}"));
                        }
                        ChatCommand::ClearTopK => {
                            session.template_mut().top_k = None;
                            terminal.print_info(&context, "top_k reset to model default");
                        }
                        ChatCommand::AddStopSequence(sequence) => {
                            let stop_sequences = session
                                .template_mut()
                                .stop_sequences
                                .get_or_insert_with(Vec::new);
                            if !stop_sequences.iter().any(|existing| existing == &sequence) {
                                stop_sequences.push(sequence.clone());
                            }
                            terminal
                                .print_info(&context, &format!("Added stop sequence: {sequence}"));
                        }
                        ChatCommand::ClearStopSequences => {
                            session.template_mut().stop_sequences = None;
                            terminal.print_info(&context, "Stop sequences cleared.");
                        }
                        ChatCommand::ListStopSequences => {
                            let sequences =
                                session.template().stop_sequences.as_deref().unwrap_or(&[]);
                            print_stop_sequences(sequences);
                        }
                        ChatCommand::Thinking(budget) => {
                            session.template_mut().thinking = budget.map(ThinkingConfig::enabled);
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
                        ChatCommand::Budget(_tokens) => {
                            terminal.print_error(&context, "budget not supported");
                        }
                        ChatCommand::ClearBudget => {
                            session.config_mut().session_budget = None;
                            terminal.print_info(&context, "Session budget cleared.");
                        }
                        ChatCommand::Caching(enabled) => {
                            session.config_mut().caching_enabled = enabled;
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
                            print_stats(&session);
                        }
                        ChatCommand::ShowConfig => {
                            print_config(&session);
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

fn print_stats<A: ChatAgent>(session: &ChatSession<A>) {
    let stats = session.stats();
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

fn print_config<A: ChatAgent>(session: &ChatSession<A>) {
    let stats = session.stats();
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
    use super::{SANDBOX_UNAVAILABLE_WARNING, SidArgs, parse_confirmation, validate_no_free_args};
    use arrrg::{CommandLine, NoExitCommandLine};

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
