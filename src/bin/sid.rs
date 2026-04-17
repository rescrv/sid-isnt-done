use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrrg::CommandLine;
use handled::SError;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use utf8path::Path;

use claudius::chat::{
    ChatAgent, ChatArgs, ChatCommand, ChatConfig, ChatSession, PlainTextRenderer, help_text,
    parse_command,
};
use claudius::{Agent, Renderer};
use claudius::{Anthropic, Model, SystemPrompt, ThinkingConfig};

use sid_isnt_done::SidAgent;

const DEFAULT_SYSTEM_PROMPT: &str = concat!(
    "You are sid, a concise coding agent with access to the current workspace mounted at /.\n",
    "Use configured tools when they help accomplish the user's request.\n",
    "Ground your answers in the files and tool results you can access, explain changes you make, and do not claim to have run commands or changed files unless you actually did."
);

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

#[tokio::main]
async fn main() {
    if let Err(err) = try_main().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

async fn try_main() -> Result<(), SError> {
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
    // Set SID_WORKSPACE_ROOT for child processes spawned by tools. This is the
    // outer binary entrypoint, so process-global mutation is acceptable here.
    unsafe {
        std::env::set_var("SID_WORKSPACE_ROOT", workspace_root.as_str());
    }
    let workspace_display = workspace_root.as_str().to_string();

    let agent = SidAgent::from_workspace_with_config_root(&workspace_root, &config_root, config)?;
    let agent_id = agent.id().to_string();
    if agent.requires_confirmation() && !confirm_manual_agent(&agent_id)? {
        println!("Aborted.");
        return Ok(());
    }

    if let Some(command) = bash_debug {
        return run_bash_debug(&agent, &agent_id, &workspace_display, &command).await;
    }

    let use_color = agent.config().use_color;

    let client = Anthropic::new(None).map_err(|err| {
        cli_error(
            "client_init_failed",
            "failed to initialize the Anthropic client",
        )
        .with_string_field("cause", &err.to_string())
    })?;
    let mut session = ChatSession::with_agent(client, agent);
    let mut rl = DefaultEditor::new().map_err(|err| {
        cli_error(
            "readline_init_failed",
            "failed to initialize the terminal editor",
        )
        .with_string_field("cause", &err.to_string())
    })?;

    let interrupted = Arc::new(AtomicBool::new(false));
    let mut renderer = PlainTextRenderer::with_color_and_interrupt(use_color, interrupted.clone());
    let context = ();

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
    println!("Type /help for commands, /quit to exit\n");

    loop {
        interrupted.store(false, Ordering::Relaxed);

        let readline = rl.readline("You: ");

        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(line);

                if let Some(cmd) = parse_command(line) {
                    match cmd {
                        ChatCommand::Quit => {
                            println!("Goodbye!");
                            break;
                        }
                        ChatCommand::Clear => {
                            session.clear();
                            renderer.print_info(&context, "Conversation cleared.");
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
                            renderer
                                .print_info(&context, &format!("Model changed to: {model_name}"));
                        }
                        ChatCommand::System(prompt) => {
                            session.template_mut().system = prompt.clone().map(SystemPrompt::from);
                            match prompt {
                                Some(prompt) => renderer.print_info(
                                    &context,
                                    &format!("System prompt set to: {prompt}"),
                                ),
                                None => renderer.print_info(&context, "System prompt cleared."),
                            }
                        }
                        ChatCommand::MaxTokens(value) => {
                            session.template_mut().max_tokens = Some(value);
                            renderer.print_info(&context, &format!("max_tokens set to {value}"));
                        }
                        ChatCommand::Temperature(value) => {
                            session.template_mut().temperature = Some(value);
                            renderer
                                .print_info(&context, &format!("temperature set to {value:.2}"));
                        }
                        ChatCommand::ClearTemperature => {
                            session.template_mut().temperature = None;
                            renderer.print_info(&context, "temperature reset to model default");
                        }
                        ChatCommand::TopP(value) => {
                            session.template_mut().top_p = Some(value);
                            renderer.print_info(&context, &format!("top_p set to {value:.2}"));
                        }
                        ChatCommand::ClearTopP => {
                            session.template_mut().top_p = None;
                            renderer.print_info(&context, "top_p reset to model default");
                        }
                        ChatCommand::TopK(value) => {
                            session.template_mut().top_k = Some(value);
                            renderer.print_info(&context, &format!("top_k set to {value}"));
                        }
                        ChatCommand::ClearTopK => {
                            session.template_mut().top_k = None;
                            renderer.print_info(&context, "top_k reset to model default");
                        }
                        ChatCommand::AddStopSequence(sequence) => {
                            let stop_sequences = session
                                .template_mut()
                                .stop_sequences
                                .get_or_insert_with(Vec::new);
                            if !stop_sequences.iter().any(|existing| existing == &sequence) {
                                stop_sequences.push(sequence.clone());
                            }
                            renderer
                                .print_info(&context, &format!("Added stop sequence: {sequence}"));
                        }
                        ChatCommand::ClearStopSequences => {
                            session.template_mut().stop_sequences = None;
                            renderer.print_info(&context, "Stop sequences cleared.");
                        }
                        ChatCommand::ListStopSequences => {
                            let sequences =
                                session.template().stop_sequences.as_deref().unwrap_or(&[]);
                            print_stop_sequences(sequences);
                        }
                        ChatCommand::Thinking(budget) => {
                            session.template_mut().thinking = budget.map(ThinkingConfig::enabled);
                            match budget {
                                Some(tokens) => renderer.print_info(
                                    &context,
                                    &format!(
                                        "Extended thinking enabled with {} token budget.",
                                        tokens
                                    ),
                                ),
                                None => {
                                    renderer.print_info(&context, "Extended thinking disabled.");
                                }
                            }
                        }
                        ChatCommand::Budget(_tokens) => {
                            renderer.print_error(&context, "budget not supported");
                        }
                        ChatCommand::ClearBudget => {
                            session.config_mut().session_budget = None;
                            renderer.print_info(&context, "Session budget cleared.");
                        }
                        ChatCommand::Caching(enabled) => {
                            session.config_mut().caching_enabled = enabled;
                            if enabled {
                                renderer.print_info(&context, "Prompt caching enabled.");
                            } else {
                                renderer.print_info(&context, "Prompt caching disabled.");
                            }
                        }
                        ChatCommand::TranscriptPath(path) => {
                            session.config_mut().transcript_path = Some(path.clone().into());
                            renderer.print_info(
                                &context,
                                &format!("Transcript auto-save set to {path}"),
                            );
                        }
                        ChatCommand::ClearTranscriptPath => {
                            session.config_mut().transcript_path = None;
                            renderer.print_info(&context, "Transcript auto-save disabled.");
                        }
                        ChatCommand::SaveTranscript(path) => {
                            match session.save_transcript_to(&path) {
                                Ok(()) => renderer
                                    .print_info(&context, &format!("Transcript saved to {path}")),
                                Err(err) => renderer.print_error(
                                    &context,
                                    &format!("Failed to save transcript: {err}"),
                                ),
                            }
                        }
                        ChatCommand::LoadTranscript(path) => {
                            match session.load_transcript_from(&path) {
                                Ok(()) => renderer.print_info(
                                    &context,
                                    &format!("Transcript loaded from {path}"),
                                ),
                                Err(err) => renderer.print_error(
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
                            renderer.print_error(&context, &message);
                        }
                    }
                    continue;
                }

                let message = claudius::MessageParam::user(line);
                if let Err(err) = session.send_message(message, &mut renderer).await {
                    renderer.print_error(&context, &err.to_string());
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!();
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("\nGoodbye!");
                break;
            }
            Err(err) => {
                renderer.print_error(&context, &format!("Input error: {err}"));
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
) -> Result<(), SError> {
    eprintln!("sid bash debug (agent: {agent_id})");
    eprintln!("workspace: {workspace_display}");

    match agent.bash(command, false).await {
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

fn resolve_sid_home(workspace_root: &Path) -> Result<Path<'static>, SError> {
    match std::env::var("SID_HOME") {
        Ok(path) if !path.is_empty() => Ok(Path::new(&path).into_owned()),
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(workspace_root.clone().into_owned()),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(cli_error("invalid_sid_home", "SID_HOME is not valid UTF-8"))
        }
    }
}

fn confirm_manual_agent(agent_id: &str) -> Result<bool, SError> {
    let mut input = String::new();
    loop {
        print!("Agent '{agent_id}' is MANUAL. Continue? [yes/no]: ");
        io::stdout().flush().map_err(|err| {
            cli_error(
                "io_error",
                "failed to flush manual-agent confirmation prompt",
            )
            .with_string_field("agent", agent_id)
            .with_string_field("cause", &err.to_string())
        })?;

        input.clear();
        if io::stdin().read_line(&mut input).map_err(|err| {
            cli_error("io_error", "failed to read manual-agent confirmation input")
                .with_string_field("agent", agent_id)
                .with_string_field("cause", &err.to_string())
        })? == 0
        {
            println!();
            return Ok(false);
        }

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
    match stats.transcript_path {
        Some(ref path) => println!(
            "      Transcript file: {}",
            Path::try_from(path.as_path())
                .map(|path| path.as_str().to_string())
                .unwrap_or_else(|_| path.display().to_string())
        ),
        None => println!("      Transcript file: (disabled)"),
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
    match stats.transcript_path {
        Some(ref path) => println!(
            "      Transcript file: {}",
            Path::try_from(path.as_path())
                .map(|path| path.as_str().to_string())
                .unwrap_or_else(|_| path.display().to_string())
        ),
        None => println!("      Transcript file: (disabled)"),
    }
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

#[cfg(test)]
mod tests {
    use super::{SidArgs, parse_confirmation, validate_no_free_args};
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
}
