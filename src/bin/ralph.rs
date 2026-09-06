//! ralph — a standalone ralph script interpreter, and the external face of
//! the `agent` and `judge` builtins.
//!
//! Invoked as `ralph` — directly or through a `#!/usr/bin/env ralph` shebang —
//! this binary runs a script through the ralph runner:
//!
//! ```text
//! ralph SCRIPT [--max-iters N] [--budget TOKENS] [--resume RUN_ID] [--] [ARGS...]
//! ```
//!
//! ralph wires `agent` and `judge` into the embedded mxsh script as symlinks
//! to this binary, so pipes and redirections behave exactly like POSIX: the
//! OS hands this process the read end of `printf '%s' "$out" | agent fix
//! ...`.  Invoked through one of those symlinks, the shim reads stdin (the
//! piped context), forwards argv plus the context through
//! `$RALPH_CONTROL_DIR` to the in-process runner, prints the response's
//! stdout/stderr, and exits with the protocol's exit code.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use claudius::chat::ChatConfig;
use utf8path::Path;

use sid_isnt_done::ralph::EXIT_TRANSPORT;
use sid_isnt_done::ralph::args::RunArgs;
use sid_isnt_done::ralph::host::SidRalphHost;
use sid_isnt_done::ralph::journal::generate_run_id;
use sid_isnt_done::ralph::runner::{
    CONTROL_DIR_ENV, RunnerOptions, ShimRequest, call_control_dir, run_ralph,
};
use sid_isnt_done::session::{SID_SESSION_DIR_ENV, SID_SESSIONS_ENV, SidSession};

/// Exit code for a malformed ralph command line (EX_USAGE).
const EXIT_USAGE: u8 = 64;

fn main() -> ExitCode {
    let mut args = std::env::args();
    let argv0 = args.next().unwrap_or_else(|| "ralph".to_string());
    let name = std::path::Path::new(&argv0)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or(argv0.clone());
    let args: Vec<String> = args.collect();
    if name == "agent" || name == "judge" {
        shim_main(name, args)
    } else {
        interpreter_main(args)
    }
}

/// Shim mode: forward one `agent`/`judge` call to the in-process runner.
fn shim_main(name: String, args: Vec<String>) -> ExitCode {
    let Ok(control_dir) = std::env::var(CONTROL_DIR_ENV) else {
        eprintln!("{name}: {CONTROL_DIR_ENV} is not set; run me from inside a ralph run");
        return ExitCode::from(EXIT_TRANSPORT as u8);
    };

    let mut context = Vec::new();
    if std::io::stdin().read_to_end(&mut context).is_err() {
        eprintln!("{name}: failed to read stdin");
        return ExitCode::from(EXIT_TRANSPORT as u8);
    }
    let context = String::from_utf8_lossy(&context).into_owned();

    let request = ShimRequest {
        name,
        args,
        context,
    };
    match call_control_dir(&PathBuf::from(control_dir), &request) {
        Ok(response) => {
            if !response.stdout.is_empty() {
                let mut stdout = std::io::stdout().lock();
                let _ = stdout.write_all(response.stdout.as_bytes());
                let _ = stdout.flush();
            }
            if !response.stderr.is_empty() {
                eprintln!("{}", response.stderr.trim_end());
            }
            let code = response.exit.clamp(0, 255) as u8;
            ExitCode::from(code)
        }
        Err(err) => {
            eprintln!("{}: control socket failure: {err}", request.name);
            ExitCode::from(EXIT_TRANSPORT as u8)
        }
    }
}

/// Interpreter mode: run a script end to end and exit with its status.
fn interpreter_main(argv: Vec<String>) -> ExitCode {
    if argv
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        println!(
            "usage: ralph SCRIPT [--max-iters N] [--budget TOKENS] [--resume RUN_ID] [--] [ARGS...]"
        );
        return ExitCode::SUCCESS;
    }
    let args = match RunArgs::parse(&argv) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("ralph: {err}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    match run_interpreter(args) {
        Ok(exit) => ExitCode::from(exit.clamp(0, 255) as u8),
        Err(err) => {
            eprintln!("ralph: {err}");
            ExitCode::from(EXIT_TRANSPORT as u8)
        }
    }
}

fn run_interpreter(args: RunArgs) -> Result<i32, String> {
    let workspace_root = Path::try_from(
        std::env::current_dir()
            .map_err(|err| format!("failed to determine the current working directory: {err}"))?,
    )
    .map_err(|err| format!("current working directory is not valid UTF-8: {err}"))?
    .into_owned();
    let config_root = resolve_config_root()?;
    let script_path = resolve_script(&args.script, &workspace_root, &config_root)?;
    let script_text = std::fs::read_to_string(&script_path)
        .map_err(|err| format!("failed to read {}: {err}", script_path.display()))?;

    // ralph journals under the sid session directory, never the workspace.
    // Launched from an interactive session, the parent exports
    // SID_SESSION_DIR; standalone, create a fresh session under SID_HOME (or,
    // when resuming, locate the session that already holds the run).
    let session_dir = resolve_session_dir(&config_root, &workspace_root, args.resume.as_deref())?;
    let runs_root = session_dir.join("runs");
    std::fs::create_dir_all(&runs_root).map_err(|err| {
        format!(
            "failed to create runs directory {}: {err}",
            runs_root.display()
        )
    })?;
    let (run_id, resume) = match args.resume.as_ref() {
        Some(id) => {
            if !runs_root.join(id).is_dir() {
                return Err(format!("no such run to --resume: {id}"));
            }
            (id.clone(), true)
        }
        None => (generate_run_id(&runs_root), false),
    };
    let run_dir = runs_root.join(&run_id);

    let interrupted = Arc::new(AtomicBool::new(false));
    install_ctrlc_handler(Arc::clone(&interrupted))?;

    let options = RunnerOptions {
        run_id: run_id.clone(),
        run_dir: run_dir.clone(),
        workspace_root: Some(PathBuf::from(workspace_root.as_str())),
        max_iters: args.max_iters,
        budget_tokens: args.budget,
        resume,
        script_args: args.script_args.clone(),
    };

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|err| format!("failed to create the async runtime: {err}"))?;
    let report = runtime.block_on(async move {
        let host = SidRalphHost::new(
            workspace_root,
            config_root,
            ChatConfig::new(),
            run_dir,
            // There is no launching interactive thread; the judge starts cold.
            Vec::new(),
            run_id,
            tokio::runtime::Handle::current(),
            Arc::clone(&interrupted),
        );
        let run_interrupted = Arc::clone(&interrupted);
        tokio::task::spawn_blocking(move || {
            run_ralph(Box::new(host), options, &script_text, &[], run_interrupted)
        })
        .await
        .map_err(|err| format!("ralph runner thread panicked: {err}"))?
    })?;

    eprint!("{}", report.summary(&args.script));
    Ok(report.exit)
}

/// The configuration root holds agents.conf and friends; ralph shares sid's
/// convention of requiring `SID_HOME` to name it.
fn resolve_config_root() -> Result<Path<'static>, String> {
    match std::env::var("SID_HOME") {
        Ok(path) if !path.is_empty() => Ok(Path::new(&path).into_owned()),
        Ok(_) | Err(std::env::VarError::NotPresent) => {
            Err("SID_HOME must be set and non-empty".to_string())
        }
        Err(std::env::VarError::NotUnicode(_)) => Err("SID_HOME is not valid UTF-8".to_string()),
    }
}

/// Resolve the sid session directory ralph journals under.  Prefer the
/// parent's `SID_SESSION_DIR` (set when launched from an interactive session).
/// Standalone, create a fresh session under `SID_HOME` for a new run, or find
/// the session already holding `resume` when resuming.
fn resolve_session_dir(
    config_root: &Path,
    workspace_root: &Path,
    resume: Option<&str>,
) -> Result<PathBuf, String> {
    if let Ok(dir) = std::env::var(SID_SESSION_DIR_ENV)
        && !dir.is_empty()
    {
        return Ok(PathBuf::from(dir));
    }
    match resume {
        Some(id) => find_run_session(&resolve_sessions_root(config_root), id),
        None => {
            let session = SidSession::create_with_workspace(config_root, workspace_root)
                .map_err(|err| format!("failed to create a session: {err}"))?;
            Ok(session.root().clone())
        }
    }
}

/// Mirror sid's sessions-root resolution: honor `SID_SESSIONS`, else fall back
/// to `<SID_HOME>/sessions`.
fn resolve_sessions_root(config_root: &Path) -> PathBuf {
    match std::env::var(SID_SESSIONS_ENV) {
        Ok(path) if !path.is_empty() => PathBuf::from(path),
        _ => PathBuf::from(config_root.as_str()).join("sessions"),
    }
}

/// Find the session directory that holds `runs/<run_id>`, so `--resume` works
/// across standalone invocations, each of which lives in its own session.
fn find_run_session(sessions_root: &std::path::Path, run_id: &str) -> Result<PathBuf, String> {
    let entries = std::fs::read_dir(sessions_root)
        .map_err(|err| format!("failed to scan sessions {}: {err}", sessions_root.display()))?;
    for entry in entries.flatten() {
        let candidate = entry.path();
        if candidate.join("runs").join(run_id).is_dir() {
            return Ok(candidate);
        }
    }
    Err(format!("no such run to --resume: {run_id}"))
}

/// Resolve a script: absolute paths as-is, otherwise relative to the
/// workspace root, then to the configuration root (where sid-init installs
/// the reference `ralph.sid`).
fn resolve_script(
    script: &str,
    workspace_root: &Path,
    config_root: &Path,
) -> Result<PathBuf, String> {
    let raw = PathBuf::from(script);
    let candidates = if raw.is_absolute() {
        vec![raw]
    } else {
        vec![
            PathBuf::from(workspace_root.as_str()).join(script),
            PathBuf::from(config_root.as_str()).join(script),
        ]
    };
    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }
    Err(format!("no such script: {script}"))
}

fn install_ctrlc_handler(interrupted: Arc<AtomicBool>) -> Result<(), String> {
    ctrlc::set_handler(move || {
        interrupted.store(true, Ordering::Relaxed);
    })
    .map_err(|err| format!("failed to install the Ctrl-C handler: {err}"))
}
