use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::ExitCode;

use sid_isnt_done::skill_inject::render_skill_blocks;

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "sid-skill-inject".to_string());
    match args.next().as_deref() {
        Some("rcvar") => {
            let prefix = env::var("RCVAR_ARGV0")
                .map_err(|_| "sid-skill-inject rcvar requires RCVAR_ARGV0".to_string())?;
            println!("{prefix}_USER_MESSAGE_FILE");
            println!("{prefix}_SKILLS_MANIFEST_FILE");
            Ok(ExitCode::SUCCESS)
        }
        Some("run") => {
            let prefix = env::var("RCVAR_ARGV0")
                .map_err(|_| "sid-skill-inject run requires RCVAR_ARGV0".to_string())?;
            let user_message_file = env_for(&prefix, "USER_MESSAGE_FILE")?;
            let skills_manifest_file = env_for(&prefix, "SKILLS_MANIFEST_FILE")?;
            let user_message = fs::read_to_string(&user_message_file).map_err(|err| {
                format!("failed to read user message file {user_message_file}: {err}")
            })?;
            let manifest = fs::read_to_string(&skills_manifest_file).map_err(|err| {
                format!("failed to read skills manifest file {skills_manifest_file}: {err}")
            })?;
            let output = render_skill_blocks(&user_message, &manifest)?;
            io::stdout()
                .write_all(output.as_bytes())
                .map_err(|err| format!("failed to write skill injection output: {err}"))?;
            Ok(ExitCode::SUCCESS)
        }
        _ => {
            eprintln!("usage: {program} [rcvar|run]");
            Ok(ExitCode::from(129))
        }
    }
}

fn env_for(prefix: &str, suffix: &str) -> Result<String, String> {
    let name = format!("{prefix}_{suffix}");
    env::var(&name).map_err(|_| format!("sid-skill-inject requires {name}"))
}
