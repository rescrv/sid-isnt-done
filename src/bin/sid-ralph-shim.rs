//! sid-ralph-shim — the external face of the `agent` and `judge` builtins.
//!
//! ralph wires `agent` and `judge` into the embedded mxsh script as symlinks
//! to this binary, so pipes and redirections behave exactly like POSIX: the
//! OS hands this process the read end of `printf '%s' "$out" | agent fix
//! ...`.  The shim reads stdin (the piped context), forwards argv plus the
//! context through `$RALPH_CONTROL_DIR` to the in-process runner, prints the
//! response's stdout/stderr, and exits with the protocol's exit code.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use sid_isnt_done::ralph::EXIT_TRANSPORT;
use sid_isnt_done::ralph::runner::{CONTROL_DIR_ENV, ShimRequest, call_control_dir};

fn main() -> ExitCode {
    let mut args = std::env::args();
    let argv0 = args.next().unwrap_or_else(|| "agent".to_string());
    let name = std::path::Path::new(&argv0)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or(argv0.clone());
    let args: Vec<String> = args.collect();

    let Ok(control_dir) = std::env::var(CONTROL_DIR_ENV) else {
        eprintln!("{name}: {CONTROL_DIR_ENV} is not set; run me from inside a sid /run");
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
