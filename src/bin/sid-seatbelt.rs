//! Run a command inside a macOS Seatbelt sandbox.
//!
//! Replaces the current process with `sandbox-exec` via exec.

use std::os::unix::process::CommandExt;
use std::process::Command;

use arrrg::CommandLine;

use sid_isnt_done::seatbelt;
use sid_isnt_done::seatbelt::WritableRoots;

#[derive(arrrg_derive::CommandLine, Debug, Default, Eq, PartialEq)]
struct SeatbeltOptions {
    #[arrrg(
        optional,
        "Colon-separated writable directories (PATH-style).",
        "DIR:DIR:..."
    )]
    writable_roots: WritableRoots,
}

fn main() {
    let (options, free) =
        SeatbeltOptions::from_command_line_relaxed("sid-seatbelt [OPTIONS] -- command [args...]");

    if free.is_empty() {
        eprintln!("error: no command specified");
        eprintln!("Usage: sid-seatbelt [--writable-roots DIR:DIR:...] -- command [args...]");
        std::process::exit(1);
    }

    if !seatbelt::sandbox_available() {
        eprintln!("error: /usr/bin/sandbox-exec not found; this binary requires macOS");
        std::process::exit(1);
    }

    let policy = seatbelt::build_policy(&options.writable_roots);
    let cache_dir = seatbelt::darwin_user_cache_dir();

    let err = Command::new("/usr/bin/sandbox-exec")
        .arg("-p")
        .arg(policy)
        .arg("-D")
        .arg(format!("DARWIN_USER_CACHE_DIR={cache_dir}"))
        .arg("--")
        .args(&free)
        .exec();

    eprintln!("error: exec failed: {err}");
    std::process::exit(1);
}
