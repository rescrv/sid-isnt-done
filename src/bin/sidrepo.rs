use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;

use sid_isnt_done::sidrepo::{SidrepoOptions, render_repo_diff};

fn usage(program: &str) {
    eprintln!("usage: {program} [REPO]");
}

fn main() -> ExitCode {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() > 2
        || args
            .get(1)
            .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        usage(args.first().map(String::as_str).unwrap_or("sidrepo"));
        return if args.len() == 2 {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(129)
        };
    }

    let repo = args.get(1).map(String::as_str).unwrap_or(".");
    match run(Path::new(repo)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) if err.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("sidrepo: {err}");
            ExitCode::from(1)
        }
    }
}

fn run(repo: &Path) -> io::Result<()> {
    let rendered = render_repo_diff(repo, SidrepoOptions::default())?;
    let mut stdout = io::stdout();
    stdout.write_all(rendered.as_bytes())?;
    stdout.flush()
}
