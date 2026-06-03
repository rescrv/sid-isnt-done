use std::io::{self, IsTerminal, Read, Write};

use sid_isnt_done::sidreview::{render_plain, run_tui};

fn main() {
    if let Err(err) = try_main() {
        eprintln!("sidreview: {err}");
        std::process::exit(1);
    }
}

fn try_main() -> io::Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    if io::stdout().is_terminal() {
        run_tui(&input)
    } else {
        let mut stdout = io::stdout();
        stdout.write_all(render_plain(&input).as_bytes())?;
        stdout.flush()
    }
}
