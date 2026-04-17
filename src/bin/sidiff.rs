use std::io::{self, IsTerminal, Read, Write};

use sid_isnt_done::sidiff::{SidiffOptions, render_diff, run_pager};

fn main() {
    if let Err(err) = try_main() {
        eprintln!("sidiff: {err}");
        std::process::exit(1);
    }
}

fn try_main() -> io::Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    let options = SidiffOptions {
        use_color: std::env::var_os("NO_COLOR").is_none(),
        ..SidiffOptions::default()
    };

    if io::stdout().is_terminal() {
        run_pager(&input, options)
    } else {
        let mut stdout = io::stdout();
        stdout.write_all(render_diff(&input, options).as_bytes())?;
        stdout.flush()
    }
}
