use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::ExitCode;

struct InitFile {
    path: &'static str,
    contents: &'static str,
    executable: bool,
}

const INIT_FILES: &[InitFile] = &[
    InitFile {
        path: "agents.conf",
        contents: include_str!("../../init/agents.conf"),
        executable: false,
    },
    InitFile {
        path: "tools.conf",
        contents: include_str!("../../init/tools.conf"),
        executable: false,
    },
    InitFile {
        path: "agents/build.md",
        contents: include_str!("../../init/agents/build.md"),
        executable: false,
    },
    InitFile {
        path: "agents/compact.md",
        contents: include_str!("../../init/agents/compact.md"),
        executable: false,
    },
    InitFile {
        path: "agents/skill-inject",
        contents: include_str!("../../init/agents/skill-inject"),
        executable: true,
    },
    InitFile {
        path: "tools/edit",
        contents: include_str!("../../init/tools/edit"),
        executable: true,
    },
    InitFile {
        path: "tools/git-diff",
        contents: include_str!("../../init/tools/git-diff"),
        executable: true,
    },
    InitFile {
        path: "tools/git-diff.json",
        contents: include_str!("../../init/tools/git-diff.json"),
        executable: false,
    },
    InitFile {
        path: "tools/git-status",
        contents: include_str!("../../init/tools/git-status"),
        executable: true,
    },
    InitFile {
        path: "tools/git-status.json",
        contents: include_str!("../../init/tools/git-status.json"),
        executable: false,
    },
    InitFile {
        path: "tools/glob",
        contents: include_str!("../../init/tools/glob"),
        executable: true,
    },
    InitFile {
        path: "tools/glob.json",
        contents: include_str!("../../init/tools/glob.json"),
        executable: false,
    },
    InitFile {
        path: "tools/read",
        contents: include_str!("../../init/tools/read"),
        executable: true,
    },
    InitFile {
        path: "tools/read.json",
        contents: include_str!("../../init/tools/read.json"),
        executable: false,
    },
    InitFile {
        path: "tools/search",
        contents: include_str!("../../init/tools/search"),
        executable: true,
    },
    InitFile {
        path: "tools/search.json",
        contents: include_str!("../../init/tools/search.json"),
        executable: false,
    },
];

fn main() -> ExitCode {
    let sid_home = match std::env::var("SID_HOME") {
        Ok(val) if !val.is_empty() => val,
        _ => {
            eprintln!("sid-init: SID_HOME must be set and non-empty");
            return ExitCode::from(1);
        }
    };

    let root = Path::new(&sid_home);

    for file in INIT_FILES {
        let dest = root.join(file.path);

        if dest.exists() {
            eprintln!("skip: {} (already exists)", dest.display());
            continue;
        }

        if let Some(parent) = dest.parent()
            && let Err(e) = fs::create_dir_all(parent)
        {
            eprintln!(
                "sid-init: cannot create directory {}: {e}",
                parent.display()
            );
            return ExitCode::from(1);
        }

        if let Err(e) = fs::write(&dest, file.contents) {
            eprintln!("sid-init: cannot write {}: {e}", dest.display());
            return ExitCode::from(1);
        }

        if file.executable
            && let Err(e) = fs::set_permissions(&dest, fs::Permissions::from_mode(0o755))
        {
            eprintln!(
                "sid-init: cannot set permissions on {}: {e}",
                dest.display()
            );
            return ExitCode::from(1);
        }

        eprintln!("create: {}", dest.display());
    }

    eprintln!("sid-init: initialized {sid_home}");
    ExitCode::SUCCESS
}
