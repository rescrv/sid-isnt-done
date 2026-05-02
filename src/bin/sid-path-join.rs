use std::process::ExitCode;

use utf8path::{Component, Path};

fn join_to_workspace_root(
    workspace_root: &Path<'_>,
    path_to_append: &Path<'_>,
) -> Result<String, &'static str> {
    if path_to_append.is_abs() {
        return Err("path to append must be relative to the workspace root");
    }
    if path_to_append
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err("path to append must not contain '..' path traversals");
    }
    Ok(workspace_root
        .join(path_to_append.as_str())
        .as_str()
        .to_string())
}

fn usage(program: &str) {
    eprintln!("usage: {program} WORKSPACE_ROOT PATH");
}

fn main() -> ExitCode {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        usage(args.first().map(String::as_str).unwrap_or("sid-path-join"));
        return ExitCode::from(129);
    }

    let workspace_root = Path::new(&args[1]);
    let path_to_append = Path::new(&args[2]);
    match join_to_workspace_root(&workspace_root, &path_to_append) {
        Ok(path) => {
            println!("{path}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::join_to_workspace_root;

    use utf8path::Path;

    #[test]
    fn joins_relative_paths() {
        let joined = join_to_workspace_root(&Path::new("/workspace"), &Path::new("src/lib.rs"))
            .expect("relative path should join");
        assert_eq!(joined, "/workspace/src/lib.rs");
    }

    #[test]
    fn rejects_absolute_paths() {
        let err = join_to_workspace_root(&Path::new("/workspace"), &Path::new("/src/lib.rs"))
            .expect_err("absolute path should be rejected");
        assert_eq!(err, "path to append must be relative to the workspace root");
    }

    #[test]
    fn rejects_parent_dir_traversals() {
        let err = join_to_workspace_root(&Path::new("/workspace"), &Path::new("src/../lib.rs"))
            .expect_err("parent dir traversal should be rejected");
        assert_eq!(err, "path to append must not contain '..' path traversals");
    }
}
