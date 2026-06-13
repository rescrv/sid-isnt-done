//! Git checkpoints: before each agent invocation the tree state is committed
//! to `refs/sid/ralph/<run-id>/<step>` so the judge can diff what changed
//! since it last looked rather than re-reading the world.
//!
//! Checkpointing never disturbs the operator's index or worktree: it uses a
//! private index file inside the run directory.

use std::path::Path;
use std::process::Command;

/// The ref name for a checkpoint.
pub fn checkpoint_ref(run_id: &str, step: u64) -> String {
    format!("refs/sid/ralph/{run_id}/{step}")
}

/// Commit the current tree state (tracked and untracked, honoring
/// .gitignore) to `refs/sid/ralph/<run-id>/<step>`.
///
/// Returns `Ok(None)` when the workspace is not a git repository, `Ok(Some(ref))`
/// on success, and `Err` for git failures.
pub fn create_checkpoint(
    workspace_root: &Path,
    run_dir: &Path,
    run_id: &str,
    step: u64,
) -> Result<Option<String>, String> {
    if !is_git_repository(workspace_root) {
        return Ok(None);
    }
    let index_file = run_dir.join("checkpoint-index");
    let index = index_file.to_string_lossy().into_owned();

    // Stage the whole tree into the private index.
    run_git(workspace_root, Some(&index), &["add", "-A", "--", "."])?;
    let tree = run_git(workspace_root, Some(&index), &["write-tree"])?;
    let tree = tree.trim();

    let mut commit_args = vec!["commit-tree".to_string(), tree.to_string()];
    if let Ok(head) = run_git(workspace_root, None, &["rev-parse", "--verify", "HEAD"]) {
        commit_args.push("-p".to_string());
        commit_args.push(head.trim().to_string());
    }
    commit_args.push("-m".to_string());
    commit_args.push(format!("sid ralph checkpoint: run {run_id} step {step}"));
    let commit_arg_refs: Vec<&str> = commit_args.iter().map(String::as_str).collect();
    let commit = run_git_env(
        workspace_root,
        None,
        &commit_arg_refs,
        &[
            ("GIT_AUTHOR_NAME", "sid-ralph"),
            ("GIT_AUTHOR_EMAIL", "ralph@sid.invalid"),
            ("GIT_COMMITTER_NAME", "sid-ralph"),
            ("GIT_COMMITTER_EMAIL", "ralph@sid.invalid"),
        ],
    )?;
    let commit = commit.trim();

    let ref_name = checkpoint_ref(run_id, step);
    run_git(workspace_root, None, &["update-ref", &ref_name, commit])?;
    Ok(Some(ref_name))
}

fn is_git_repository(workspace_root: &Path) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(workspace_root)
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn run_git(
    workspace_root: &Path,
    index_file: Option<&str>,
    args: &[&str],
) -> Result<String, String> {
    run_git_env(workspace_root, index_file, args, &[])
}

fn run_git_env(
    workspace_root: &Path,
    index_file: Option<&str>,
    args: &[&str],
    env: &[(&str, &str)],
) -> Result<String, String> {
    let mut command = Command::new("git");
    command.arg("-C").arg(workspace_root).args(args);
    if let Some(index) = index_file {
        command.env("GIT_INDEX_FILE", index);
    }
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command
        .output()
        .map_err(|err| format!("failed to spawn git: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sid-ralph-checkpoint-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn git(dir: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_AUTHOR_NAME", "test")
            .env("GIT_AUTHOR_EMAIL", "test@test.invalid")
            .env("GIT_COMMITTER_NAME", "test")
            .env("GIT_COMMITTER_EMAIL", "test@test.invalid")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    #[test]
    fn ref_name_shape() {
        assert_eq!(
            checkpoint_ref("2026-06-10T14-22-07", 3),
            "refs/sid/ralph/2026-06-10T14-22-07/3"
        );
    }

    #[test]
    fn non_repo_is_skipped() {
        let dir = temp_dir("nonrepo");
        let run_dir = dir.join("run");
        fs::create_dir_all(&run_dir).unwrap();
        // Guard: the temp dir must not be inside a git repository for this
        // test to be meaningful; if it is, skip.
        if is_git_repository(&dir) {
            fs::remove_dir_all(&dir).unwrap();
            return;
        }
        assert_eq!(create_checkpoint(&dir, &run_dir, "r", 1).unwrap(), None);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn checkpoints_capture_untracked_files_without_touching_the_index() {
        let dir = temp_dir("repo");
        let run_dir = dir.join(".run");
        fs::create_dir_all(&run_dir).unwrap();
        git(&dir, &["init", "-q"]);
        fs::write(dir.join("tracked.txt"), "v1\n").unwrap();
        git(&dir, &["add", "tracked.txt"]);
        git(&dir, &["commit", "-q", "-m", "initial"]);

        fs::write(dir.join("tracked.txt"), "v2\n").unwrap();
        fs::write(dir.join("untracked.txt"), "new\n").unwrap();

        let reference = create_checkpoint(&dir, &run_dir, "run-1", 1)
            .unwrap()
            .expect("checkpoint in a git repo");
        assert_eq!(reference, "refs/sid/ralph/run-1/1");

        // The checkpoint contains both the modification and the untracked file.
        let shown = git(&dir, &["show", &format!("{reference}:tracked.txt")]);
        assert_eq!(shown, "v2\n");
        let shown = git(&dir, &["show", &format!("{reference}:untracked.txt")]);
        assert_eq!(shown, "new\n");

        // The operator's index is untouched: untracked.txt is still untracked.
        let status = git(&dir, &["status", "--porcelain"]);
        assert!(status.contains("?? untracked.txt"), "status: {status}");

        // The checkpoint commit has HEAD as parent, so `git diff <ref>` works.
        let parent = git(&dir, &["rev-parse", &format!("{reference}^")]);
        let head = git(&dir, &["rev-parse", "HEAD"]);
        assert_eq!(parent, head);

        // A second checkpoint gets its own ref.
        let second = create_checkpoint(&dir, &run_dir, "run-1", 2)
            .unwrap()
            .unwrap();
        assert_eq!(second, "refs/sid/ralph/run-1/2");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn checkpoints_work_on_unborn_head() {
        let dir = temp_dir("unborn");
        let run_dir = dir.join(".run");
        fs::create_dir_all(&run_dir).unwrap();
        git(&dir, &["init", "-q"]);
        fs::write(dir.join("file.txt"), "data\n").unwrap();
        let reference = create_checkpoint(&dir, &run_dir, "run-2", 1)
            .unwrap()
            .unwrap();
        let shown = git(&dir, &["show", &format!("{reference}:file.txt")]);
        assert_eq!(shown, "data\n");
        fs::remove_dir_all(&dir).unwrap();
    }
}
