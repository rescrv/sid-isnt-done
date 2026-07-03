//! Repository snapshot renderer for `sidreview`.
//!
//! `sidrepo` turns a git working tree into a synthetic unified diff.  Each
//! text file is rendered as a newly-added file, which lets `sidreview` reuse
//! its existing per-file folding, progress, and syntax-highlighted pure
//! addition path for whole-repository review.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Controls how a repository snapshot is rendered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SidrepoOptions {
    /// Include untracked, non-ignored files alongside tracked files.
    pub include_untracked: bool,
}

impl Default for SidrepoOptions {
    fn default() -> Self {
        Self {
            include_untracked: true,
        }
    }
}

/// Render `repo` as a synthetic unified diff suitable for `sidreview`.
///
/// The repository is discovered with `git rev-parse --show-toplevel`.
/// Tracked files are always included.  When
/// [`SidrepoOptions::include_untracked`] is true, untracked files that are not
/// ignored by git are included as well.  Binary files are represented as binary
/// diff entries instead of embedding their bytes.
pub fn render_repo_diff(repo: &Path, options: SidrepoOptions) -> io::Result<String> {
    let root = git_toplevel(repo)?;
    let paths = git_ls_files(&root, options)?;
    let mut out = String::new();

    for path in paths {
        let full_path = root.join(&path);
        let metadata = match fs::symlink_metadata(&full_path) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err),
        };

        if metadata.file_type().is_symlink() {
            let target = fs::read_link(&full_path)?;
            append_text_file_diff(&mut out, &path, "120000", &target.to_string_lossy());
            continue;
        }

        if !metadata.is_file() {
            continue;
        }

        let mode = regular_file_mode(&metadata);
        let bytes = fs::read(&full_path)?;
        let Some(text) = text_file_content(&bytes) else {
            append_binary_file_diff(&mut out, &path, mode);
            continue;
        };
        append_text_file_diff(&mut out, &path, mode, text);
    }

    Ok(out)
}

fn git_toplevel(repo: &Path) -> io::Result<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "--show-toplevel"])
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let message = if stderr.is_empty() {
            format!("not a git repository: {}", repo.display())
        } else {
            stderr
        };
        return Err(io::Error::new(io::ErrorKind::InvalidInput, message));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let root = stdout.trim_end_matches(['\r', '\n']);
    if root.is_empty() {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "git returned an empty repository root",
        ))
    } else {
        Ok(PathBuf::from(root))
    }
}

fn git_ls_files(root: &Path, options: SidrepoOptions) -> io::Result<BTreeSet<String>> {
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z", "--cached"]);
    if options.include_untracked {
        command.args(["--others", "--exclude-standard"]);
    }

    let output = command.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let message = if stderr.is_empty() {
            "git ls-files failed".to_string()
        } else {
            stderr
        };
        return Err(io::Error::other(message));
    }

    let mut paths = BTreeSet::new();
    for raw_path in output.stdout.split(|byte| *byte == 0) {
        if raw_path.is_empty() {
            continue;
        }
        let path = std::str::from_utf8(raw_path).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("git returned a non-UTF-8 path: {err}"),
            )
        })?;
        paths.insert(path.to_string());
    }
    Ok(paths)
}

fn text_file_content(bytes: &[u8]) -> Option<&str> {
    if bytes.contains(&0) {
        return None;
    }
    std::str::from_utf8(bytes).ok()
}

fn append_binary_file_diff(out: &mut String, path: &str, mode: &str) {
    append_file_header(out, path, mode);
    out.push_str("Binary files /dev/null and ");
    out.push_str(&quote_diff_path("b/", path));
    out.push_str(" differ\n");
}

fn append_text_file_diff(out: &mut String, path: &str, mode: &str, text: &str) {
    append_file_header(out, path, mode);
    out.push_str("--- /dev/null\n");
    out.push_str("+++ ");
    out.push_str(&quote_diff_path("b/", path));
    out.push('\n');
    let line_count = logical_line_count(text);
    let start = if line_count == 0 { 0 } else { 1 };
    out.push_str(&format!("@@ -0,0 +{start},{line_count} @@\n"));
    append_added_lines(out, text);
}

fn append_file_header(out: &mut String, path: &str, mode: &str) {
    out.push_str("diff --git ");
    out.push_str(&quote_diff_path("a/", path));
    out.push(' ');
    out.push_str(&quote_diff_path("b/", path));
    out.push('\n');
    out.push_str("new file mode ");
    out.push_str(mode);
    out.push('\n');
    out.push_str("index 0000000..0000000\n");
}

fn logical_line_count(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    text.as_bytes()
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + usize::from(!text.ends_with('\n'))
}

fn append_added_lines(out: &mut String, text: &str) {
    let mut rest = text;
    while let Some(newline) = rest.find('\n') {
        out.push('+');
        out.push_str(&rest[..newline]);
        out.push('\n');
        rest = &rest[newline + 1..];
    }
    if !rest.is_empty() {
        out.push('+');
        out.push_str(rest);
        out.push('\n');
        out.push_str("\\ No newline at end of file\n");
    }
}

fn quote_diff_path(prefix: &str, path: &str) -> String {
    let full = format!("{prefix}{path}");
    if !needs_quoted_path(&full) {
        return full;
    }

    let mut quoted = String::from("\"");
    for ch in full.chars() {
        match ch {
            '\\' => quoted.push_str("\\\\"),
            '"' => quoted.push_str("\\\""),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            ch if ch.is_control() => {
                quoted.push_str(&format!("\\{:03o}", ch as u32));
            }
            ch => quoted.push(ch),
        }
    }
    quoted.push('"');
    quoted
}

fn needs_quoted_path(path: &str) -> bool {
    path.chars()
        .any(|ch| ch.is_whitespace() || ch == '"' || ch == '\\' || ch.is_control())
}

#[cfg(unix)]
fn regular_file_mode(metadata: &fs::Metadata) -> &'static str {
    use std::os::unix::fs::PermissionsExt;

    if metadata.permissions().mode() & 0o111 != 0 {
        "100755"
    } else {
        "100644"
    }
}

#[cfg(not(unix))]
fn regular_file_mode(_metadata: &fs::Metadata) -> &'static str {
    "100644"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sidiff::parse_unified_diff;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> io::Result<Self> {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("sidrepo-{name}-{}-{unique}", process::id()));
            fs::create_dir(&path)?;
            Ok(Self { path })
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn run_git(cwd: &Path, args: &[&str]) -> bool {
        Command::new("git")
            .current_dir(cwd)
            .args(args)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn renders_tracked_and_untracked_text_files_as_additions() {
        let temp = TestDir::new("tracked-untracked").unwrap();
        if !run_git(&temp.path, &["init"]) {
            return;
        }

        fs::write(temp.path.join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(temp.path.join("tracked.rs"), "fn main() {}\n").unwrap();
        fs::write(temp.path.join("untracked.txt"), "hello").unwrap();
        fs::write(temp.path.join("ignored.txt"), "ignored\n").unwrap();
        assert!(run_git(&temp.path, &["add", ".gitignore", "tracked.rs"]));

        let rendered = render_repo_diff(&temp.path, SidrepoOptions::default()).unwrap();
        assert!(rendered.contains("+++ b/tracked.rs"));
        assert!(rendered.contains("+fn main() {}"));
        assert!(rendered.contains("+++ b/untracked.txt"));
        assert!(rendered.contains("+hello"));
        assert!(rendered.contains("\\ No newline at end of file"));
        assert!(!rendered.contains("+++ b/ignored.txt"));

        let diff = parse_unified_diff(&rendered);
        let paths = diff
            .files
            .iter()
            .filter_map(|file| file.new_path.as_deref())
            .collect::<Vec<_>>();
        assert_eq!(paths, vec![".gitignore", "tracked.rs", "untracked.txt"]);
    }

    #[test]
    fn renders_binary_files_without_embedding_bytes() {
        let temp = TestDir::new("binary").unwrap();
        if !run_git(&temp.path, &["init"]) {
            return;
        }

        fs::write(temp.path.join("image.bin"), b"abc\0def").unwrap();

        let rendered = render_repo_diff(&temp.path, SidrepoOptions::default()).unwrap();
        assert!(rendered.contains("diff --git a/image.bin b/image.bin"));
        assert!(rendered.contains("Binary files /dev/null and b/image.bin differ"));
        assert!(!rendered.contains("+abc"));
    }

    #[test]
    fn quotes_paths_with_spaces_for_sidreview_parsing() {
        let temp = TestDir::new("spaces").unwrap();
        if !run_git(&temp.path, &["init"]) {
            return;
        }

        fs::write(temp.path.join("file with spaces.rs"), "pub fn f() {}\n").unwrap();

        let rendered = render_repo_diff(&temp.path, SidrepoOptions::default()).unwrap();
        assert!(rendered.contains("\"a/file with spaces.rs\""));
        let diff = parse_unified_diff(&rendered);
        assert_eq!(
            diff.files[0].new_path.as_deref(),
            Some("file with spaces.rs")
        );
    }

    #[test]
    fn tracked_only_mode_skips_untracked_files() {
        let temp = TestDir::new("tracked-only").unwrap();
        if !run_git(&temp.path, &["init"]) {
            return;
        }

        fs::write(temp.path.join("tracked.txt"), "tracked\n").unwrap();
        fs::write(temp.path.join("untracked.txt"), "untracked\n").unwrap();
        assert!(run_git(&temp.path, &["add", "tracked.txt"]));

        let rendered = render_repo_diff(
            &temp.path,
            SidrepoOptions {
                include_untracked: false,
            },
        )
        .unwrap();
        assert!(rendered.contains("tracked.txt"));
        assert!(!rendered.contains("untracked.txt"));
    }
}
