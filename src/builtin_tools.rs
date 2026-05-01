use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::path::{Path as StdPath, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use claudius::FileSystem;
use serde::Deserialize;
use serde_json::{Map, Value};
use utf8path::Path;

use crate::config::TOOL_PROTOCOL_VERSION;
use crate::sidiff::render_diff_preview_with_value;
use crate::tool_protocol::{
    ToolRequestEnvelope, ToolResultEnvelope, ToolResultError, ToolResultOutput,
};

static PREVIEW_DIFF_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
const NULL_DEVICE: &str = "/dev/null";

#[cfg(windows)]
const NULL_DEVICE: &str = "NUL";

#[derive(Debug)]
struct ToolInvocation {
    request_file: PathBuf,
    result_file: PathBuf,
    workspace_root: Path<'static>,
    temp_dir: PathBuf,
}

impl ToolInvocation {
    fn from_env() -> io::Result<Self> {
        let request_file = PathBuf::from(required_env_var("REQUEST_FILE")?);
        let result_file = PathBuf::from(required_env_var("RESULT_FILE")?);
        let workspace_root = required_env_var("WORKSPACE_ROOT")?;
        let workspace_root = Path::new(&workspace_root).into_owned();
        let temp_dir = env::var_os("TEMP_DIR")
            .or_else(|| env::var_os("TMPDIR"))
            .map(PathBuf::from)
            .unwrap_or_else(env::temp_dir);
        Ok(Self {
            request_file,
            result_file,
            workspace_root,
            temp_dir,
        })
    }

    fn read_request(&self) -> io::Result<ToolRequestEnvelope> {
        let request = fs::read_to_string(&self.request_file)?;
        serde_json::from_str(&request)
            .map_err(|err| invalid_data("failed to parse request file", err))
    }

    fn write_success(&self, request_id: &str, text: String) -> io::Result<()> {
        self.write_result(ToolResultEnvelope {
            protocol_version: TOOL_PROTOCOL_VERSION,
            request_id: request_id.to_string(),
            ok: true,
            output: Some(ToolResultOutput {
                kind: "text".to_string(),
                text: Some(text),
            }),
            error: None,
        })
    }

    fn write_failure(&self, request_id: &str, failure: ToolFailure) -> io::Result<()> {
        self.write_result(ToolResultEnvelope {
            protocol_version: TOOL_PROTOCOL_VERSION,
            request_id: request_id.to_string(),
            ok: false,
            output: None,
            error: Some(ToolResultError {
                code: Some(failure.code.to_string()),
                message: Some(failure.message),
            }),
        })
    }

    fn write_result(&self, result: ToolResultEnvelope) -> io::Result<()> {
        let payload = serde_json::to_vec_pretty(&result)
            .map_err(|err| invalid_data("failed to serialize result file", err))?;
        fs::write(&self.result_file, payload)
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ToolFailure {
    code: &'static str,
    message: String,
}

impl ToolFailure {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct EditorCommand {
    command: String,
}

#[derive(Debug, Deserialize)]
struct ViewRequest {
    path: String,
    view_range: Option<(u32, u32)>,
}

#[derive(Debug, Deserialize)]
struct StrReplaceRequest {
    path: String,
    old_str: String,
    new_str: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InsertRequest {
    path: String,
    insert_line: u32,
    insert_text: Option<String>,
    new_str: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CreateRequest {
    path: String,
    file_text: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChangedLineRange {
    old_start: usize,
    old_end: usize,
    new_start: usize,
    new_end: usize,
}

pub async fn run_sid_editor_tool() -> io::Result<()> {
    let invocation = ToolInvocation::from_env()?;
    let request = invocation.read_request()?;
    if request.protocol_version != TOOL_PROTOCOL_VERSION {
        return invocation.write_failure(
            &request.request_id,
            ToolFailure::new(
                "unsupported_tool_protocol_version",
                format!(
                    "unsupported request protocol version {}",
                    request.protocol_version
                ),
            ),
        );
    }

    let mode = env::args().nth(1).unwrap_or_else(|| "run".to_string());
    if mode == "confirm" {
        let preview = preview_editor_command(
            &invocation.workspace_root,
            &invocation.temp_dir,
            &request.invocation.input,
        )
        .map_err(|failure| io::Error::new(io::ErrorKind::InvalidInput, failure.message))?;
        println!("{preview}");
        return Ok(());
    }
    if mode != "run" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported sid-editor-tool mode: {mode}"),
        ));
    }

    match execute_editor_command(&invocation.workspace_root, &request.invocation.input).await {
        Ok(output) => invocation.write_success(&request.request_id, output),
        Err(failure) => invocation.write_failure(&request.request_id, failure),
    }
}

fn preview_editor_command(
    workspace_root: &Path<'_>,
    temp_dir: &StdPath,
    input: &Map<String, Value>,
) -> Result<String, ToolFailure> {
    preview_editor_command_with_diff_command(workspace_root, temp_dir, input, env::var_os("DIFF"))
}

fn preview_editor_command_with_diff_command(
    workspace_root: &Path<'_>,
    temp_dir: &StdPath,
    input: &Map<String, Value>,
    diff_command: Option<OsString>,
) -> Result<String, ToolFailure> {
    let command = parse_request_input::<EditorCommand>(input)?;
    match command.command.as_str() {
        "view" => {
            let request = parse_request_input::<ViewRequest>(input)?;
            let range = request
                .view_range
                .map(|(start, end)| format!(" lines {start}-{end}"))
                .unwrap_or_default();
            Ok(format!("View workspace file: {}{}", request.path, range))
        }
        "str_replace" => {
            let request = parse_request_input::<StrReplaceRequest>(input)?;
            let new_str = request.new_str.as_deref().unwrap_or("");
            let before = read_workspace_file_for_edit(workspace_root, &request.path)?;
            let count = before.matches(&request.old_str).count();
            if count == 0 {
                return Err(ToolFailure::new("unsupported", "old_str not found in file"));
            }
            if count > 1 {
                return Err(ToolFailure::new(
                    "unsupported",
                    "old_str found in file more than once",
                ));
            }
            let after = before.replacen(&request.old_str, new_str, 1);
            render_mutating_preview(
                temp_dir,
                &request.path,
                "str_replace",
                Some(&before),
                &after,
                diff_command,
            )
        }
        "insert" => {
            let request = parse_request_input::<InsertRequest>(input)?;
            let text = request
                .insert_text
                .or(request.new_str)
                .ok_or_else(|| ToolFailure::new("invalid_input", "missing insert_text field"))?;
            let before = read_workspace_file_for_edit(workspace_root, &request.path)?;
            let after = preview_insert(&before, request.insert_line, &text)
                .map_err(map_filesystem_error)?;
            render_mutating_preview(
                temp_dir,
                &request.path,
                "insert",
                Some(&before),
                &after,
                diff_command,
            )
        }
        "create" => {
            let request = parse_request_input::<CreateRequest>(input)?;
            let path = sanitize_editor_path(workspace_root, &request.path)?;
            if path.exists() {
                return Err(ToolFailure::new("already_exists", "EEXISTS:  file exists"));
            }
            render_mutating_preview(
                temp_dir,
                &request.path,
                "create",
                None,
                &request.file_text,
                diff_command,
            )
        }
        _ => Err(ToolFailure::new(
            "unsupported_command",
            format!("{} is not a supported editor command", command.command),
        )),
    }
}

fn render_mutating_preview(
    temp_dir: &StdPath,
    path: &str,
    operation: &str,
    before: Option<&str>,
    after: &str,
    diff_command: Option<OsString>,
) -> Result<String, ToolFailure> {
    let display_path = normalize_editor_path(path);
    let lines = changed_line_range(before, after);
    let rendered_diff = build_unified_diff(temp_dir, &display_path, before, after)
        .map_err(map_filesystem_error)?
        .map(
            |unified_diff| match render_diff_preview_with_value(&unified_diff, diff_command) {
                Ok(rendered) if !rendered.trim().is_empty() => rendered,
                Ok(_) => unified_diff,
                Err(err) => {
                    format!("diff renderer failed: {err}\n\n{unified_diff}")
                }
            },
        );

    let before_len = before.map(str::len).unwrap_or(0);
    let after_len = after.len();
    let mut preview = String::new();
    preview.push_str("Edit preview\n");
    preview.push_str(&format!("file: {display_path}\n"));
    preview.push_str(&format!("operation: {operation}\n"));
    if let Some(line_range) = format_line_range(lines) {
        preview.push_str(&format!("line range: {line_range}\n"));
    }
    preview.push_str(&format!(
        "file bytes: {before_len} -> {after_len} ({})",
        format_signed_delta(before_len, after_len)
    ));
    if let Some(rendered_diff) = rendered_diff {
        preview.push_str("\n\n");
        preview.push_str(rendered_diff.trim_end());
    }
    Ok(preview)
}

fn normalize_editor_path(path: &str) -> String {
    Path::from(path)
        .as_str()
        .trim_start_matches('/')
        .to_string()
}

fn sanitize_editor_path(workspace_root: &Path<'_>, path: &str) -> Result<PathBuf, ToolFailure> {
    let path = Path::from(path);
    if path
        .components()
        .any(|component| matches!(component, utf8path::Component::AppDefined))
    {
        return Err(ToolFailure::new(
            "unsupported",
            "viewing // paths is not supported",
        ));
    }
    if path
        .components()
        .any(|component| matches!(component, utf8path::Component::ParentDir))
    {
        return Err(ToolFailure::new("unsupported", ".. path name prohibited"));
    }
    Ok(PathBuf::from(
        workspace_root
            .join(path.as_str().trim_start_matches('/'))
            .as_str(),
    ))
}

fn read_workspace_file_for_edit(
    workspace_root: &Path<'_>,
    path: &str,
) -> Result<String, ToolFailure> {
    let path = sanitize_editor_path(workspace_root, path)?;
    if !path.is_file() {
        return Err(ToolFailure::new(
            "unsupported",
            "editing non-standard file types is not supported",
        ));
    }
    fs::read_to_string(path).map_err(map_filesystem_error)
}

fn preview_insert(content: &str, insert_line: u32, insert_text: &str) -> io::Result<String> {
    let mut lines = content
        .split_terminator('\n')
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    let insert_idx = insert_line as usize;
    if insert_idx > lines.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "insert_line out of range",
        ));
    }
    lines.insert(insert_idx, insert_text.to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    Ok(out)
}

fn changed_line_range(before: Option<&str>, after: &str) -> ChangedLineRange {
    let before_text = before.unwrap_or("");
    let before_lines = diff_lines(before_text);
    let after_lines = diff_lines(after);
    let prefix = common_prefix_len(&before_lines, &after_lines);
    let suffix = common_suffix_len(&before_lines[prefix..], &after_lines[prefix..]);
    ChangedLineRange {
        old_start: prefix,
        old_end: before_lines.len().saturating_sub(suffix),
        new_start: prefix,
        new_end: after_lines.len().saturating_sub(suffix),
    }
}

fn diff_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        Vec::new()
    } else {
        text.split_terminator('\n').collect()
    }
}

fn common_prefix_len(before: &[&str], after: &[&str]) -> usize {
    let mut prefix = 0;
    let max_prefix = usize::min(before.len(), after.len());
    while prefix < max_prefix && before[prefix] == after[prefix] {
        prefix += 1;
    }
    prefix
}

fn common_suffix_len(before: &[&str], after: &[&str]) -> usize {
    let mut suffix = 0;
    let max_suffix = usize::min(before.len(), after.len());
    while suffix < max_suffix
        && before[before.len() - 1 - suffix] == after[after.len() - 1 - suffix]
    {
        suffix += 1;
    }
    suffix
}

fn format_line_range(lines: ChangedLineRange) -> Option<String> {
    let old_count = lines.old_end.saturating_sub(lines.old_start);
    let new_count = lines.new_end.saturating_sub(lines.new_start);
    match (old_count, new_count) {
        (0, 0) => None,
        (0, _) => Some(format!(
            "new {} added",
            line_span(lines.new_start, lines.new_end)
        )),
        (_, 0) => Some(format!(
            "old {} removed",
            line_span(lines.old_start, lines.old_end)
        )),
        _ => Some(format!(
            "old {} -> new {}",
            line_span(lines.old_start, lines.old_end),
            line_span(lines.new_start, lines.new_end)
        )),
    }
}

fn line_span(start: usize, end: usize) -> String {
    let first = start + 1;
    let last = end;
    if first == last {
        first.to_string()
    } else {
        format!("{first}-{last}")
    }
}

fn format_signed_delta(before_len: usize, after_len: usize) -> String {
    let delta = after_len as i64 - before_len as i64;
    if delta >= 0 {
        format!("+{delta}")
    } else {
        delta.to_string()
    }
}

fn build_unified_diff(
    temp_dir: &StdPath,
    path: &str,
    before: Option<&str>,
    after: &str,
) -> io::Result<Option<String>> {
    build_unified_diff_with_program(temp_dir, OsStr::new("diff"), path, before, after)
}

fn build_unified_diff_with_program(
    temp_dir: &StdPath,
    program: &OsStr,
    path: &str,
    before: Option<&str>,
    after: &str,
) -> io::Result<Option<String>> {
    let workspace = PreviewDiffWorkspace::new(temp_dir, before, after)?;
    let old_label = before
        .map(|_| format!("a/{path}"))
        .unwrap_or_else(|| NULL_DEVICE.to_string());
    let new_label = format!("b/{path}");
    let output = match Command::new(program)
        .arg("-u")
        .arg("-L")
        .arg(&old_label)
        .arg("-L")
        .arg(&new_label)
        .arg(workspace.before_path())
        .arg(workspace.after_path())
        .output()
    {
        Ok(output) => output,
        Err(_) => return Ok(None),
    };

    match output.status.code() {
        Some(0) | Some(1) => Ok(Some(String::from_utf8_lossy(&output.stdout).into_owned())),
        _ => {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let detail = if stderr.is_empty() {
                output.status.to_string()
            } else {
                format!("{}: {stderr}", output.status)
            };
            Err(io::Error::other(format!("diff command failed: {detail}")))
        }
    }
}

struct PreviewDiffWorkspace {
    root: PathBuf,
    before: Option<PathBuf>,
    after: PathBuf,
}

impl PreviewDiffWorkspace {
    fn new(temp_dir: &StdPath, before: Option<&str>, after: &str) -> io::Result<Self> {
        fs::create_dir_all(temp_dir)?;
        let root = create_preview_diff_dir(temp_dir);
        fs::create_dir(&root)?;
        let before = if let Some(content) = before {
            let path = root.join("before");
            fs::write(&path, content)?;
            Some(path)
        } else {
            None
        };
        let after_path = root.join("after");
        fs::write(&after_path, after)?;
        Ok(Self {
            root,
            before,
            after: after_path,
        })
    }

    fn before_path(&self) -> &StdPath {
        self.before
            .as_deref()
            .unwrap_or_else(|| StdPath::new(NULL_DEVICE))
    }

    fn after_path(&self) -> &StdPath {
        &self.after
    }
}

impl Drop for PreviewDiffWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn create_preview_diff_dir(temp_dir: &StdPath) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = PREVIEW_DIFF_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    temp_dir.join(format!(
        "sid-editor-preview-{}-{timestamp}-{sequence}",
        std::process::id()
    ))
}

async fn execute_editor_command(
    workspace_root: &Path<'_>,
    input: &Map<String, Value>,
) -> Result<String, ToolFailure> {
    let command = parse_request_input::<EditorCommand>(input)?;
    match command.command.as_str() {
        "view" => {
            let request = parse_request_input::<ViewRequest>(input)?;
            workspace_root
                .view(&request.path, request.view_range)
                .await
                .map_err(map_filesystem_error)
        }
        "str_replace" => {
            let request = parse_request_input::<StrReplaceRequest>(input)?;
            workspace_root
                .str_replace(
                    &request.path,
                    &request.old_str,
                    request.new_str.as_deref().unwrap_or(""),
                )
                .await
                .map_err(map_filesystem_error)
        }
        "insert" => {
            let request = parse_request_input::<InsertRequest>(input)?;
            let text = request
                .insert_text
                .or(request.new_str)
                .ok_or_else(|| ToolFailure::new("invalid_input", "missing insert_text field"))?;
            workspace_root
                .insert(&request.path, request.insert_line, &text)
                .await
                .map_err(map_filesystem_error)
        }
        "create" => {
            let request = parse_request_input::<CreateRequest>(input)?;
            workspace_root
                .create(&request.path, &request.file_text)
                .await
                .map_err(map_filesystem_error)
        }
        _ => Err(ToolFailure::new(
            "unsupported_command",
            format!("{} is not a supported editor command", command.command),
        )),
    }
}

fn parse_request_input<T: for<'de> Deserialize<'de>>(
    input: &Map<String, Value>,
) -> Result<T, ToolFailure> {
    serde_json::from_value(Value::Object(input.clone()))
        .map_err(|err| ToolFailure::new("invalid_input", err.to_string()))
}

fn map_filesystem_error(err: io::Error) -> ToolFailure {
    let code = match err.kind() {
        io::ErrorKind::InvalidInput => "invalid_input",
        io::ErrorKind::AlreadyExists => "already_exists",
        io::ErrorKind::NotFound => "not_found",
        io::ErrorKind::PermissionDenied => "permission_denied",
        io::ErrorKind::Unsupported => "unsupported",
        _ => "io_error",
    };
    ToolFailure::new(code, err.to_string())
}

fn required_env_var(name: &str) -> io::Result<String> {
    env::var(name)
        .map_err(|err| io::Error::new(io::ErrorKind::NotFound, format!("missing {name}: {err}")))
}

fn invalid_data(context: &str, err: impl std::error::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{context}: {err}"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::*;
    use crate::test_support::unique_temp_dir;

    #[test]
    fn editor_tool_replaces_text_in_workspace() {
        let root = unique_temp_dir("editor-tool");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("file.txt").as_str(), "hello old world\n").unwrap();

        let input = json!({
            "command": "str_replace",
            "path": "file.txt",
            "old_str": "old",
            "new_str": "new"
        });
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let output = runtime
            .block_on(execute_editor_command(
                &root,
                input.as_object().expect("input must be an object"),
            ))
            .expect("editor command should succeed");
        assert_eq!(output, "success");
        assert_eq!(
            fs::read_to_string(root.join("file.txt").as_str()).unwrap(),
            "hello new world\n"
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn editor_tool_views_workspace_file() {
        let root = unique_temp_dir("editor-tool");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("file.txt").as_str(), "line one\nline two\n").unwrap();

        let input = json!({
            "command": "view",
            "path": "file.txt"
        });
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let output = runtime
            .block_on(execute_editor_command(
                &root,
                input.as_object().expect("input must be an object"),
            ))
            .expect("editor command should succeed");
        assert_eq!(output, "line one\nline two\n\n");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn editor_tool_absolute_path_is_workspace_rooted() {
        let root = unique_temp_dir("editor-tool");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("foo").as_str(), "workspace foo\n").unwrap();

        let input = json!({
            "command": "view",
            "path": "/foo"
        });
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let output = runtime
            .block_on(execute_editor_command(
                &root,
                input.as_object().expect("input must be an object"),
            ))
            .expect("editor command should succeed");
        assert_eq!(output, "workspace foo\n\n");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn editor_tool_previews_mutating_commands() {
        let root = unique_temp_dir("editor-tool");
        let temp_dir = StdPath::new(root.as_str());
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("file.txt").as_str(), "line one\nline two\n").unwrap();

        let create = json!({
            "command": "create",
            "path": "new.txt",
            "file_text": "hello"
        });
        let create_preview = preview_editor_command_with_diff_command(
            &root,
            temp_dir,
            create.as_object().unwrap(),
            Some(OsString::from("")),
        )
        .unwrap();
        assert!(create_preview.contains("file: new.txt"));
        assert!(create_preview.contains("operation: create"));
        assert!(create_preview.contains("line range: new 1 added"));
        assert!(create_preview.contains("file bytes: 0 -> 5 (+5)"));
        assert!(create_preview.contains("--- /dev/null"));
        assert!(create_preview.contains("+++ b/new.txt"));
        assert!(create_preview.contains("@@ -0,0 +1 @@"));
        assert!(create_preview.contains("+hello"));

        let replace = json!({
            "command": "str_replace",
            "path": "file.txt",
            "old_str": "line two",
            "new_str": "changed line"
        });
        let replace_preview = preview_editor_command_with_diff_command(
            &root,
            temp_dir,
            replace.as_object().unwrap(),
            Some(OsString::from("")),
        )
        .unwrap();
        assert!(replace_preview.contains("file: file.txt"));
        assert!(replace_preview.contains("operation: str_replace"));
        assert!(replace_preview.contains("line range: old 2 -> new 2"));
        assert!(replace_preview.contains("file bytes: 18 -> 22 (+4)"));
        assert!(replace_preview.contains("--- a/file.txt"));
        assert!(replace_preview.contains("+++ b/file.txt"));
        assert!(replace_preview.contains("-line two"));
        assert!(replace_preview.contains("+changed line"));

        let insert = json!({
            "command": "insert",
            "path": "file.txt",
            "insert_line": 1,
            "insert_text": "abc"
        });
        let insert_preview = preview_editor_command_with_diff_command(
            &root,
            temp_dir,
            insert.as_object().unwrap(),
            Some(OsString::from("")),
        )
        .unwrap();
        assert!(insert_preview.contains("file: file.txt"));
        assert!(insert_preview.contains("operation: insert"));
        assert!(insert_preview.contains("line range: new 2 added"));
        assert!(insert_preview.contains("file bytes: 18 -> 22 (+4)"));
        assert!(insert_preview.contains("@@ -1,2 +1,3 @@"));
        assert!(insert_preview.contains(" line one"));
        assert!(insert_preview.contains("+abc"));
        assert!(insert_preview.contains(" line two"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn editor_tool_preview_preserves_unchanged_middle_lines_in_multiline_replace() {
        let root = unique_temp_dir("editor-tool");
        let temp_dir = StdPath::new(root.as_str());
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("file.txt").as_str(), "before\nkeep\nafter\n").unwrap();

        let replace = json!({
            "command": "str_replace",
            "path": "file.txt",
            "old_str": "before\nkeep\nafter",
            "new_str": "start\nkeep\nend"
        });
        let preview = preview_editor_command_with_diff_command(
            &root,
            temp_dir,
            replace.as_object().unwrap(),
            Some(OsString::from("")),
        )
        .unwrap();

        assert!(preview.contains("@@ -1,3 +1,3 @@"));
        assert!(preview.contains("-before"));
        assert!(preview.contains("+start"));
        assert!(preview.contains(" keep"));
        assert!(!preview.contains("-keep"));
        assert!(!preview.contains("+keep"));
        assert!(preview.contains("-after"));
        assert!(preview.contains("+end"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn editor_tool_previews_view_range() {
        let root = unique_temp_dir("editor-tool");
        let temp_dir = StdPath::new(root.as_str());
        let input = json!({
            "command": "view",
            "path": "file.txt",
            "view_range": [2, 4]
        });
        let preview = preview_editor_command(&root, temp_dir, input.as_object().unwrap()).unwrap();
        assert_eq!(preview, "View workspace file: file.txt lines 2-4");
    }

    #[test]
    fn missing_diff_binary_omits_preview_diff() {
        let root = unique_temp_dir("editor-tool");
        let temp_dir = StdPath::new(root.as_str());
        let diff = build_unified_diff_with_program(
            temp_dir,
            OsStr::new("sid-diff-command-does-not-exist"),
            "file.txt",
            Some("before\n"),
            "after\n",
        )
        .unwrap();
        assert_eq!(diff, None);
    }
}
