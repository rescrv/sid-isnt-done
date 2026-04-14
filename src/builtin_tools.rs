use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::Output;

use claudius::FileSystem;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::process::Command;
use utf8path::Path;

use crate::config::TOOL_PROTOCOL_VERSION;
use crate::initialize_sid_workspace_root_env;

#[derive(Debug, Deserialize)]
struct ToolRequestEnvelope {
    protocol_version: u32,
    request_id: String,
    invocation: ToolRequestInvocation,
}

#[derive(Debug, Deserialize)]
struct ToolRequestInvocation {
    input: Map<String, Value>,
}

#[derive(Debug, Serialize)]
struct ToolResultEnvelope {
    protocol_version: u32,
    request_id: String,
    ok: bool,
    output: Option<ToolResultOutput>,
    error: Option<ToolResultError>,
}

#[derive(Debug, Serialize)]
struct ToolResultOutput {
    kind: String,
    text: Option<String>,
}

#[derive(Debug, Serialize)]
struct ToolResultError {
    code: Option<String>,
    message: Option<String>,
}

#[derive(Debug)]
struct ToolInvocation {
    request_file: PathBuf,
    result_file: PathBuf,
    workspace_root: Path<'static>,
}

impl ToolInvocation {
    fn from_env() -> io::Result<Self> {
        let request_file = PathBuf::from(required_env_var("REQUEST_FILE")?);
        let result_file = PathBuf::from(required_env_var("RESULT_FILE")?);
        let workspace_root = required_env_var("WORKSPACE_ROOT")?;
        let workspace_root = Path::new(&workspace_root).into_owned();
        initialize_sid_workspace_root_env(&workspace_root);
        Ok(Self {
            request_file,
            result_file,
            workspace_root,
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
struct BashRequest {
    command: String,
    #[serde(default)]
    restart: bool,
}

pub async fn run_sid_bash_tool() -> io::Result<()> {
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

    match parse_request_input::<BashRequest>(&request.invocation.input) {
        Ok(command) => {
            let _ = command.restart;
            match execute_bash_command(&invocation.workspace_root, &command.command).await {
                Ok(output) => invocation.write_success(&request.request_id, output),
                Err(failure) => invocation.write_failure(&request.request_id, failure),
            }
        }
        Err(failure) => invocation.write_failure(&request.request_id, failure),
    }
}

async fn execute_bash_command(
    workspace_root: &Path<'_>,
    command: &str,
) -> Result<String, ToolFailure> {
    let shell = "/bin/bash";
    let output = Command::new(shell)
        .arg("-lc")
        .arg(command)
        .current_dir(workspace_root.as_str())
        .output()
        .await
        .map_err(|err| {
            ToolFailure::new(
                "bash_launch_failed",
                format!("failed to launch shell command via {shell}: {err}"),
            )
        })?;

    render_bash_output(output)
}

fn render_bash_output(output: Output) -> Result<String, ToolFailure> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut rendered = String::new();
    if !stdout.is_empty() {
        rendered.push_str(&stdout);
        if !stdout.ends_with('\n') {
            rendered.push('\n');
        }
    }
    if !stderr.is_empty() {
        rendered.push_str("stderr:\n");
        rendered.push_str(&stderr);
        if !stderr.ends_with('\n') {
            rendered.push('\n');
        }
    }

    if output.status.success() {
        if rendered.is_empty() {
            rendered.push_str("success\n");
        }
        Ok(rendered)
    } else {
        rendered.push_str(&format!("exit status: {}\n", output.status));
        Err(ToolFailure::new("bash_exit_status", rendered))
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

    match execute_editor_command(&invocation.workspace_root, &request.invocation.input).await {
        Ok(output) => invocation.write_success(&request.request_id, output),
        Err(failure) => invocation.write_failure(&request.request_id, failure),
    }
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
    fn bash_tool_runs_in_workspace() {
        let root = unique_temp_dir("bash-tool");
        fs::create_dir_all(root.as_str()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let output = runtime
            .block_on(execute_bash_command(&root, "pwd"))
            .expect("bash command should succeed");
        let canonical = std::fs::canonicalize(root.as_str()).unwrap();
        let canonical = Path::try_from(canonical).unwrap().into_owned();
        assert_eq!(output, format!("{}\n", canonical.as_str()));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn bash_tool_reports_exit_status() {
        let root = unique_temp_dir("bash-tool");
        fs::create_dir_all(root.as_str()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let failure = runtime
            .block_on(execute_bash_command(&root, "printf nope; exit 7"))
            .expect_err("bash command should fail");
        assert_eq!(failure.code, "bash_exit_status");
        assert!(failure.message.contains("nope"));
        assert!(failure.message.contains("exit status"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

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
}
