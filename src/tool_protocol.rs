/// Wire types for the sid tool protocol.
///
/// The protocol is a request/result JSON exchange between the harness (which
/// serializes requests and deserializes results) and the tool binary (which
/// deserializes requests and serializes results).  Both sides of the protocol
/// share the same types.
use std::fs;
use std::path::Path as StdPath;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use handled::SError;
use serde::{Deserialize, Serialize};

use crate::config::TOOL_PROTOCOL_VERSION;

/// JSON envelope written by the harness before launching a tool.
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolRequestEnvelope {
    /// Protocol version for this request.
    pub protocol_version: u32,
    /// Unique identifier for matching the result back to this request.
    pub request_id: String,
    /// Identifies the canonical tool being invoked.
    pub tool: ToolRequestTool,
    /// Per-call invocation context.
    pub invocation: ToolRequestInvocation,
    /// Agent identity.
    pub agent: ToolRequestAgent,
    /// Workspace context.
    pub workspace: ToolRequestWorkspace,
    /// Scratch file paths.
    pub files: ToolRequestFiles,
}

/// Identifies the canonical tool being invoked.
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolRequestTool {
    /// Canonical tool identifier after alias resolution.
    pub id: String,
}

/// Per-call invocation context written into the request envelope.
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolRequestInvocation {
    /// API-level tool-use identifier.
    pub tool_use_id: String,
    /// Arbitrary JSON input parameters from the model.
    pub input: serde_json::Map<String, serde_json::Value>,
}

/// Agent identity included in the request envelope.
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolRequestAgent {
    /// Agent identifier.
    pub id: String,
}

/// Workspace context included in the request envelope.
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolRequestWorkspace {
    /// Absolute path to the workspace root.
    pub root: String,
    /// Current working directory for the tool invocation.
    pub cwd: String,
}

/// Scratch file paths included in the request envelope.
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolRequestFiles {
    /// Temporary directory for this invocation.
    pub scratch_dir: String,
    /// Process temporary directory for this invocation.
    pub temp_dir: String,
    /// Path where the tool must write its result JSON.
    pub result_file: String,
}

/// JSON envelope written by the tool after execution.
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolResultEnvelope {
    /// Protocol version echoed from the request.
    pub protocol_version: u32,
    /// Request identifier echoed from the request.
    pub request_id: String,
    /// `true` for success, `false` for error.
    pub ok: bool,
    /// Present when `ok` is `true`.
    pub output: Option<ToolResultOutput>,
    /// Present when `ok` is `false`.
    pub error: Option<ToolResultError>,
}

/// Successful output payload inside a result envelope.
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolResultOutput {
    /// Output format; currently only `"text"` is supported.
    pub kind: String,
    /// The text payload.
    pub text: Option<String>,
}

/// Error payload inside a result envelope.
#[derive(Debug, Deserialize, Serialize)]
pub struct ToolResultError {
    /// Machine-readable error code.
    pub code: Option<String>,
    /// Human-readable error message.
    pub message: Option<String>,
}

fn protocol_error(code: &str, message: &str) -> SError {
    SError::new("tool-protocol")
        .with_code(code)
        .with_message(message)
}

/// Validate and extract the text output from a tool result envelope.
pub fn extract_tool_output(
    display_name: &str,
    request_id: &str,
    result: ToolResultEnvelope,
) -> Result<String, String> {
    if result.protocol_version != TOOL_PROTOCOL_VERSION {
        return Err(format!(
            "tool '{display_name}' protocol error: unsupported result protocol version {0}",
            result.protocol_version
        ));
    }
    if result.request_id != request_id {
        return Err(format!(
            "tool '{display_name}' protocol error: request_id mismatch (expected {request_id}, got {result_request_id})",
            result_request_id = result.request_id
        ));
    }

    if result.ok {
        let Some(output) = result.output else {
            return Err(format!(
                "tool '{display_name}' protocol error: missing success output"
            ));
        };
        if output.kind != "text" {
            return Err(format!(
                "tool '{display_name}' protocol error: unsupported output kind '{0}'",
                output.kind
            ));
        }
        let Some(text) = output.text else {
            return Err(format!(
                "tool '{display_name}' protocol error: missing output.text"
            ));
        };
        return Ok(text);
    }

    let Some(error) = result.error else {
        return Err(format!(
            "tool '{display_name}' protocol error: missing error object"
        ));
    };
    let Some(message) = error.message else {
        return Err(format!(
            "tool '{display_name}' protocol error: missing error.message"
        ));
    };
    let _ = error.code;
    Err(message)
}

/// Serialize a value as pretty-printed JSON and write it to `path`.
pub fn write_json_file(path: &StdPath, value: &impl Serialize) -> Result<(), SError> {
    let path_display = path.to_string_lossy();
    let payload = serde_json::to_vec_pretty(value).map_err(|err| {
        protocol_error("json_serialize_error", "failed to serialize JSON file")
            .with_string_field("path", path_display.as_ref())
            .with_string_field("cause", &err.to_string())
    })?;
    fs::write(path, payload).map_err(|err| {
        protocol_error("io_error", "failed to write JSON file")
            .with_string_field("path", path_display.as_ref())
            .with_string_field("cause", &err.to_string())
    })
}

/// Read and deserialize a tool result envelope from `path`.
pub fn read_tool_result(path: &StdPath) -> Result<ToolResultEnvelope, SError> {
    let path_display = path.to_string_lossy();
    let payload = fs::read_to_string(path).map_err(|err| {
        protocol_error("io_error", "failed to read tool result file")
            .with_string_field("path", path_display.as_ref())
            .with_string_field("cause", &err.to_string())
    })?;
    serde_json::from_str(&payload).map_err(|err| {
        protocol_error(
            "invalid_tool_result_json",
            "failed to parse tool result file",
        )
        .with_string_field("path", path_display.as_ref())
        .with_string_field("cause", &err.to_string())
    })
}

/// Generate a unique request identifier for a tool invocation.
pub fn next_request_id() -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("sidreq_{timestamp}_{}_{}", std::process::id(), sequence)
}
