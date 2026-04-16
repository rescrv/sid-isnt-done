/// Tool invocation runtime for executing rc-conf-based tools.
///
/// Handles the lifecycle of invoking external tools: constructing request
/// envelopes, preparing the rc-conf overlay, launching the tool process,
/// and reading back the result.
use std::collections::HashMap;
use std::fs;
use std::path::Path as StdPath;
use std::process::Stdio;

use handled::SError;
use rc_conf::{RcConf, var_name_from_service, var_prefix_from_service};
use tokio::process::Command;
use utf8path::Path;

use crate::config::{TOOL_PROTOCOL_VERSION, TOOLS_CONF_FILE, TOOLS_DIR};
use crate::tool_protocol::{
    ToolRequestAgent, ToolRequestEnvelope, ToolRequestFiles, ToolRequestInvocation,
    ToolRequestTool, ToolRequestWorkspace, create_tool_scratch_dir, extract_tool_output,
    next_request_id, read_tool_result, write_json_file,
};

/// Minimal context needed by the tool invocation runtime.
///
/// Decouples the tool runtime from the full agent type so the invocation
/// pipeline does not depend on agent construction.
pub(crate) struct ToolRuntimeContext<'a> {
    /// Agent identifier included in request envelopes.
    pub(crate) agent_id: &'a str,
    /// Root directory where rc-conf tool configuration lives.
    pub(crate) config_root: &'a Path<'a>,
    /// Workspace root used as cwd and included in request envelopes.
    pub(crate) workspace_root: &'a Path<'a>,
}

#[derive(Debug)]
struct ToolRcRuntime {
    rc_conf_path: String,
    rc_d_path: String,
    bindings: HashMap<String, String>,
}

struct ToolOverlayContext<'a> {
    request_file: &'a StdPath,
    result_file: &'a StdPath,
    scratch_dir: &'a StdPath,
    rc_conf_path: &'a str,
    rc_d_path: &'a str,
}

/// Invoke an rc-conf-based tool and return its text output.
///
/// Constructs a request envelope, writes it to a scratch directory, launches the
/// tool process with the appropriate rc-conf bindings, and reads back the result.
pub(crate) async fn invoke_rc_tool_text(
    display_name: &str,
    rc_service_name: &str,
    canonical_id: &str,
    executable_path: &Path<'_>,
    context: &ToolRuntimeContext<'_>,
    tool_use_id: &str,
    input: serde_json::Map<String, serde_json::Value>,
) -> Result<String, String> {
    let request_id = next_request_id();
    let scratch_dir = create_tool_scratch_dir(&request_id).map_err(|err| {
        format!(
            "tool '{}' failed to create scratch directory: {}",
            display_name, err
        )
    })?;
    let request_file = scratch_dir.join("request.json");
    let result_file = scratch_dir.join("result.json");

    let request = ToolRequestEnvelope {
        protocol_version: TOOL_PROTOCOL_VERSION,
        request_id: request_id.clone(),
        tool: ToolRequestTool {
            id: canonical_id.to_string(),
        },
        invocation: ToolRequestInvocation {
            tool_use_id: tool_use_id.to_string(),
            input,
        },
        agent: ToolRequestAgent {
            id: context.agent_id.to_string(),
        },
        workspace: ToolRequestWorkspace {
            root: context.workspace_root.as_str().to_string(),
            cwd: context.workspace_root.as_str().to_string(),
        },
        files: ToolRequestFiles {
            scratch_dir: scratch_dir.to_string_lossy().into_owned(),
            result_file: result_file.to_string_lossy().into_owned(),
        },
    };

    write_json_file(&request_file, &request).map_err(|err| {
        format!(
            "tool '{}' failed to write request.json: {}",
            display_name, err
        )
    })?;

    let runtime = prepare_tool_rc_runtime(
        display_name,
        rc_service_name,
        executable_path,
        context,
        &request_file,
        &result_file,
        &scratch_dir,
    )
    .map_err(|err| {
        format!(
            "tool '{}' failed to prepare rc invocation: {}",
            display_name, err
        )
    })?;

    let status = Command::new(executable_path.as_str())
        .arg("run")
        .current_dir(context.workspace_root.as_str())
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .envs(&runtime.bindings)
        .env("RCVAR_ARGV0", var_name_from_service(rc_service_name))
        .env("RC_CONF_PATH", &runtime.rc_conf_path)
        .env("RC_D_PATH", &runtime.rc_d_path)
        .status()
        .await
        .map_err(|err| format!("tool '{}' failed to launch: {}", display_name, err))?;
    if !status.success() {
        return Err(format!(
            "tool '{}' exited with status {}",
            display_name, status
        ));
    }

    let result = read_tool_result(&result_file)
        .map_err(|err| format!("tool '{}' protocol error: {}", display_name, err))?;
    extract_tool_output(display_name, &request_id, result)
}

fn prepare_tool_rc_runtime(
    display_name: &str,
    rc_service_name: &str,
    executable_path: &Path,
    context: &ToolRuntimeContext<'_>,
    request_file: &StdPath,
    result_file: &StdPath,
    scratch_dir: &StdPath,
) -> Result<ToolRcRuntime, SError> {
    let tools_conf_path = context.config_root.join(TOOLS_CONF_FILE);
    let rc_d_path = context.config_root.join(TOOLS_DIR);
    let overlay_path = scratch_dir.join("tool-invoke.conf");
    let base_rc_conf = RcConf::parse(tools_conf_path.as_str()).map_err(|err| {
        tool_runtime_error(display_name, "rc_conf_error", "failed to parse tools.conf")
            .with_string_field("path", tools_conf_path.as_str())
            .with_string_field("cause", &format!("{err:?}"))
    })?;
    let services = base_rc_conf.list().map_err(|err| {
        tool_runtime_error(
            display_name,
            "rc_conf_error",
            "failed to list configured tools",
        )
        .with_string_field("path", tools_conf_path.as_str())
        .with_string_field("cause", &format!("{err:?}"))
    })?;
    let services = services.collect::<Vec<_>>();
    let rc_conf_path = format!("{}:{}", tools_conf_path.as_str(), overlay_path.display());
    let rc_d_path = rc_d_path.as_str().to_string();
    let overlay_context = ToolOverlayContext {
        request_file,
        result_file,
        scratch_dir,
        rc_conf_path: &rc_conf_path,
        rc_d_path: &rc_d_path,
    };
    let overlay = render_tool_rc_overlay(&base_rc_conf, &services, context, &overlay_context);
    fs::write(&overlay_path, overlay).map_err(|err| {
        tool_runtime_error(display_name, "io_error", "failed to write tool rc overlay")
            .with_string_field("path", overlay_path.to_string_lossy().as_ref())
            .with_string_field("cause", &err.to_string())
    })?;
    let rc_conf = RcConf::parse(&rc_conf_path).map_err(|err| {
        tool_runtime_error(
            display_name,
            "rc_conf_error",
            "failed to parse tool rc overlay",
        )
        .with_string_field("path", &rc_conf_path)
        .with_string_field("cause", &format!("{err:?}"))
    })?;
    let bindings = rc_conf
        .bind_for_invoke(rc_service_name, executable_path)
        .map_err(|err| {
            tool_runtime_error(
                display_name,
                "rc_conf_error",
                "failed to bind rcvars for tool invocation",
            )
            .with_string_field("path", executable_path.as_str())
            .with_string_field("cause", &format!("{err:?}"))
        })?;

    Ok(ToolRcRuntime {
        rc_conf_path,
        rc_d_path,
        bindings,
    })
}

fn render_tool_rc_overlay(
    rc_conf: &RcConf,
    services: &[String],
    context: &ToolRuntimeContext<'_>,
    overlay_context: &ToolOverlayContext<'_>,
) -> String {
    let request_file = overlay_context.request_file.to_string_lossy().into_owned();
    let result_file = overlay_context.result_file.to_string_lossy().into_owned();
    let scratch_dir = overlay_context.scratch_dir.to_string_lossy().into_owned();
    let workspace_root = context.workspace_root.as_str().to_string();
    let tool_protocol = TOOL_PROTOCOL_VERSION.to_string();
    let mut overlay = String::new();

    for service in services {
        let prefix = var_prefix_from_service(service);
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}REQUEST_FILE"),
            &request_file,
        );
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}RESULT_FILE"), &result_file);
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}SCRATCH_DIR"), &scratch_dir);
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}WORKSPACE_ROOT"),
            &workspace_root,
        );
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}AGENT_ID"), context.agent_id);
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}TOOL_ID"),
            rc_conf.resolve_alias(service),
        );
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}TOOL_NAME"), service);
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}TOOL_PROTOCOL"),
            &tool_protocol,
        );
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}RC_CONF_PATH"),
            overlay_context.rc_conf_path,
        );
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}RC_D_PATH"),
            overlay_context.rc_d_path,
        );
    }

    overlay
}

fn append_rc_conf_assignment(output: &mut String, name: &str, value: &str) {
    output.push_str(name);
    output.push('=');
    output.push_str(&shvar::quote(vec![value.to_string()]));
    output.push('\n');
}

fn tool_runtime_error(tool: &str, code: &str, message: &str) -> SError {
    SError::new("tool-runtime")
        .with_code(code)
        .with_message(message)
        .with_string_field("tool", tool)
}
