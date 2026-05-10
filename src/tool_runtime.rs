/// Tool invocation runtime for executing rc-conf-based tools.
///
/// Handles the lifecycle of invoking external tools: constructing request
/// envelopes, preparing the rc-conf overlay, launching the tool process,
/// and reading back the result.
use std::collections::HashMap;
use std::fs;
use std::io;
use std::io::ErrorKind;
use std::path::Path as StdPath;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use handled::SError;
use rc_conf::{RcConf, var_name_from_service, var_prefix_from_service};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use utf8path::Path;

use crate::config::{TOOL_PROTOCOL_VERSION, TOOLS_CONF_FILE, TOOLS_DIR};
use crate::raw_protocol::{
    ToolOutputEvent, has_active_tool_output_observer, notify_tool_output_observer,
};
use crate::seatbelt;
use crate::seatbelt::WritableRoots;
use crate::session::{self, SidSession, ToolFinishEvent, ToolStartEvent, ToolStreamJournal};
use crate::tool_protocol::{
    ToolRequestAgent, ToolRequestEnvelope, ToolRequestFiles, ToolRequestInvocation,
    ToolRequestTool, ToolRequestWorkspace, extract_tool_output, next_request_id, read_tool_result,
    write_json_file,
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
    /// Directories the sandboxed tool may write to.
    pub(crate) writable_roots: &'a WritableRoots,
    /// Session artifact root for logs and ordered tool directories.
    pub(crate) session: Option<&'a SidSession>,
}

#[derive(Debug)]
struct ToolRcRuntime {
    rc_conf_path: String,
    rc_d_path: String,
    bindings: HashMap<String, String>,
}

#[derive(Debug)]
pub(crate) struct PreparedRcToolInvocation {
    display_name: String,
    rc_service_name: String,
    canonical_id: String,
    executable_path: Path<'static>,
    config_root: Path<'static>,
    workspace_root: Path<'static>,
    agent_id: String,
    tool_use_id: String,
    request_id: String,
    sequence: u64,
    tool_dir: PathBuf,
    result_file: PathBuf,
    temp_dir: PathBuf,
    session_id: Option<String>,
    session_dir: Option<PathBuf>,
    sessions_root: Option<PathBuf>,
    runtime: ToolRcRuntime,
    /// Maximum execution time, or `None` for no timeout.
    timeout: Option<Duration>,
}

struct ToolOverlayContext<'a> {
    request_file: &'a StdPath,
    result_file: &'a StdPath,
    scratch_dir: &'a StdPath,
    temp_dir: &'a StdPath,
    rc_conf_path: &'a str,
    rc_d_path: &'a str,
}

/// Invoke an rc-conf-based tool and return its text output.
///
/// Constructs a request envelope, writes it to a scratch directory, launches the
/// tool process with the appropriate rc-conf bindings, and reads back the result.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn invoke_rc_tool_text(
    display_name: &str,
    rc_service_name: &str,
    canonical_id: &str,
    executable_path: &Path<'_>,
    context: &ToolRuntimeContext<'_>,
    tool_use_id: &str,
    input: serde_json::Map<String, serde_json::Value>,
    timeout: Option<Duration>,
) -> Result<String, String> {
    let prepared = prepare_rc_tool_invocation(
        display_name,
        rc_service_name,
        canonical_id,
        executable_path,
        context,
        tool_use_id,
        input,
        timeout,
    )?;
    run_prepared_rc_tool_text(&prepared, context.writable_roots, context.session).await
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_rc_tool_invocation(
    display_name: &str,
    rc_service_name: &str,
    canonical_id: &str,
    executable_path: &Path<'_>,
    context: &ToolRuntimeContext<'_>,
    tool_use_id: &str,
    input: serde_json::Map<String, serde_json::Value>,
    timeout: Option<Duration>,
) -> Result<PreparedRcToolInvocation, String> {
    let request_id = next_request_id();
    let invocation_dirs = create_tool_invocation_dirs(context, &request_id).map_err(|err| {
        format!("tool '{display_name}' failed to create invocation directories: {err}")
    })?;
    let scratch_dir = invocation_dirs.scratch_dir;
    let temp_dir = invocation_dirs.temp_dir;
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
            temp_dir: temp_dir.to_string_lossy().into_owned(),
            result_file: result_file.to_string_lossy().into_owned(),
        },
    };

    write_json_file(&request_file, &request)
        .map_err(|err| format!("tool '{display_name}' failed to write request.json: {err}"))?;

    let runtime = prepare_tool_rc_runtime(
        display_name,
        rc_service_name,
        executable_path,
        context,
        &request_file,
        &result_file,
        &scratch_dir,
        &temp_dir,
    )
    .map_err(|err| format!("tool '{display_name}' failed to prepare rc invocation: {err}"))?;

    Ok(PreparedRcToolInvocation {
        display_name: display_name.to_string(),
        rc_service_name: rc_service_name.to_string(),
        canonical_id: canonical_id.to_string(),
        executable_path: executable_path.clone().into_owned(),
        config_root: context.config_root.clone().into_owned(),
        workspace_root: context.workspace_root.clone().into_owned(),
        agent_id: context.agent_id.to_string(),
        tool_use_id: tool_use_id.to_string(),
        request_id,
        sequence: invocation_dirs.sequence,
        tool_dir: invocation_dirs.root,
        result_file,
        temp_dir,
        session_id: context.session.map(|session| session.id().to_string()),
        session_dir: context.session.map(|session| session.root().clone()),
        sessions_root: context
            .session
            .map(|session| session.sessions_root().clone()),
        runtime,
        timeout,
    })
}

fn create_tool_invocation_dirs(
    context: &ToolRuntimeContext<'_>,
    request_id: &str,
) -> Result<crate::session::ToolInvocationDirs, SError> {
    if let Some(session) = context.session {
        return session.create_tool_invocation_dirs(request_id);
    }

    let scratch_dir = create_standalone_tool_runtime_dir(request_id)?;
    let temp_dir = scratch_dir.join("tmp");
    fs::create_dir_all(&temp_dir).map_err(|err| {
        SError::new("tool-protocol")
            .with_code("io_error")
            .with_message("failed to create tool temporary directory")
            .with_string_field("path", temp_dir.to_string_lossy().as_ref())
            .with_string_field("request_id", request_id)
            .with_string_field("cause", &err.to_string())
    })?;
    Ok(crate::session::ToolInvocationDirs {
        sequence: 1,
        root: scratch_dir.clone(),
        scratch_dir,
        temp_dir,
    })
}

pub(crate) async fn run_prepared_rc_tool_text(
    prepared: &PreparedRcToolInvocation,
    writable_roots: &WritableRoots,
    session: Option<&SidSession>,
) -> Result<String, String> {
    if let Some(session) = session
        && let Err(err) = session.log_tool_start(ToolStartEvent {
            tool_seq: prepared.sequence,
            request_id: &prepared.request_id,
            tool: &prepared.display_name,
            canonical_tool: &prepared.canonical_id,
            tool_use_id: &prepared.tool_use_id,
            agent: &prepared.agent_id,
            scratch_dir: &prepared.tool_dir,
        })
    {
        let _ = cleanup_prepared_rc_tool(prepared, true);
        return Err(format!(
            "tool '{}' failed to log start: {}",
            prepared.display_name, err
        ));
    }

    let (mut result, report) =
        run_prepared_rc_tool_text_inner(prepared, writable_roots, session).await;
    let failed = result.is_err();
    let scratch_preserved = session::should_keep_tool_scratch(failed);
    let cleanup_error = if scratch_preserved {
        None
    } else {
        cleanup_prepared_rc_tool(prepared, failed).err()
    };
    if let Some(err) = cleanup_error.as_ref()
        && result.is_ok()
    {
        result = Err(err.clone());
    }
    let success = result.is_ok();
    let error = result.as_ref().err().map(String::as_str);

    if let Some(session) = session {
        session
            .log_tool_finish(ToolFinishEvent {
                tool_seq: prepared.sequence,
                request_id: &prepared.request_id,
                status: report.status.as_deref(),
                exit_code: report.exit_code,
                success,
                result_ok: report.result_ok,
                output_len: report.output_len,
                error,
                scratch_preserved,
                scratch_dir: scratch_preserved.then_some(prepared.tool_dir.as_path()),
                cleanup_error: cleanup_error.as_deref(),
            })
            .map_err(|err| {
                format!(
                    "tool '{}' failed to log finish: {}",
                    prepared.display_name, err
                )
            })?;
    }

    result
}

/// Attempt a graceful shutdown of a tool child process.
///
/// Sends SIGTERM first, waits up to 5 seconds for voluntary exit, then
/// falls back to SIGKILL.  This gives tools a chance to clean up temporary
/// files, release locks, or flush partial output before being forcefully
/// terminated.
async fn kill_gracefully(child: &mut tokio::process::Child) {
    const GRACE_PERIOD: Duration = Duration::from_secs(5);

    if let Some(pid) = child.id() {
        #[cfg(unix)]
        {
            // SAFETY: sending a signal to a known-live process id is safe.
            unsafe {
                libc::kill(pid as libc::pid_t, libc::SIGTERM);
            }
        }
        match tokio::time::timeout(GRACE_PERIOD, child.wait()).await {
            Ok(_) => {}
            Err(_) => {
                child.kill().await.ok();
            }
        }
    } else {
        child.kill().await.ok();
    }
}

async fn run_prepared_rc_tool_text_inner(
    prepared: &PreparedRcToolInvocation,
    writable_roots: &WritableRoots,
    session: Option<&SidSession>,
) -> (Result<String, String>, ToolRunReport) {
    let mut report = ToolRunReport::default();

    if let Err(err) = clear_stale_result_file(prepared) {
        return (Err(err), report);
    }
    let mut cmd = prepared_rc_tool_command(prepared, "run", writable_roots);
    cmd.kill_on_drop(true);
    let mut child = match cmd
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return (
                Err(format!(
                    "tool '{}' failed to launch: {}",
                    prepared.display_name, err
                )),
                report,
            );
        }
    };
    let stdout = child
        .stdout
        .take()
        .expect("child stdout should be piped before spawn");
    let stderr = child
        .stderr
        .take()
        .expect("child stderr should be piped before spawn");
    let journal = session.map(SidSession::tool_stream_journal);
    let suppress_terminal_output = has_active_tool_output_observer();
    let stdout_terminal: Box<dyn AsyncWrite + Unpin + Send> = if suppress_terminal_output {
        Box::new(tokio::io::sink())
    } else {
        Box::new(tokio::io::stdout())
    };
    let stderr_terminal: Box<dyn AsyncWrite + Unpin + Send> = if suppress_terminal_output {
        Box::new(tokio::io::sink())
    } else {
        Box::new(tokio::io::stderr())
    };
    let stdout_task = tokio::spawn(tee_child_output(
        stdout,
        stdout_terminal,
        journal.clone(),
        prepared.sequence,
        "stdout",
        prepared.request_id.clone(),
        prepared.display_name.clone(),
        prepared.tool_use_id.clone(),
    ));
    let stderr_task = tokio::spawn(tee_child_output(
        stderr,
        stderr_terminal,
        journal,
        prepared.sequence,
        "stderr",
        prepared.request_id.clone(),
        prepared.display_name.clone(),
        prepared.tool_use_id.clone(),
    ));

    let wait_result = match prepared.timeout {
        Some(limit) => tokio::time::timeout(limit, child.wait())
            .await
            .map_err(|_| {
                format!(
                    "tool '{}' timed out after {}s",
                    prepared.display_name,
                    limit.as_secs(),
                )
            }),
        None => Ok(child.wait().await),
    };

    let status = match wait_result {
        Ok(Ok(status)) => status,
        Ok(Err(err)) => {
            return (
                Err(format!(
                    "tool '{}' failed to wait: {}",
                    prepared.display_name, err
                )),
                report,
            );
        }
        Err(timeout_msg) => {
            kill_gracefully(&mut child).await;
            // Pipes close when child is killed; let tee tasks finish.
            let _ = stdout_task.await;
            let _ = stderr_task.await;
            report.status = Some("timeout".to_string());
            return (Err(timeout_msg), report);
        }
    };
    report.status = Some(status.to_string());
    report.exit_code = status.code();
    if let Err(err) = await_output_log(prepared, "stdout", stdout_task).await {
        return (Err(err), report);
    }
    if let Err(err) = await_output_log(prepared, "stderr", stderr_task).await {
        return (Err(err), report);
    }
    if !status.success() {
        return (
            Err(format!(
                "tool '{}' exited with status {}",
                prepared.display_name, status
            )),
            report,
        );
    }

    let result = match read_tool_result(&prepared.result_file) {
        Ok(result) => result,
        Err(err) => {
            return (
                Err(format!(
                    "tool '{}' protocol error: {}",
                    prepared.display_name, err
                )),
                report,
            );
        }
    };
    report.result_ok = Some(result.ok);
    match extract_tool_output(&prepared.display_name, &prepared.request_id, result) {
        Ok(text) => {
            report.output_len = Some(text.len());
            (Ok(text), report)
        }
        Err(err) => (Err(err), report),
    }
}

pub(crate) async fn render_rc_tool_confirmation_preview(
    prepared: &PreparedRcToolInvocation,
    session: Option<&SidSession>,
) -> Result<String, String> {
    const CONFIRM_TIMEOUT: Duration = Duration::from_secs(5);

    let readonly_roots = WritableRoots::default();
    let mut cmd = prepared_rc_tool_command(prepared, "confirm", &readonly_roots);
    cmd.kill_on_drop(true);
    let output = tokio::time::timeout(
        CONFIRM_TIMEOUT,
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await
    .map_err(|_| {
        format!(
            "tool '{}' confirm timed out after {}s",
            prepared.display_name,
            CONFIRM_TIMEOUT.as_secs()
        )
    })?
    .map_err(|err| {
        format!(
            "tool '{}' failed to launch confirm: {}",
            prepared.display_name, err
        )
    })?;

    if let Some(session) = session {
        let journal = session.tool_stream_journal();
        journal
            .append(prepared.sequence, "confirm_stdout", &output.stdout)
            .map_err(|err| {
                format!(
                    "tool '{}' failed to log confirm stdout: {}",
                    prepared.display_name, err
                )
            })?;
        journal
            .append(prepared.sequence, "confirm_stderr", &output.stderr)
            .map_err(|err| {
                format!(
                    "tool '{}' failed to log confirm stderr: {}",
                    prepared.display_name, err
                )
            })?;
    }
    if !output.status.success() {
        return Err(format!(
            "tool '{}' confirm exited with status {}",
            prepared.display_name, output.status
        ));
    }

    let preview = String::from_utf8_lossy(&output.stdout)
        .trim_end()
        .to_string();
    if preview.trim().is_empty() {
        return Err(format!(
            "tool '{}' confirm produced no preview",
            prepared.display_name
        ));
    }
    Ok(preview)
}

pub(crate) fn cleanup_prepared_rc_tool(
    prepared: &PreparedRcToolInvocation,
    failed: bool,
) -> Result<bool, String> {
    if session::should_keep_tool_scratch(failed) {
        return Ok(true);
    }
    match fs::remove_dir_all(&prepared.tool_dir) {
        Ok(()) => Ok(false),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(false),
        Err(err) => Err(format!(
            "tool '{}' failed to clean scratch directory {}: {}",
            prepared.display_name,
            prepared.tool_dir.display(),
            err
        )),
    }
}

#[derive(Default)]
struct ToolRunReport {
    status: Option<String>,
    exit_code: Option<i32>,
    result_ok: Option<bool>,
    output_len: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
async fn tee_child_output<R, W>(
    mut reader: R,
    mut terminal: W,
    journal: Option<ToolStreamJournal>,
    tool_seq: u64,
    stream: &'static str,
    request_id: String,
    tool_name: String,
    tool_use_id: String,
) -> io::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buffer = [0u8; 8192];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        terminal.write_all(&buffer[..read]).await?;
        terminal.flush().await?;
        if let Some(journal) = journal.as_ref() {
            journal
                .append(tool_seq, stream, &buffer[..read])
                .map_err(|err| io::Error::other(err.to_string()))?;
        }
        let (text, data_b64) = match std::str::from_utf8(&buffer[..read]) {
            Ok(text) => (Some(text.to_string()), None),
            Err(_) => (None, Some(BASE64_STANDARD.encode(&buffer[..read]))),
        };
        notify_tool_output_observer(&ToolOutputEvent {
            request_id: request_id.clone(),
            tool_name: tool_name.clone(),
            tool_use_id: tool_use_id.clone(),
            stream: stream.to_string(),
            text,
            data_b64,
        });
    }
    Ok(())
}

async fn await_output_log(
    prepared: &PreparedRcToolInvocation,
    stream: &str,
    task: tokio::task::JoinHandle<io::Result<()>>,
) -> Result<(), String> {
    match task.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(format!(
            "tool '{}' failed to log {}: {}",
            prepared.display_name, stream, err
        )),
        Err(err) => Err(format!(
            "tool '{}' failed to join {} logger: {}",
            prepared.display_name, stream, err
        )),
    }
}

fn prepared_rc_tool_command(
    prepared: &PreparedRcToolInvocation,
    subcommand: &str,
    writable_roots: &WritableRoots,
) -> tokio::process::Command {
    let read_roots = seatbelt::service_read_roots(
        StdPath::new(prepared.config_root.as_str()),
        StdPath::new(prepared.executable_path.as_str()),
    );
    let mut cmd = seatbelt::sandboxed_command_with_read_roots(
        prepared.executable_path.as_str(),
        &[subcommand],
        writable_roots,
        &read_roots,
    );
    cmd.current_dir(prepared.workspace_root.as_str())
        .envs(&prepared.runtime.bindings)
        .env("TMPDIR", &prepared.temp_dir)
        .env("TEMP", &prepared.temp_dir)
        .env("TMP", &prepared.temp_dir)
        .env("PAGER", "cat")
        .env(
            "RCVAR_ARGV0",
            var_name_from_service(&prepared.rc_service_name),
        )
        .env("RC_CONF_PATH", &prepared.runtime.rc_conf_path)
        .env("RC_D_PATH", &prepared.runtime.rc_d_path);
    if let Some(session_id) = prepared.session_id.as_ref() {
        cmd.env(crate::session::SID_SESSION_ID_ENV, session_id);
    }
    if let Some(session_dir) = prepared.session_dir.as_ref() {
        cmd.env(crate::session::SID_SESSION_DIR_ENV, session_dir);
    }
    if let Some(sessions_root) = prepared.sessions_root.as_ref() {
        cmd.env(crate::session::SID_SESSIONS_ENV, sessions_root);
    }
    cmd
}

fn clear_stale_result_file(prepared: &PreparedRcToolInvocation) -> Result<(), String> {
    match fs::remove_file(&prepared.result_file) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!(
            "tool '{}' failed to clear stale result.json: {}",
            prepared.display_name, err
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_tool_rc_runtime(
    display_name: &str,
    rc_service_name: &str,
    executable_path: &Path,
    context: &ToolRuntimeContext<'_>,
    request_file: &StdPath,
    result_file: &StdPath,
    scratch_dir: &StdPath,
    temp_dir: &StdPath,
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
        temp_dir,
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
    let temp_dir = overlay_context.temp_dir.to_string_lossy().into_owned();
    let workspace_root = context.workspace_root.as_str().to_string();
    let tool_protocol = TOOL_PROTOCOL_VERSION.to_string();
    let session_id = context.session.map(SidSession::id);
    let session_dir = context
        .session
        .map(|session| session.root().to_string_lossy().into_owned());
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
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}TEMP_DIR"), &temp_dir);
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}TMPDIR"), &temp_dir);
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}WORKSPACE_ROOT"),
            &workspace_root,
        );
        if let Some(session_id) = session_id {
            append_rc_conf_assignment(&mut overlay, &format!("{prefix}SESSION_ID"), session_id);
        }
        if let Some(session_dir) = session_dir.as_ref() {
            append_rc_conf_assignment(&mut overlay, &format!("{prefix}SESSION_DIR"), session_dir);
        }
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

fn create_standalone_tool_runtime_dir(request_id: &str) -> Result<PathBuf, SError> {
    let parent = std::env::temp_dir().join("sid-runtime-tools");
    fs::create_dir_all(&parent).map_err(|err| {
        tool_runtime_error(
            "standalone",
            "io_error",
            "failed to create tool runtime root",
        )
        .with_string_field("path", parent.to_string_lossy().as_ref())
        .with_string_field("request_id", request_id)
        .with_string_field("cause", &err.to_string())
    })?;
    let scratch_dir = parent.join(request_id);
    fs::create_dir(&scratch_dir).map_err(|err| {
        tool_runtime_error(
            "standalone",
            "io_error",
            "failed to create tool runtime directory",
        )
        .with_string_field("path", scratch_dir.to_string_lossy().as_ref())
        .with_string_field("request_id", request_id)
        .with_string_field("cause", &err.to_string())
    })?;
    Ok(scratch_dir)
}

fn tool_runtime_error(tool: &str, code: &str, message: &str) -> SError {
    SError::new("tool-runtime")
        .with_code(code)
        .with_message(message)
        .with_string_field("tool", tool)
}
