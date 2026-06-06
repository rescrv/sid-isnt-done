//! Container runtime abstraction and `sid-container` process helpers.
//!
//! This module keeps container management independent from a particular
//! executable while preserving the command shapes used by the `sid-container`
//! helper.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, ExitStatus};

use serde::Deserialize;

/// Default executable name for the `sid-container` helper.
pub const DEFAULT_SIDCC_CONTAINER_BIN: &str = "sid-container";

/// Desired state for a container to be launched by a [`ContainerRuntime`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunContainerRequest {
    /// Stable runtime identifier to assign to the container.
    pub id: String,
    /// Image reference passed to the container runtime.
    pub image: String,
    /// Number of CPUs requested for the container, or zero for runtime default.
    pub cpus: u32,
    /// Memory limit in MiB, or zero for runtime default.
    pub memory_mib: u32,
    /// Labels applied to the container.
    pub labels: BTreeMap<String, String>,
    /// Command and arguments appended after the image reference.
    pub command: Vec<String>,
}

/// Lifecycle state reported by a container runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContainerStatus {
    /// The container exists but has not started running.
    Created,
    /// The container is currently running.
    Running,
    /// The container has stopped.
    Stopped,
    /// A runtime-specific state not modeled by this crate.
    Other(String),
}

impl ContainerStatus {
    /// Convert a runtime status string into the nearest modeled status.
    pub fn parse(status: &str) -> Self {
        match status {
            "created" => Self::Created,
            "running" => Self::Running,
            "stopped" => Self::Stopped,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Container metadata normalized from a runtime-specific listing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContainerInstance {
    /// Stable runtime identifier for the container.
    pub id: String,
    /// Current lifecycle state.
    pub status: ContainerStatus,
    /// Image reference used to create the container.
    pub image_reference: String,
    /// Labels attached to the container.
    pub labels: BTreeMap<String, String>,
    /// CPU allocation reported by the runtime.
    pub cpus: u32,
    /// Memory allocation reported by the runtime, in MiB.
    pub memory_mib: u32,
}

/// Minimal container operations required by sid-managed runtimes.
pub trait ContainerRuntime {
    /// List containers visible to the runtime.
    fn list(&mut self) -> Result<Vec<ContainerInstance>, ContainerRuntimeError>;

    /// Launch a detached container from the supplied request.
    fn run(&mut self, request: &RunContainerRequest) -> Result<(), ContainerRuntimeError>;

    /// Stop a running container by runtime identifier.
    fn stop(&mut self, id: &str) -> Result<(), ContainerRuntimeError>;

    /// Delete a container by runtime identifier.
    fn delete(&mut self, id: &str) -> Result<(), ContainerRuntimeError>;
}

/// [`ContainerRuntime`] implementation backed by an operating-system command.
#[derive(Clone, Debug)]
pub struct OsContainerRuntime {
    container_bin: String,
}

impl OsContainerRuntime {
    /// Create a runtime that invokes `container_bin` for each operation.
    pub fn new(container_bin: impl Into<String>) -> Self {
        Self {
            container_bin: container_bin.into(),
        }
    }

    fn command_output(&self, args: Vec<String>) -> Result<String, ContainerRuntimeError> {
        container_command_output(&self.container_bin, args).map_err(ContainerRuntimeError::from)
    }
}

impl ContainerRuntime for OsContainerRuntime {
    fn list(&mut self) -> Result<Vec<ContainerInstance>, ContainerRuntimeError> {
        let stdout = self.command_output(vec![
            "list".to_string(),
            "--all".to_string(),
            "--format".to_string(),
            "json".to_string(),
        ])?;
        let raw = serde_json::from_str::<Vec<RawContainer>>(&stdout)
            .map_err(ContainerRuntimeError::Decode)?;
        Ok(raw.into_iter().map(ContainerInstance::from).collect())
    }

    fn run(&mut self, request: &RunContainerRequest) -> Result<(), ContainerRuntimeError> {
        let mut args = vec![
            "run".to_string(),
            "--detach".to_string(),
            "--name".to_string(),
            request.id.clone(),
        ];
        if request.cpus > 0 {
            args.push("--cpus".to_string());
            args.push(request.cpus.to_string());
        }
        if request.memory_mib > 0 {
            args.push("--memory".to_string());
            args.push(format!("{}M", request.memory_mib));
        }
        for (key, value) in &request.labels {
            args.push("--label".to_string());
            args.push(format!("{key}={value}"));
        }
        args.push(request.image.clone());
        args.extend(request.command.iter().cloned());
        self.command_output(args)?;
        Ok(())
    }

    fn stop(&mut self, id: &str) -> Result<(), ContainerRuntimeError> {
        self.command_output(vec!["stop".to_string(), id.to_string()])?;
        Ok(())
    }

    fn delete(&mut self, id: &str) -> Result<(), ContainerRuntimeError> {
        self.command_output(vec!["delete".to_string(), id.to_string()])?;
        Ok(())
    }
}

/// Error produced while translating runtime operations into container commands.
#[derive(Debug)]
pub enum ContainerRuntimeError {
    /// The runtime command could not be spawned or waited on.
    Io(std::io::Error),
    /// The runtime command exited unsuccessfully.
    CommandFailed {
        /// Executable path or name that was invoked.
        command: String,
        /// Arguments passed to the executable.
        args: Vec<String>,
        /// Exit status returned by the process.
        status: ExitStatus,
        /// Standard error captured from the process.
        stderr: String,
    },
    /// Container list output was not valid JSON in the expected shape.
    Decode(serde_json::Error),
}

/// Error produced by a direct container helper command invocation.
#[derive(Debug)]
pub enum ContainerCommandError {
    /// The command could not be spawned or waited on.
    Io(std::io::Error),
    /// The command exited unsuccessfully.
    CommandFailed {
        /// Executable path or name that was invoked.
        command: String,
        /// Arguments passed to the executable.
        args: Vec<String>,
        /// Exit status returned by the process.
        status: ExitStatus,
        /// Standard output captured from the process.
        stdout: String,
        /// Standard error captured from the process.
        stderr: String,
    },
}

impl From<ContainerCommandError> for ContainerRuntimeError {
    fn from(err: ContainerCommandError) -> Self {
        match err {
            ContainerCommandError::Io(err) => Self::Io(err),
            ContainerCommandError::CommandFailed {
                command,
                args,
                status,
                stderr,
                ..
            } => Self::CommandFailed {
                command,
                args,
                status,
                stderr,
            },
        }
    }
}

impl std::fmt::Display for ContainerRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "container runtime I/O error: {err}"),
            Self::CommandFailed {
                command,
                args,
                status,
                stderr,
            } => {
                write!(
                    f,
                    "container command failed: {command} {} ({status})",
                    args.join(" ")
                )?;
                if !stderr.is_empty() {
                    write!(f, ": {stderr}")?;
                }
                Ok(())
            }
            Self::Decode(err) => write!(f, "failed to decode container JSON: {err}"),
        }
    }
}

impl std::error::Error for ContainerRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Decode(err) => Some(err),
            Self::CommandFailed { .. } => None,
        }
    }
}

impl std::fmt::Display for ContainerCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "container command I/O error: {err}"),
            Self::CommandFailed {
                command,
                args,
                status,
                stdout,
                stderr,
            } => {
                write!(
                    f,
                    "{command} {} failed with status {status}",
                    args.join(" ")
                )?;
                match (stdout.is_empty(), stderr.is_empty()) {
                    (true, true) => Ok(()),
                    (false, true) => write!(f, ": {stdout}"),
                    (true, false) => write!(f, ": {stderr}"),
                    (false, false) => write!(f, ": {stderr}; stdout: {stdout}"),
                }
            }
        }
    }
}

impl std::error::Error for ContainerCommandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::CommandFailed { .. } => None,
        }
    }
}

/// Invoke a container helper command and return trimmed standard output.
///
/// # Errors
///
/// Returns [`ContainerCommandError::Io`] when the process cannot be executed,
/// and [`ContainerCommandError::CommandFailed`] when it exits unsuccessfully.
pub fn container_command_output(
    container_bin: &str,
    args: Vec<String>,
) -> Result<String, ContainerCommandError> {
    let output = Command::new(container_bin)
        .args(args.iter().map(String::as_str))
        .output()
        .map_err(ContainerCommandError::Io)?;
    if !output.status.success() {
        return Err(ContainerCommandError::CommandFailed {
            command: container_bin.to_string(),
            args,
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).trim().to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Resolve the default `sid-container` executable path.
///
/// When a sibling executable exists next to the current process, that path is
/// returned. Otherwise this falls back to [`DEFAULT_SIDCC_CONTAINER_BIN`].
pub fn default_sid_container_bin() -> String {
    if let Ok(current_exe) = std::env::current_exe()
        && let Some(parent) = current_exe.parent()
    {
        let candidate = parent.join(format!(
            "{DEFAULT_SIDCC_CONTAINER_BIN}{}",
            std::env::consts::EXE_SUFFIX
        ));
        if candidate.exists() {
            return candidate.display().to_string();
        }
    }
    DEFAULT_SIDCC_CONTAINER_BIN.to_string()
}

/// Launch `sid-container run` and extract the listener endpoint it prints.
///
/// # Errors
///
/// Returns an error string when the helper command fails or when its output is
/// not exactly one valid raw listener endpoint.
pub fn launch_sid_container(container_bin: &str, run_args: &[String]) -> Result<String, String> {
    let mut args = Vec::with_capacity(run_args.len() + 1);
    args.push("run".to_string());
    args.extend(run_args.iter().cloned());

    let stdout = container_command_output(container_bin, args).map_err(|err| err.to_string())?;
    parse_sid_container_socket_output(&stdout)
}

/// Parse the raw listener endpoint printed by `sid-container run`.
///
/// # Errors
///
/// Returns an error when the output is empty, contains multiple non-empty
/// lines, or is not a `tcp://HOST:PORT` or `unix:///absolute/path` endpoint.
pub fn parse_sid_container_socket_output(stdout: &str) -> Result<String, String> {
    let mut paths = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let Some(path) = paths.next() else {
        return Err("sid-container did not print a raw listener endpoint".to_string());
    };
    if paths.next().is_some() {
        return Err(
            "sid-container printed multiple non-empty lines; expected one raw listener endpoint"
                .to_string(),
        );
    }
    validate_sid_container_socket_path(path)?;
    Ok(path.to_string())
}

fn validate_sid_container_socket_path(path: &str) -> Result<(), String> {
    if let Some(address) = path.strip_prefix("tcp://") {
        return validate_tcp_endpoint(address)
            .map_err(|err| format!("sid-container printed {path:?}; {err}"));
    }
    if let Some(path_without_scheme) = path.strip_prefix("unix://")
        && Path::new(path_without_scheme).is_absolute()
    {
        return Ok(());
    }
    Err(format!(
        "sid-container printed {path:?}; expected tcp://HOST:PORT or unix:///absolute/path"
    ))
}

fn validate_tcp_endpoint(address: &str) -> Result<(), String> {
    if address.is_empty() || address.contains('/') {
        return Err("expected tcp://HOST:PORT".to_string());
    }
    let Some((host, port)) = address.rsplit_once(':') else {
        return Err("expected tcp://HOST:PORT".to_string());
    };
    if host.is_empty() {
        return Err("TCP endpoint host must not be empty".to_string());
    }
    let port = port
        .parse::<u16>()
        .map_err(|_| "TCP endpoint port must be in 1..=65535".to_string())?;
    if port == 0 {
        return Err("TCP endpoint port must be greater than zero".to_string());
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RawContainer {
    configuration: RawConfiguration,
    status: String,
}

#[derive(Debug, Default, Deserialize)]
struct RawConfiguration {
    id: String,
    #[serde(default)]
    labels: BTreeMap<String, String>,
    #[serde(default)]
    image: RawImage,
    #[serde(default)]
    resources: RawResources,
}

#[derive(Debug, Default, Deserialize)]
struct RawImage {
    #[serde(default)]
    reference: String,
}

#[derive(Debug, Default, Deserialize)]
struct RawResources {
    #[serde(default)]
    cpus: u32,
    #[serde(rename = "memoryInBytes", default)]
    memory_in_bytes: u64,
}

impl From<RawContainer> for ContainerInstance {
    fn from(raw: RawContainer) -> Self {
        Self {
            id: raw.configuration.id,
            status: ContainerStatus::parse(&raw.status),
            image_reference: raw.configuration.image.reference,
            labels: raw.configuration.labels,
            cpus: raw.configuration.resources.cpus,
            memory_mib: bytes_to_mib(raw.configuration.resources.memory_in_bytes),
        }
    }
}

fn bytes_to_mib(bytes: u64) -> u32 {
    u32::try_from(bytes / (1024 * 1024)).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_container_list_json() {
        let json = r#"
[
  {
    "configuration": {
      "id": "sid-probe",
      "labels": {
        "sid.managed": "true",
        "sid.host_id": "host_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "resources": {
        "cpus": 4,
        "memoryInBytes": 1073741824
      },
      "image": {
        "reference": "docker.io/library/debian:trixie"
      }
    },
    "status": "running"
  }
]
"#;
        let parsed = serde_json::from_str::<Vec<RawContainer>>(json).unwrap();
        let containers: Vec<ContainerInstance> =
            parsed.into_iter().map(ContainerInstance::from).collect();
        assert_eq!(
            containers,
            vec![ContainerInstance {
                id: "sid-probe".to_string(),
                status: ContainerStatus::Running,
                image_reference: "docker.io/library/debian:trixie".to_string(),
                labels: BTreeMap::from([
                    (
                        "sid.host_id".to_string(),
                        "host_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string()
                    ),
                    ("sid.managed".to_string(), "true".to_string()),
                ]),
                cpus: 4,
                memory_mib: 1024,
            }]
        );
    }

    #[test]
    fn parse_sid_container_socket_output_accepts_listener_specs() {
        assert_eq!(
            parse_sid_container_socket_output("unix:///tmp/sid/sid.sock\n").unwrap(),
            "unix:///tmp/sid/sid.sock"
        );
        assert_eq!(
            parse_sid_container_socket_output("tcp://127.0.0.1:45450\n").unwrap(),
            "tcp://127.0.0.1:45450"
        );
    }

    #[test]
    fn parse_sid_container_socket_output_rejects_invalid_output() {
        assert!(parse_sid_container_socket_output("\n").is_err());
        assert!(parse_sid_container_socket_output("/tmp/sid/sid.sock\n").is_err());
        assert!(parse_sid_container_socket_output("relative.sock\n").is_err());
        assert!(parse_sid_container_socket_output("tcp://127.0.0.1\n").is_err());
        assert!(parse_sid_container_socket_output("tcp://:45450\n").is_err());
        assert!(parse_sid_container_socket_output("/tmp/one.sock\n/tmp/two.sock\n").is_err());
    }
}
