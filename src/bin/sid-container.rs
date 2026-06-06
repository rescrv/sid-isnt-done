use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrrg::{CommandLine, NoExitCommandLine};
use arrrg_derive::CommandLine;
use serde_json::json;
use sid_isnt_done::containers::container_command_output;

const DEFAULT_CONTAINER_BIN: &str = "container";
const DEFAULT_CONTAINER_SOCKET_DIR: &str = "/run/sid";
const DEFAULT_HOST_TCP_ADDRESS: &str = "127.0.0.1";
const DEFAULT_CONTAINER_TCP_ADDRESS: &str = "0.0.0.0";
const DEFAULT_CONTAINER_TCP_PORT: u16 = 8890;
const DEFAULT_OVERLAY_REGISTRY: &str = "ghcr.io";
const DEFAULT_SID_BIN: &str = "sid";
const DEFAULT_SOCKET_ROOT: &str = "/tmp/sid";
const DEFAULT_SOCKET_NAME: &str = "sid.sock";
const DEFAULT_TIMEOUT_MS: u64 = 10_000;
const DEFAULT_NAME_IMAGE_CHARS: usize = 48;
const DEFAULT_SID_HOME_ENV: &str = "SID_HOME=/.sid";
const DEFAULT_INHERITED_ENV: &[&str] = &["CLAUDIUS_API_KEY", "ANTHROPIC_API_KEY"];
const CONTAINER_SECRET_DIR: &str = "/run/sid-secrets";

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd"
))]
const UNIX_SOCKET_PATH_MAX_BYTES: usize = 104;
#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd"
)))]
const UNIX_SOCKET_PATH_MAX_BYTES: usize = 108;

const MANAGED_LABEL: &str = "sid.managed";
const NAME_LABEL: &str = "sid.name";
const IMAGE_LABEL: &str = "sid.image";
const SOCKET_LABEL: &str = "sid.socket";
const CONTAINER_SOCKET_LABEL: &str = "sid.container_socket";

const TOP_USAGE: &str = "\
USAGE: sid-container <command> [options]

Commands:
  build    Compose prebuilt scratch overlays onto a base image
  run      Start one sid container and print its raw listener endpoint";

const BUILD_USAGE: &str = "\
USAGE: sid-container build [options] <name> <base-image> [overlay...]";

const RUN_USAGE: &str = "\
USAGE: sid-container run [options] <image> [sid-arg...]";

#[derive(Clone, Debug, Default, Eq, PartialEq, CommandLine)]
struct BuildOptions {
    #[arrrg(optional, "Path to the macOS container CLI", "PATH")]
    container_bin: String,
    #[arrrg(
        optional,
        "Registry used to resolve @owner/repo overlay shorthand",
        "REGISTRY"
    )]
    registry: String,
    #[arrrg(
        optional,
        "Also write the generated Containerfile to this path",
        "PATH"
    )]
    containerfile: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, CommandLine)]
struct RunOptions {
    #[arrrg(optional, "Container name to create", "NAME")]
    name: String,
    #[arrrg(optional, "Path to the macOS container CLI", "PATH")]
    container_bin: String,
    #[arrrg(optional, "sid executable inside the container", "PATH")]
    sid_bin: String,
    #[arrrg(optional, "Listener transport to expose: tcp or unix", "TRANSPORT")]
    transport: String,
    #[arrrg(optional, "Host address for published TCP listeners", "HOST")]
    host_address: String,
    #[arrrg(optional, "Host TCP port to publish; 0 chooses a free port", "PORT")]
    host_port: u16,
    #[arrrg(optional, "Container address for sid TCP listeners", "HOST")]
    container_address: String,
    #[arrrg(optional, "Container TCP port for sid TCP listeners", "PORT")]
    container_port: u16,
    #[arrrg(optional, "Host directory for the Unix socket", "PATH")]
    socket_dir: Option<String>,
    #[arrrg(optional, "Socket file name", "NAME")]
    socket_name: String,
    #[arrrg(optional, "Mount point for the socket directory", "PATH")]
    container_socket_dir: String,
    #[arrrg(
        optional,
        "Milliseconds to wait for Unix socket creation with --transport unix",
        "MS"
    )]
    timeout_ms: u64,
    #[arrrg(optional, "Pass a CPU limit through to container run", "COUNT")]
    cpus: Option<String>,
    #[arrrg(optional, "Pass a memory limit through to container run", "SIZE")]
    memory: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BuildRequest {
    name: String,
    base_image: String,
    overlays: Vec<String>,
    resolved_overlays: Vec<String>,
    container_bin: String,
    containerfile: Option<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RunRequest {
    name: String,
    image: String,
    container_bin: String,
    sid_bin: String,
    inherited_env: Vec<InheritedEnv>,
    transport: ListenTransport,
    host_address: String,
    host_port: u16,
    container_address: String,
    container_port: u16,
    socket_dir: Option<PathBuf>,
    socket_name: String,
    container_socket_dir: PathBuf,
    timeout: Duration,
    cpus: Option<String>,
    memory: Option<String>,
    sid_args: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ListenTransport {
    Tcp,
    Unix,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InheritedEnv {
    name: String,
    spec: String,
    file_source: Option<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LaunchPaths {
    host_endpoint: String,
    host_secret_dir: PathBuf,
    container_endpoint: String,
    wait_target: WaitTarget,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum WaitTarget {
    Tcp {
        host: String,
        port: u16,
    },
    Unix {
        host_socket_dir: PathBuf,
        host_socket_path: PathBuf,
        container_socket_path: PathBuf,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CommandResult {
    BuiltImage(String),
    Endpoint(String),
}

impl CommandResult {
    fn print(&self) {
        match self {
            Self::BuiltImage(image) => println!("{image}"),
            Self::Endpoint(endpoint) => println!("{endpoint}"),
        }
    }
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if is_top_level_help(&args) {
        println!("{TOP_USAGE}");
        return;
    }
    if is_subcommand_help(&args, "build") {
        print_arrrg_usage::<BuildOptions>(BUILD_USAGE);
        return;
    }
    if is_subcommand_help(&args, "run") {
        print_arrrg_usage::<RunOptions>(RUN_USAGE);
        return;
    }

    match dispatch(args) {
        Ok(result) => result.print(),
        Err(err) => {
            eprintln!("sid-container: {err}");
            std::process::exit(1);
        }
    }
}

fn dispatch(args: Vec<String>) -> Result<CommandResult, String> {
    let (command, rest) = split_subcommand(args, &["build", "run"])?;
    match command.as_str() {
        "build" => build(parse_build_request(&rest)?).map(CommandResult::BuiltImage),
        "run" => run(parse_run_request(&rest)?).map(CommandResult::Endpoint),
        _ => unreachable!("split_subcommand returned an unknown command"),
    }
}

fn split_subcommand(args: Vec<String>, commands: &[&str]) -> Result<(String, Vec<String>), String> {
    let Some((command, rest)) = args.split_first() else {
        return Err(format!(
            "missing subcommand; expected one of: {}\n{TOP_USAGE}",
            commands.join(", ")
        ));
    };
    if commands.iter().any(|expected| command == expected) {
        Ok((command.clone(), rest.to_vec()))
    } else {
        Err(format!(
            "unknown subcommand {command:?}; expected one of: {}\n{TOP_USAGE}",
            commands.join(", ")
        ))
    }
}

fn parse_build_request(args: &[String]) -> Result<BuildRequest, String> {
    let (options, free) = parse_arrrg::<BuildOptions>(BUILD_USAGE, args)?;
    if free.len() < 2 {
        return Err(format!(
            "build requires <name> and <base-image>\n{BUILD_USAGE}"
        ));
    }
    let name = free[0].clone();
    let base_image = free[1].clone();
    let overlays = free[2..].to_vec();
    validate_nonempty(&name, "build image name")?;
    validate_containerfile_image_ref(&base_image, "base image")?;
    for overlay in &overlays {
        validate_nonempty(overlay, "overlay")?;
    }

    let registry = default_string(options.registry, DEFAULT_OVERLAY_REGISTRY);
    let resolved_overlays = overlays
        .iter()
        .map(|overlay| resolve_overlay_ref(overlay, &registry))
        .collect::<Result<Vec<_>, _>>()?;
    for overlay in &resolved_overlays {
        validate_containerfile_image_ref(overlay, "overlay")?;
    }

    Ok(BuildRequest {
        name,
        base_image,
        overlays,
        resolved_overlays,
        container_bin: default_string(options.container_bin, DEFAULT_CONTAINER_BIN),
        containerfile: options.containerfile.map(PathBuf::from),
    })
}

fn parse_run_request(args: &[String]) -> Result<RunRequest, String> {
    let (options, free) = parse_arrrg::<RunOptions>(RUN_USAGE, args)?;
    if free.is_empty() {
        return Err(format!("run requires <image>\n{RUN_USAGE}"));
    }
    let image = free[0].clone();
    let sid_args = free[1..].to_vec();

    validate_nonempty(&image, "image")?;
    let name = if options.name.is_empty() {
        default_container_name(&image)?
    } else {
        options.name
    };
    validate_container_name(&name)?;
    let transport = parse_transport(&default_string(options.transport, "tcp"))?;
    let host_address = default_string(options.host_address, DEFAULT_HOST_TCP_ADDRESS);
    validate_tcp_host(&host_address, "--host-address")?;
    let host_port = options.host_port;
    let container_address =
        default_string(options.container_address, DEFAULT_CONTAINER_TCP_ADDRESS);
    validate_tcp_host(&container_address, "--container-address")?;
    let container_port = default_u16(options.container_port, DEFAULT_CONTAINER_TCP_PORT);
    validate_tcp_port(container_port, "--container-port")?;
    let socket_name = default_string(options.socket_name, DEFAULT_SOCKET_NAME);
    validate_socket_name(&socket_name)?;
    let container_socket_dir = PathBuf::from(default_string(
        options.container_socket_dir,
        DEFAULT_CONTAINER_SOCKET_DIR,
    ));
    validate_absolute_path(&container_socket_dir, "--container-socket-dir")?;
    if let Some(cpus) = &options.cpus {
        validate_nonempty(cpus, "--cpus")?;
    }
    if let Some(memory) = &options.memory {
        validate_nonempty(memory, "--memory")?;
    }

    Ok(RunRequest {
        name,
        image,
        container_bin: default_string(options.container_bin, DEFAULT_CONTAINER_BIN),
        sid_bin: default_string(options.sid_bin, DEFAULT_SID_BIN),
        inherited_env: default_inherited_env()?,
        transport,
        host_address,
        host_port,
        container_address,
        container_port,
        socket_dir: options.socket_dir.map(PathBuf::from),
        socket_name,
        container_socket_dir,
        timeout: Duration::from_millis(default_u64(options.timeout_ms, DEFAULT_TIMEOUT_MS)),
        cpus: options.cpus,
        memory: options.memory,
        sid_args,
    })
}

fn parse_arrrg<T>(usage: &str, args: &[String]) -> Result<(T, Vec<String>), String>
where
    T: CommandLine,
{
    let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
    let (wrapped, free) = NoExitCommandLine::<T>::from_arguments_relaxed(usage, &refs);
    let (options, messages, status) = wrapped.into_parts();
    if status != 0 {
        return Err(messages.join("\n"));
    }
    Ok((options, free))
}

fn print_arrrg_usage<T>(usage: &str)
where
    T: CommandLine,
{
    let args = ["--help"];
    let (wrapped, _) = NoExitCommandLine::<T>::from_arguments_relaxed(usage, &args);
    let (_, messages, _) = wrapped.into_parts();
    println!("{}", messages.join("\n"));
}

fn build(request: BuildRequest) -> Result<String, String> {
    let context_dir = create_temp_context_dir(&request.name)?;
    let result = build_with_context(&request, &context_dir);
    let cleanup = fs::remove_dir_all(&context_dir);
    match (result, cleanup) {
        (Ok(image), _) => Ok(image),
        (Err(err), Ok(())) => Err(err),
        (Err(err), Err(cleanup_err)) => Err(format!(
            "{err}; additionally failed to remove temporary build context {}: {cleanup_err}",
            context_dir.display()
        )),
    }
}

fn build_with_context(request: &BuildRequest, context_dir: &Path) -> Result<String, String> {
    fs::create_dir_all(context_dir).map_err(|err| {
        format!(
            "failed to create build context {}: {err}",
            context_dir.display()
        )
    })?;
    let containerfile = context_dir.join("Containerfile");
    let metadata = build_metadata(request);
    let metadata_path = context_dir.join("sid-image.json");
    fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&metadata)
            .map_err(|err| format!("failed to encode build metadata: {err}"))?,
    )
    .map_err(|err| {
        format!(
            "failed to write build metadata {}: {err}",
            metadata_path.display()
        )
    })?;

    let containerfile_text = render_containerfile(request);
    fs::write(&containerfile, containerfile_text.as_bytes()).map_err(|err| {
        format!(
            "failed to write Containerfile {}: {err}",
            containerfile.display()
        )
    })?;
    if let Some(copy_path) = &request.containerfile {
        if let Some(parent) = copy_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create Containerfile parent {}: {err}",
                    parent.display()
                )
            })?;
        }
        fs::write(copy_path, containerfile_text.as_bytes()).map_err(|err| {
            format!(
                "failed to write requested Containerfile {}: {err}",
                copy_path.display()
            )
        })?;
    }

    let build_args = build_container_args(request, &containerfile, context_dir);
    container_command_output(&request.container_bin, build_args).map_err(|err| err.to_string())?;
    Ok(request.name.clone())
}

fn build_metadata(request: &BuildRequest) -> serde_json::Value {
    json!({
        "name": request.name,
        "base": request.base_image,
        "overlays": request.overlays,
        "resolved_overlays": request.resolved_overlays,
    })
}

fn render_containerfile(request: &BuildRequest) -> String {
    let mut out = String::new();
    for (idx, overlay) in request.resolved_overlays.iter().enumerate() {
        out.push_str(&format!("FROM {overlay} AS sid_overlay_{idx}\n"));
    }
    if !request.resolved_overlays.is_empty() {
        out.push('\n');
    }
    out.push_str(&format!("FROM {}\n", request.base_image));
    for idx in 0..request.resolved_overlays.len() {
        out.push_str(&format!("COPY --from=sid_overlay_{idx} / /\n"));
    }
    out.push_str("COPY sid-image.json /etc/sid-image.json\n");
    out.push_str(&format!(
        "LABEL sid.base={}\n",
        containerfile_quote(&request.base_image)
    ));
    out.push_str(&format!(
        "LABEL sid.overlays={}\n",
        containerfile_quote(&request.overlays.join(" "))
    ));
    out.push_str(&format!(
        "LABEL sid.resolved_overlays={}\n",
        containerfile_quote(&request.resolved_overlays.join(" "))
    ));
    out
}

fn build_container_args(
    request: &BuildRequest,
    containerfile: &Path,
    context_dir: &Path,
) -> Vec<String> {
    vec![
        "build".to_string(),
        "--tag".to_string(),
        request.name.clone(),
        "--file".to_string(),
        containerfile.display().to_string(),
        context_dir.display().to_string(),
    ]
}

fn run(request: RunRequest) -> Result<String, String> {
    let paths = launch_paths(&request)?;
    prepare_listen_target(&paths)?;
    let _staged_secrets = stage_secret_files(&request, &paths)?;
    run_container_and_wait(&request, &paths)
}

fn run_container_and_wait(request: &RunRequest, paths: &LaunchPaths) -> Result<String, String> {
    let container_id = run_container(request, paths)?;
    wait_for_listen_target(&paths.wait_target, request.timeout).map_err(|err| {
        let diagnostics = container_startup_diagnostics(request, &container_id);
        listen_wait_failed_message(
            err,
            &container_id,
            &paths.host_endpoint,
            request.timeout,
            diagnostics.as_deref(),
        )
    })?;
    Ok(paths.host_endpoint.clone())
}

fn launch_paths(request: &RunRequest) -> Result<LaunchPaths, String> {
    let host_secret_dir = std::env::temp_dir()
        .join("sid")
        .join(format!("{}-secrets", request.name));
    match request.transport {
        ListenTransport::Tcp => {
            let host_port = if request.host_port == 0 {
                allocate_tcp_port(&request.host_address)?
            } else {
                request.host_port
            };
            let host_endpoint = tcp_endpoint(&request.host_address, host_port);
            let container_endpoint =
                tcp_endpoint(&request.container_address, request.container_port);
            Ok(LaunchPaths {
                host_endpoint,
                host_secret_dir,
                container_endpoint,
                wait_target: WaitTarget::Tcp {
                    host: request.host_address.clone(),
                    port: host_port,
                },
            })
        }
        ListenTransport::Unix => {
            let host_socket_dir = request
                .socket_dir
                .clone()
                .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET_ROOT).join(&request.name));
            fs::create_dir_all(&host_socket_dir).map_err(|err| {
                format!(
                    "failed to create socket directory {}: {err}",
                    host_socket_dir.display()
                )
            })?;
            let host_socket_dir = host_socket_dir.canonicalize().map_err(|err| {
                format!(
                    "failed to canonicalize socket directory {}: {err}",
                    host_socket_dir.display()
                )
            })?;
            let host_socket_path = host_socket_dir.join(&request.socket_name);
            validate_unix_socket_path_len(&host_socket_path, "host socket path")?;
            let container_socket_path = request.container_socket_dir.join(&request.socket_name);
            validate_unix_socket_path_len(&container_socket_path, "container socket path")?;
            Ok(LaunchPaths {
                host_endpoint: host_socket_path.display().to_string(),
                host_secret_dir,
                container_endpoint: format!("unix://{}", container_socket_path.display()),
                wait_target: WaitTarget::Unix {
                    host_socket_dir,
                    host_socket_path,
                    container_socket_path,
                },
            })
        }
    }
}

fn prepare_listen_target(paths: &LaunchPaths) -> Result<(), String> {
    match &paths.wait_target {
        WaitTarget::Tcp { .. } => Ok(()),
        WaitTarget::Unix {
            host_socket_path, ..
        } => prepare_socket_path(host_socket_path),
    }
}

fn prepare_socket_path(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    if is_socket(path)? {
        fs::remove_file(path)
            .map_err(|err| format!("failed to remove stale socket {}: {err}", path.display()))?;
        return Ok(());
    }
    Err(format!(
        "refusing to replace non-socket path {}",
        path.display()
    ))
}

fn stage_secret_files(request: &RunRequest, paths: &LaunchPaths) -> Result<bool, String> {
    let secret_envs = request
        .inherited_env
        .iter()
        .filter(|env| env.file_source.is_some())
        .collect::<Vec<_>>();
    if secret_envs.is_empty() {
        return Ok(false);
    }

    if paths.host_secret_dir.exists() {
        fs::remove_dir_all(&paths.host_secret_dir).map_err(|err| {
            format!(
                "failed to remove stale secret directory {}: {err}",
                paths.host_secret_dir.display()
            )
        })?;
    }
    fs::create_dir_all(&paths.host_secret_dir).map_err(|err| {
        format!(
            "failed to create secret directory {}: {err}",
            paths.host_secret_dir.display()
        )
    })?;

    for env in secret_envs {
        let source = env
            .file_source
            .as_ref()
            .expect("filtered for file-backed environment variables");
        let staged_path = paths.host_secret_dir.join(&env.name);
        fs::copy(source, &staged_path).map_err(|err| {
            format!(
                "failed to stage API key file {} as {}: {err}",
                source.display(),
                staged_path.display()
            )
        })?;
        restrict_secret_file_permissions(&staged_path)?;
    }

    Ok(true)
}

#[cfg(unix)]
fn restrict_secret_file_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(|err| {
        format!(
            "failed to restrict staged API key file permissions {}: {err}",
            path.display()
        )
    })
}

#[cfg(not(unix))]
fn restrict_secret_file_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
fn cleanup_secret_files(paths: &LaunchPaths, staged: bool) -> Result<(), String> {
    if !staged || !paths.host_secret_dir.exists() {
        return Ok(());
    }
    fs::remove_dir_all(&paths.host_secret_dir).map_err(|err| {
        format!(
            "failed to remove secret directory {}: {err}",
            paths.host_secret_dir.display()
        )
    })
}

fn run_container(request: &RunRequest, paths: &LaunchPaths) -> Result<String, String> {
    let run_args = run_container_args(request, paths);
    container_command_output(&request.container_bin, run_args).map_err(|err| err.to_string())
}

fn run_container_args(request: &RunRequest, paths: &LaunchPaths) -> Vec<String> {
    let mut args = vec![
        "run".to_string(),
        "--detach".to_string(),
        "--name".to_string(),
        request.name.clone(),
    ];
    if let Some(cpus) = &request.cpus {
        args.push("--cpus".to_string());
        args.push(cpus.clone());
    }
    if let Some(memory) = &request.memory {
        args.push("--memory".to_string());
        args.push(memory.clone());
    }
    args.push("--env".to_string());
    args.push(DEFAULT_SID_HOME_ENV.to_string());
    for env in &request.inherited_env {
        args.push("--env".to_string());
        args.push(env.spec.clone());
    }
    if request
        .inherited_env
        .iter()
        .any(|env| env.file_source.is_some())
    {
        args.push("--mount".to_string());
        args.push(secret_mount_arg(&paths.host_secret_dir));
    }
    for (key, value) in run_labels(request, paths) {
        args.push("--label".to_string());
        args.push(format!("{key}={value}"));
    }
    match &paths.wait_target {
        WaitTarget::Tcp { host, port } => {
            args.push("--publish".to_string());
            args.push(format!("{}:{}:{}/tcp", host, port, request.container_port));
        }
        WaitTarget::Unix {
            host_socket_dir, ..
        } => {
            args.push("--volume".to_string());
            args.push(format!(
                "{}:{}",
                host_socket_dir.display(),
                request.container_socket_dir.display()
            ));
        }
    }
    args.push(request.image.clone());
    args.push(request.sid_bin.clone());
    args.extend(request.sid_args.iter().cloned());
    args.push("--listen".to_string());
    args.push(paths.container_endpoint.clone());
    args
}

fn default_inherited_env() -> Result<Vec<InheritedEnv>, String> {
    let mut out = Vec::new();
    for name in DEFAULT_INHERITED_ENV {
        match std::env::var(name) {
            Ok(value) if value.is_empty() => {}
            Ok(value) => out.push(resolve_inherited_env(name, &value)?),
            Err(std::env::VarError::NotPresent) => {}
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(format!("{name} must be valid Unicode to pass to container"));
            }
        }
    }
    Ok(out)
}

fn resolve_inherited_env(name: &str, value: &str) -> Result<InheritedEnv, String> {
    let Some(path) = api_key_file_path(value)? else {
        return Ok(InheritedEnv {
            name: name.to_string(),
            spec: name.to_string(),
            file_source: None,
        });
    };
    let source = path.canonicalize().map_err(|err| {
        format!(
            "{name} references unreadable API key file {}: {err}",
            path.display()
        )
    })?;
    if !source.is_file() {
        return Err(format!(
            "{name} references {}, which is not a regular file",
            source.display()
        ));
    }
    let target = container_secret_path(name);
    Ok(InheritedEnv {
        name: name.to_string(),
        spec: format!("{name}=file://{}", target.display()),
        file_source: Some(source),
    })
}

fn api_key_file_path(value: &str) -> Result<Option<PathBuf>, String> {
    let Some(path) = value.strip_prefix("file://") else {
        return Ok(None);
    };
    if path.is_empty() {
        return Err("API key file URL must include a path".to_string());
    }
    let path = PathBuf::from(path);
    if path.is_absolute() {
        Ok(Some(path))
    } else {
        std::env::current_dir()
            .map(|cwd| Some(cwd.join(path)))
            .map_err(|err| format!("failed to resolve relative API key file path: {err}"))
    }
}

fn container_secret_path(name: &str) -> PathBuf {
    PathBuf::from(CONTAINER_SECRET_DIR).join(name)
}

fn secret_mount_arg(host_secret_dir: &Path) -> String {
    format!(
        "type=bind,source={},target={},readonly",
        host_secret_dir.display(),
        CONTAINER_SECRET_DIR
    )
}

fn run_labels(request: &RunRequest, paths: &LaunchPaths) -> Vec<(&'static str, String)> {
    vec![
        (MANAGED_LABEL, "true".to_string()),
        (NAME_LABEL, request.name.clone()),
        (IMAGE_LABEL, request.image.clone()),
        (SOCKET_LABEL, paths.host_endpoint.clone()),
        (CONTAINER_SOCKET_LABEL, paths.container_endpoint.clone()),
    ]
}

fn wait_for_listen_target(target: &WaitTarget, timeout: Duration) -> Result<(), String> {
    match target {
        WaitTarget::Tcp { .. } => Ok(()),
        WaitTarget::Unix {
            host_socket_path, ..
        } => wait_for_socket(host_socket_path, timeout),
    }
}

fn wait_for_socket(path: &Path, timeout: Duration) -> Result<(), String> {
    let start = Instant::now();
    loop {
        if path.exists() {
            if is_socket(path)? {
                return Ok(());
            }
            return Err(format!(
                "expected socket at {}, found non-socket path",
                path.display()
            ));
        }
        if start.elapsed() >= timeout {
            return Err(format!("timed out waiting for socket {}", path.display()));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(unix)]
fn is_socket(path: &Path) -> Result<bool, String> {
    use std::os::unix::fs::FileTypeExt;

    fs::symlink_metadata(path)
        .map(|metadata| metadata.file_type().is_socket())
        .map_err(|err| format!("failed to inspect {}: {err}", path.display()))
}

#[cfg(not(unix))]
fn is_socket(_path: &Path) -> Result<bool, String> {
    Err("Unix-domain sockets are not supported on this platform".to_string())
}

fn resolve_overlay_ref(overlay: &str, registry: &str) -> Result<String, String> {
    if let Some(shorthand) = overlay.strip_prefix('@') {
        validate_nonempty(shorthand, "overlay shorthand")?;
        if shorthand.starts_with('/') || !shorthand.contains('/') {
            return Err(format!(
                "overlay shorthand {overlay:?} must look like @owner/name"
            ));
        }
        if shorthand.contains('@') || has_explicit_tag(shorthand) {
            Ok(format!("{registry}/{shorthand}"))
        } else {
            Ok(format!("{registry}/{shorthand}:latest"))
        }
    } else {
        Ok(overlay.to_string())
    }
}

fn has_explicit_tag(reference: &str) -> bool {
    let last_slash = reference.rfind('/').map(|idx| idx + 1).unwrap_or(0);
    reference[last_slash..].contains(':')
}

fn containerfile_quote(value: &str) -> String {
    let mut quoted = String::from("\"");
    for ch in value.chars() {
        match ch {
            '\\' => quoted.push_str("\\\\"),
            '"' => quoted.push_str("\\\""),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            other => quoted.push(other),
        }
    }
    quoted.push('"');
    quoted
}

fn create_temp_context_dir(name: &str) -> Result<PathBuf, String> {
    let millis = current_time_millis()?;
    Ok(std::env::temp_dir().join(format!(
        "sid-container-build-{}-{}-{millis}",
        std::process::id(),
        sanitize_for_path(name)
    )))
}

fn default_container_name(image: &str) -> Result<String, String> {
    let millis = current_time_millis()?;
    let image = sanitize_for_path(image)
        .chars()
        .take(DEFAULT_NAME_IMAGE_CHARS)
        .collect::<String>();
    let image = image.trim_matches('_');
    let image = if image.is_empty() { "container" } else { image };
    Ok(format!("sid-{image}-{}-{millis}", std::process::id()))
}

fn current_time_millis() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("system clock is before Unix epoch: {err}"))
        .map(|duration| duration.as_millis())
}

fn sanitize_for_path(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn display_container_id(container_id: &str) -> &str {
    if container_id.is_empty() {
        "<unknown>"
    } else {
        container_id
    }
}

fn default_string(value: String, default: &str) -> String {
    if value.is_empty() {
        default.to_string()
    } else {
        value
    }
}

fn default_u64(value: u64, default: u64) -> u64 {
    if value == 0 { default } else { value }
}

fn default_u16(value: u16, default: u16) -> u16 {
    if value == 0 { default } else { value }
}

fn parse_transport(value: &str) -> Result<ListenTransport, String> {
    match value {
        "tcp" => Ok(ListenTransport::Tcp),
        "unix" => Ok(ListenTransport::Unix),
        _ => Err("--transport must be either tcp or unix".to_string()),
    }
}

fn tcp_endpoint(host: &str, port: u16) -> String {
    format!("tcp://{host}:{port}")
}

fn allocate_tcp_port(host: &str) -> Result<u16, String> {
    let listener = std::net::TcpListener::bind((host, 0))
        .map_err(|err| format!("failed to allocate a free TCP port on {host}: {err}"))?;
    listener
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|err| format!("failed to inspect allocated TCP port: {err}"))
}

fn validate_container_name(name: &str) -> Result<(), String> {
    validate_nonempty(name, "--name")?;
    if name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.')
    {
        Ok(())
    } else {
        Err(
            "--name may contain only ASCII letters, digits, dots, underscores, and dashes"
                .to_string(),
        )
    }
}

fn validate_socket_name(name: &str) -> Result<(), String> {
    validate_nonempty(name, "--socket-name")?;
    if name.contains('/') {
        return Err("--socket-name must be a file name, not a path".to_string());
    }
    Ok(())
}

fn validate_tcp_host(host: &str, label: &str) -> Result<(), String> {
    validate_nonempty(host, label)?;
    if host.chars().any(|ch| ch.is_whitespace() || ch == '/') {
        Err(format!("{label} must be a TCP host or IP address"))
    } else {
        Ok(())
    }
}

fn validate_tcp_port(port: u16, label: &str) -> Result<(), String> {
    if port == 0 {
        Err(format!("{label} must be greater than zero"))
    } else {
        Ok(())
    }
}

fn validate_containerfile_image_ref(value: &str, label: &str) -> Result<(), String> {
    validate_nonempty(value, label)?;
    if value
        .chars()
        .any(|ch| ch.is_whitespace() || ch.is_control())
    {
        return Err(format!(
            "{label} must be a single container image reference without whitespace or control characters"
        ));
    }
    Ok(())
}

fn validate_absolute_path(path: &Path, option: &str) -> Result<(), String> {
    if path.is_absolute() {
        Ok(())
    } else {
        Err(format!("{option} must be an absolute path"))
    }
}

#[cfg(unix)]
fn validate_unix_socket_path_len(path: &Path, label: &str) -> Result<(), String> {
    use std::os::unix::ffi::OsStrExt;

    let len = path.as_os_str().as_bytes().len();
    if len < UNIX_SOCKET_PATH_MAX_BYTES {
        Ok(())
    } else {
        Err(format!(
            "{label} {} is {len} bytes; Unix socket paths must be shorter than {UNIX_SOCKET_PATH_MAX_BYTES} bytes on this platform. Use --socket-dir with a shorter directory or --socket-name with a shorter file name.",
            path.display()
        ))
    }
}

#[cfg(not(unix))]
fn validate_unix_socket_path_len(_path: &Path, _label: &str) -> Result<(), String> {
    Ok(())
}

fn validate_nonempty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must not be empty"))
    } else {
        Ok(())
    }
}

fn listen_wait_failed_message(
    err: String,
    container_id: &str,
    endpoint: &str,
    timeout: Duration,
    diagnostics: Option<&str>,
) -> String {
    let mut message = format!(
        "{err}; container {} was started detached and left running for debugging because it did not expose {} within {}ms",
        display_container_id(container_id),
        endpoint,
        timeout.as_millis()
    );
    if let Some(diagnostics) = diagnostics {
        message.push_str("; ");
        message.push_str(diagnostics);
    }
    message
}

fn container_startup_diagnostics(request: &RunRequest, container_id: &str) -> Option<String> {
    if container_id.trim().is_empty() {
        return None;
    }
    let mut parts = Vec::new();
    if let Ok(Some(status)) = inspect_container_status(&request.container_bin, container_id) {
        parts.push(format!("container status: {status}"));
    }
    if let Ok(logs) = container_command_output(
        &request.container_bin,
        vec!["logs".to_string(), container_id.to_string()],
    ) {
        let logs = logs.trim();
        if !logs.is_empty() {
            parts.push(format!("container logs:\n{}", truncate_log_output(logs)));
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("; "))
    }
}

#[derive(serde::Deserialize)]
struct RawInspectContainer {
    status: String,
}

fn inspect_container_status(
    container_bin: &str,
    container_id: &str,
) -> Result<Option<String>, String> {
    let stdout = container_command_output(
        container_bin,
        vec!["inspect".to_string(), container_id.to_string()],
    )
    .map_err(|err| err.to_string())?;
    let containers = serde_json::from_str::<Vec<RawInspectContainer>>(&stdout)
        .map_err(|err| format!("failed to decode container inspect output: {err}"))?;
    Ok(containers
        .into_iter()
        .next()
        .map(|container| container.status))
}

fn truncate_log_output(logs: &str) -> String {
    const MAX_LOG_BYTES: usize = 4096;
    if logs.len() <= MAX_LOG_BYTES {
        return logs.to_string();
    }
    let mut split = logs.len() - MAX_LOG_BYTES;
    while !logs.is_char_boundary(split) {
        split += 1;
    }
    format!("<truncated>\n{}", &logs[split..])
}

fn is_top_level_help(args: &[String]) -> bool {
    args.is_empty() || matches!(args, [help] if help == "-h" || help == "--help")
}

fn is_subcommand_help(args: &[String], command: &str) -> bool {
    matches!(args, [cmd, help] if cmd == command && (help == "-h" || help == "--help"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| arg.to_string()).collect()
    }

    #[test]
    fn parse_build_resolves_overlay_shorthand() {
        let request = parse_build_request(&strings(&[
            "--container-bin",
            "container",
            "sid-dev",
            "rust:trixie",
            "@rescrv/blue",
            "@rescrv/sid-isnt-done:v1",
            "example.com/direct/overlay:tag",
        ]))
        .unwrap();
        assert_eq!(request.name, "sid-dev");
        assert_eq!(request.base_image, "rust:trixie");
        assert_eq!(
            request.resolved_overlays,
            vec![
                "ghcr.io/rescrv/blue:latest",
                "ghcr.io/rescrv/sid-isnt-done:v1",
                "example.com/direct/overlay:tag",
            ]
        );
    }

    #[test]
    fn parse_build_uses_custom_overlay_registry() {
        let request = parse_build_request(&strings(&[
            "--registry",
            "registry.example.com",
            "sid-dev",
            "debian:trixie",
            "@rescrv/sid-isnt-done",
        ]))
        .unwrap();
        assert_eq!(
            request.resolved_overlays,
            vec!["registry.example.com/rescrv/sid-isnt-done:latest"]
        );
    }

    #[test]
    fn parse_build_rejects_base_image_with_containerfile_control_chars() {
        let err =
            parse_build_request(&strings(&["sid-dev", "alpine\nRUN echo injected"])).unwrap_err();
        assert!(err.contains("base image"));
        assert!(err.contains("without whitespace or control characters"));
    }

    #[test]
    fn parse_build_rejects_overlay_with_containerfile_control_chars() {
        let err = parse_build_request(&strings(&[
            "sid-dev",
            "alpine",
            "example.com/overlay\nRUN echo injected",
        ]))
        .unwrap_err();
        assert!(err.contains("overlay"));
        assert!(err.contains("without whitespace or control characters"));
    }

    #[test]
    fn render_containerfile_copies_overlays_in_order() {
        let request = BuildRequest {
            name: "sid-dev".to_string(),
            base_image: "rust:trixie".to_string(),
            overlays: vec![
                "@rescrv/blue".to_string(),
                "@rescrv/sid-isnt-done".to_string(),
            ],
            resolved_overlays: vec![
                "ghcr.io/rescrv/blue:latest".to_string(),
                "ghcr.io/rescrv/sid-isnt-done:latest".to_string(),
            ],
            container_bin: "container".to_string(),
            containerfile: None,
        };
        let rendered = render_containerfile(&request);
        assert!(rendered.contains("FROM ghcr.io/rescrv/blue:latest AS sid_overlay_0\n"));
        assert!(rendered.contains("FROM ghcr.io/rescrv/sid-isnt-done:latest AS sid_overlay_1\n"));
        assert!(rendered.contains("FROM rust:trixie\n"));
        assert!(rendered.contains("COPY --from=sid_overlay_0 / /\n"));
        assert!(rendered.contains("COPY --from=sid_overlay_1 / /\n"));
        assert!(rendered.contains("COPY sid-image.json /etc/sid-image.json\n"));
    }

    #[test]
    fn build_container_args_use_apple_container_build_flags() {
        let request = BuildRequest {
            name: "sid-dev".to_string(),
            base_image: "rust:trixie".to_string(),
            overlays: Vec::new(),
            resolved_overlays: Vec::new(),
            container_bin: "container".to_string(),
            containerfile: None,
        };
        assert_eq!(
            build_container_args(
                &request,
                Path::new("/tmp/sid/Containerfile"),
                Path::new("/tmp/sid")
            ),
            vec![
                "build",
                "--tag",
                "sid-dev",
                "--file",
                "/tmp/sid/Containerfile",
                "/tmp/sid",
            ]
        );
    }

    #[test]
    fn parse_run_treats_remaining_free_args_as_sid_args() {
        let request = parse_run_request(&strings(&[
            "--name",
            "sid-a",
            "--timeout-ms",
            "2500",
            "sid-dev",
            "--config",
            "/etc/sid/config.toml",
        ]))
        .unwrap();
        assert_eq!(request.name, "sid-a");
        assert_eq!(request.image, "sid-dev");
        assert_eq!(request.transport, ListenTransport::Tcp);
        assert_eq!(request.container_port, DEFAULT_CONTAINER_TCP_PORT);
        assert_eq!(request.timeout, Duration::from_millis(2500));
        assert_eq!(request.sid_args, vec!["--config", "/etc/sid/config.toml"]);
    }

    #[test]
    fn parse_run_generates_name_when_omitted() {
        let request = parse_run_request(&strings(&["ghcr.io/rescrv/sid-dev:latest"])).unwrap();
        assert!(request.name.starts_with("sid-ghcr.io_rescrv_sid-dev"));
        validate_container_name(&request.name).unwrap();
    }

    #[test]
    fn launch_paths_uses_tcp_by_default() {
        let request = RunRequest {
            name: "sid-a".to_string(),
            image: "sid-dev".to_string(),
            container_bin: DEFAULT_CONTAINER_BIN.to_string(),
            sid_bin: DEFAULT_SID_BIN.to_string(),
            inherited_env: Vec::new(),
            transport: ListenTransport::Tcp,
            host_address: DEFAULT_HOST_TCP_ADDRESS.to_string(),
            host_port: 45450,
            container_address: DEFAULT_CONTAINER_TCP_ADDRESS.to_string(),
            container_port: DEFAULT_CONTAINER_TCP_PORT,
            socket_dir: None,
            socket_name: DEFAULT_SOCKET_NAME.to_string(),
            container_socket_dir: PathBuf::from(DEFAULT_CONTAINER_SOCKET_DIR),
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
            cpus: None,
            memory: None,
            sid_args: Vec::new(),
        };

        let paths = launch_paths(&request).unwrap();
        assert_eq!(paths.host_endpoint, "tcp://127.0.0.1:45450");
        assert_eq!(paths.container_endpoint, "tcp://0.0.0.0:8890");
        assert_eq!(
            paths.wait_target,
            WaitTarget::Tcp {
                host: "127.0.0.1".to_string(),
                port: 45450
            }
        );
    }

    #[test]
    fn unix_launch_paths_use_short_default_socket_root() {
        let name = format!(
            "sid-test-{}-{}",
            std::process::id(),
            current_time_millis().unwrap()
        );
        let request = RunRequest {
            name: name.clone(),
            image: "sid-dev".to_string(),
            container_bin: DEFAULT_CONTAINER_BIN.to_string(),
            sid_bin: DEFAULT_SID_BIN.to_string(),
            inherited_env: Vec::new(),
            transport: ListenTransport::Unix,
            host_address: DEFAULT_HOST_TCP_ADDRESS.to_string(),
            host_port: 0,
            container_address: DEFAULT_CONTAINER_TCP_ADDRESS.to_string(),
            container_port: DEFAULT_CONTAINER_TCP_PORT,
            socket_dir: None,
            socket_name: DEFAULT_SOCKET_NAME.to_string(),
            container_socket_dir: PathBuf::from(DEFAULT_CONTAINER_SOCKET_DIR),
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
            cpus: None,
            memory: None,
            sid_args: Vec::new(),
        };

        let paths = launch_paths(&request).unwrap();
        let WaitTarget::Unix {
            host_socket_dir,
            host_socket_path,
            ..
        } = &paths.wait_target
        else {
            panic!("expected unix wait target");
        };
        assert!(host_socket_dir.ends_with(Path::new("sid").join(&name)));
        validate_unix_socket_path_len(host_socket_path, "host socket path").unwrap();

        let _ = fs::remove_dir_all(host_socket_dir);
    }

    #[test]
    fn validate_unix_socket_path_rejects_overlong_paths() {
        let path = PathBuf::from(format!("/{}", "s".repeat(UNIX_SOCKET_PATH_MAX_BYTES)));
        let err = validate_unix_socket_path_len(&path, "host socket path").unwrap_err();
        assert!(err.contains("host socket path"));
        assert!(err.contains("Unix socket paths must be shorter"));
    }

    #[test]
    fn socket_wait_error_documents_debug_container_lifetime() {
        let message = listen_wait_failed_message(
            "timed out waiting for socket /tmp/sid.sock".to_string(),
            "sid-a",
            "/tmp/sid.sock",
            Duration::from_millis(250),
            None,
        );
        assert!(message.contains("left running for debugging"));
        assert!(message.contains("sid-a"));
        assert!(message.contains("250ms"));
    }

    #[test]
    fn socket_wait_error_includes_startup_diagnostics() {
        let message = listen_wait_failed_message(
            "timed out waiting for socket /tmp/sid.sock".to_string(),
            "sid-a",
            "/tmp/sid.sock",
            Duration::from_millis(250),
            Some("container status: stopped"),
        );
        assert!(message.contains("container status: stopped"));
    }

    #[test]
    fn tcp_listen_target_wait_does_not_probe_listener() {
        wait_for_listen_target(
            &WaitTarget::Tcp {
                host: "127.0.0.1".to_string(),
                port: 1,
            },
            Duration::from_millis(0),
        )
        .unwrap();
    }

    #[test]
    fn resolve_inherited_env_passes_direct_values_by_name() {
        assert_eq!(
            resolve_inherited_env("CLAUDIUS_API_KEY", "sk-test").unwrap(),
            InheritedEnv {
                name: "CLAUDIUS_API_KEY".to_string(),
                spec: "CLAUDIUS_API_KEY".to_string(),
                file_source: None,
            }
        );
    }

    #[test]
    fn resolve_inherited_env_remaps_file_urls() {
        let dir = std::env::temp_dir().join(format!(
            "sid-container-test-remap-{}-{}",
            std::process::id(),
            current_time_millis().unwrap()
        ));
        fs::create_dir_all(&dir).unwrap();
        let key_path = dir.join("anthropic-key");
        fs::write(&key_path, "sk-test").unwrap();

        let inherited = resolve_inherited_env(
            "CLAUDIUS_API_KEY",
            &format!("file://{}", key_path.display()),
        )
        .unwrap();
        let target = PathBuf::from("/run/sid-secrets/CLAUDIUS_API_KEY");
        assert_eq!(
            inherited,
            InheritedEnv {
                name: "CLAUDIUS_API_KEY".to_string(),
                spec: format!("CLAUDIUS_API_KEY=file://{}", target.display()),
                file_source: Some(key_path.canonicalize().unwrap()),
            }
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stage_secret_files_copies_file_backed_env_to_private_dir() {
        let dir = std::env::temp_dir().join(format!(
            "sid-container-test-stage-{}-{}",
            std::process::id(),
            current_time_millis().unwrap()
        ));
        fs::create_dir_all(&dir).unwrap();
        let key_path = dir.join("anthropic-key");
        fs::write(&key_path, "sk-test\n").unwrap();

        let request = RunRequest {
            name: "sid-a".to_string(),
            image: "sid-dev".to_string(),
            container_bin: DEFAULT_CONTAINER_BIN.to_string(),
            sid_bin: DEFAULT_SID_BIN.to_string(),
            inherited_env: vec![
                resolve_inherited_env(
                    "CLAUDIUS_API_KEY",
                    &format!("file://{}", key_path.display()),
                )
                .unwrap(),
            ],
            transport: ListenTransport::Tcp,
            host_address: DEFAULT_HOST_TCP_ADDRESS.to_string(),
            host_port: 45450,
            container_address: DEFAULT_CONTAINER_TCP_ADDRESS.to_string(),
            container_port: DEFAULT_CONTAINER_TCP_PORT,
            socket_dir: None,
            socket_name: "agent.sock".to_string(),
            container_socket_dir: PathBuf::from("/agent"),
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
            cpus: None,
            memory: None,
            sid_args: Vec::new(),
        };
        let paths = LaunchPaths {
            host_endpoint: "tcp://127.0.0.1:45450".to_string(),
            host_secret_dir: dir.join("secrets"),
            container_endpoint: "tcp://0.0.0.0:8890".to_string(),
            wait_target: WaitTarget::Tcp {
                host: "127.0.0.1".to_string(),
                port: 45450,
            },
        };

        assert!(stage_secret_files(&request, &paths).unwrap());
        assert_eq!(
            fs::read_to_string(paths.host_secret_dir.join("CLAUDIUS_API_KEY")).unwrap(),
            "sk-test\n"
        );
        let args = run_container_args(&request, &paths);
        assert!(args.contains(&"--mount".to_string()));
        assert!(args.contains(&format!(
            "type=bind,source={},target=/run/sid-secrets,readonly",
            paths.host_secret_dir.display()
        )));
        cleanup_secret_files(&paths, true).unwrap();
        assert!(!paths.host_secret_dir.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_container_args_label_mount_and_listen() {
        let request = RunRequest {
            name: "sid-a".to_string(),
            image: "sid-dev".to_string(),
            container_bin: DEFAULT_CONTAINER_BIN.to_string(),
            sid_bin: DEFAULT_SID_BIN.to_string(),
            inherited_env: vec![InheritedEnv {
                name: "CLAUDIUS_API_KEY".to_string(),
                spec: "CLAUDIUS_API_KEY".to_string(),
                file_source: None,
            }],
            transport: ListenTransport::Unix,
            host_address: DEFAULT_HOST_TCP_ADDRESS.to_string(),
            host_port: 0,
            container_address: DEFAULT_CONTAINER_TCP_ADDRESS.to_string(),
            container_port: DEFAULT_CONTAINER_TCP_PORT,
            socket_dir: None,
            socket_name: "agent.sock".to_string(),
            container_socket_dir: PathBuf::from("/agent"),
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
            cpus: Some("2".to_string()),
            memory: Some("4G".to_string()),
            sid_args: vec!["--config".to_string(), "/etc/sid/config.toml".to_string()],
        };
        let paths = LaunchPaths {
            host_endpoint: "/tmp/sid/sid-a/agent.sock".to_string(),
            host_secret_dir: PathBuf::from("/tmp/sid/sid-a-secrets"),
            container_endpoint: "unix:///agent/agent.sock".to_string(),
            wait_target: WaitTarget::Unix {
                host_socket_dir: PathBuf::from("/tmp/sid/sid-a"),
                host_socket_path: PathBuf::from("/tmp/sid/sid-a/agent.sock"),
                container_socket_path: PathBuf::from("/agent/agent.sock"),
            },
        };
        let args = run_container_args(&request, &paths);
        assert_eq!(
            args[0..8],
            [
                "run", "--detach", "--name", "sid-a", "--cpus", "2", "--memory", "4G"
            ]
        );
        assert!(args.contains(&"--label".to_string()));
        assert!(args.contains(&"sid.managed=true".to_string()));
        assert!(args.contains(&"sid.image=sid-dev".to_string()));
        assert!(args.contains(&"--env".to_string()));
        assert!(args.contains(&DEFAULT_SID_HOME_ENV.to_string()));
        assert!(args.contains(&"CLAUDIUS_API_KEY".to_string()));
        assert!(args.contains(&"--volume".to_string()));
        assert!(args.contains(&"/tmp/sid/sid-a:/agent".to_string()));
        assert_eq!(args[args.len() - 2], "--listen");
        assert_eq!(args[args.len() - 1], "unix:///agent/agent.sock");
    }

    #[test]
    fn run_container_args_publish_tcp_and_listen_tcp() {
        let request = RunRequest {
            name: "sid-a".to_string(),
            image: "sid-dev".to_string(),
            container_bin: DEFAULT_CONTAINER_BIN.to_string(),
            sid_bin: DEFAULT_SID_BIN.to_string(),
            inherited_env: Vec::new(),
            transport: ListenTransport::Tcp,
            host_address: DEFAULT_HOST_TCP_ADDRESS.to_string(),
            host_port: 45450,
            container_address: DEFAULT_CONTAINER_TCP_ADDRESS.to_string(),
            container_port: DEFAULT_CONTAINER_TCP_PORT,
            socket_dir: None,
            socket_name: DEFAULT_SOCKET_NAME.to_string(),
            container_socket_dir: PathBuf::from(DEFAULT_CONTAINER_SOCKET_DIR),
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
            cpus: None,
            memory: None,
            sid_args: Vec::new(),
        };
        let paths = LaunchPaths {
            host_endpoint: "tcp://127.0.0.1:45450".to_string(),
            host_secret_dir: PathBuf::from("/tmp/sid/sid-a-secrets"),
            container_endpoint: "tcp://0.0.0.0:8890".to_string(),
            wait_target: WaitTarget::Tcp {
                host: "127.0.0.1".to_string(),
                port: 45450,
            },
        };
        let args = run_container_args(&request, &paths);
        assert!(args.contains(&"--env".to_string()));
        assert!(args.contains(&DEFAULT_SID_HOME_ENV.to_string()));
        assert!(args.contains(&"--publish".to_string()));
        assert!(args.contains(&"127.0.0.1:45450:8890/tcp".to_string()));
        assert!(!args.contains(&"--volume".to_string()));
        assert!(args.contains(&"sid.socket=tcp://127.0.0.1:45450".to_string()));
        assert_eq!(args[args.len() - 2], "--listen");
        assert_eq!(args[args.len() - 1], "tcp://0.0.0.0:8890");
    }
}
