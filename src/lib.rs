pub mod builtin_tools;
pub mod config;
#[cfg(test)]
pub(crate) mod test_support;

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::ops::ControlFlow;
use std::path::{Path as StdPath, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use claudius::chat::{ChatAgent, ChatConfig};
use claudius::{
    Agent, Anthropic, Error, FileSystem, IntermediateToolResult, Model, SystemPrompt,
    ThinkingConfig, Tool, ToolBash20250124, ToolCallback, ToolParam, ToolResult, ToolResultBlock,
    ToolTextEditor20250728, ToolUnionParam, ToolUseBlock,
};
use handled::SError;
use rc_conf::{RcConf, SwitchPosition, var_name_from_service, var_prefix_from_service};
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use utf8path::Path;

use crate::config::{
    AGENTS_CONF_FILE, AgentConfig, Config, TOOL_PROTOCOL_VERSION, TOOLS_CONF_FILE, TOOLS_DIR,
    ToolConfig, is_valid_anthropic_tool_name, resolve_canonical_tool_id,
};
const DEFAULT_AGENT_ID: &str = "sid";

pub fn initialize_sid_workspace_root_env(workspace_root: &Path) {
    let effective_workspace_root = std::env::var("SID_WORKSPACE_ROOT")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| workspace_root.as_str().to_string());
    unsafe {
        std::env::set_var("SID_WORKSPACE_ROOT", effective_workspace_root);
    }
}

pub struct SidAgent {
    id: String,
    enabled: SwitchPosition,
    config: ChatConfig,
    tools: Vec<Arc<dyn Tool<Self>>>,
    builtin_bindings: BuiltinToolBindings,
    config_root: Path<'static>,
    filesystem: Path<'static>,
}

impl SidAgent {
    pub fn new(config: ChatConfig, filesystem: Path<'static>) -> Self {
        Self::new_with_roots(config, filesystem.clone(), filesystem)
    }

    fn new_with_roots(
        config: ChatConfig,
        config_root: Path<'static>,
        filesystem: Path<'static>,
    ) -> Self {
        Self::with_parts(
            DEFAULT_AGENT_ID.to_string(),
            SwitchPosition::Yes,
            config,
            vec![],
            BuiltinToolBindings::default(),
            config_root,
            filesystem,
        )
    }

    pub fn from_workspace(root: &Path, fallback: ChatConfig) -> Result<Self, SError> {
        Self::from_workspace_with_config_root(root, root, fallback)
    }

    pub fn from_workspace_with_config_root(
        workspace_root: &Path,
        config_root: &Path,
        fallback: ChatConfig,
    ) -> Result<Self, SError> {
        if !workspace_has_config(config_root) {
            return Ok(Self::new_with_roots(
                fallback,
                config_root.clone().into_owned(),
                workspace_root.clone().into_owned(),
            ));
        }
        let config = Config::load(config_root)?;
        let agent = default_agent_id(&config)?;
        Self::from_loaded_config(
            &config,
            &agent,
            Some(&fallback),
            workspace_root.clone().into_owned(),
        )
    }

    pub fn from_workspace_agent(
        root: &Path,
        agent: &str,
        fallback: ChatConfig,
    ) -> Result<Self, SError> {
        Self::from_workspace_agent_with_config_root(root, root, agent, fallback)
    }

    pub fn from_workspace_agent_with_config_root(
        workspace_root: &Path,
        config_root: &Path,
        agent: &str,
        fallback: ChatConfig,
    ) -> Result<Self, SError> {
        if !workspace_has_config(config_root) {
            return if agent == DEFAULT_AGENT_ID {
                Ok(Self::new_with_roots(
                    fallback,
                    config_root.clone().into_owned(),
                    workspace_root.clone().into_owned(),
                ))
            } else {
                Err(missing_agent_error(agent))
            };
        }
        let config = Config::load(config_root)?;
        Self::from_loaded_config(
            &config,
            agent,
            Some(&fallback),
            workspace_root.clone().into_owned(),
        )
    }

    pub fn from_config(
        config: &Config,
        agent: &str,
        filesystem: Path<'static>,
    ) -> Result<Self, SError> {
        Self::from_loaded_config(config, agent, None, filesystem)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn requires_confirmation(&self) -> bool {
        self.enabled == SwitchPosition::Manual
    }

    fn from_loaded_config(
        config: &Config,
        agent: &str,
        fallback: Option<&ChatConfig>,
        filesystem: Path<'static>,
    ) -> Result<Self, SError> {
        let agent_config = config
            .agents
            .get(agent)
            .ok_or_else(|| missing_agent_error(agent))?;
        if !agent_config.enabled.can_be_started() {
            return Err(disabled_agent_error(agent, agent_config.enabled));
        }

        let built_tools = build_tools(config, agent_config)?;
        let chat_config = merged_chat_config(agent_config, fallback);
        Ok(Self::with_parts(
            agent.to_string(),
            agent_config.enabled,
            chat_config,
            built_tools.tools,
            built_tools.builtin_bindings,
            config.root.clone(),
            filesystem,
        ))
    }

    fn with_parts(
        id: String,
        enabled: SwitchPosition,
        config: ChatConfig,
        tools: Vec<Arc<dyn Tool<Self>>>,
        builtin_bindings: BuiltinToolBindings,
        config_root: Path<'static>,
        filesystem: Path<'static>,
    ) -> Self {
        Self {
            id,
            enabled,
            config,
            tools,
            builtin_bindings,
            config_root,
            filesystem,
        }
    }

    async fn run_bash_command(
        &self,
        command: &str,
        restart: bool,
    ) -> Result<String, std::io::Error> {
        let Some(binding) = self.builtin_bindings.bash.as_ref() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "bash is not supported",
            ));
        };

        let input = serde_json::Map::from_iter([
            (
                "command".to_string(),
                serde_json::Value::String(command.to_string()),
            ),
            ("restart".to_string(), serde_json::Value::Bool(restart)),
        ]);
        let tool_use_id = synthetic_tool_use_id(&binding.service_name);
        self.invoke_rc_tool(binding, &tool_use_id, input).await
    }

    async fn invoke_rc_tool(
        &self,
        binding: &RcToolBinding,
        tool_use_id: &str,
        input: serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, std::io::Error> {
        invoke_rc_tool_text(
            &binding.service_name,
            &binding.service_name,
            &binding.canonical_id,
            &binding.executable_path,
            self,
            tool_use_id,
            input,
        )
        .await
        .map_err(std::io::Error::other)
    }

    async fn default_text_editor(&self, tool_use: ToolUseBlock) -> Result<String, std::io::Error> {
        #[derive(serde::Deserialize)]
        struct Command {
            command: String,
        }
        let cmd: Command = serde_json::from_value(tool_use.input.clone())?;
        match cmd.command.as_str() {
            "view" => {
                #[derive(serde::Deserialize)]
                struct ViewTool {
                    path: String,
                    view_range: Option<(u32, u32)>,
                }
                let args: ViewTool = serde_json::from_value(tool_use.input)?;
                self.view(&args.path, args.view_range).await
            }
            "str_replace" => {
                #[derive(serde::Deserialize)]
                struct StrReplaceTool {
                    path: String,
                    old_str: String,
                    new_str: Option<String>,
                }
                let args: StrReplaceTool = serde_json::from_value(tool_use.input)?;
                let new_str = args.new_str.as_deref().unwrap_or("");
                self.str_replace(&args.path, &args.old_str, new_str).await
            }
            "insert" => {
                #[derive(serde::Deserialize)]
                struct InsertTool {
                    path: String,
                    insert_line: u32,
                    insert_text: Option<String>,
                    new_str: Option<String>,
                }
                let args: InsertTool = serde_json::from_value(tool_use.input)?;
                let text = args.insert_text.or(args.new_str).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "missing insert_text field",
                    )
                })?;
                self.insert(&args.path, args.insert_line, &text).await
            }
            "create" => {
                #[derive(serde::Deserialize)]
                struct CreateTool {
                    path: String,
                    file_text: String,
                }
                let args: CreateTool = serde_json::from_value(tool_use.input)?;
                self.create(&args.path, &args.file_text).await
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!("{} is not a supported tool", tool_use.name),
            )),
        }
    }
}

#[async_trait::async_trait]
impl Agent for SidAgent {
    fn stream_label(&self) -> String {
        self.id.clone()
    }

    async fn max_tokens(&self) -> u32 {
        self.config.max_tokens()
    }

    async fn model(&self) -> Model {
        self.config.model()
    }

    async fn stop_sequences(&self) -> Option<Vec<String>> {
        let sequences = self.config.stop_sequences();
        if sequences.is_empty() {
            None
        } else {
            Some(sequences.to_vec())
        }
    }

    async fn system(&self) -> Option<SystemPrompt> {
        let prompt = self.config.template.system.as_ref()?;
        Some(prompt.clone())
    }

    async fn temperature(&self) -> Option<f32> {
        self.config.template.temperature
    }

    async fn thinking(&self) -> Option<ThinkingConfig> {
        self.config.template.thinking
    }

    async fn tools(&self) -> Vec<Arc<dyn Tool<Self>>> {
        self.tools.clone()
    }

    async fn top_k(&self) -> Option<u32> {
        self.config.template.top_k
    }

    async fn top_p(&self) -> Option<f32> {
        self.config.template.top_p
    }

    async fn filesystem(&self) -> Option<&dyn FileSystem> {
        Some(&self.filesystem)
    }

    async fn text_editor(&self, tool_use: ToolUseBlock) -> Result<String, std::io::Error> {
        let Some(binding) = self.builtin_bindings.edit.as_ref() else {
            return self.default_text_editor(tool_use).await;
        };
        let input = tool_use.input.as_object().cloned().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "text editor input must be a JSON object",
            )
        })?;
        self.invoke_rc_tool(binding, &tool_use.id, input).await
    }

    async fn bash(&self, command: &str, restart: bool) -> Result<String, std::io::Error> {
        self.run_bash_command(command, restart).await
    }
}

impl ChatAgent for SidAgent {
    fn config(&self) -> &ChatConfig {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ChatConfig {
        &mut self.config
    }
}

fn sid_agent_error(code: &str, message: &str) -> SError {
    SError::new("sid-agent")
        .with_code(code)
        .with_message(message)
}

fn tool_runtime_error(tool: &str, code: &str, message: &str) -> SError {
    sid_agent_error(code, message).with_string_field("tool", tool)
}

#[derive(Clone, Debug, Default)]
struct BuiltinToolBindings {
    bash: Option<RcToolBinding>,
    edit: Option<RcToolBinding>,
}

#[derive(Clone, Debug)]
struct RcToolBinding {
    service_name: String,
    canonical_id: String,
    executable_path: Path<'static>,
}

struct BuiltTools {
    tools: Vec<Arc<dyn Tool<SidAgent>>>,
    builtin_bindings: BuiltinToolBindings,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BuiltinToolKind {
    Bash,
    Edit,
}

impl BuiltinToolKind {
    fn from_canonical_id(canonical_id: &str) -> Option<Self> {
        match canonical_id {
            "bash" => Some(Self::Bash),
            "edit" => Some(Self::Edit),
            _ => None,
        }
    }

    fn tool(self) -> Arc<dyn Tool<SidAgent>> {
        match self {
            Self::Bash => Arc::new(ToolBash20250124::new()) as Arc<dyn Tool<SidAgent>>,
            Self::Edit => Arc::new(ToolTextEditor20250728::new()) as Arc<dyn Tool<SidAgent>>,
        }
    }

    fn bind(self, builtins: &mut BuiltinToolBindings, binding: RcToolBinding) {
        match self {
            Self::Bash if builtins.bash.is_none() => builtins.bash = Some(binding),
            Self::Edit if builtins.edit.is_none() => builtins.edit = Some(binding),
            _ => {}
        }
    }
}

#[derive(Clone, Debug)]
struct ExternalTool {
    name: String,
    canonical_id: String,
    executable_path: Path<'static>,
    description: String,
    input_schema: serde_json::Value,
}

impl ExternalTool {
    fn from_config(name: String, canonical_id: String, tool: &ToolConfig) -> Self {
        let manifest = tool
            .manifest
            .as_ref()
            .expect("external tools require a manifest");
        Self {
            name,
            canonical_id,
            executable_path: tool.executable_path.clone(),
            description: manifest.description.clone(),
            input_schema: manifest.input_schema.clone(),
        }
    }
}

impl Tool<SidAgent> for ExternalTool {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn callback(&self) -> Box<dyn ToolCallback<SidAgent> + '_> {
        Box::new(ExternalToolCallback { tool: self.clone() })
    }

    fn to_param(&self) -> ToolUnionParam {
        ToolUnionParam::CustomTool(
            ToolParam::new(self.name.clone(), self.input_schema.clone())
                .with_description(self.description.clone()),
        )
    }
}

#[derive(Clone, Debug)]
struct ExternalToolCallback {
    tool: ExternalTool,
}

#[async_trait::async_trait]
impl ToolCallback<SidAgent> for ExternalToolCallback {
    async fn compute_tool_result(
        &self,
        _client: &Anthropic,
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
    ) -> Box<dyn IntermediateToolResult> {
        Box::new(invoke_external_tool(&self.tool, agent, tool_use).await)
    }

    async fn apply_tool_result(
        &self,
        _client: &Anthropic,
        _agent: &mut SidAgent,
        _tool_use: &ToolUseBlock,
        intermediate: Box<dyn IntermediateToolResult>,
    ) -> ToolResult {
        let Some(intermediate) = intermediate.as_any().downcast_ref::<ToolResult>() else {
            return ControlFlow::Break(Error::unknown(
                "intermediate tool result fails to deserialize",
            ));
        };
        intermediate.clone()
    }
}

#[derive(Debug, Serialize)]
struct ToolRequestEnvelope {
    protocol_version: u32,
    request_id: String,
    tool: ToolRequestTool,
    invocation: ToolRequestInvocation,
    agent: ToolRequestAgent,
    workspace: ToolRequestWorkspace,
    files: ToolRequestFiles,
}

#[derive(Debug, Serialize)]
struct ToolRequestTool {
    id: String,
}

#[derive(Debug, Serialize)]
struct ToolRequestInvocation {
    tool_use_id: String,
    input: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ToolRequestAgent {
    id: String,
}

#[derive(Debug, Serialize)]
struct ToolRequestWorkspace {
    root: String,
    cwd: String,
}

#[derive(Debug, Serialize)]
struct ToolRequestFiles {
    scratch_dir: String,
    result_file: String,
}

#[derive(Debug, Deserialize)]
struct ToolResultEnvelope {
    protocol_version: u32,
    request_id: String,
    ok: bool,
    output: Option<ToolResultOutput>,
    error: Option<ToolResultError>,
}

#[derive(Debug, Deserialize)]
struct ToolResultOutput {
    kind: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ToolResultError {
    code: Option<String>,
    message: Option<String>,
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

async fn invoke_external_tool(
    tool: &ExternalTool,
    agent: &SidAgent,
    tool_use: &ToolUseBlock,
) -> ToolResult {
    let input = match tool_use.input.as_object() {
        Some(input) => input.clone(),
        None => {
            return tool_error_result(
                &tool_use.id,
                format!(
                    "tool '{}' protocol error: tool input must be a JSON object",
                    tool.name
                ),
            );
        }
    };

    match invoke_rc_tool_text(
        &tool.name,
        &tool.name,
        &tool.canonical_id,
        &tool.executable_path,
        agent,
        &tool_use.id,
        input,
    )
    .await
    {
        Ok(text) => tool_success_result(&tool_use.id, text),
        Err(message) => tool_error_result(&tool_use.id, message),
    }
}

async fn invoke_rc_tool_text(
    display_name: &str,
    rc_service_name: &str,
    canonical_id: &str,
    executable_path: &Path<'_>,
    agent: &SidAgent,
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
            id: agent.id.clone(),
        },
        workspace: ToolRequestWorkspace {
            root: agent.filesystem.as_str().to_string(),
            cwd: agent.filesystem.as_str().to_string(),
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
        agent,
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
        .current_dir(agent.filesystem.as_str())
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
    agent: &SidAgent,
    request_file: &StdPath,
    result_file: &StdPath,
    scratch_dir: &StdPath,
) -> Result<ToolRcRuntime, SError> {
    let tools_conf_path = agent.config_root.join(TOOLS_CONF_FILE);
    let rc_d_path = agent.config_root.join(TOOLS_DIR);
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
    let overlay = render_tool_rc_overlay(&base_rc_conf, &services, agent, &overlay_context);
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
    agent: &SidAgent,
    overlay_context: &ToolOverlayContext<'_>,
) -> String {
    let request_file = overlay_context.request_file.to_string_lossy().into_owned();
    let result_file = overlay_context.result_file.to_string_lossy().into_owned();
    let scratch_dir = overlay_context.scratch_dir.to_string_lossy().into_owned();
    let workspace_root = agent.filesystem.as_str().to_string();
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
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}AGENT_ID"), &agent.id);
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

fn extract_tool_output(
    display_name: &str,
    request_id: &str,
    result: ToolResultEnvelope,
) -> Result<String, String> {
    if result.protocol_version != TOOL_PROTOCOL_VERSION {
        return Err(format!(
            "tool '{}' protocol error: unsupported result protocol version {}",
            display_name, result.protocol_version
        ));
    }
    if result.request_id != request_id {
        return Err(format!(
            "tool '{}' protocol error: request_id mismatch (expected {}, got {})",
            display_name, request_id, result.request_id
        ));
    }

    if result.ok {
        let Some(output) = result.output else {
            return Err(format!(
                "tool '{}' protocol error: missing success output",
                display_name
            ));
        };
        if output.kind != "text" {
            return Err(format!(
                "tool '{}' protocol error: unsupported output kind '{}'",
                display_name, output.kind
            ));
        }
        let Some(text) = output.text else {
            return Err(format!(
                "tool '{}' protocol error: missing output.text",
                display_name
            ));
        };
        return Ok(text);
    }

    let Some(error) = result.error else {
        return Err(format!(
            "tool '{}' protocol error: missing error object",
            display_name
        ));
    };
    let Some(message) = error.message else {
        return Err(format!(
            "tool '{}' protocol error: missing error.message",
            display_name
        ));
    };
    let _ = error.code;
    Err(message)
}

fn write_json_file(path: &StdPath, value: &impl Serialize) -> Result<(), SError> {
    let path_display = path.to_string_lossy();
    let payload = serde_json::to_vec_pretty(value).map_err(|err| {
        sid_agent_error("json_serialize_error", "failed to serialize JSON file")
            .with_string_field("path", path_display.as_ref())
            .with_string_field("cause", &err.to_string())
    })?;
    fs::write(path, payload).map_err(|err| {
        sid_agent_error("io_error", "failed to write JSON file")
            .with_string_field("path", path_display.as_ref())
            .with_string_field("cause", &err.to_string())
    })
}

fn read_tool_result(path: &StdPath) -> Result<ToolResultEnvelope, SError> {
    let path_display = path.to_string_lossy();
    let payload = fs::read_to_string(path).map_err(|err| {
        sid_agent_error("io_error", "failed to read tool result file")
            .with_string_field("path", path_display.as_ref())
            .with_string_field("cause", &err.to_string())
    })?;
    serde_json::from_str(&payload).map_err(|err| {
        sid_agent_error(
            "invalid_tool_result_json",
            "failed to parse tool result file",
        )
        .with_string_field("path", path_display.as_ref())
        .with_string_field("cause", &err.to_string())
    })
}

fn next_request_id() -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("sidreq_{timestamp}_{}_{}", std::process::id(), sequence)
}

fn synthetic_tool_use_id(name: &str) -> String {
    format!("toolu_builtin_{name}_{}", next_request_id())
}

fn create_tool_scratch_dir(request_id: &str) -> Result<PathBuf, SError> {
    let parent = std::env::temp_dir().join("sid-tool");
    fs::create_dir_all(&parent).map_err(|err| {
        sid_agent_error("io_error", "failed to create tool scratch root")
            .with_string_field("path", parent.to_string_lossy().as_ref())
            .with_string_field("request_id", request_id)
            .with_string_field("cause", &err.to_string())
    })?;
    let scratch_dir = parent.join(request_id);
    fs::create_dir(&scratch_dir).map_err(|err| {
        sid_agent_error("io_error", "failed to create tool scratch directory")
            .with_string_field("path", scratch_dir.to_string_lossy().as_ref())
            .with_string_field("request_id", request_id)
            .with_string_field("cause", &err.to_string())
    })?;
    Ok(scratch_dir)
}

fn tool_success_result(tool_use_id: &str, message: String) -> ToolResult {
    ControlFlow::Continue(Ok(
        ToolResultBlock::new(tool_use_id.to_string()).with_string_content(message)
    ))
}

fn tool_error_result(tool_use_id: &str, message: String) -> ToolResult {
    ControlFlow::Continue(Err(ToolResultBlock::new(tool_use_id.to_string())
        .with_string_content(message)
        .with_error(true)))
}

fn workspace_has_config(root: &Path) -> bool {
    root.join(AGENTS_CONF_FILE).is_file() || root.join(TOOLS_CONF_FILE).is_file()
}

fn default_agent_id(config: &Config) -> Result<String, SError> {
    if let Some(agent) = config
        .agents
        .values()
        .find(|agent| agent.enabled == SwitchPosition::Yes)
    {
        return Ok(agent.id.clone());
    }
    if let Some(agent) = config
        .agents
        .values()
        .find(|agent| agent.enabled == SwitchPosition::Manual)
    {
        return Ok(agent.id.clone());
    }

    Err(SError::new("sid-agent")
        .with_code("no_startable_agents")
        .with_message("workspace config defines no runnable agents"))
}

fn build_tools(config: &Config, agent_config: &AgentConfig) -> Result<BuiltTools, SError> {
    let mut tools = Vec::new();
    let mut seen = BTreeSet::new();
    let mut builtin_bindings = BuiltinToolBindings::default();
    for tool_name in &agent_config.tools {
        let tool_config = config.tools.get(tool_name).ok_or_else(|| {
            SError::new("sid-agent")
                .with_code("unknown_tool")
                .with_message("agent references an undefined tool")
                .with_string_field("agent", &agent_config.id)
                .with_string_field("tool", tool_name)
        })?;
        if !tool_config.enabled.can_be_started() {
            return Err(disabled_tool_error(
                &agent_config.id,
                tool_name,
                tool_config.enabled,
            ));
        }
        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, tool_name)?;
        let Some(builtin_kind) = BuiltinToolKind::from_canonical_id(&canonical_id) else {
            let tool = ExternalTool::from_config(
                exposed_tool_name(&agent_config.id, tool_name)?,
                canonical_id,
                tool_config,
            );
            if seen.insert(tool.name.clone()) {
                tools.push(Arc::new(tool) as Arc<dyn Tool<SidAgent>>);
            }
            continue;
        };
        let tool = builtin_kind.tool();
        let exposed_name = tool.name();
        if seen.insert(exposed_name) {
            builtin_kind.bind(
                &mut builtin_bindings,
                RcToolBinding {
                    service_name: tool_name.clone(),
                    canonical_id,
                    executable_path: tool_config.executable_path.clone(),
                },
            );
            tools.push(tool);
        }
    }
    Ok(BuiltTools {
        tools,
        builtin_bindings,
    })
}

fn exposed_tool_name(agent: &str, tool_name: &str) -> Result<String, SError> {
    if is_valid_anthropic_tool_name(tool_name) {
        Ok(tool_name.to_string())
    } else {
        Err(SError::new("sid-agent")
            .with_code("invalid_tool_name")
            .with_message("tool name exposed to the model is not legal")
            .with_string_field("agent", agent)
            .with_string_field("tool", tool_name)
            .with_string_field("name", tool_name))
    }
}

fn merged_chat_config(agent_config: &AgentConfig, fallback: Option<&ChatConfig>) -> ChatConfig {
    let mut merged = fallback.cloned().unwrap_or_else(ChatConfig::new);
    let defaults = ChatConfig::new();
    let agent = &agent_config.chat_config;

    if agent.template.model != defaults.template.model {
        merged.template.model = agent.template.model.clone();
    }
    if agent.template.system.is_some() {
        merged.template.system = agent.template.system.clone();
    }
    if agent.template.max_tokens != defaults.template.max_tokens {
        merged.template.max_tokens = agent.template.max_tokens;
    }
    if agent.template.temperature != defaults.template.temperature {
        merged.template.temperature = agent.template.temperature;
    }
    if agent.template.top_p != defaults.template.top_p {
        merged.template.top_p = agent.template.top_p;
    }
    if agent.template.top_k != defaults.template.top_k {
        merged.template.top_k = agent.template.top_k;
    }
    if agent.template.stop_sequences != defaults.template.stop_sequences {
        merged.template.stop_sequences = agent.template.stop_sequences.clone();
    }
    if agent.template.thinking != defaults.template.thinking {
        merged.template.thinking = agent.template.thinking;
    }
    if agent.use_color != defaults.use_color {
        merged.use_color = agent.use_color;
    }
    if agent.session_budget.is_some() {
        merged.session_budget = agent.session_budget.clone();
    }
    if agent.transcript_path.is_some() {
        merged.transcript_path = agent.transcript_path.clone();
    }
    if agent.caching_enabled != defaults.caching_enabled {
        merged.caching_enabled = agent.caching_enabled;
    }

    merged
}

fn missing_agent_error(agent: &str) -> SError {
    SError::new("sid-agent")
        .with_code("unknown_agent")
        .with_message("requested agent is not defined")
        .with_string_field("agent", agent)
}

fn disabled_agent_error(agent: &str, enabled: SwitchPosition) -> SError {
    SError::new("sid-agent")
        .with_code("disabled_agent")
        .with_message("requested agent is disabled")
        .with_string_field("agent", agent)
        .with_string_field("enabled", &format!("{enabled:?}"))
}

fn disabled_tool_error(agent: &str, tool: &str, enabled: SwitchPosition) -> SError {
    SError::new("sid-agent")
        .with_code("disabled_tool")
        .with_message("agent references a disabled tool")
        .with_string_field("agent", agent)
        .with_string_field("tool", tool)
        .with_string_field("enabled", &format!("{enabled:?}"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use claudius::{
        KnownModel, ToolBash20250124, ToolResultBlockContent, ToolTextEditor20250728,
        ToolUnionParam,
    };
    use serde_json::json;

    use super::*;
    use crate::test_support::{
        make_executable, temp_config_root, unique_temp_dir, write_default_tool_manifest,
    };

    #[test]
    fn from_config_uses_agent_prompt_and_tools() {
        let root = temp_config_root("agent");
        write_sample_config(&root);

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        assert_eq!(agent.id(), "build");
        assert_eq!(agent.stream_label(), "build".to_string());
        assert_eq!(
            agent.config.system_prompt_text(),
            Some("# Build\n\nYou are an expert builder.\n")
        );

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tool_names = runtime
            .block_on(agent.tools())
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>();
        assert_eq!(tool_names, vec!["format".to_string(), "shell".to_string()]);
        assert!(!agent.requires_confirmation());

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_workspace_prefers_workspace_agent_and_fills_from_fallback() {
        let root = temp_config_root("agent");
        write_sample_config(&root);

        let fallback = ChatConfig::new()
            .with_model(Model::Known(KnownModel::ClaudeHaiku45))
            .with_system_prompt("fallback system".to_string())
            .with_max_tokens(2048);
        let agent = SidAgent::from_workspace(&root, fallback).unwrap();

        assert_eq!(agent.id(), "build");
        assert_eq!(
            agent.config.system_prompt_text(),
            Some("# Build\n\nYou are an expert builder.\n")
        );
        assert_eq!(
            agent.config.model(),
            Model::Known(KnownModel::ClaudeHaiku45)
        );
        assert_eq!(agent.config.max_tokens(), 2048);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tool_names = runtime
            .block_on(agent.tools())
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>();
        assert_eq!(tool_names, vec!["format".to_string(), "shell".to_string()]);
        assert!(!agent.requires_confirmation());

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_workspace_with_config_root_loads_config_from_sid_home() {
        let config_root = temp_config_root("config-root");
        write_sample_config_with_fmt_script(
            &config_root,
            &capturing_tool_script(
                "format via sid_home",
                "sid-home-request-capture.json",
                "sid-home-env-capture.json",
            ),
        );
        let workspace_root = unique_temp_dir("workspace-root");
        fs::create_dir_all(workspace_root.as_str()).unwrap();

        let fallback = ChatConfig::new()
            .with_system_prompt("fallback system".to_string())
            .with_max_tokens(2048);
        let agent =
            SidAgent::from_workspace_with_config_root(&workspace_root, &config_root, fallback)
                .unwrap();
        assert_eq!(agent.id(), "build");
        assert_eq!(
            agent.config.system_prompt_text(),
            Some("# Build\n\nYou are an expert builder.\n")
        );
        assert_eq!(agent.config.max_tokens(), 2048);

        let config = Config::load(&config_root).unwrap();
        let tool_config = config.tools.get("format").unwrap();
        let exposed_name = exposed_tool_name("build", "format").unwrap();
        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, "format").unwrap();
        let tool = ExternalTool::from_config(exposed_name.clone(), canonical_id, tool_config);
        let tool_use = ToolUseBlock::new(
            "toolu_sid_home_123",
            exposed_name,
            json!({ "paths": ["src/lib.rs"] }),
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(invoke_external_tool(&tool, &agent, &tool_use));
        assert_eq!(unwrap_success_text(result), "format via sid_home");

        let request: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(
                workspace_root
                    .join("sid-home-request-capture.json")
                    .as_str(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(request["workspace"]["root"], json!(workspace_root.as_str()));
        assert_eq!(request["workspace"]["cwd"], json!(workspace_root.as_str()));

        let env: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(workspace_root.join("sid-home-env-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(env["workspace_root"], json!(workspace_root.as_str()));
        assert_eq!(env["rc_d_path"], json!(config_root.join("tools").as_str()));
        assert!(
            env["rc_conf_path"]
                .as_str()
                .unwrap()
                .starts_with(config_root.join("tools.conf").as_str())
        );

        fs::remove_dir_all(config_root.as_str()).unwrap();
        fs::remove_dir_all(workspace_root.as_str()).unwrap();
    }

    #[test]
    fn from_config_exposes_builtin_bash_and_edit_tools() {
        let root = temp_config_root("agent");
        write_builtin_config(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tool_names = runtime
            .block_on(agent.tools())
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>();
        assert_eq!(
            tool_names,
            vec![
                "bash".to_string(),
                "str_replace_based_edit_tool".to_string(),
            ]
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn builtin_tools_without_manifests_use_claudius_union_params() {
        let root = temp_config_root("agent");
        write_builtin_config_without_manifests(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tools = runtime.block_on(agent.tools());
        let params = tools
            .iter()
            .map(|tool| (tool.name(), tool.to_param()))
            .collect::<Vec<_>>();
        assert_eq!(
            params,
            vec![
                (
                    "bash".to_string(),
                    ToolUnionParam::Bash20250124(ToolBash20250124::new()),
                ),
                (
                    "str_replace_based_edit_tool".to_string(),
                    ToolUnionParam::TextEditor20250728(ToolTextEditor20250728::new()),
                ),
            ]
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_workspace_falls_back_without_config_files() {
        let root = unique_temp_dir("agent");
        fs::create_dir_all(root.as_str()).unwrap();

        let fallback = ChatConfig::new()
            .with_system_prompt("fallback system".to_string())
            .with_max_tokens(2048);
        let agent = SidAgent::from_workspace(&root, fallback).unwrap();

        assert_eq!(agent.id(), DEFAULT_AGENT_ID);
        assert_eq!(agent.config.system_prompt_text(), Some("fallback system"));
        assert_eq!(agent.config.max_tokens(), 2048);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        assert!(runtime.block_on(agent.tools()).is_empty());
        assert!(!agent.requires_confirmation());

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn manual_agents_require_confirmation() {
        let root = temp_config_root("agent");
        write_sample_config(&root);

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "plan", root.clone()).unwrap();

        assert_eq!(agent.id(), "plan");
        assert!(agent.requires_confirmation());

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_rejects_unknown_tools() {
        let root = unique_temp_dir("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='missing'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=YES\n").unwrap();
        write_tool_contract(
            &root,
            "fmt",
            "Format files in the workspace.",
            "#!/bin/sh\nexit 0\n",
        );
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let err = SidAgent::from_config(&config, "build", root.clone())
            .err()
            .expect("unknown tools should fail")
            .to_string();
        assert!(err.contains("unknown_tool"));
        assert!(err.contains("missing"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_rejects_invalid_tool_names() {
        let root = unique_temp_dir("agent");
        let invalid_name = "a".repeat(65);
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            format!("build_ENABLED=YES\nbuild_TOOLS='{invalid_name}'\n"),
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            format!("fmt_ENABLED=YES\n{invalid_name}_ALIASES=fmt\n"),
        )
        .unwrap();
        write_tool_contract(
            &root,
            "fmt",
            "Format files in the workspace.",
            "#!/bin/sh\nexit 0\n",
        );
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let err = SidAgent::from_config(&config, "build", root.clone())
            .err()
            .expect("invalid tool names should fail")
            .to_string();
        assert!(err.contains("invalid_tool_name"));
        assert!(err.contains(&invalid_name));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn builtin_bash_uses_rc_tool_runtime() {
        let root = temp_config_root("agent");
        write_builtin_config(
            &root,
            &capturing_tool_script(
                "bash via rc",
                "bash-request-capture.json",
                "bash-env-capture.json",
            ),
            "#!/bin/sh\nexit 0\n",
        );

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(agent.bash("printf hi", true)).unwrap();
        assert_eq!(result, "bash via rc");

        let request: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("bash-request-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(request["tool"]["id"], json!("bash"));
        assert_eq!(
            request["invocation"]["input"],
            json!({ "command": "printf hi", "restart": true })
        );

        let env: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("bash-env-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(env["tool_id"], json!("bash"));
        assert_eq!(env["tool_name"], json!("bash"));
        assert_eq!(env["workspace_root"], json!(root.as_str()));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn builtin_text_editor_uses_rc_tool_runtime() {
        let root = temp_config_root("agent");
        write_builtin_config(
            &root,
            "#!/bin/sh\nexit 0\n",
            &capturing_tool_script(
                "edit via rc",
                "edit-request-capture.json",
                "edit-env-capture.json",
            ),
        );

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tool_use = ToolUseBlock::new(
            "toolu_edit_123",
            "str_replace_based_edit_tool",
            json!({
                "command": "str_replace",
                "path": "src/lib.rs",
                "old_str": "old",
                "new_str": "new"
            }),
        );
        let result = runtime.block_on(agent.text_editor(tool_use)).unwrap();
        assert_eq!(result, "edit via rc");

        let request: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("edit-request-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(request["tool"]["id"], json!("edit"));
        assert_eq!(
            request["invocation"]["tool_use_id"],
            json!("toolu_edit_123")
        );
        assert_eq!(
            request["invocation"]["input"],
            json!({
                "command": "str_replace",
                "path": "src/lib.rs",
                "old_str": "old",
                "new_str": "new"
            })
        );

        let env: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("edit-env-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(env["tool_id"], json!("edit"));
        assert_eq!(env["tool_name"], json!("edit"));
        assert_eq!(env["workspace_root"], json!(root.as_str()));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_writes_request_and_returns_success() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            &success_tool_script("Formatted 3 files.", true),
        );

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        assert_eq!(unwrap_success_text(result), "Formatted 3 files.");

        let request: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("request-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(request["protocol_version"], json!(TOOL_PROTOCOL_VERSION));
        assert_eq!(request["tool"]["id"], json!("fmt"));
        assert_eq!(request["invocation"]["tool_use_id"], json!("toolu_123"));
        assert_eq!(
            request["invocation"]["input"],
            json!({ "paths": ["src/lib.rs"] })
        );
        assert_eq!(request["agent"]["id"], json!("build"));
        assert_eq!(request["workspace"]["root"], json!(root.as_str()));
        assert_eq!(request["workspace"]["cwd"], json!(root.as_str()));
        let scratch_dir = request["files"]["scratch_dir"].as_str().unwrap();
        let result_file = request["files"]["result_file"].as_str().unwrap();
        assert!(scratch_dir.starts_with('/'));
        assert!(result_file.starts_with(scratch_dir));
        assert!(result_file.ends_with("/result.json"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_exposes_execution_env_and_ignores_terminal_output() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            &environment_capturing_tool_script("tool result only"),
        );

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        assert_eq!(unwrap_success_text(result), "tool result only");

        let env: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("env-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(env["protocol"], json!(TOOL_PROTOCOL_VERSION.to_string()));
        assert_eq!(env["workspace_root"], json!(root.as_str()));
        assert_eq!(env["agent_id"], json!("build"));
        assert_eq!(env["tool_id"], json!("fmt"));
        assert_eq!(env["tool_name"], json!("format"));
        assert_eq!(env["rc_d_path"], json!(root.join("tools").as_str()));
        let scratch_dir = env["scratch_dir"].as_str().unwrap();
        assert!(scratch_dir.starts_with('/'));
        assert!(
            env["request_file"]
                .as_str()
                .unwrap()
                .ends_with("/request.json")
        );
        assert!(
            env["result_file"]
                .as_str()
                .unwrap()
                .ends_with("/result.json")
        );
        let overlay_path = env["rc_conf_path"]
            .as_str()
            .unwrap()
            .rsplit(':')
            .next()
            .unwrap();
        let overlay = fs::read_to_string(overlay_path).unwrap();
        assert!(overlay.contains("format_REQUEST_FILE="));
        assert!(overlay.contains("fmt_REQUEST_FILE="));
        assert!(overlay.contains("format_TOOL_ID=fmt"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_handled_model_visible_error() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            &handled_error_tool_script("paths must not be empty"),
        );

        let result = invoke_configured_tool(&root, "format", json!({ "paths": [] }));
        assert_eq!(unwrap_error_text(result), "paths must not be empty");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_process_error_when_tool_exits_nonzero() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &nonzero_exit_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("exited with status"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_result_protocol_version_is_unsupported() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &unsupported_protocol_version_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("unsupported result protocol version"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_output_kind_is_unsupported() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &unsupported_output_kind_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("unsupported output kind"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_success_output_is_missing() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &missing_success_output_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("missing success output"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_output_text_is_missing() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &missing_output_text_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("missing output.text"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_result_is_missing() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, "#!/bin/sh\nexit 0\n");

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("protocol error"));
        assert!(error.contains("failed to read tool result file"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_result_is_malformed() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            "#!/bin/sh\nprintf 'not json' >\"$RESULT_FILE\"\n",
        );

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("protocol error"));
        assert!(error.contains("failed to parse tool result file"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_error_object_is_missing() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &missing_error_object_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": [] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("missing error object"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_error_message_is_missing() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &missing_error_message_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": [] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("missing error.message"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_request_id_mismatches() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &mismatched_request_id_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("request_id mismatch"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_input_is_not_an_object() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            &success_tool_script("Formatted 3 files.", false),
        );

        let result = invoke_configured_tool(&root, "format", json!(["src/lib.rs"]));
        let error = unwrap_error_text(result);
        assert!(error.contains("tool input must be a JSON object"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    fn write_sample_config(root: &Path) {
        write_sample_config_with_fmt_script(root, "#!/bin/sh\nexit 0\n");
    }

    fn write_builtin_config(root: &Path, bash_script: &str, edit_script: &str) {
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            r#"
build_ENABLED="YES"
build_TOOLS='bash edit'
"#,
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            r#"
bash_ENABLED="YES"
edit_ENABLED="YES"
"#,
        )
        .unwrap();
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        write_tool_contract(root, "bash", "Run a shell command.", bash_script);
        write_tool_contract(root, "edit", "Edit workspace files.", edit_script);
    }

    fn write_builtin_config_without_manifests(root: &Path, bash_script: &str, edit_script: &str) {
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            r#"
build_ENABLED="YES"
build_TOOLS='bash edit'
"#,
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            r#"
bash_ENABLED="YES"
edit_ENABLED="YES"
"#,
        )
        .unwrap();
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        write_tool_runtime(root, "bash", bash_script);
        write_tool_runtime(root, "edit", edit_script);
    }

    fn write_sample_config_with_fmt_script(root: &Path, fmt_script: &str) {
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            r#"
ROLE='principal engineer'
build_ENABLED="YES"
plan_ENABLED="MANUAL"
evil_ENABLED="NO"

build_NAME="Let's go ${ROLE}"
build_DESC="buildit"
build_TOOLS='format shell'

plan_MODEL=claude-sonnet-4-5
plan_SYSTEM="You are ${ROLE}"
plan_MAX_TOKENS=8192
"#,
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            r#"
fmt_ENABLED="YES"
shell_ENABLED="YES"

format_INHERIT="YES"
format_ALIASES="fmt"
"#,
        )
        .unwrap();
        fs::write(
            root.join("agents/build.md").as_str(),
            "# Build\n\nYou are an expert builder.\n",
        )
        .unwrap();
        write_tool_contract(root, "fmt", "Format files in the workspace.", fmt_script);
        write_tool_contract(root, "shell", "Run a shell command.", "#!/bin/sh\nexit 0\n");
    }

    fn invoke_configured_tool(
        root: &Path,
        tool_name: &str,
        input: serde_json::Value,
    ) -> ToolResult {
        let config = Config::load(root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone().into_owned()).unwrap();
        let tool_config = config.tools.get(tool_name).unwrap();
        let exposed_name = exposed_tool_name("build", tool_name).unwrap();
        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, tool_name).unwrap();
        let tool = ExternalTool::from_config(exposed_name.clone(), canonical_id, tool_config);
        let tool_use = ToolUseBlock::new("toolu_123", exposed_name, input);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(invoke_external_tool(&tool, &agent, &tool_use))
    }

    fn unwrap_success_text(result: ToolResult) -> String {
        match result {
            ControlFlow::Continue(Ok(block)) => tool_block_text(block),
            other => panic!("expected successful tool result, got {other:?}"),
        }
    }

    fn unwrap_error_text(result: ToolResult) -> String {
        match result {
            ControlFlow::Continue(Err(block)) => {
                assert_eq!(block.is_error, Some(true));
                tool_block_text(block)
            }
            other => panic!("expected errored tool result, got {other:?}"),
        }
    }

    fn tool_block_text(block: ToolResultBlock) -> String {
        match block.content.unwrap() {
            ToolResultBlockContent::String(text) => text,
            other => panic!("expected string tool result content, got {other:?}"),
        }
    }

    fn success_tool_script(text: &str, capture_request: bool) -> String {
        let capture = if capture_request {
            "cp \"$REQUEST_FILE\" \"$WORKSPACE_ROOT/request-capture.json\"\n"
        } else {
            ""
        };
        format!(
            "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\n{capture}cat >\"$RESULT_FILE\" <<EOF\n{{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{{\"kind\":\"text\",\"text\":\"{text}\"}}}}\nEOF\n"
        )
    }

    fn handled_error_tool_script(message: &str) -> String {
        format!(
            "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":false,\"error\":{{\"code\":\"invalid_input\",\"message\":\"{message}\"}}}}\nEOF\n"
        )
    }

    fn environment_capturing_tool_script(text: &str) -> String {
        format!(
            "#!/bin/sh\nprintf 'stdout from tool\\n'\nprintf 'stderr from tool\\n' >&2\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$WORKSPACE_ROOT/env-capture.json\" <<EOF\n{{\"protocol\":\"$TOOL_PROTOCOL\",\"request_file\":\"$REQUEST_FILE\",\"result_file\":\"$RESULT_FILE\",\"scratch_dir\":\"$SCRATCH_DIR\",\"workspace_root\":\"$WORKSPACE_ROOT\",\"agent_id\":\"$AGENT_ID\",\"tool_id\":\"$TOOL_ID\",\"tool_name\":\"$TOOL_NAME\",\"rc_conf_path\":\"$RC_CONF_PATH\",\"rc_d_path\":\"$RC_D_PATH\"}}\nEOF\ncat >\"$RESULT_FILE\" <<EOF\n{{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{{\"kind\":\"text\",\"text\":\"{text}\"}}}}\nEOF\n"
        )
    }

    fn capturing_tool_script(text: &str, request_capture: &str, env_capture: &str) -> String {
        format!(
            "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncp \"$REQUEST_FILE\" \"$WORKSPACE_ROOT/{request_capture}\"\ncat >\"$WORKSPACE_ROOT/{env_capture}\" <<EOF\n{{\"protocol\":\"$TOOL_PROTOCOL\",\"request_file\":\"$REQUEST_FILE\",\"result_file\":\"$RESULT_FILE\",\"scratch_dir\":\"$SCRATCH_DIR\",\"workspace_root\":\"$WORKSPACE_ROOT\",\"agent_id\":\"$AGENT_ID\",\"tool_id\":\"$TOOL_ID\",\"tool_name\":\"$TOOL_NAME\",\"rc_conf_path\":\"$RC_CONF_PATH\",\"rc_d_path\":\"$RC_D_PATH\"}}\nEOF\ncat >\"$RESULT_FILE\" <<EOF\n{{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{{\"kind\":\"text\",\"text\":\"{text}\"}}}}\nEOF\n"
        )
    }

    fn nonzero_exit_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{\"kind\":\"text\",\"text\":\"ignored\"}}\nEOF\nexit 9\n".to_string()
    }

    fn mismatched_request_id_tool_script() -> String {
        "#!/bin/sh\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"wrong-request\",\"ok\":true,\"output\":{\"kind\":\"text\",\"text\":\"ignored\"}}\nEOF\n".to_string()
    }

    fn unsupported_protocol_version_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":2,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{\"kind\":\"text\",\"text\":\"ignored\"}}\nEOF\n".to_string()
    }

    fn unsupported_output_kind_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{\"kind\":\"json\",\"text\":\"ignored\"}}\nEOF\n".to_string()
    }

    fn missing_success_output_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true}\nEOF\n".to_string()
    }

    fn missing_output_text_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{\"kind\":\"text\"}}\nEOF\n".to_string()
    }

    fn missing_error_object_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":false}\nEOF\n".to_string()
    }

    fn missing_error_message_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":false,\"error\":{\"code\":\"invalid_input\"}}\nEOF\n".to_string()
    }

    fn write_tool_contract(root: &Path, tool: &str, description: &str, body: &str) {
        write_tool_runtime(root, tool, body);
        write_default_tool_manifest(root, tool, description);
    }

    fn write_tool_runtime(root: &Path, tool: &str, body: &str) {
        fs::create_dir_all(root.join(TOOLS_DIR).as_str()).unwrap();
        let executable = root.join(format!("{TOOLS_DIR}/{tool}")).into_owned();
        let implementation = root.join(format!("{TOOLS_DIR}/{tool}.impl")).into_owned();
        fs::write(implementation.as_str(), body).unwrap();
        make_executable(&implementation);
        fs::write(
            executable.as_str(),
            format!(
                "#!/bin/sh\nset -eu\n\nlookup() {{\n    printenv \"$1\"\n}}\n\nPREFIX=${{RCVAR_ARGV0:?missing RCVAR_ARGV0}}\n\ncase \"${{1:-}}\" in\nrcvar)\n    printf '%s\\n' \\\n        \"${{PREFIX}}_REQUEST_FILE\" \\\n        \"${{PREFIX}}_RESULT_FILE\" \\\n        \"${{PREFIX}}_SCRATCH_DIR\" \\\n        \"${{PREFIX}}_WORKSPACE_ROOT\" \\\n        \"${{PREFIX}}_AGENT_ID\" \\\n        \"${{PREFIX}}_TOOL_ID\" \\\n        \"${{PREFIX}}_TOOL_NAME\" \\\n        \"${{PREFIX}}_TOOL_PROTOCOL\" \\\n        \"${{PREFIX}}_RC_CONF_PATH\" \\\n        \"${{PREFIX}}_RC_D_PATH\"\n    ;;\nrun)\n    shift\n    export REQUEST_FILE=\"$(lookup \"${{PREFIX}}_REQUEST_FILE\")\"\n    export RESULT_FILE=\"$(lookup \"${{PREFIX}}_RESULT_FILE\")\"\n    export SCRATCH_DIR=\"$(lookup \"${{PREFIX}}_SCRATCH_DIR\")\"\n    export WORKSPACE_ROOT=\"$(lookup \"${{PREFIX}}_WORKSPACE_ROOT\")\"\n    export AGENT_ID=\"$(lookup \"${{PREFIX}}_AGENT_ID\")\"\n    export TOOL_ID=\"$(lookup \"${{PREFIX}}_TOOL_ID\")\"\n    export TOOL_NAME=\"$(lookup \"${{PREFIX}}_TOOL_NAME\")\"\n    export TOOL_PROTOCOL=\"$(lookup \"${{PREFIX}}_TOOL_PROTOCOL\")\"\n    export RC_CONF_PATH=\"$(lookup \"${{PREFIX}}_RC_CONF_PATH\")\"\n    export RC_D_PATH=\"$(lookup \"${{PREFIX}}_RC_D_PATH\")\"\n    exec {} \"$@\"\n    ;;\n*)\n    echo \"usage: $0 [rcvar|run]\" >&2\n    exit 129\n    ;;\nesac\n",
                shvar::quote_string(implementation.as_str())
            ),
        )
        .unwrap();
        make_executable(&executable);
    }
}
