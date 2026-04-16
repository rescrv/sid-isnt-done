pub mod builtin_tools;
pub mod config;
mod filesystem;
#[cfg(test)]
pub(crate) mod test_support;
mod tool_protocol;
mod tool_runtime;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;

use claudius::chat::{ChatAgent, ChatConfig};
use claudius::{
    Agent, Anthropic, BashPtyConfig, BashPtyResult, BashPtySession, Error, FileSystem,
    IntermediateToolResult, Model, MountHierarchy, SystemPrompt, ThinkingConfig, Tool,
    ToolBash20250124, ToolCallback, ToolParam, ToolResult, ToolResultBlock, ToolTextEditor20250728,
    ToolUnionParam, ToolUseBlock,
};
use handled::SError;
use rc_conf::SwitchPosition;
use tokio::sync::Mutex;
use utf8path::Path;

use crate::config::{
    AGENTS_CONF_FILE, AgentConfig, Config, SkillConfig, TOOLS_CONF_FILE, ToolConfig,
    is_valid_anthropic_tool_name, resolve_canonical_tool_id,
};
use crate::filesystem::{build_agent_filesystem, build_default_filesystem, resolve_agent_skills};
use crate::tool_runtime::ToolRuntimeContext;
const DEFAULT_AGENT_ID: &str = "sid";

pub struct SidAgent {
    id: String,
    enabled: SwitchPosition,
    config: ChatConfig,
    tools: Vec<Arc<dyn Tool<Self>>>,
    builtin_bindings: BuiltinToolBindings,
    config_root: Path<'static>,
    workspace_root: Path<'static>,
    filesystem: MountHierarchy,
    bash_session: Mutex<Option<BashPtySession>>,
}

impl SidAgent {
    pub fn new(config: ChatConfig, workspace_root: Path<'static>) -> Self {
        Self::new_with_roots(config, workspace_root.clone(), workspace_root)
    }

    fn new_with_roots(
        config: ChatConfig,
        config_root: Path<'static>,
        workspace_root: Path<'static>,
    ) -> Self {
        let filesystem = build_default_filesystem(&workspace_root);
        Self::with_parts(
            DEFAULT_AGENT_ID.to_string(),
            SwitchPosition::Yes,
            config,
            vec![],
            BuiltinToolBindings::default(),
            config_root,
            workspace_root,
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
        workspace_root: Path<'static>,
    ) -> Result<Self, SError> {
        let agent_config = config
            .agents
            .get(agent)
            .ok_or_else(|| missing_agent_error(agent))?;
        if !agent_config.enabled.can_be_started() {
            return Err(disabled_agent_error(agent, agent_config.enabled));
        }

        let built_tools = build_tools(config, agent_config)?;
        let mut chat_config = merged_chat_config(agent_config, fallback);
        let skills = resolve_agent_skills(config, agent_config)?;
        let filesystem = build_agent_filesystem(&workspace_root, config, agent_config)?;
        if !skills.is_empty() {
            append_skill_index_to_system_prompt(&mut chat_config, &skills);
        }
        Ok(Self::with_parts(
            agent.to_string(),
            agent_config.enabled,
            chat_config,
            built_tools.tools,
            built_tools.builtin_bindings,
            config.root.clone(),
            workspace_root,
            filesystem,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn with_parts(
        id: String,
        enabled: SwitchPosition,
        config: ChatConfig,
        tools: Vec<Arc<dyn Tool<Self>>>,
        builtin_bindings: BuiltinToolBindings,
        config_root: Path<'static>,
        workspace_root: Path<'static>,
        filesystem: MountHierarchy,
    ) -> Self {
        Self {
            id,
            enabled,
            config,
            tools,
            builtin_bindings,
            config_root,
            workspace_root,
            filesystem,
            bash_session: Mutex::new(None),
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

        match binding.enabled {
            SwitchPosition::Yes => {}
            SwitchPosition::No => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "bash is disabled",
                ));
            }
            SwitchPosition::Manual => {
                let input_value = serde_json::json!({
                    "command": command,
                    "restart": restart,
                });
                match confirm_manual_tool_call("bash", &input_value) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            "bash call denied by operator",
                        ));
                    }
                    Err(err) => {
                        return Err(std::io::Error::other(err));
                    }
                }
            }
        }

        let mut session = self.bash_session.lock().await;
        if session.is_none() {
            *session = Some(BashPtySession::new(self.bash_pty_config()).await?);
        }

        let result = {
            let session = session
                .as_mut()
                .expect("bash PTY session should be initialized");
            session.run(command, restart).await
        };
        match result {
            Ok(result) => render_bash_pty_result(result),
            Err(err) => {
                *session = None;
                Err(err)
            }
        }
    }

    fn bash_pty_config(&self) -> BashPtyConfig {
        let mut env: BTreeMap<String, String> = std::env::vars().collect();
        env.insert(
            "SID_WORKSPACE_ROOT".to_string(),
            self.workspace_root.as_str().to_string(),
        );
        BashPtyConfig {
            cwd: PathBuf::from(self.workspace_root.as_str()),
            env,
            ..BashPtyConfig::default()
        }
    }

    async fn invoke_rc_tool(
        &self,
        binding: &RcToolBinding,
        tool_use_id: &str,
        input: serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, std::io::Error> {
        let context = ToolRuntimeContext {
            agent_id: &self.id,
            config_root: &self.config_root,
            workspace_root: &self.workspace_root,
        };
        tool_runtime::invoke_rc_tool_text(
            &binding.service_name,
            &binding.service_name,
            &binding.canonical_id,
            &binding.executable_path,
            &context,
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
        match binding.enabled {
            SwitchPosition::Yes => {}
            SwitchPosition::No => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "edit is disabled",
                ));
            }
            SwitchPosition::Manual => match confirm_manual_tool_call("edit", &tool_use.input) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "edit call denied by operator",
                    ));
                }
                Err(err) => {
                    return Err(std::io::Error::other(err));
                }
            },
        }
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

#[derive(Clone, Debug, Default)]
struct BuiltinToolBindings {
    bash: Option<BuiltinBashBinding>,
    edit: Option<RcToolBinding>,
}

#[derive(Clone, Debug)]
struct BuiltinBashBinding {
    enabled: SwitchPosition,
}

#[derive(Clone, Debug)]
struct RcToolBinding {
    service_name: String,
    canonical_id: String,
    enabled: SwitchPosition,
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
}

#[derive(Clone, Debug)]
struct ExternalTool {
    name: String,
    canonical_id: String,
    enabled: SwitchPosition,
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
        let executable_path = tool
            .executable_path
            .clone()
            .expect("external tools require an executable path");
        Self {
            name,
            canonical_id,
            enabled: tool.enabled,
            executable_path,
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

async fn invoke_external_tool(
    tool: &ExternalTool,
    agent: &SidAgent,
    tool_use: &ToolUseBlock,
) -> ToolResult {
    match tool.enabled {
        SwitchPosition::Yes => {}
        SwitchPosition::No => {
            return tool_error_result(&tool_use.id, format!("tool '{}' is disabled", tool.name));
        }
        SwitchPosition::Manual => match confirm_manual_tool_call(&tool.name, &tool_use.input) {
            Ok(true) => {}
            Ok(false) => {
                return tool_error_result(
                    &tool_use.id,
                    format!("tool '{}' call denied by operator", tool.name),
                );
            }
            Err(err) => {
                return tool_error_result(&tool_use.id, err);
            }
        },
    }

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

    let context = ToolRuntimeContext {
        agent_id: &agent.id,
        config_root: &agent.config_root,
        workspace_root: &agent.workspace_root,
    };
    match tool_runtime::invoke_rc_tool_text(
        &tool.name,
        &tool.name,
        &tool.canonical_id,
        &tool.executable_path,
        &context,
        &tool_use.id,
        input,
    )
    .await
    {
        Ok(text) => tool_success_result(&tool_use.id, text),
        Err(message) => tool_error_result(&tool_use.id, message),
    }
}

/// Prompt the operator to confirm a manual tool call via stdin/stdout.
///
/// Returns `Ok(true)` when the operator approves, `Ok(false)` when denied or
/// stdin reaches EOF, and `Err` on I/O failure.
fn confirm_manual_tool_call(tool_name: &str, input: &serde_json::Value) -> Result<bool, String> {
    use std::io::{self, Write};

    let input_display =
        serde_json::to_string_pretty(input).unwrap_or_else(|_| format!("{input:?}"));

    let mut buf = String::new();
    loop {
        print!("Tool '{tool_name}' is MANUAL.\n{input_display}\nAllow this call? [yes/no]: ");
        io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush manual-tool confirmation prompt: {err}"))?;

        buf.clear();
        if io::stdin()
            .read_line(&mut buf)
            .map_err(|err| format!("failed to read manual-tool confirmation input: {err}"))?
            == 0
        {
            println!();
            return Ok(false);
        }

        match parse_tool_confirmation(&buf) {
            Some(answer) => return Ok(answer),
            None => println!("Please answer yes or no."),
        }
    }
}

fn parse_tool_confirmation(input: &str) -> Option<bool> {
    match input.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Some(true),
        "n" | "no" => Some(false),
        _ => None,
    }
}

fn render_bash_pty_result(result: BashPtyResult) -> Result<String, std::io::Error> {
    let mut rendered = result.output;
    if result.status.success() {
        if rendered.is_empty() {
            rendered.push_str("success\n");
        }
        Ok(rendered)
    } else {
        if !rendered.is_empty() && !rendered.ends_with('\n') {
            rendered.push('\n');
        }
        rendered.push_str(&format!("{}\n", result.status));
        Err(std::io::Error::other(rendered))
    }
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

/// Determine the default agent to use when none is explicitly requested.
///
/// Prefers the explicit `DEFAULT_AGENT` setting from agents.conf when present.
/// Falls back to the first enabled agent, then the first manual agent.
fn default_agent_id(config: &Config) -> Result<String, SError> {
    if let Some(default) = config.default_agent.as_ref() {
        let agent_config = config.agents.get(default).ok_or_else(|| {
            SError::new("sid-agent")
                .with_code("invalid_default_agent")
                .with_message("DEFAULT_AGENT names an undefined agent")
                .with_string_field("default_agent", default)
        })?;
        if !agent_config.enabled.can_be_started() {
            return Err(SError::new("sid-agent")
                .with_code("disabled_default_agent")
                .with_message("DEFAULT_AGENT names a disabled agent")
                .with_string_field("default_agent", default)
                .with_string_field("enabled", &format!("{:?}", agent_config.enabled)));
        }
        return Ok(default.clone());
    }

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
            match builtin_kind {
                BuiltinToolKind::Bash if builtin_bindings.bash.is_none() => {
                    builtin_bindings.bash = Some(BuiltinBashBinding {
                        enabled: tool_config.enabled,
                    });
                }
                BuiltinToolKind::Edit if builtin_bindings.edit.is_none() => {
                    builtin_bindings.edit = Some(RcToolBinding {
                        service_name: tool_name.clone(),
                        canonical_id,
                        enabled: tool_config.enabled,
                        executable_path: tool_config
                            .executable_path
                            .clone()
                            .expect("built-in edit tool requires an executable path"),
                    });
                }
                _ => {}
            }
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

/// Append a skill index to the system prompt so the model knows what skills are
/// available and where they are mounted.
fn append_skill_index_to_system_prompt(chat_config: &mut ChatConfig, skills: &[&SkillConfig]) {
    let mut index = String::from("\n\nAvailable skills (mounted read-only under /skills/):\n");
    for skill in skills {
        index.push_str(&format!("  - /skills/{}/SKILL.md\n", skill.id));
    }
    let existing = chat_config.system_prompt_text().unwrap_or("").to_string();
    chat_config.set_system_prompt(Some(format!("{existing}{index}")));
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
    use crate::config::{TOOL_PROTOCOL_VERSION, TOOLS_DIR};
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
    fn parse_tool_confirmation_accepts_yes() {
        assert_eq!(parse_tool_confirmation("yes"), Some(true));
        assert_eq!(parse_tool_confirmation("YES"), Some(true));
        assert_eq!(parse_tool_confirmation("y"), Some(true));
        assert_eq!(parse_tool_confirmation("Y"), Some(true));
        assert_eq!(parse_tool_confirmation("  yes  \n"), Some(true));
    }

    #[test]
    fn parse_tool_confirmation_accepts_no() {
        assert_eq!(parse_tool_confirmation("no"), Some(false));
        assert_eq!(parse_tool_confirmation("NO"), Some(false));
        assert_eq!(parse_tool_confirmation("n"), Some(false));
        assert_eq!(parse_tool_confirmation("N"), Some(false));
        assert_eq!(parse_tool_confirmation("  no  \n"), Some(false));
    }

    #[test]
    fn parse_tool_confirmation_rejects_other() {
        assert_eq!(parse_tool_confirmation(""), None);
        assert_eq!(parse_tool_confirmation("maybe"), None);
        assert_eq!(parse_tool_confirmation("yep"), None);
        assert_eq!(parse_tool_confirmation("nope"), None);
    }

    #[test]
    fn manual_external_tool_stores_enabled_state() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='fmt'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=MANUAL\n").unwrap();
        write_tool_contract(&root, "fmt", "Format files.", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let tool_config = config.tools.get("fmt").unwrap();
        assert_eq!(tool_config.enabled, SwitchPosition::Manual);

        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, "fmt").unwrap();
        let tool = ExternalTool::from_config("fmt".to_string(), canonical_id, tool_config);
        assert_eq!(tool.enabled, SwitchPosition::Manual);

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn manual_builtin_binding_stores_enabled_state() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash edit'\n",
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            "bash_ENABLED=MANUAL\nedit_ENABLED=MANUAL\n",
        )
        .unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        write_tool_runtime(&root, "edit", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        assert_eq!(
            agent.builtin_bindings.bash.as_ref().unwrap().enabled,
            SwitchPosition::Manual,
        );
        assert_eq!(
            agent.builtin_bindings.edit.as_ref().unwrap().enabled,
            SwitchPosition::Manual,
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_mounts_skills_as_read_only_files() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/rust").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/python").as_str()).unwrap();

        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_SKILLS='rust python'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(
            root.join("agents/build.md").as_str(),
            "# Build\n\nYou are a builder.\n",
        )
        .unwrap();
        fs::write(
            root.join("skills/rust/SKILL.md").as_str(),
            "# Rust Skill\n\nWrite safe Rust.\n",
        )
        .unwrap();
        fs::write(
            root.join("skills/python/SKILL.md").as_str(),
            "# Python Skill\n\nWrite clean Python.\n",
        )
        .unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let rust_content = runtime
            .block_on(agent.filesystem.view("/skills/rust/SKILL.md", None))
            .unwrap();
        assert!(
            rust_content.contains("Write safe Rust."),
            "skill content should be viewable: {rust_content}"
        );
        let python_content = runtime
            .block_on(agent.filesystem.view("/skills/python/SKILL.md", None))
            .unwrap();
        assert!(
            python_content.contains("Write clean Python."),
            "skill content should be viewable: {python_content}"
        );

        let replace_err = runtime
            .block_on(agent.filesystem.str_replace(
                "/skills/rust/SKILL.md",
                "Write safe Rust.",
                "Replaced.",
            ))
            .unwrap_err();
        assert_eq!(
            replace_err.kind(),
            std::io::ErrorKind::PermissionDenied,
            "skill files should be read-only: {replace_err}"
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_mounts_all_skills_with_wildcard() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/docs").as_str()).unwrap();

        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_SKILLS='*'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(root.join("skills/docs/SKILL.md").as_str(), "# Docs Skill\n").unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let docs_content = runtime
            .block_on(agent.filesystem.view("/skills/docs/SKILL.md", None))
            .unwrap();
        assert!(
            docs_content.contains("# Docs Skill"),
            "wildcard should mount all skills: {docs_content}"
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_no_skills_mounts_workspace_only() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();

        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let content = runtime
            .block_on(agent.filesystem.view("/agents/build.md", None))
            .unwrap();
        assert!(
            content.contains("# Build"),
            "workspace files should be viewable: {content}"
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_rejects_unknown_skill_references() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/rust").as_str()).unwrap();

        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_SKILLS='rust missing'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(root.join("skills/rust/SKILL.md").as_str(), "# Rust Skill\n").unwrap();

        let config = Config::load(&root).unwrap();
        let err = SidAgent::from_config(&config, "build", root.clone())
            .err()
            .expect("unknown skills should fail")
            .to_string();
        assert!(err.contains("unknown_skill"), "error: {err}");
        assert!(err.contains("missing"), "error: {err}");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn explicit_default_agent_is_preferred() {
        let root = temp_config_root("agent");
        write_sample_config(&root);
        fs::write(
            root.join("agents.conf").as_str(),
            concat!(
                "DEFAULT_AGENT=plan\n",
                "ROLE='principal engineer'\n",
                "build_ENABLED=\"YES\"\n",
                "plan_ENABLED=\"MANUAL\"\n",
                "evil_ENABLED=\"NO\"\n",
                "build_TOOLS='format shell'\n",
                "plan_MODEL=claude-sonnet-4-5\n",
                "plan_SYSTEM=\"You are ${ROLE}\"\n",
                "plan_MAX_TOKENS=8192\n",
            ),
        )
        .unwrap();

        let config = Config::load(&root).unwrap();
        let agent_id = default_agent_id(&config).unwrap();
        assert_eq!(agent_id, "plan");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn invalid_default_agent_is_rejected_at_config_load() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "DEFAULT_AGENT=nonexistent\nbuild_ENABLED=YES\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "").unwrap();
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let err = Config::load(&root)
            .expect_err("invalid DEFAULT_AGENT should fail")
            .to_string();
        assert!(err.contains("invalid_default_agent"), "error: {err}");
        assert!(err.contains("nonexistent"), "error: {err}");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn skills_are_advertised_in_system_prompt() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/rust").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/python").as_str()).unwrap();

        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_SKILLS='rust python'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(root.join("skills/rust/SKILL.md").as_str(), "# Rust Skill\n").unwrap();
        fs::write(
            root.join("skills/python/SKILL.md").as_str(),
            "# Python Skill\n",
        )
        .unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let prompt = agent.config.system_prompt_text().unwrap();
        assert!(
            prompt.contains("/skills/rust/SKILL.md"),
            "system prompt should list rust skill: {prompt}"
        );
        assert!(
            prompt.contains("/skills/python/SKILL.md"),
            "system prompt should list python skill: {prompt}"
        );

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
    fn builtin_bash_persists_shell_state_across_calls() {
        let root = temp_config_root("agent");
        write_builtin_config(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        let canonical_agents = Path::try_from(
            fs::canonicalize(root.join("agents").as_str())
                .expect("canonicalize agents directory should succeed"),
        )
        .expect("canonical agents path should be valid UTF-8")
        .into_owned();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(agent.bash("export FOO=bar\nf() { printf hi; }\ncd agents", true))
            .unwrap();
        let result = runtime
            .block_on(agent.bash("printf '%s:%s:%s' \"$FOO\" \"$(f)\" \"$PWD\"", false))
            .unwrap();
        assert_eq!(
            result.trim_end(),
            format!("bar:hi:{}", canonical_agents.as_str()),
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn builtin_bash_restart_resets_shell_state() {
        let root = temp_config_root("agent");
        write_builtin_config(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        let canonical_root = Path::try_from(
            fs::canonicalize(root.as_str()).expect("canonicalize root directory should succeed"),
        )
        .expect("canonical root path should be valid UTF-8")
        .into_owned();
        let canonical_agents = Path::try_from(
            fs::canonicalize(root.join("agents").as_str())
                .expect("canonicalize agents directory should succeed"),
        )
        .expect("canonical agents path should be valid UTF-8")
        .into_owned();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(agent.bash("export FOO=bar\ncd agents", true))
            .unwrap();
        let persisted = runtime
            .block_on(agent.bash("printf '%s:%s' \"$FOO\" \"$PWD\"", false))
            .unwrap();
        assert_eq!(
            persisted.trim_end(),
            format!("bar:{}", canonical_agents.as_str()),
        );

        let restarted = runtime
            .block_on(agent.bash("printf '%s:%s' \"${FOO-unset}\" \"$PWD\"", true))
            .unwrap();
        assert_eq!(
            restarted.trim_end(),
            format!("unset:{}", canonical_root.as_str())
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn two_agents_with_different_workspace_roots_have_no_crosstalk() {
        let root_a = temp_config_root("agent-a");
        let root_b = temp_config_root("agent-b");
        write_builtin_config(&root_a, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        write_builtin_config(&root_b, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        let canonical_a = Path::try_from(
            fs::canonicalize(root_a.as_str()).expect("canonicalize root_a should succeed"),
        )
        .expect("canonical root_a path should be valid UTF-8")
        .into_owned();
        let canonical_b = Path::try_from(
            fs::canonicalize(root_b.as_str()).expect("canonicalize root_b should succeed"),
        )
        .expect("canonical root_b path should be valid UTF-8")
        .into_owned();

        let config_a = Config::load(&root_a).unwrap();
        let agent_a = SidAgent::from_config(&config_a, "build", root_a.clone()).unwrap();
        let config_b = Config::load(&root_b).unwrap();
        let agent_b = SidAgent::from_config(&config_b, "build", root_b.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let pwd_a = runtime.block_on(agent_a.bash("pwd", true)).unwrap();
        let pwd_b = runtime.block_on(agent_b.bash("pwd", true)).unwrap();

        assert_eq!(pwd_a.trim_end(), canonical_a.as_str());
        assert_eq!(pwd_b.trim_end(), canonical_b.as_str());
        assert_ne!(
            canonical_a.as_str(),
            canonical_b.as_str(),
            "the two workspace roots must be distinct"
        );

        fs::remove_dir_all(root_a.as_str()).unwrap();
        fs::remove_dir_all(root_b.as_str()).unwrap();
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
