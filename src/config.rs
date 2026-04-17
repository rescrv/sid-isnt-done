use std::collections::BTreeMap;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use claudius::chat::ChatConfig;
use claudius::{Model, ThinkingConfig};
use handled::SError;
use rc_conf::{RcConf, SwitchPosition};
use serde::Deserialize;
use shvar::VariableProvider;
use utf8path::Path;

pub const DEFAULT_THINKING_BUDGET: u32 = 1024;
pub const AGENTS_CONF_FILE: &str = "agents.conf";
pub const TOOLS_CONF_FILE: &str = "tools.conf";
pub const AGENTS_DIR: &str = "agents";
pub const TOOLS_DIR: &str = "tools";
pub const SKILLS_DIR: &str = "skills";
pub const SKILL_FILE: &str = "SKILL.md";
pub const TOOL_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug)]
pub struct Config {
    pub root: Path<'static>,
    pub default_agent: Option<String>,
    pub agents: BTreeMap<String, AgentConfig>,
    pub tools: BTreeMap<String, ToolConfig>,
    pub skills: BTreeMap<String, SkillConfig>,
    pub(crate) tools_rc_conf: RcConf,
}

impl Config {
    pub fn load(root: &Path) -> Result<Self, SError> {
        let root = root.clone().into_owned();
        let agents_conf_path = root.join(AGENTS_CONF_FILE);
        let tools_conf_path = root.join(TOOLS_CONF_FILE);

        require_file(&agents_conf_path, AGENTS_CONF_FILE)?;
        require_file(&tools_conf_path, TOOLS_CONF_FILE)?;

        let agents_rc_conf = parse_rc_conf(&agents_conf_path)?;
        let tools_rc_conf = parse_rc_conf(&tools_conf_path)?;
        let agent_names = collect_names_from_rc_conf(&agents_rc_conf)?;
        let tool_names = collect_names_from_rc_conf(&tools_rc_conf)?;
        let skills_dirs = resolve_skills_dirs(&root);
        let skills = load_skills(&skills_dirs)?;
        Self::from_parts(
            root,
            agents_rc_conf,
            &agent_names,
            tools_rc_conf,
            &tool_names,
            skills,
        )
    }

    fn from_parts(
        root: Path<'static>,
        agents_rc_conf: RcConf,
        agent_names: &[String],
        tools_rc_conf: RcConf,
        tool_names: &[String],
        skills: BTreeMap<String, SkillConfig>,
    ) -> Result<Self, SError> {
        let agents_dir = root.join(AGENTS_DIR).into_owned();
        let tools_dir = root.join(TOOLS_DIR).into_owned();

        let default_agent = resolve_default_agent(&agents_rc_conf, agent_names)?;

        let mut agents = BTreeMap::new();
        for agent_name in agent_names {
            let agent = AgentConfig::from_rc_conf(&agents_dir, &agents_rc_conf, agent_name)?;
            agents.insert(agent_name.clone(), agent);
        }

        let tools = resolve_tool_configs(&tools_dir, &tools_rc_conf, tool_names)?;

        Ok(Self {
            root,
            default_agent,
            agents,
            tools,
            skills,
            tools_rc_conf,
        })
    }
}

#[derive(Debug)]
pub struct AgentConfig {
    pub id: String,
    pub enabled: SwitchPosition,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub tools: Vec<String>,
    pub skills: Vec<String>,
    pub prompt_path: Path<'static>,
    pub prompt_markdown: Option<String>,
    pub chat_config: ChatConfig,
}

impl AgentConfig {
    fn from_rc_conf(agents_dir: &Path, rc_conf: &RcConf, agent: &str) -> Result<Self, SError> {
        let provider = rc_conf.variable_provider_for(agent).map_err(|err| {
            SError::new("config")
                .with_code("rc_conf_error")
                .with_message("failed to derive agent config from rc_conf")
                .with_string_field("agent", agent)
                .with_string_field("cause", &format!("{err:?}"))
        })?;

        let enabled = rc_conf.service_switch(agent);
        let display_name = lookup_expanded(&provider, agent, "NAME")?;
        let description = lookup_expanded(&provider, agent, "DESC")?;
        let tools = lookup_split_field(&provider, agent, "TOOLS")?;
        let skills = lookup_split_field(&provider, agent, "SKILLS")?;

        let prompt_path = resolve_agent_prompt_path(agents_dir, rc_conf, agent);
        let prompt_markdown = if prompt_path.exists() {
            Some(read_utf8_file(&prompt_path, agent, "prompt")?)
        } else {
            None
        };

        let mut chat_config = ChatConfig::new();
        if let Some(prompt) = prompt_markdown.as_ref() {
            chat_config.set_system_prompt(Some(prompt.clone()));
        }
        apply_chat_config_overrides(&mut chat_config, &provider, agent)?;

        Ok(Self {
            id: agent.to_string(),
            enabled,
            display_name,
            description,
            tools,
            skills,
            prompt_path,
            prompt_markdown,
            chat_config,
        })
    }
}

#[derive(Debug)]
pub struct ToolConfig {
    pub id: String,
    pub enabled: SwitchPosition,
    pub confirm_preview: bool,
    pub executable_path: Option<Path<'static>>,
    pub manifest_path: Path<'static>,
    pub manifest: Option<ToolManifest>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolManifest {
    pub protocol_version: u32,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Per-skill configuration loaded from a markdown file in the skills directory.
#[derive(Debug)]
pub struct SkillConfig {
    /// Skill identifier derived from the filename without the `.md` extension.
    pub id: String,
    /// Absolute path to the skill markdown file.
    pub path: Path<'static>,
    /// Markdown content of the skill file.
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct ToolManifestFile {
    protocol_version: u32,
    description: String,
    input_schema: serde_json::Value,
}

fn resolve_tool_configs(
    tools_dir: &Path,
    rc_conf: &RcConf,
    tool_names: &[String],
) -> Result<BTreeMap<String, ToolConfig>, SError> {
    let mut resolved_ids = BTreeMap::new();
    for tool in tool_names {
        let canonical_id = resolve_canonical_tool_id(rc_conf, tool)?;
        validate_anthropic_tool_name(&canonical_id)?;
        resolved_ids.insert(tool.clone(), canonical_id);
    }

    let mut canonical_metadata = BTreeMap::new();
    for canonical_id in resolved_ids.values() {
        if canonical_metadata.contains_key(canonical_id) {
            continue;
        }
        let executable_path = if builtin_tool_executable_is_optional(canonical_id) {
            None
        } else {
            let executable_path = tools_dir.join(canonical_id).into_owned();
            require_tool_executable(canonical_id, &executable_path)?;
            Some(executable_path)
        };

        let manifest_path = tools_dir.join(format!("{canonical_id}.json")).into_owned();
        let manifest = if manifest_path.is_file() {
            Some(load_tool_manifest(canonical_id, &manifest_path)?)
        } else if builtin_tool_manifest_is_optional(canonical_id) {
            None
        } else {
            return Err(SError::new("config")
                .with_code("missing_tool_manifest")
                .with_message("required tool manifest does not exist")
                .with_string_field("tool", canonical_id)
                .with_string_field("manifest_path", manifest_path.as_str()));
        };
        canonical_metadata.insert(
            canonical_id.clone(),
            (executable_path, manifest_path, manifest),
        );
    }

    let mut tools = BTreeMap::new();
    for tool_id in tool_names {
        let enabled = resolve_tool_switch(rc_conf, tool_id)?;
        let confirm_preview = resolve_tool_confirm_preview(rc_conf, tool_id)?;
        let canonical_id = resolved_ids
            .get(tool_id)
            .expect("resolved tool id should exist")
            .clone();
        let (executable_path, manifest_path, manifest) = canonical_metadata
            .get(&canonical_id)
            .expect("canonical metadata should exist");
        tools.insert(
            tool_id.clone(),
            ToolConfig {
                id: tool_id.clone(),
                enabled,
                confirm_preview,
                executable_path: executable_path.clone(),
                manifest_path: manifest_path.clone(),
                manifest: manifest.clone(),
            },
        );
    }

    Ok(tools)
}

pub(crate) fn resolve_canonical_tool_id(rc_conf: &RcConf, tool: &str) -> Result<String, SError> {
    let services = collect_names_from_rc_conf(rc_conf)?;
    let canonical_id = rc_conf.resolve_alias(tool).to_string();
    if services.iter().any(|service| service == &canonical_id) {
        Ok(canonical_id)
    } else {
        Err(SError::new("config")
            .with_code("unknown_tool")
            .with_message("tool alias resolves to an undefined tool")
            .with_string_field("tool", tool)
            .with_string_field("alias_target", &canonical_id))
    }
}

fn resolve_tool_switch(rc_conf: &RcConf, tool: &str) -> Result<SwitchPosition, SError> {
    Ok(rc_conf.service_switch(tool))
}

fn resolve_tool_confirm_preview(rc_conf: &RcConf, tool: &str) -> Result<bool, SError> {
    let provider = rc_conf.variable_provider_for(tool).map_err(|err| {
        SError::new("config")
            .with_code("rc_conf_error")
            .with_message("failed to derive tool config from rc_conf")
            .with_string_field("tool", tool)
            .with_string_field("cause", &format!("{err:?}"))
    })?;
    let Some(value) = lookup_expanded(&provider, tool, "CONFIRM")? else {
        return Ok(false);
    };
    parse_bool_field(tool, "CONFIRM", &value)
}

pub(crate) fn is_valid_anthropic_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == b'_' || ch == b'-')
}

fn validate_anthropic_tool_name(name: &str) -> Result<(), SError> {
    if is_valid_anthropic_tool_name(name) {
        Ok(())
    } else {
        Err(SError::new("config")
            .with_code("invalid_tool_id")
            .with_message("tool id is not a legal Anthropic tool name")
            .with_string_field("tool", name)
            .with_string_field("reason", "expected 1-64 ASCII letters, digits, '_' or '-'"))
    }
}

fn require_tool_executable(tool: &str, path: &Path) -> Result<(), SError> {
    require_file_with_code(
        path,
        "missing_tool_executable",
        "required tool executable does not exist",
        "tool",
        tool,
        "path",
    )?;
    #[cfg(unix)]
    {
        let metadata = fs::metadata(path.as_str()).map_err(|err| {
            SError::new("config")
                .with_code("io_error")
                .with_message("failed to inspect tool executable")
                .with_string_field("tool", tool)
                .with_string_field("path", path.as_str())
                .with_string_field("cause", &err.to_string())
        })?;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(SError::new("config")
                .with_code("tool_not_executable")
                .with_message("tool executable is not marked executable")
                .with_string_field("tool", tool)
                .with_string_field("path", path.as_str()));
        }
    }
    Ok(())
}

fn load_tool_manifest(tool: &str, path: &Path) -> Result<ToolManifest, SError> {
    let raw = read_utf8_file(path, tool, "manifest")?;
    let manifest: ToolManifestFile = serde_json::from_str(&raw).map_err(|err| {
        SError::new("config")
            .with_code("invalid_tool_manifest_json")
            .with_message("failed to parse tool manifest")
            .with_string_field("tool", tool)
            .with_string_field("path", path.as_str())
            .with_string_field("cause", &err.to_string())
    })?;
    if manifest.protocol_version != TOOL_PROTOCOL_VERSION {
        return Err(SError::new("config")
            .with_code("unsupported_tool_protocol_version")
            .with_message("tool manifest declares an unsupported protocol version")
            .with_string_field("tool", tool)
            .with_string_field("path", path.as_str())
            .with_string_field("protocol_version", &manifest.protocol_version.to_string()));
    }
    if manifest.description.trim().is_empty() {
        return Err(SError::new("config")
            .with_code("invalid_tool_manifest")
            .with_message("tool manifest description must not be empty")
            .with_string_field("tool", tool)
            .with_string_field("path", path.as_str()));
    }
    if !manifest.input_schema.is_object() {
        return Err(SError::new("config")
            .with_code("invalid_tool_manifest")
            .with_message("tool manifest input_schema must be a JSON object")
            .with_string_field("tool", tool)
            .with_string_field("path", path.as_str()));
    }

    Ok(ToolManifest {
        protocol_version: manifest.protocol_version,
        description: manifest.description,
        input_schema: manifest.input_schema,
    })
}

fn builtin_tool_manifest_is_optional(tool: &str) -> bool {
    matches!(tool, "bash" | "edit")
}

fn builtin_tool_executable_is_optional(tool: &str) -> bool {
    matches!(tool, "bash")
}

fn resolve_skills_dirs(root: &Path) -> Vec<Path<'static>> {
    match std::env::var("SID_SKILLS_PATH") {
        Ok(path) if !path.is_empty() => path
            .split(':')
            .filter(|component| !component.is_empty())
            .map(|component| Path::new(component).into_owned())
            .collect(),
        _ => vec![root.join(SKILLS_DIR).into_owned()],
    }
}

fn load_skills(dirs: &[Path<'static>]) -> Result<BTreeMap<String, SkillConfig>, SError> {
    let mut skills = BTreeMap::new();
    for dir in dirs {
        if !std::path::Path::new(dir.as_str()).is_dir() {
            continue;
        }
        let entries = fs::read_dir(dir.as_str()).map_err(|err| {
            SError::new("config")
                .with_code("io_error")
                .with_message("failed to read skills directory")
                .with_string_field("path", dir.as_str())
                .with_string_field("cause", &err.to_string())
        })?;
        for entry in entries {
            let entry = entry.map_err(|err| {
                SError::new("config")
                    .with_code("io_error")
                    .with_message("failed to read skills directory entry")
                    .with_string_field("path", dir.as_str())
                    .with_string_field("cause", &err.to_string())
            })?;
            let entry_path = entry.path();
            if !entry_path.is_dir() {
                continue;
            }
            let skill_file = entry_path.join(SKILL_FILE);
            if !skill_file.is_file() {
                continue;
            }
            let dir_name = entry.file_name();
            let skill_name = dir_name.to_string_lossy();
            if skill_name.is_empty() {
                continue;
            }
            if skills.contains_key(skill_name.as_ref()) {
                continue;
            }
            let path = Path::try_from(skill_file)
                .map_err(|err| {
                    SError::new("config")
                        .with_code("invalid_skill_path")
                        .with_message("skill path is not valid UTF-8")
                        .with_string_field("cause", &format!("{err:?}"))
                })?
                .into_owned();
            let content = read_utf8_file(&path, &skill_name, "skill")?;
            skills.insert(
                skill_name.to_string(),
                SkillConfig {
                    id: skill_name.to_string(),
                    path,
                    content,
                },
            );
        }
    }
    Ok(skills)
}

fn apply_chat_config_overrides(
    chat_config: &mut ChatConfig,
    provider: &impl VariableProvider,
    agent: &str,
) -> Result<(), SError> {
    if let Some(model) = lookup_expanded(provider, agent, "MODEL")? {
        let model = model
            .parse()
            .unwrap_or_else(|_| Model::Custom(model.clone()));
        chat_config.set_model(model);
    }
    if let Some(system_prompt) = lookup_expanded(provider, agent, "SYSTEM")? {
        chat_config.set_system_prompt(Some(system_prompt));
    }
    if let Some(max_tokens) = lookup_expanded(provider, agent, "MAX_TOKENS")? {
        chat_config.set_max_tokens(parse_u32_field(agent, "MAX_TOKENS", &max_tokens)?);
    }
    if let Some(temperature) = lookup_expanded(provider, agent, "TEMPERATURE")? {
        chat_config.set_temperature(Some(parse_unit_interval_field(
            agent,
            "TEMPERATURE",
            &temperature,
        )?));
    }
    if let Some(top_p) = lookup_expanded(provider, agent, "TOP_P")? {
        chat_config.set_top_p(Some(parse_unit_interval_field(agent, "TOP_P", &top_p)?));
    }
    if let Some(top_k) = lookup_expanded(provider, agent, "TOP_K")? {
        chat_config.set_top_k(Some(parse_u32_field(agent, "TOP_K", &top_k)?));
    }
    if let Some(stop_sequences) = lookup_expanded(provider, agent, "STOP_SEQUENCES")? {
        let stop_sequences = shvar::split(&stop_sequences).map_err(|err| {
            invalid_config_field(agent, "STOP_SEQUENCES", &stop_sequences, format!("{err:?}"))
        })?;
        if stop_sequences.is_empty() {
            chat_config.template.stop_sequences = None;
        } else {
            chat_config.template.stop_sequences = Some(stop_sequences);
        }
    }
    if let Some(thinking) = lookup_expanded(provider, agent, "THINKING")? {
        chat_config.template.thinking =
            parse_thinking_budget(agent, "THINKING", &thinking)?.map(ThinkingConfig::enabled);
    }
    if let Some(use_color) = lookup_expanded(provider, agent, "USE_COLOR")? {
        chat_config.use_color = parse_bool_field(agent, "USE_COLOR", &use_color)?;
    }
    if let Some(no_color) = lookup_expanded(provider, agent, "NO_COLOR")? {
        chat_config.use_color = !parse_bool_field(agent, "NO_COLOR", &no_color)?;
    }
    if let Some(session_budget) = lookup_expanded(provider, agent, "SESSION_BUDGET")? {
        chat_config.set_session_budget(Some(parse_u64_field(
            agent,
            "SESSION_BUDGET",
            &session_budget,
        )?));
    }
    if let Some(transcript_path) = lookup_expanded(provider, agent, "TRANSCRIPT_PATH")? {
        chat_config.transcript_path = Some(transcript_path.into());
    }
    if let Some(caching_enabled) = lookup_expanded(provider, agent, "CACHING_ENABLED")? {
        chat_config.caching_enabled = parse_bool_field(agent, "CACHING_ENABLED", &caching_enabled)?;
    }

    Ok(())
}

/// Read the explicit `DEFAULT_AGENT` global variable from agents.conf when present.
///
/// If the variable is set, its value must name one of the defined agents.
fn resolve_default_agent(
    agents_rc_conf: &RcConf,
    agent_names: &[String],
) -> Result<Option<String>, SError> {
    let Some(value) = agents_rc_conf.lookup("DEFAULT_AGENT") else {
        return Ok(None);
    };
    let value = value.trim().to_string();
    if value.is_empty() {
        return Ok(None);
    }
    if !agent_names.contains(&value) {
        return Err(SError::new("config")
            .with_code("invalid_default_agent")
            .with_message("DEFAULT_AGENT names an undefined agent")
            .with_string_field("default_agent", &value));
    }
    Ok(Some(value))
}

fn collect_names_from_rc_conf(rc_conf: &RcConf) -> Result<Vec<String>, SError> {
    Ok(rc_conf
        .list()
        .map_err(|err| {
            SError::new("config")
                .with_code("rc_conf_error")
                .with_message("failed to list configured names")
                .with_string_field("cause", &format!("{err:?}"))
        })?
        .collect())
}

fn parse_rc_conf(path: &Path) -> Result<RcConf, SError> {
    RcConf::parse(path.as_str()).map_err(|err| {
        SError::new("config")
            .with_code("rc_conf_error")
            .with_message("failed to parse rc_conf file")
            .with_string_field("path", path.as_str())
            .with_string_field("cause", &format!("{err:?}"))
    })
}

fn require_file(path: &Path, label: &str) -> Result<(), SError> {
    require_file_with_code(
        path,
        "missing_config_file",
        "required config file does not exist",
        "file",
        label,
        "path",
    )
}

fn require_file_with_code(
    path: &Path,
    code: &str,
    message: &str,
    name_field: &str,
    name: &str,
    path_field: &str,
) -> Result<(), SError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(SError::new("config")
            .with_code(code)
            .with_message(message)
            .with_string_field(name_field, name)
            .with_string_field(path_field, path.as_str()))
    }
}

fn read_utf8_file(path: &Path, name: &str, field: &str) -> Result<String, SError> {
    fs::read_to_string(path.as_str()).map_err(|err| {
        SError::new("config")
            .with_code("io_error")
            .with_message("failed to read config file")
            .with_string_field("name", name)
            .with_string_field("field", field)
            .with_string_field("path", path.as_str())
            .with_string_field("cause", &err.to_string())
    })
}

fn resolve_agent_prompt_path(agents_dir: &Path, rc_conf: &RcConf, agent: &str) -> Path<'static> {
    for candidate in rc_conf.alias_lookup_order(agent).0 {
        let path = agents_dir.join(format!("{candidate}.md")).into_owned();
        if path.is_file() {
            return path;
        }
    }
    agents_dir.join(format!("{agent}.md")).into_owned()
}

fn lookup_expanded(
    provider: &impl VariableProvider,
    scope: &str,
    key: &str,
) -> Result<Option<String>, SError> {
    let Some(value) = provider.lookup(key) else {
        return Ok(None);
    };
    let expanded = expand_config_value(provider, scope, key, &value)?;
    Ok(Some(expanded))
}

fn lookup_split_field(
    provider: &impl VariableProvider,
    scope: &str,
    key: &str,
) -> Result<Vec<String>, SError> {
    let Some(value) = lookup_expanded(provider, scope, key)? else {
        return Ok(vec![]);
    };
    shvar::split(&value).map_err(|err| invalid_config_field(scope, key, &value, format!("{err:?}")))
}

fn parse_u32_field(scope: &str, field: &str, value: &str) -> Result<u32, SError> {
    value
        .trim()
        .parse::<u32>()
        .map_err(|err| invalid_config_field(scope, field, value, err.to_string()))
}

fn parse_u64_field(scope: &str, field: &str, value: &str) -> Result<u64, SError> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|err| invalid_config_field(scope, field, value, err.to_string()))
}

fn parse_unit_interval_field(scope: &str, field: &str, value: &str) -> Result<f32, SError> {
    let parsed = value
        .trim()
        .parse::<f32>()
        .map_err(|err| invalid_config_field(scope, field, value, err.to_string()))?;
    if parsed.is_finite() && (0.0..=1.0).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(invalid_config_field(
            scope,
            field,
            value,
            "expected a finite value between 0.0 and 1.0",
        ))
    }
}

fn parse_bool_field(scope: &str, field: &str, value: &str) -> Result<bool, SError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" | "enable" | "enabled" => Ok(true),
        "0" | "off" | "false" | "no" | "disable" | "disabled" => Ok(false),
        _ => Err(invalid_config_field(
            scope,
            field,
            value,
            "expected one of yes/no, true/false, on/off, or 1/0",
        )),
    }
}

fn parse_thinking_budget(scope: &str, field: &str, value: &str) -> Result<Option<u32>, SError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "false" | "no" | "disable" | "disabled" => Ok(None),
        "on" | "true" | "yes" | "enable" | "enabled" => Ok(Some(DEFAULT_THINKING_BUDGET)),
        _ => parse_u32_field(scope, field, value).map(Some),
    }
}

fn invalid_config_field(
    scope: &str,
    field: &str,
    value: &str,
    reason: impl Into<String>,
) -> SError {
    let reason = reason.into();
    SError::new("config")
        .with_code("invalid_config_field")
        .with_message("failed to derive config from rc_conf")
        .with_string_field("scope", scope)
        .with_string_field("field", field)
        .with_string_field("value", value)
        .with_string_field("reason", &reason)
}

fn expand_config_value(
    provider: &impl VariableProvider,
    scope: &str,
    field: &str,
    value: &str,
) -> Result<String, SError> {
    let mut current = value.to_string();
    for _ in 0..128 {
        let next = expand_config_value_once(provider, scope, field, &current)?;
        if next == current {
            return Ok(next);
        }
        current = next;
    }
    Err(invalid_config_field(
        scope,
        field,
        value,
        "variable expansion exceeded recursion limit",
    ))
}

fn expand_config_value_once(
    provider: &impl VariableProvider,
    scope: &str,
    field: &str,
    value: &str,
) -> Result<String, SError> {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '$' {
            output.push(ch);
            continue;
        }
        match chars.peek().copied() {
            Some('$') => {
                output.push('$');
                chars.next();
            }
            Some('{') => {
                chars.next();
                let ident = parse_braced_identifier(&mut chars, scope, field, value)?;
                output.push_str(&provider.lookup(&ident).unwrap_or_default());
            }
            Some(next) if is_identifier_start(next) => {
                let ident = parse_identifier(&mut chars);
                output.push_str(&provider.lookup(&ident).unwrap_or_default());
            }
            _ => {
                output.push('$');
            }
        }
    }
    Ok(output)
}

fn parse_braced_identifier(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    scope: &str,
    field: &str,
    value: &str,
) -> Result<String, SError> {
    let Some(first) = chars.next() else {
        return Err(invalid_config_field(
            scope,
            field,
            value,
            "unterminated variable expansion",
        ));
    };
    if !is_identifier_start(first) {
        return Err(invalid_config_field(
            scope,
            field,
            value,
            "invalid variable name in expansion",
        ));
    }

    let mut ident = String::from(first);
    loop {
        match chars.next() {
            Some('}') => return Ok(ident),
            Some(ch) if is_identifier_continue(ch) => ident.push(ch),
            Some(_) => {
                return Err(invalid_config_field(
                    scope,
                    field,
                    value,
                    "invalid variable name in expansion",
                ));
            }
            None => {
                return Err(invalid_config_field(
                    scope,
                    field,
                    value,
                    "unterminated variable expansion",
                ));
            }
        }
    }
}

fn parse_identifier(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut ident = String::new();
    while let Some(ch) = chars.peek().copied() {
        if !is_identifier_continue(ch) {
            break;
        }
        ident.push(ch);
        chars.next();
    }
    ident
}

fn is_identifier_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_identifier_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use claudius::KnownModel;

    use super::*;
    use crate::test_support::{
        unique_temp_dir, write_default_tool_manifest, write_tool_manifest,
        write_tool_manifest_with_schema, write_tool_script,
    };

    #[test]
    fn load_config_from_readme_style_files() {
        let root = unique_temp_dir("config");
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
build_TOOLS='format bash'
build_SKILLS='* "rust docs"'

plan_MODEL=claude-sonnet-4-5
plan_SYSTEM="You are ${ROLE}"
plan_MAX_TOKENS=8192
plan_TEMPERATURE=0.7
plan_TOP_P=0.9
plan_TOP_K=40
plan_STOP_SEQUENCES='END "two words"'
plan_THINKING=on
plan_NO_COLOR=yes
plan_SESSION_BUDGET=50000
plan_TRANSCRIPT_PATH='logs/${ROLE}.json'
plan_CACHING_ENABLED=off
"#,
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            r#"
fmt_ENABLED="YES"
bash_ENABLED="YES"

format_INHERIT="YES"
format_ALIASES="fmt"
"#,
        )
        .unwrap();
        write_tool_contract(&root, "fmt", "Format files in the workspace.");
        write_tool_contract(&root, "bash", "Run a shell command.");
        fs::write(
            root.join("agents/build.md").as_str(),
            "# Build\n\nYou are an expert builder.\n",
        )
        .unwrap();
        fs::write(
            root.join("agents/plan.md").as_str(),
            "# Plan\n\nYou are an expert planner.\n",
        )
        .unwrap();

        let config = Config::load(&root).unwrap();

        assert_eq!(config.agents.len(), 4);
        assert_eq!(config.tools.len(), 3);

        let build = config.agents.get("build").unwrap();
        assert_eq!(build.enabled, SwitchPosition::Yes);
        assert_eq!(
            build.display_name.as_deref(),
            Some("Let's go principal engineer")
        );
        assert_eq!(build.description.as_deref(), Some("buildit"));
        assert_eq!(build.tools, vec!["format".to_string(), "bash".to_string()]);
        assert_eq!(build.skills, vec!["*".to_string(), "rust docs".to_string()]);
        assert_eq!(
            build.prompt_markdown.as_deref(),
            Some("# Build\n\nYou are an expert builder.\n")
        );
        assert_eq!(
            build.chat_config.system_prompt_text(),
            Some("# Build\n\nYou are an expert builder.\n")
        );

        let plan = config.agents.get("plan").unwrap();
        assert_eq!(plan.enabled, SwitchPosition::Manual);
        assert_eq!(
            plan.chat_config.model(),
            Model::Known(KnownModel::ClaudeSonnet45)
        );
        assert_eq!(
            plan.chat_config.system_prompt_text(),
            Some("You are principal engineer")
        );
        assert_eq!(plan.chat_config.max_tokens(), 8192);
        assert_eq!(plan.chat_config.template.temperature, Some(0.7));
        assert_eq!(plan.chat_config.template.top_p, Some(0.9));
        assert_eq!(plan.chat_config.template.top_k, Some(40));
        assert_eq!(
            plan.chat_config.stop_sequences(),
            &["END".to_string(), "two words".to_string()]
        );
        assert_eq!(
            plan.chat_config.thinking_budget(),
            Some(DEFAULT_THINKING_BUDGET)
        );
        assert!(!plan.chat_config.use_color);
        assert!(plan.chat_config.session_budget.is_some());
        assert_eq!(
            plan.chat_config
                .transcript_path
                .as_deref()
                .and_then(|path| Path::try_from(path).ok())
                .map(Path::into_owned),
            Some(Path::new("logs/principal engineer.json").into_owned())
        );
        assert!(!plan.chat_config.caching_enabled);

        let evil = config.agents.get("evil").unwrap();
        assert_eq!(evil.enabled, SwitchPosition::No);
        assert!(evil.prompt_markdown.is_none());
        assert_eq!(evil.prompt_path, root.join("agents/evil.md"));

        let plan_caching = config.agents.get("plan-caching").unwrap();
        assert_eq!(plan_caching.enabled, SwitchPosition::No);
        assert!(plan_caching.prompt_markdown.is_none());
        assert_eq!(
            plan_caching.prompt_path,
            root.join("agents/plan-caching.md")
        );

        let fmt = config.tools.get("fmt").unwrap();
        assert_eq!(fmt.enabled, SwitchPosition::Yes);
        assert!(!fmt.confirm_preview);
        assert_eq!(fmt.executable_path, Some(root.join("tools/fmt")));
        assert_eq!(fmt.manifest_path, root.join("tools/fmt.json"));
        let fmt_manifest = fmt.manifest.as_ref().unwrap();
        assert_eq!(fmt_manifest.protocol_version, TOOL_PROTOCOL_VERSION);
        assert_eq!(fmt_manifest.description, "Format files in the workspace.");

        let format = config.tools.get("format").unwrap();
        assert_eq!(format.enabled, SwitchPosition::Yes);
        assert_eq!(format.executable_path, Some(root.join("tools/fmt")));
        assert_eq!(format.manifest_path, root.join("tools/fmt.json"));
        assert_eq!(
            format.manifest.as_ref().unwrap().description,
            "Format files in the workspace."
        );

        let bash = config.tools.get("bash").unwrap();
        assert_eq!(bash.enabled, SwitchPosition::Yes);
        assert!(!bash.confirm_preview);
        assert!(bash.executable_path.is_none());
    }

    #[test]
    fn tool_confirm_preview_defaults_and_parses_bool() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='fmt plain'\n",
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            "fmt_ENABLED=YES\nfmt_CONFIRM=YES\nplain_ENABLED=YES\n",
        )
        .unwrap();
        write_tool_contract(&root, "fmt", "Format files.");
        write_tool_contract(&root, "plain", "Plain tool.");

        let config = Config::load(&root).unwrap();
        assert!(config.tools.get("fmt").unwrap().confirm_preview);
        assert!(!config.tools.get("plain").unwrap().confirm_preview);

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn invalid_tool_confirm_preview_bool_is_an_error() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='fmt'\n",
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            "fmt_ENABLED=YES\nfmt_CONFIRM=maybe\n",
        )
        .unwrap();
        write_tool_contract(&root, "fmt", "Format files.");

        let err = Config::load(&root)
            .expect_err("invalid CONFIRM value should fail")
            .to_string();
        assert!(err.contains("CONFIRM"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn missing_top_level_config_file_is_an_error() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "build_ENABLED=YES\n").unwrap();

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("missing_config_file"));
        assert!(err.contains("tools.conf"));
    }

    #[test]
    fn invalid_agent_field_is_reported() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOP_P=wat\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "").unwrap();
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("invalid_config_field"));
        assert!(err.contains("TOP_P"));
        assert!(err.contains("wat"));
    }

    #[test]
    fn missing_tool_executable_is_reported_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=YES\n").unwrap();
        write_tool_manifest(&root, "fmt", TOOL_PROTOCOL_VERSION, "Format files.");

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("missing_tool_executable"));
        assert!(err.contains("fmt"));
    }

    #[test]
    fn missing_tool_manifest_is_reported_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=YES\n").unwrap();
        write_tool_script(&root, "fmt", "#!/bin/sh\nexit 0\n");

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("missing_tool_manifest"));
        assert!(err.contains("fmt"));
    }

    #[test]
    fn missing_builtin_tool_manifest_is_allowed_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            "bash_ENABLED=YES\nedit_ENABLED=YES\n",
        )
        .unwrap();
        write_tool_script(&root, "edit", "#!/bin/sh\nexit 0\n");

        let config = Config::load(&root).unwrap();
        let bash = config.tools.get("bash").unwrap();
        assert!(bash.executable_path.is_none());
        assert_eq!(bash.manifest_path, root.join("tools/bash.json"));
        assert!(bash.manifest.is_none());

        let edit = config.tools.get("edit").unwrap();
        assert_eq!(edit.executable_path, Some(root.join("tools/edit")));
        assert_eq!(edit.manifest_path, root.join("tools/edit.json"));
        assert!(edit.manifest.is_none());
    }

    #[test]
    fn missing_builtin_edit_executable_is_reported_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            "bash_ENABLED=YES\nedit_ENABLED=YES\n",
        )
        .unwrap();

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("missing_tool_executable"));
        assert!(err.contains("edit"));
    }

    #[test]
    fn invalid_tool_manifest_json_is_reported_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=YES\n").unwrap();
        write_tool_script(&root, "fmt", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("tools/fmt.json").as_str(), "{ not valid json").unwrap();

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("invalid_tool_manifest_json"));
    }

    #[test]
    fn unsupported_tool_protocol_version_is_reported_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=YES\n").unwrap();
        write_tool_script(&root, "fmt", "#!/bin/sh\nexit 0\n");
        write_tool_manifest(&root, "fmt", 2, "Format files.");

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("unsupported_tool_protocol_version"));
    }

    #[test]
    fn invalid_tool_id_is_reported_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        let long_name = "a".repeat(65);
        fs::write(
            root.join("tools.conf").as_str(),
            format!("{long_name}_ENABLED=YES\n"),
        )
        .unwrap();

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("invalid_tool_id"));
        assert!(err.contains(&long_name));
    }

    #[cfg(unix)]
    #[test]
    fn tool_must_be_executable_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=YES\n").unwrap();
        fs::create_dir_all(root.join("tools").as_str()).unwrap();
        let executable = root.join("tools/fmt").into_owned();
        fs::write(executable.as_str(), "#!/bin/sh\nexit 0\n").unwrap();
        write_tool_manifest(&root, "fmt", TOOL_PROTOCOL_VERSION, "Format files.");

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("tool_not_executable"));
        assert!(err.contains("fmt"));
    }

    #[test]
    fn empty_tool_description_is_reported_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=YES\n").unwrap();
        write_tool_script(&root, "fmt", "#!/bin/sh\nexit 0\n");
        write_tool_manifest(&root, "fmt", TOOL_PROTOCOL_VERSION, "   ");

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("invalid_tool_manifest"));
        assert!(err.contains("description"));
    }

    #[test]
    fn non_object_input_schema_is_reported_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=YES\n").unwrap();
        write_tool_script(&root, "fmt", "#!/bin/sh\nexit 0\n");
        write_tool_manifest_with_schema(
            &root,
            "fmt",
            TOOL_PROTOCOL_VERSION,
            "Format files.",
            serde_json::json!("not an object"),
        );

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("invalid_tool_manifest"));
        assert!(err.contains("input_schema"));
    }

    #[test]
    fn undefined_tool_alias_target_is_reported_during_config_load() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            "format_ALIASES=fmt\nformat_INHERIT=YES\n",
        )
        .unwrap();

        let err = Config::load(&root).unwrap_err().to_string();
        assert!(err.contains("unknown_tool"));
        assert!(err.contains("fmt"));
    }

    #[test]
    fn load_skills_from_skills_directory() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/rust").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/python").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/empty-dir").as_str()).unwrap();

        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(root.join("tools.conf").as_str(), "").unwrap();
        fs::write(
            root.join("skills/rust/SKILL.md").as_str(),
            "# Rust\n\nYou are a Rust expert.\n",
        )
        .unwrap();
        fs::write(
            root.join("skills/python/SKILL.md").as_str(),
            "# Python\n\nYou know Python.\n",
        )
        .unwrap();
        // Bare file in skills/ should be ignored.
        fs::write(root.join("skills/notes.txt").as_str(), "not a skill").unwrap();
        // Subdirectory without SKILL.md should be ignored.

        let config = Config::load(&root).unwrap();
        assert_eq!(config.skills.len(), 2);

        let rust_skill = config.skills.get("rust").unwrap();
        assert_eq!(rust_skill.id, "rust");
        assert_eq!(rust_skill.content, "# Rust\n\nYou are a Rust expert.\n");
        assert_eq!(rust_skill.path, root.join("skills/rust/SKILL.md"));

        let python_skill = config.skills.get("python").unwrap();
        assert_eq!(python_skill.id, "python");
        assert_eq!(python_skill.content, "# Python\n\nYou know Python.\n");
    }

    #[test]
    fn missing_skills_directory_produces_empty_skills() {
        let root = unique_temp_dir("config");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("agents.conf").as_str(), "").unwrap();
        fs::write(root.join("tools.conf").as_str(), "").unwrap();

        let config = Config::load(&root).unwrap();
        assert!(config.skills.is_empty());
    }

    #[test]
    fn load_skills_earlier_directory_wins() {
        let dir_a = unique_temp_dir("skills-a");
        let dir_b = unique_temp_dir("skills-b");
        fs::create_dir_all(dir_a.join("rust").as_str()).unwrap();
        fs::create_dir_all(dir_a.join("go").as_str()).unwrap();
        fs::create_dir_all(dir_b.join("rust").as_str()).unwrap();
        fs::create_dir_all(dir_b.join("python").as_str()).unwrap();

        fs::write(dir_a.join("rust/SKILL.md").as_str(), "# Rust from A\n").unwrap();
        fs::write(dir_a.join("go/SKILL.md").as_str(), "# Go from A\n").unwrap();
        fs::write(
            dir_b.join("rust/SKILL.md").as_str(),
            "# Rust from B (should be shadowed)\n",
        )
        .unwrap();
        fs::write(dir_b.join("python/SKILL.md").as_str(), "# Python from B\n").unwrap();

        let dirs = vec![dir_a.clone(), dir_b.clone()];
        let skills = load_skills(&dirs).unwrap();
        assert_eq!(skills.len(), 3);

        let rust_skill = skills.get("rust").unwrap();
        assert_eq!(rust_skill.content, "# Rust from A\n");
        assert_eq!(rust_skill.path, dir_a.join("rust/SKILL.md"));

        assert_eq!(skills.get("go").unwrap().content, "# Go from A\n");

        let python_skill = skills.get("python").unwrap();
        assert_eq!(python_skill.content, "# Python from B\n");
        assert_eq!(python_skill.path, dir_b.join("python/SKILL.md"));
    }

    #[test]
    fn load_skills_skips_nonexistent_directories() {
        let dir_exists = unique_temp_dir("skills-exists");
        let dir_missing = unique_temp_dir("skills-missing");
        fs::create_dir_all(dir_exists.join("rust").as_str()).unwrap();
        fs::write(dir_exists.join("rust/SKILL.md").as_str(), "# Rust\n").unwrap();

        let dirs = vec![dir_missing, dir_exists];
        let skills = load_skills(&dirs).unwrap();
        assert_eq!(skills.len(), 1);
        assert!(skills.contains_key("rust"));
    }

    fn write_tool_contract(root: &Path, tool: &str, description: &str) {
        write_tool_script(root, tool, "#!/bin/sh\nexit 0\n");
        write_default_tool_manifest(root, tool, description);
    }
}
