//! Runtime user-instruction injection.

use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path as StdPath, PathBuf};
use std::process::Stdio;

use claudius::chat::ChatConfig;
use claudius::{ContentBlock, MessageParam, MessageParamContent, MessageRole, TextBlock};
use handled::SError;
use rc_conf::{RcConf, var_name_from_service, var_prefix_from_service};
use utf8path::Path;

use crate::config::{
    AGENTS_CONF_FILE, AGENTS_DIR, AGENTS_MD_FILE, AGENTS_MD_PATH_ENV, AgentConfig, Config,
    SkillConfig,
};
use crate::seatbelt::WritableRoots;
use crate::session;
use crate::tool_protocol::next_request_id;

#[derive(Clone, Debug)]
pub(crate) struct UserInstructionSettings {
    agents_md_enabled: bool,
    agents_md_path: Option<String>,
    hook: Option<UserInstructionHook>,
}

impl Default for UserInstructionSettings {
    fn default() -> Self {
        Self {
            agents_md_enabled: true,
            agents_md_path: None,
            hook: None,
        }
    }
}

pub(crate) fn disabled_user_instruction_settings() -> UserInstructionSettings {
    UserInstructionSettings {
        agents_md_enabled: false,
        agents_md_path: None,
        hook: None,
    }
}

#[derive(Clone, Debug)]
struct UserInstructionHook {
    service_name: String,
    executable_path: Path<'static>,
}

pub(crate) struct UserInstructionRuntimeContext<'a> {
    pub(crate) agent_id: &'a str,
    pub(crate) config_root: &'a Path<'a>,
    pub(crate) workspace_root: &'a Path<'a>,
    pub(crate) session: Option<&'a session::SidSession>,
    pub(crate) latest_user_message: Option<&'a str>,
    pub(crate) skills: &'a [SkillConfig],
}

struct HookDirs {
    scratch_dir: PathBuf,
    temp_dir: PathBuf,
}

struct HookOverlayContext<'a> {
    scratch_dir: &'a StdPath,
    temp_dir: &'a StdPath,
    user_message_file: &'a StdPath,
    skills_manifest_file: &'a StdPath,
    skills_dir: &'a StdPath,
    rc_conf_path: &'a str,
    rc_d_path: &'a str,
    agents_md_path: &'a str,
}

struct HookTurnFiles {
    user_message_file: PathBuf,
    skills_manifest_file: PathBuf,
    skills_dir: PathBuf,
}

struct ResolvedAgentsMdComponent {
    path: PathBuf,
    workspace_relative: bool,
}

pub(crate) fn resolve_user_instruction_settings(
    config: &Config,
    agent_config: &AgentConfig,
) -> Result<UserInstructionSettings, SError> {
    let hook = if agent_config.user_instructions_enabled {
        agent_config
            .user_instructions_hook
            .as_deref()
            .map(|service| resolve_user_instruction_hook(config, service))
            .transpose()?
    } else {
        None
    };
    Ok(UserInstructionSettings {
        agents_md_enabled: agent_config.agents_md_enabled,
        agents_md_path: agent_config.agents_md_path.clone(),
        hook,
    })
}

pub(crate) fn append_agents_md_to_system_prompt(
    chat_config: &mut ChatConfig,
    settings: &UserInstructionSettings,
    workspace_root: &Path,
) -> Result<(), SError> {
    if !settings.agents_md_enabled {
        return Ok(());
    }

    let files = resolve_agents_md_files(settings, workspace_root)
        .map_err(|err| user_instruction_runtime_error("invalid_agents_md_path", &err))?;
    let text = read_agents_md_files(&files)
        .map_err(|err| user_instruction_runtime_error("io_error", &err))?;
    let text = text.trim_end();
    if text.is_empty() {
        return Ok(());
    }

    let existing = chat_config.system_prompt_text().unwrap_or("").to_string();
    chat_config.set_system_prompt(Some(format!(
        "{existing}\n\n# User instructions from AGENTS.md\n\n{text}\n"
    )));
    Ok(())
}

pub(crate) async fn build_user_instruction_block(
    settings: &UserInstructionSettings,
    context: &UserInstructionRuntimeContext<'_>,
) -> Result<Option<String>, String> {
    let Some(hook) = settings.hook.as_ref() else {
        return Ok(None);
    };

    let agents_md_files = resolve_agents_md_files(settings, context.workspace_root)?;
    let agents_md_path = join_paths_for_env(&agents_md_files);
    let mut sections = String::new();

    let output = run_user_instruction_hook(hook, context, &agents_md_path).await?;
    append_section(
        &mut sections,
        &format!("# User instructions from hook {}", hook.service_name),
        &output,
    );

    if sections.is_empty() {
        return Ok(None);
    }

    Ok(Some(format!(
        "# sid injected user instructions\n\nDirect system and operator instructions take precedence over these injected user instructions.\n\n{}",
        sections.trim_end()
    )))
}

pub(crate) fn append_user_instruction_block(
    messages: &mut [MessageParam],
    instructions: String,
) -> bool {
    let Some(message) = messages.last_mut() else {
        return false;
    };
    if message.role != MessageRole::User || message_contains_tool_result(message) {
        return false;
    }
    append_text_block(message, instructions);
    true
}

fn append_text_block(message: &mut MessageParam, text: String) {
    let block = ContentBlock::Text(TextBlock::new(text));
    match &mut message.content {
        MessageParamContent::String(existing) => {
            let original = std::mem::take(existing);
            message.content = MessageParamContent::Array(vec![
                ContentBlock::Text(TextBlock::new(original)),
                block,
            ]);
        }
        MessageParamContent::Array(blocks) => blocks.push(block),
    }
}

fn message_contains_tool_result(message: &MessageParam) -> bool {
    match &message.content {
        MessageParamContent::String(_) => false,
        MessageParamContent::Array(blocks) => blocks.iter().any(ContentBlock::is_tool_result),
    }
}

fn append_section(output: &mut String, header: &str, body: &str) {
    let body = body.trim_end();
    if body.trim().is_empty() {
        return;
    }
    if !output.is_empty() {
        output.push_str("\n\n");
    }
    output.push_str(header);
    output.push_str("\n\n");
    output.push_str(body);
    output.push('\n');
}

fn resolve_user_instruction_hook(
    config: &Config,
    service_name: &str,
) -> Result<UserInstructionHook, SError> {
    let agents_dir = config.root.join(AGENTS_DIR).into_owned();
    let executable_path =
        resolve_agent_executable_path(&agents_dir, &config.agents_rc_conf, service_name);
    require_hook_executable(service_name, &executable_path)?;
    Ok(UserInstructionHook {
        service_name: service_name.to_string(),
        executable_path,
    })
}

fn resolve_agent_executable_path(
    agents_dir: &Path,
    rc_conf: &RcConf,
    service_name: &str,
) -> Path<'static> {
    for candidate in rc_conf.alias_lookup_order(service_name).0 {
        let path = agents_dir.join(candidate).into_owned();
        if path.is_file() {
            return path;
        }
    }
    agents_dir.join(service_name).into_owned()
}

fn require_hook_executable(service_name: &str, path: &Path) -> Result<(), SError> {
    if !path.is_file() {
        return Err(user_instruction_config_error(
            "missing_user_instructions_hook",
            "configured user-instructions hook executable does not exist",
        )
        .with_string_field("hook", service_name)
        .with_string_field("path", path.as_str()));
    }
    #[cfg(unix)]
    {
        let metadata = fs::metadata(path.as_str()).map_err(|err| {
            user_instruction_config_error(
                "io_error",
                "failed to inspect user-instructions hook executable",
            )
            .with_string_field("hook", service_name)
            .with_string_field("path", path.as_str())
            .with_string_field("cause", &err.to_string())
        })?;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(user_instruction_config_error(
                "user_instructions_hook_not_executable",
                "configured user-instructions hook is not marked executable",
            )
            .with_string_field("hook", service_name)
            .with_string_field("path", path.as_str()));
        }
    }
    Ok(())
}

fn user_instruction_config_error(code: &str, message: &str) -> SError {
    SError::new("config").with_code(code).with_message(message)
}

fn user_instruction_runtime_error(code: &str, message: &str) -> SError {
    SError::new("user-instructions")
        .with_code(code)
        .with_message(message)
}

fn resolve_agents_md_files(
    settings: &UserInstructionSettings,
    workspace_root: &Path,
) -> Result<Vec<PathBuf>, String> {
    let path_spec = settings
        .agents_md_path
        .clone()
        .or_else(|| nonempty_env(AGENTS_MD_PATH_ENV))
        .unwrap_or_else(|| format!("./{AGENTS_MD_FILE}"));
    let mut files = Vec::new();
    for component in path_spec
        .split(':')
        .filter(|component| !component.is_empty())
    {
        let resolved = resolve_agents_md_component(component, workspace_root)?;
        if resolved.path.is_file() {
            if resolved.workspace_relative {
                require_workspace_relative_agents_md_file(
                    component,
                    &resolved.path,
                    workspace_root,
                )?;
            }
            files.push(resolved.path);
        }
    }
    Ok(files)
}

fn resolve_agents_md_component(
    component: &str,
    workspace_root: &Path,
) -> Result<ResolvedAgentsMdComponent, String> {
    let path = if let Some(rest) = component.strip_prefix("~/") {
        let home = std::env::var("HOME")
            .map_err(|_| format!("failed to expand {component}: HOME is not set"))?;
        PathBuf::from(home).join(rest)
    } else {
        PathBuf::from(component)
    };
    if path.is_absolute() {
        Ok(ResolvedAgentsMdComponent {
            path,
            workspace_relative: false,
        })
    } else {
        let relative = normalize_workspace_relative_path(component, &path)?;
        Ok(ResolvedAgentsMdComponent {
            path: PathBuf::from(workspace_root.as_str()).join(relative),
            workspace_relative: true,
        })
    }
}

fn nonempty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .and_then(|value| (!value.trim().is_empty()).then_some(value))
}

fn normalize_workspace_relative_path(component: &str, path: &StdPath) -> Result<PathBuf, String> {
    let mut cleaned = PathBuf::new();
    for path_component in path.components() {
        match path_component {
            std::path::Component::CurDir => {}
            std::path::Component::Normal(part) => cleaned.push(part),
            std::path::Component::ParentDir => {
                if !cleaned.pop() {
                    return Err(format!(
                        "AGENTS.md path component {component:?} escapes the workspace root"
                    ));
                }
            }
            std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                return Err(format!(
                    "AGENTS.md path component {component:?} is not workspace-relative"
                ));
            }
        }
    }
    Ok(cleaned)
}

fn require_workspace_relative_agents_md_file(
    component: &str,
    path: &StdPath,
    workspace_root: &Path,
) -> Result<(), String> {
    let workspace = fs::canonicalize(workspace_root.as_str()).map_err(|err| {
        format!(
            "failed to canonicalize workspace root {} for AGENTS.md path {component:?}: {err}",
            workspace_root.as_str()
        )
    })?;
    let file = fs::canonicalize(path).map_err(|err| {
        format!(
            "failed to canonicalize AGENTS.md path {} from component {component:?}: {err}",
            path.display()
        )
    })?;
    if file.starts_with(&workspace) {
        Ok(())
    } else {
        Err(format!(
            "AGENTS.md path component {component:?} resolves outside the workspace root"
        ))
    }
}

fn read_agents_md_files(files: &[PathBuf]) -> Result<String, String> {
    let mut output = String::new();
    for file in files {
        let content = fs::read_to_string(file)
            .map_err(|err| format!("failed to read AGENTS.md file {}: {}", file.display(), err))?;
        if !output.is_empty() {
            output.push_str("\n\n");
        }
        output.push_str(content.trim_end());
        output.push('\n');
    }
    Ok(output)
}

fn join_paths_for_env(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.to_string_lossy())
        .collect::<Vec<_>>()
        .join(":")
}

fn write_hook_turn_files(
    hook: &UserInstructionHook,
    context: &UserInstructionRuntimeContext<'_>,
    dirs: &HookDirs,
) -> Result<HookTurnFiles, String> {
    let user_message_file = dirs.scratch_dir.join("user-message.txt");
    fs::write(
        &user_message_file,
        context.latest_user_message.unwrap_or_default(),
    )
    .map_err(|err| {
        format!(
            "user-instructions hook '{}' failed to write user message file {}: {}",
            hook.service_name,
            user_message_file.display(),
            err
        )
    })?;

    let skills_dir = dirs.scratch_dir.join("skills");
    fs::create_dir_all(&skills_dir).map_err(|err| {
        format!(
            "user-instructions hook '{}' failed to create skills directory {}: {}",
            hook.service_name,
            skills_dir.display(),
            err
        )
    })?;

    let skills_manifest_file = dirs.scratch_dir.join("skills.tsv");
    let mut manifest = String::new();
    for (idx, skill) in context.skills.iter().enumerate() {
        if !is_skill_mention_id(&skill.id) {
            continue;
        }
        let skill_dir = skills_dir.join(idx.to_string());
        fs::create_dir_all(&skill_dir).map_err(|err| {
            format!(
                "user-instructions hook '{}' failed to create skill scratch directory {}: {}",
                hook.service_name,
                skill_dir.display(),
                err
            )
        })?;
        let content_file = skill_dir.join("SKILL.md");
        fs::write(&content_file, &skill.content).map_err(|err| {
            format!(
                "user-instructions hook '{}' failed to write skill file {}: {}",
                hook.service_name,
                content_file.display(),
                err
            )
        })?;
        manifest.push_str(&skill.id);
        manifest.push('\t');
        manifest.push_str(&format!("/skills/{}/SKILL.md", skill.id));
        manifest.push('\t');
        manifest.push_str(&content_file.to_string_lossy());
        manifest.push('\n');
    }
    fs::write(&skills_manifest_file, manifest).map_err(|err| {
        format!(
            "user-instructions hook '{}' failed to write skills manifest {}: {}",
            hook.service_name,
            skills_manifest_file.display(),
            err
        )
    })?;

    Ok(HookTurnFiles {
        user_message_file,
        skills_manifest_file,
        skills_dir,
    })
}

fn is_skill_mention_id(id: &str) -> bool {
    !id.is_empty()
        && !is_common_env_var(id)
        && id.bytes().all(
            |byte| matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-' | b':'),
        )
}

fn is_common_env_var(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "PATH"
            | "HOME"
            | "USER"
            | "SHELL"
            | "PWD"
            | "TMPDIR"
            | "TEMP"
            | "TMP"
            | "LANG"
            | "TERM"
            | "XDG_CONFIG_HOME"
    )
}

async fn run_user_instruction_hook(
    hook: &UserInstructionHook,
    context: &UserInstructionRuntimeContext<'_>,
    agents_md_path: &str,
) -> Result<String, String> {
    let dirs = create_hook_dirs().map_err(|err| {
        format!(
            "user-instructions hook '{}' failed to create scratch directory: {}",
            hook.service_name, err
        )
    })?;
    let result = run_user_instruction_hook_inner(hook, context, agents_md_path, &dirs).await;
    let cleanup_result = fs::remove_dir_all(&dirs.scratch_dir);
    match (result, cleanup_result) {
        (Ok(output), Ok(())) => Ok(output),
        (Ok(_), Err(err)) => Err(format!(
            "user-instructions hook '{}' failed to clean scratch directory {}: {}",
            hook.service_name,
            dirs.scratch_dir.display(),
            err
        )),
        (Err(err), _) => Err(err),
    }
}

async fn run_user_instruction_hook_inner(
    hook: &UserInstructionHook,
    context: &UserInstructionRuntimeContext<'_>,
    agents_md_path: &str,
    dirs: &HookDirs,
) -> Result<String, String> {
    let prepared = prepare_hook_runtime(hook, context, agents_md_path, dirs)?;
    let mut writable_roots = WritableRoots::default();
    writable_roots.push(dirs.scratch_dir.to_string_lossy().into_owned());
    let mut cmd = crate::seatbelt::sandboxed_command(
        hook.executable_path.as_str(),
        &["run"],
        &writable_roots,
    );
    cmd.current_dir(context.workspace_root.as_str())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .envs(&prepared.bindings)
        .env("TMPDIR", &dirs.temp_dir)
        .env("TEMP", &dirs.temp_dir)
        .env("TMP", &dirs.temp_dir)
        .env("PAGER", "cat")
        .env("RCVAR_ARGV0", var_name_from_service(&hook.service_name))
        .env("RC_CONF_PATH", &prepared.rc_conf_path)
        .env("RC_D_PATH", &prepared.rc_d_path);
    if let Some(session) = context.session {
        cmd.env(session::SID_SESSION_ID_ENV, session.id())
            .env(session::SID_SESSION_DIR_ENV, session.root())
            .env(session::SID_SESSIONS_ENV, session.sessions_root());
    }

    let output = cmd.output().await.map_err(|err| {
        format!(
            "user-instructions hook '{}' failed to launch: {}",
            hook.service_name, err
        )
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim_end();
        let detail = if stderr.is_empty() {
            String::new()
        } else {
            format!(": {stderr}")
        };
        return Err(format!(
            "user-instructions hook '{}' exited with status {}{}",
            hook.service_name, output.status, detail
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

struct PreparedHookRuntime {
    rc_conf_path: String,
    rc_d_path: String,
    bindings: std::collections::HashMap<String, String>,
}

fn prepare_hook_runtime(
    hook: &UserInstructionHook,
    context: &UserInstructionRuntimeContext<'_>,
    agents_md_path: &str,
    dirs: &HookDirs,
) -> Result<PreparedHookRuntime, String> {
    let turn_files = write_hook_turn_files(hook, context, dirs)?;
    let agents_conf_path = context.config_root.join(AGENTS_CONF_FILE);
    let rc_d_path = context.config_root.join(AGENTS_DIR);
    let overlay_path = dirs.scratch_dir.join("user-instructions-hook.conf");
    let base_rc_conf = RcConf::parse(agents_conf_path.as_str()).map_err(|err| {
        format!(
            "user-instructions hook '{}' failed to parse agents.conf {}: {:?}",
            hook.service_name,
            agents_conf_path.as_str(),
            err
        )
    })?;
    let mut services = base_rc_conf
        .list()
        .map_err(|err| {
            format!(
                "user-instructions hook '{}' failed to list configured agents: {:?}",
                hook.service_name, err
            )
        })?
        .collect::<Vec<_>>();
    if !services.iter().any(|service| service == &hook.service_name) {
        services.push(hook.service_name.clone());
    }
    let rc_conf_path = format!("{}:{}", agents_conf_path.as_str(), overlay_path.display());
    let rc_d_path = rc_d_path.as_str().to_string();
    let overlay_context = HookOverlayContext {
        scratch_dir: &dirs.scratch_dir,
        temp_dir: &dirs.temp_dir,
        user_message_file: &turn_files.user_message_file,
        skills_manifest_file: &turn_files.skills_manifest_file,
        skills_dir: &turn_files.skills_dir,
        rc_conf_path: &rc_conf_path,
        rc_d_path: &rc_d_path,
        agents_md_path,
    };
    let overlay = render_hook_overlay(&services, hook, context, &overlay_context);
    fs::write(&overlay_path, overlay).map_err(|err| {
        format!(
            "user-instructions hook '{}' failed to write rc overlay {}: {}",
            hook.service_name,
            overlay_path.display(),
            err
        )
    })?;
    let rc_conf = RcConf::parse(&rc_conf_path).map_err(|err| {
        format!(
            "user-instructions hook '{}' failed to parse rc overlay {}: {:?}",
            hook.service_name, rc_conf_path, err
        )
    })?;
    let bindings = rc_conf
        .bind_for_invoke(&hook.service_name, &hook.executable_path)
        .map_err(|err| {
            format!(
                "user-instructions hook '{}' failed to bind rcvars: {:?}",
                hook.service_name, err
            )
        })?;
    Ok(PreparedHookRuntime {
        rc_conf_path,
        rc_d_path,
        bindings,
    })
}

fn render_hook_overlay(
    services: &[String],
    hook: &UserInstructionHook,
    context: &UserInstructionRuntimeContext<'_>,
    overlay_context: &HookOverlayContext<'_>,
) -> String {
    let scratch_dir = overlay_context.scratch_dir.to_string_lossy().into_owned();
    let temp_dir = overlay_context.temp_dir.to_string_lossy().into_owned();
    let user_message_file = overlay_context
        .user_message_file
        .to_string_lossy()
        .into_owned();
    let skills_manifest_file = overlay_context
        .skills_manifest_file
        .to_string_lossy()
        .into_owned();
    let skills_dir = overlay_context.skills_dir.to_string_lossy().into_owned();
    let config_root = context.config_root.as_str();
    let workspace_root = context.workspace_root.as_str();
    let mut overlay = String::new();
    for service in services {
        let prefix = var_prefix_from_service(service);
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}WORKSPACE_ROOT"),
            workspace_root,
        );
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}CONFIG_ROOT"), config_root);
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}AGENT_ID"), context.agent_id);
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}HOOK_NAME"),
            &hook.service_name,
        );
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}AGENTS_MD_PATH"),
            overlay_context.agents_md_path,
        );
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}USER_MESSAGE_FILE"),
            &user_message_file,
        );
        append_rc_conf_assignment(
            &mut overlay,
            &format!("{prefix}SKILLS_MANIFEST_FILE"),
            &skills_manifest_file,
        );
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}SKILLS_DIR"), &skills_dir);
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}SCRATCH_DIR"), &scratch_dir);
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}TEMP_DIR"), &temp_dir);
        append_rc_conf_assignment(&mut overlay, &format!("{prefix}TMPDIR"), &temp_dir);
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

fn create_hook_dirs() -> Result<HookDirs, std::io::Error> {
    let parent = std::env::temp_dir().join("sid-user-instructions");
    fs::create_dir_all(&parent)?;
    for _ in 0..8 {
        let scratch_dir = parent.join(next_request_id());
        match fs::create_dir(&scratch_dir) {
            Ok(()) => {
                let temp_dir = scratch_dir.join("tmp");
                fs::create_dir_all(&temp_dir)?;
                return Ok(HookDirs {
                    scratch_dir,
                    temp_dir,
                });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "failed to allocate unique user-instructions hook directory",
    ))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path as StdPath;

    use super::*;
    use crate::test_support::temp_config_root;

    #[test]
    fn agents_md_path_allows_internal_parent_components() {
        let root = temp_config_root("agents-md-path");
        fs::create_dir_all(root.join("local").as_str()).unwrap();
        fs::write(root.join("AGENTS.md").as_str(), "Workspace rules.\n").unwrap();
        let settings = UserInstructionSettings {
            agents_md_enabled: true,
            agents_md_path: Some("local/../AGENTS.md".to_string()),
            hook: None,
        };

        let files = resolve_agents_md_files(&settings, &root).unwrap();

        assert_eq!(files, vec![StdPath::new(root.as_str()).join("AGENTS.md")]);

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn agents_md_path_rejects_relative_parent_escape() {
        let root = temp_config_root("agents-md-path");
        let outside = temp_config_root("agents-md-outside");
        fs::write(outside.join("AGENTS.md").as_str(), "Outside rules.\n").unwrap();
        let outside_name = StdPath::new(outside.as_str())
            .file_name()
            .unwrap()
            .to_string_lossy();
        let settings = UserInstructionSettings {
            agents_md_enabled: true,
            agents_md_path: Some(format!("../{outside_name}/AGENTS.md")),
            hook: None,
        };

        let err = resolve_agents_md_files(&settings, &root).unwrap_err();

        assert!(err.contains("escapes the workspace root"), "{err}");

        fs::remove_dir_all(root.as_str()).unwrap();
        fs::remove_dir_all(outside.as_str()).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn agents_md_path_rejects_relative_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = temp_config_root("agents-md-path");
        let outside = temp_config_root("agents-md-outside");
        fs::create_dir_all(root.join("local").as_str()).unwrap();
        fs::write(outside.join("AGENTS.md").as_str(), "Outside rules.\n").unwrap();
        symlink(
            outside.join("AGENTS.md").as_str(),
            root.join("local/AGENTS.md").as_str(),
        )
        .unwrap();
        let settings = UserInstructionSettings {
            agents_md_enabled: true,
            agents_md_path: Some("local/AGENTS.md".to_string()),
            hook: None,
        };

        let err = resolve_agents_md_files(&settings, &root).unwrap_err();

        assert!(err.contains("resolves outside the workspace root"), "{err}");

        fs::remove_dir_all(root.as_str()).unwrap();
        fs::remove_dir_all(outside.as_str()).unwrap();
    }
}
