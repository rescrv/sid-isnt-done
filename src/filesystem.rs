/// Filesystem mounting and skill resolution for agent sessions.
///
/// Builds the virtual filesystem hierarchy that agents use to access workspace
/// files and skill documents.  Skills are mounted read-only under
/// `/skills/<skill_id>/`.
use std::path::Path as StdPath;

use claudius::{MountHierarchy, Permissions};
use handled::SError;
use utf8path::Path;

use crate::config::{AgentConfig, Config, SkillConfig};

/// Build a default filesystem hierarchy with only the workspace root mounted at `/`.
pub(crate) fn build_default_filesystem(workspace_root: &Path) -> MountHierarchy {
    let mut hierarchy = MountHierarchy::default();
    hierarchy
        .mount(
            "/".into(),
            Permissions::ReadWrite,
            workspace_root.clone().into_owned(),
        )
        .expect("root mount at / must succeed as the first mount");
    hierarchy
}

/// Build a filesystem hierarchy with the workspace root at `/` and agent skills
/// mounted read-only under `/skills/<skill_id>/`.
pub(crate) fn build_agent_filesystem(
    workspace_root: &Path,
    config: &Config,
    agent_config: &AgentConfig,
) -> Result<MountHierarchy, SError> {
    let mut hierarchy = build_default_filesystem(workspace_root);
    let skills = resolve_agent_skills(config, agent_config)?;
    for skill in skills {
        mount_skill(&mut hierarchy, skill)?;
    }
    Ok(hierarchy)
}

/// Resolve which skills an agent has access to based on its skill configuration.
///
/// Returns an error if the agent references a skill name that is not defined in
/// the loaded configuration, unless the wildcard `*` is used.
pub(crate) fn resolve_agent_skills<'a>(
    config: &'a Config,
    agent_config: &AgentConfig,
) -> Result<Vec<&'a SkillConfig>, SError> {
    if agent_config.skills.is_empty() {
        return Ok(vec![]);
    }
    let include_all = agent_config.skills.iter().any(|s| s == "*");
    if include_all {
        return Ok(config.skills.values().collect());
    }
    let mut skills = Vec::new();
    for name in &agent_config.skills {
        let skill = config.skills.get(name).ok_or_else(|| {
            SError::new("sid-agent")
                .with_code("unknown_skill")
                .with_message("agent references an undefined skill")
                .with_string_field("agent", &agent_config.id)
                .with_string_field("skill", name)
        })?;
        skills.push(skill);
    }
    Ok(skills)
}

/// Mount a skill's directory as read-only under `/skills/<skill_id>/`.
fn mount_skill(hierarchy: &mut MountHierarchy, skill: &SkillConfig) -> Result<(), SError> {
    let parent = skill_parent_dir(&skill.path)?;
    let mount_path = format!("/skills/{}", skill.id);
    hierarchy
        .mount(
            Path::new(&mount_path).into_owned(),
            Permissions::ReadOnly,
            parent,
        )
        .map_err(|err| {
            filesystem_error("mount_error", &err).with_string_field("skill", &skill.id)
        })?;
    Ok(())
}

/// Extract the parent directory from a skill's file path.
fn skill_parent_dir(skill_path: &Path) -> Result<Path<'static>, SError> {
    let std_path = StdPath::new(skill_path.as_str());
    let parent = std_path.parent().ok_or_else(|| {
        filesystem_error("invalid_skill_path", "skill path has no parent directory")
            .with_string_field("path", skill_path.as_str())
    })?;
    Path::try_from(parent.to_path_buf())
        .map(Path::into_owned)
        .map_err(|err| {
            filesystem_error("invalid_skill_path", "skill parent path is not valid UTF-8")
                .with_string_field("path", skill_path.as_str())
                .with_string_field("cause", &format!("{err:?}"))
        })
}

fn filesystem_error(code: &str, message: &str) -> SError {
    SError::new("sid-agent")
        .with_code(code)
        .with_message(message)
}
