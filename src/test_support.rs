use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use utf8path::Path;

use crate::config::{TOOL_PROTOCOL_VERSION, TOOLS_DIR};

pub(crate) fn temp_config_root(prefix: &str) -> Path<'static> {
    let root = unique_temp_dir(prefix);
    fs::create_dir_all(root.as_str()).unwrap();
    root
}

pub(crate) fn unique_temp_dir(prefix: &str) -> Path<'static> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let sequence = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    Path::try_from(
        std::env::temp_dir().join(format!("sid-isnt-done-{prefix}-{timestamp}-{sequence}")),
    )
    .unwrap()
    .into_owned()
}

pub(crate) fn write_tool_script(root: &Path, tool: &str, body: &str) {
    fs::create_dir_all(root.join(TOOLS_DIR).as_str()).unwrap();
    let path = root.join(format!("{TOOLS_DIR}/{tool}")).into_owned();
    fs::write(path.as_str(), body).unwrap();
    make_executable(&path);
}

pub(crate) fn write_tool_manifest(
    root: &Path,
    tool: &str,
    protocol_version: u32,
    description: &str,
) {
    write_tool_manifest_with_schema(
        root,
        tool,
        protocol_version,
        description,
        serde_json::json!({
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["paths"]
        }),
    );
}

pub(crate) fn write_default_tool_manifest(root: &Path, tool: &str, description: &str) {
    write_tool_manifest(root, tool, TOOL_PROTOCOL_VERSION, description);
}

pub(crate) fn write_tool_manifest_with_schema(
    root: &Path,
    tool: &str,
    protocol_version: u32,
    description: &str,
    input_schema: serde_json::Value,
) {
    fs::create_dir_all(root.join(TOOLS_DIR).as_str()).unwrap();
    fs::write(
        root.join(format!("{TOOLS_DIR}/{tool}.json")).as_str(),
        serde_json::to_vec_pretty(&serde_json::json!({
            "protocol_version": protocol_version,
            "description": description,
            "input_schema": input_schema
        }))
        .unwrap(),
    )
    .unwrap();
}

#[cfg(unix)]
pub(crate) fn make_executable(path: &Path) {
    let mut permissions = fs::metadata(path.as_str()).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path.as_str(), permissions).unwrap();
}

#[cfg(not(unix))]
pub(crate) fn make_executable(_path: &Path) {}
