use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use handled::SError;
use serde::Serialize;
use utf8path::Path;

pub const SESSIONS_DIR: &str = "sessions";
pub const SID_SESSIONS_ENV: &str = "SID_SESSIONS";
pub const SID_SESSION_DIR_ENV: &str = "SID_SESSION_DIR";
pub const SID_SESSION_ID_ENV: &str = "SID_SESSION_ID";

#[derive(Debug)]
pub struct SidSession {
    id: String,
    sessions_root: PathBuf,
    root: PathBuf,
    api_dir: PathBuf,
    tools_dir: PathBuf,
    next_api_request: AtomicU64,
    next_api_response: AtomicU64,
    next_tool_invocation: AtomicU64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ToolInvocationDirs {
    pub(crate) sequence: u64,
    pub(crate) root: PathBuf,
    pub(crate) scratch_dir: PathBuf,
    pub(crate) temp_dir: PathBuf,
}

impl SidSession {
    pub fn create(config_root: &Path) -> Result<Self, SError> {
        Self::create_in(resolve_sessions_root(config_root)?)
    }

    pub(crate) fn create_in(sessions_root: PathBuf) -> Result<Self, SError> {
        fs::create_dir_all(&sessions_root).map_err(|err| {
            session_error("io_error", "failed to create sessions directory")
                .with_string_field("path", sessions_root.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())
        })?;

        for _ in 0..8 {
            let id = generate_uuid_v4()?;
            let root = sessions_root.join(&id);
            match fs::create_dir(&root) {
                Ok(()) => return Self::from_created_root(id, sessions_root, root),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => {
                    return Err(
                        session_error("io_error", "failed to create session directory")
                            .with_string_field("path", root.to_string_lossy().as_ref())
                            .with_string_field("cause", &err.to_string()),
                    );
                }
            }
        }

        Err(session_error(
            "uuid_collision",
            "failed to allocate a unique session UUID",
        ))
    }

    fn from_created_root(
        id: String,
        sessions_root: PathBuf,
        root: PathBuf,
    ) -> Result<Self, SError> {
        let api_dir = root.join("api");
        let tools_dir = root.join("tools");
        let bash_tmp_dir = root.join("bash-tmp");
        fs::create_dir_all(&api_dir).map_err(|err| {
            session_error("io_error", "failed to create session API log directory")
                .with_string_field("path", api_dir.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())
        })?;
        fs::create_dir_all(&tools_dir).map_err(|err| {
            session_error("io_error", "failed to create session tool directory")
                .with_string_field("path", tools_dir.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())
        })?;
        fs::create_dir_all(&bash_tmp_dir).map_err(|err| {
            session_error(
                "io_error",
                "failed to create session bash temporary directory",
            )
            .with_string_field("path", bash_tmp_dir.to_string_lossy().as_ref())
            .with_string_field("cause", &err.to_string())
        })?;

        let session = Self {
            id,
            sessions_root,
            root,
            api_dir,
            tools_dir,
            next_api_request: AtomicU64::new(0),
            next_api_response: AtomicU64::new(0),
            next_tool_invocation: AtomicU64::new(0),
        };
        session.write_metadata()?;
        Ok(session)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn sessions_root(&self) -> &PathBuf {
        &self.sessions_root
    }

    pub fn root(&self) -> &PathBuf {
        &self.root
    }

    pub fn transcript_path(&self) -> PathBuf {
        self.root.join("transcript.json")
    }

    pub(crate) fn create_tool_invocation_dirs(
        &self,
        request_id: &str,
    ) -> Result<ToolInvocationDirs, SError> {
        let sequence = self.next_tool_invocation.fetch_add(1, Ordering::Relaxed) + 1;
        let root = self.tools_dir.join(format!("{sequence:06}-{request_id}"));
        let scratch_dir = root.join("scratch");
        let temp_dir = root.join("tmp");
        fs::create_dir_all(&scratch_dir).map_err(|err| {
            session_error("io_error", "failed to create tool scratch directory")
                .with_string_field("path", scratch_dir.to_string_lossy().as_ref())
                .with_string_field("request_id", request_id)
                .with_string_field("cause", &err.to_string())
        })?;
        fs::create_dir_all(&temp_dir).map_err(|err| {
            session_error("io_error", "failed to create tool temporary directory")
                .with_string_field("path", temp_dir.to_string_lossy().as_ref())
                .with_string_field("request_id", request_id)
                .with_string_field("cause", &err.to_string())
        })?;
        Ok(ToolInvocationDirs {
            sequence,
            root,
            scratch_dir,
            temp_dir,
        })
    }

    pub(crate) fn log_api_request(&self, value: &impl Serialize) -> Result<(), SError> {
        let sequence = self.next_api_request.fetch_add(1, Ordering::Relaxed) + 1;
        self.write_json(
            self.api_dir.join(format!("{sequence:06}-request.json")),
            value,
        )
    }

    pub(crate) fn log_api_response(&self, value: &impl Serialize) -> Result<(), SError> {
        let sequence = self.next_api_response.fetch_add(1, Ordering::Relaxed) + 1;
        self.write_json(
            self.api_dir.join(format!("{sequence:06}-response.json")),
            value,
        )
    }

    fn write_metadata(&self) -> Result<(), SError> {
        #[derive(Serialize)]
        struct Metadata<'a> {
            id: &'a str,
            sessions_root: String,
            session_dir: String,
            created_unix_nanos: u128,
        }

        let created_unix_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let metadata = Metadata {
            id: &self.id,
            sessions_root: self.sessions_root.to_string_lossy().into_owned(),
            session_dir: self.root.to_string_lossy().into_owned(),
            created_unix_nanos,
        };
        self.write_json(self.root.join("session.json"), &metadata)
    }

    fn write_json(&self, path: PathBuf, value: &impl Serialize) -> Result<(), SError> {
        let payload = serde_json::to_vec_pretty(value).map_err(|err| {
            session_error("json_serialize_error", "failed to serialize session log")
                .with_string_field("path", path.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())
        })?;
        fs::write(&path, payload).map_err(|err| {
            session_error("io_error", "failed to write session log")
                .with_string_field("path", path.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())
        })
    }
}

fn resolve_sessions_root(config_root: &Path) -> Result<PathBuf, SError> {
    match std::env::var(SID_SESSIONS_ENV) {
        Ok(path) if !path.is_empty() => Ok(PathBuf::from(path)),
        Ok(_) | Err(std::env::VarError::NotPresent) => {
            Ok(PathBuf::from(config_root.as_str()).join(SESSIONS_DIR))
        }
        Err(std::env::VarError::NotUnicode(_)) => Err(session_error(
            "invalid_sid_sessions",
            "SID_SESSIONS is not valid UTF-8",
        )),
    }
}

fn generate_uuid_v4() -> Result<String, SError> {
    let mut bytes = [0u8; 16];
    fill_random_bytes(&mut bytes)?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Ok(format_uuid(&bytes))
}

#[cfg(unix)]
fn fill_random_bytes(bytes: &mut [u8]) -> Result<(), SError> {
    let mut random = fs::File::open("/dev/urandom").map_err(|err| {
        session_error("random_error", "failed to open system random device")
            .with_string_field("path", "/dev/urandom")
            .with_string_field("cause", &err.to_string())
    })?;
    random.read_exact(bytes).map_err(|err| {
        session_error("random_error", "failed to read system random bytes")
            .with_string_field("path", "/dev/urandom")
            .with_string_field("cause", &err.to_string())
    })
}

#[cfg(not(unix))]
fn fill_random_bytes(bytes: &mut [u8]) -> Result<(), SError> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let mut state = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        ^ ((std::process::id() as u128) << 64)
        ^ NEXT_ID.fetch_add(1, Ordering::Relaxed) as u128;
    for byte in bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = state as u8;
        state = state.rotate_left(11);
    }
    Ok(())
}

fn format_uuid(bytes: &[u8; 16]) -> String {
    let mut uuid = String::with_capacity(36);
    for (idx, byte) in bytes.iter().enumerate() {
        if matches!(idx, 4 | 6 | 8 | 10) {
            uuid.push('-');
        }
        uuid.push(nibble_to_hex(byte >> 4));
        uuid.push(nibble_to_hex(byte & 0x0f));
    }
    uuid
}

fn nibble_to_hex(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'a' + (nibble - 10)) as char,
        _ => unreachable!("nibble should be in range"),
    }
}

fn session_error(code: &str, message: &str) -> SError {
    SError::new("sid-session")
        .with_code(code)
        .with_message(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_temp_dir;

    #[test]
    fn session_create_allocates_uuid_directory() {
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let session = SidSession::create_in(sessions_root.clone()).unwrap();

        assert_eq!(session.id().len(), 36);
        assert_eq!(session.id().as_bytes()[14], b'4');
        assert!(matches!(
            session.id().as_bytes()[19],
            b'8' | b'9' | b'a' | b'b'
        ));
        assert_eq!(session.root(), &sessions_root.join(session.id()));
        assert!(session.root().join("session.json").is_file());
        assert!(session.root().join("api").is_dir());
        assert!(session.root().join("tools").is_dir());

        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn tool_invocation_dirs_are_ordered_within_session() {
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let session = SidSession::create_in(sessions_root.clone()).unwrap();

        let first = session.create_tool_invocation_dirs("sidreq_first").unwrap();
        let second = session
            .create_tool_invocation_dirs("sidreq_second")
            .unwrap();

        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert!(first.root.ends_with("000001-sidreq_first"));
        assert!(second.root.ends_with("000002-sidreq_second"));
        assert_eq!(first.scratch_dir, first.root.join("scratch"));
        assert_eq!(first.temp_dir, first.root.join("tmp"));

        fs::remove_dir_all(sessions_root).unwrap();
    }
}
