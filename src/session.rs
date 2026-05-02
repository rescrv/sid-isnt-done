//! Session lifecycle management.
//!
//! A [`SidSession`] represents a single interactive run of the agent.  Each
//! session occupies a timestamped directory under the workspace's sessions
//! root and maintains append-only journals for events, API calls, and tool
//! stream fragments.  Sessions can be created fresh, created as compacted
//! continuations of an earlier session, or resumed from their on-disk state.

use std::fs;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path as StdPath;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicU64, Ordering};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use handled::SError;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use utf8path::Path;

/// Default subdirectory name under the config root where sessions are stored.
pub const SESSIONS_DIR: &str = "sessions";
/// Environment variable that overrides the sessions root directory.
pub const SID_SESSIONS_ENV: &str = "SID_SESSIONS";
/// Environment variable exported to tools with the active session directory.
pub const SID_SESSION_DIR_ENV: &str = "SID_SESSION_DIR";
/// Environment variable exported to tools with the active session identifier.
pub const SID_SESSION_ID_ENV: &str = "SID_SESSION_ID";
/// When set, tool scratch directories are preserved after successful invocations.
pub const SID_KEEP_TOOL_SCRATCH_ENV: &str = "SID_KEEP_TOOL_SCRATCH";
/// When set, tool scratch directories are preserved after failed invocations.
pub const SID_KEEP_FAILED_TOOL_SCRATCH_ENV: &str = "SID_KEEP_FAILED_TOOL_SCRATCH";

const SESSION_METADATA_FILE: &str = "session.json";
const TRANSCRIPT_FILE: &str = "transcript.json";
const EVENTS_JOURNAL_FILE: &str = "events.jsonl";
const API_JOURNAL_FILE: &str = "api.jsonl";
const TOOL_STREAMS_JOURNAL_FILE: &str = "tool-streams.jsonl";
const BASH_STATE_FILE: &str = "bash-state.sh";

/// Identity snapshot of the agent that produced a compacted session summary.
///
/// Stored in the session metadata so that a resumed session can reconstruct
/// a memory-expert agent for follow-up questions.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct CompactionExpertConfig {
    /// Agent identifier at the time of compaction, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Model that produced the compacted summary.
    pub model: String,
    /// System prompt the compacting agent was using.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
}

/// Pointer back to the parent session from which a compacted session was derived.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct CompactionProvenance {
    /// Session identifier of the parent session.
    pub session_id: String,
    /// Filesystem path to the parent session directory.
    pub session_dir: String,
    /// Expert configuration for the agent that wrote the summary.
    pub expert: CompactionExpertConfig,
}

/// On-disk representation of a single interactive agent session.
///
/// See the [module-level documentation](self) for an overview of the session
/// lifecycle.
#[derive(Debug)]
pub struct SidSession {
    id: String,
    sessions_root: PathBuf,
    root: PathBuf,
    tmp_dir: PathBuf,
    bash_tmp_dir: PathBuf,
    events_path: PathBuf,
    api_path: PathBuf,
    tool_streams_path: PathBuf,
    bash_state_path: PathBuf,
    compaction_provenance: Option<CompactionProvenance>,
    counters: SessionCounters,
    journal_lock: StdMutex<()>,
    stream_lock: Arc<StdMutex<()>>,
}

#[derive(Debug, Default)]
struct SessionCounters {
    event_entry: AtomicU64,
    api_entry: AtomicU64,
    api_call: AtomicU64,
    active_api_call: AtomicU64,
    tool_invocation: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SessionCounterState {
    event_entry: u64,
    api_entry: u64,
    api_call: u64,
    tool_invocation: u64,
}

impl SessionCounters {
    fn from_state(state: SessionCounterState) -> Self {
        Self {
            event_entry: AtomicU64::new(state.event_entry),
            api_entry: AtomicU64::new(state.api_entry),
            api_call: AtomicU64::new(state.api_call),
            active_api_call: AtomicU64::new(state.api_call),
            tool_invocation: AtomicU64::new(state.tool_invocation),
        }
    }

    fn next_event_entry(&self) -> u64 {
        self.event_entry.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn next_api_entry(&self) -> u64 {
        self.api_entry.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn start_api_call(&self) -> u64 {
        let api_seq = self.api_call.fetch_add(1, Ordering::Relaxed) + 1;
        self.active_api_call.store(api_seq, Ordering::Relaxed);
        api_seq
    }

    fn current_api_call(&self) -> u64 {
        let api_seq = self.active_api_call.load(Ordering::Relaxed);
        if api_seq == 0 {
            self.start_api_call()
        } else {
            api_seq
        }
    }

    fn next_tool_invocation(&self) -> u64 {
        self.tool_invocation.fetch_add(1, Ordering::Relaxed) + 1
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ToolInvocationDirs {
    pub(crate) sequence: u64,
    pub(crate) root: PathBuf,
    pub(crate) scratch_dir: PathBuf,
    pub(crate) temp_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub(crate) struct ToolStreamJournal {
    path: PathBuf,
    lock: Arc<StdMutex<()>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ToolStartEvent<'a> {
    pub(crate) tool_seq: u64,
    pub(crate) request_id: &'a str,
    pub(crate) tool: &'a str,
    pub(crate) canonical_tool: &'a str,
    pub(crate) tool_use_id: &'a str,
    pub(crate) agent: &'a str,
    pub(crate) scratch_dir: &'a StdPath,
}

#[derive(Clone, Debug)]
pub(crate) struct ToolFinishEvent<'a> {
    pub(crate) tool_seq: u64,
    pub(crate) request_id: &'a str,
    pub(crate) status: Option<&'a str>,
    pub(crate) exit_code: Option<i32>,
    pub(crate) success: bool,
    pub(crate) result_ok: Option<bool>,
    pub(crate) output_len: Option<usize>,
    pub(crate) error: Option<&'a str>,
    pub(crate) scratch_preserved: bool,
    pub(crate) scratch_dir: Option<&'a StdPath>,
    pub(crate) cleanup_error: Option<&'a str>,
}

#[derive(Clone, Debug)]
struct SessionTimestamp {
    id: String,
    created_at: String,
    created_unix_micros: i128,
}

#[derive(Debug, Deserialize)]
struct SessionMetadata {
    id: String,
    #[serde(default)]
    compacted_from: Option<CompactionProvenance>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EventJournalState {
    max_seq: u64,
    max_tool_seq: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ApiJournalState {
    max_seq: u64,
    max_api_seq: u64,
}

impl SidSession {
    /// Create a fresh session under the default sessions root for `config_root`.
    ///
    /// # Errors
    ///
    /// Returns an error when the sessions directory cannot be created or a
    /// unique session identifier cannot be allocated.
    pub fn create(config_root: &Path) -> Result<Self, SError> {
        Self::create_in(resolve_sessions_root(config_root)?)
    }

    /// Create a compacted continuation session linked to a parent via `provenance`.
    ///
    /// # Errors
    ///
    /// Returns an error when the sessions directory cannot be created or a
    /// unique session identifier cannot be allocated.
    pub fn create_compacted(
        config_root: &Path,
        provenance: CompactionProvenance,
    ) -> Result<Self, SError> {
        Self::create_compacted_in(resolve_sessions_root(config_root)?, provenance)
    }

    /// Resume an existing session identified by `spec`.
    ///
    /// `spec` can be an absolute path to a session directory, a path to its
    /// `session.json` metadata file, or a bare session name relative to the
    /// default sessions root.
    ///
    /// # Errors
    ///
    /// Returns an error when the session directory does not exist or its
    /// metadata cannot be read.
    pub fn resume(config_root: &Path, spec: &str) -> Result<Self, SError> {
        let default_sessions_root = resolve_sessions_root(config_root)?;
        let root = resolve_existing_session_root(&default_sessions_root, spec)?;
        let sessions_root = root
            .parent()
            .map(StdPath::to_path_buf)
            .unwrap_or(default_sessions_root);
        Self::resume_from_root(sessions_root, root)
    }

    pub(crate) fn create_in(sessions_root: PathBuf) -> Result<Self, SError> {
        Self::create_in_with_provenance(sessions_root, None)
    }

    pub(crate) fn create_compacted_in(
        sessions_root: PathBuf,
        provenance: CompactionProvenance,
    ) -> Result<Self, SError> {
        Self::create_in_with_provenance(sessions_root, Some(provenance))
    }

    fn create_in_with_provenance(
        sessions_root: PathBuf,
        provenance: Option<CompactionProvenance>,
    ) -> Result<Self, SError> {
        fs::create_dir_all(&sessions_root).map_err(|err| {
            session_error("io_error", "failed to create sessions directory")
                .with_string_field("path", sessions_root.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())
        })?;

        for _ in 0..8 {
            let timestamp = now_session_timestamp();
            let root = sessions_root.join(&timestamp.id);
            match fs::create_dir(&root) {
                Ok(()) => {
                    return Self::from_created_root(
                        timestamp,
                        sessions_root,
                        root,
                        provenance.clone(),
                    );
                }
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
            "session_id_collision",
            "failed to allocate a unique timestamp session id",
        ))
    }

    #[cfg(test)]
    pub(crate) fn resume_in(sessions_root: PathBuf, spec: &str) -> Result<Self, SError> {
        let root = resolve_existing_session_root(&sessions_root, spec)?;
        let sessions_root = root
            .parent()
            .map(StdPath::to_path_buf)
            .unwrap_or(sessions_root);
        Self::resume_from_root(sessions_root, root)
    }

    fn from_created_root(
        timestamp: SessionTimestamp,
        sessions_root: PathBuf,
        root: PathBuf,
        compaction_provenance: Option<CompactionProvenance>,
    ) -> Result<Self, SError> {
        let tmp_dir = root.join("tmp");
        let bash_tmp_dir = tmp_dir.join("bash");
        fs::create_dir_all(&bash_tmp_dir).map_err(|err| {
            session_error(
                "io_error",
                "failed to create session runtime temporary directory",
            )
            .with_string_field("path", bash_tmp_dir.to_string_lossy().as_ref())
            .with_string_field("cause", &err.to_string())
        })?;

        let session = Self {
            id: timestamp.id,
            sessions_root,
            root: root.clone(),
            tmp_dir,
            bash_tmp_dir,
            events_path: root.join(EVENTS_JOURNAL_FILE),
            api_path: root.join(API_JOURNAL_FILE),
            tool_streams_path: root.join(TOOL_STREAMS_JOURNAL_FILE),
            bash_state_path: root.join(BASH_STATE_FILE),
            compaction_provenance,
            counters: SessionCounters::default(),
            journal_lock: StdMutex::new(()),
            stream_lock: Arc::new(StdMutex::new(())),
        };
        session.write_metadata(&timestamp.created_at, timestamp.created_unix_micros)?;
        session.log_session_start()?;
        Ok(session)
    }

    fn resume_from_root(sessions_root: PathBuf, root: PathBuf) -> Result<Self, SError> {
        let metadata = read_session_metadata(&root)?;
        let tmp_dir = root.join("tmp");
        let bash_tmp_dir = tmp_dir.join("bash");
        fs::create_dir_all(&bash_tmp_dir).map_err(|err| {
            session_error(
                "io_error",
                "failed to create session runtime temporary directory",
            )
            .with_string_field("path", bash_tmp_dir.to_string_lossy().as_ref())
            .with_string_field("cause", &err.to_string())
        })?;

        let counters = load_session_counter_state(&root, &tmp_dir)?;
        let session = Self {
            id: metadata.id,
            sessions_root,
            root: root.clone(),
            tmp_dir,
            bash_tmp_dir,
            events_path: root.join(EVENTS_JOURNAL_FILE),
            api_path: root.join(API_JOURNAL_FILE),
            tool_streams_path: root.join(TOOL_STREAMS_JOURNAL_FILE),
            bash_state_path: root.join(BASH_STATE_FILE),
            compaction_provenance: metadata.compacted_from,
            counters: SessionCounters::from_state(counters),
            journal_lock: StdMutex::new(()),
            stream_lock: Arc::new(StdMutex::new(())),
        };
        session.log_session_resume()?;
        Ok(session)
    }

    /// Return the session's unique identifier (a timestamp-based string).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Return the parent directory that contains all session directories.
    pub fn sessions_root(&self) -> &PathBuf {
        &self.sessions_root
    }

    /// Return the on-disk root directory for this session.
    pub fn root(&self) -> &PathBuf {
        &self.root
    }

    /// Return the compaction provenance if this session was created via compaction.
    pub fn compaction_provenance(&self) -> Option<&CompactionProvenance> {
        self.compaction_provenance.as_ref()
    }

    /// Return the path to the transcript JSON file for this session.
    pub fn transcript_path(&self) -> PathBuf {
        self.root.join(TRANSCRIPT_FILE)
    }

    pub(crate) fn bash_tmp_dir(&self) -> &PathBuf {
        &self.bash_tmp_dir
    }

    pub(crate) fn bash_state_path(&self) -> &PathBuf {
        &self.bash_state_path
    }

    /// Read the saved bash session state, if any.
    ///
    /// Returns `Ok(None)` when no state file exists or the file is empty.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure.
    pub fn read_bash_state(&self) -> Result<Option<String>, SError> {
        match fs::read_to_string(&self.bash_state_path) {
            Ok(state) if state.trim().is_empty() => Ok(None),
            Ok(state) => Ok(Some(state)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(session_error("io_error", "failed to read bash state")
                .with_string_field("path", self.bash_state_path.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())),
        }
    }

    /// Persist bash session state to disk for later resumption.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure.
    pub fn write_bash_state(&self, state: &str) -> Result<(), SError> {
        fs::write(&self.bash_state_path, state).map_err(|err| {
            session_error("io_error", "failed to write bash state")
                .with_string_field("path", self.bash_state_path.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())
        })
    }

    pub(crate) fn clear_bash_state(&self) -> Result<(), SError> {
        match fs::remove_file(&self.bash_state_path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(session_error("io_error", "failed to clear bash state")
                .with_string_field("path", self.bash_state_path.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())),
        }
    }

    pub(crate) fn tool_stream_journal(&self) -> ToolStreamJournal {
        ToolStreamJournal {
            path: self.tool_streams_path.clone(),
            lock: self.stream_lock.clone(),
        }
    }

    pub(crate) fn create_tool_invocation_dirs(
        &self,
        request_id: &str,
    ) -> Result<ToolInvocationDirs, SError> {
        let sequence = self.counters.next_tool_invocation();
        let root = self.tmp_dir.join(format!("tool-{sequence:06}"));
        let scratch_dir = root.clone();
        let temp_dir = root.join("tmp");
        fs::create_dir(&root).map_err(|err| {
            session_error("io_error", "failed to create tool scratch directory")
                .with_string_field("path", root.to_string_lossy().as_ref())
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
        let api_seq = self.counters.start_api_call();
        self.log_api_payload("request", api_seq, value)
    }

    pub(crate) fn log_api_response(&self, value: &impl Serialize) -> Result<(), SError> {
        let api_seq = self.counters.current_api_call();
        self.log_api_payload("response", api_seq, value)
    }

    pub(crate) fn log_tool_start(&self, event: ToolStartEvent<'_>) -> Result<(), SError> {
        #[derive(Serialize)]
        struct ToolStartRecord {
            seq: u64,
            ts: String,
            kind: &'static str,
            tool_seq: u64,
            request_id: String,
            tool: String,
            canonical_tool: String,
            tool_use_id: String,
            agent: String,
            scratch_dir: String,
        }

        let record = ToolStartRecord {
            seq: self.next_event_seq(),
            ts: now_created_at(),
            kind: "tool_start",
            tool_seq: event.tool_seq,
            request_id: event.request_id.to_string(),
            tool: event.tool.to_string(),
            canonical_tool: event.canonical_tool.to_string(),
            tool_use_id: event.tool_use_id.to_string(),
            agent: event.agent.to_string(),
            scratch_dir: event.scratch_dir.to_string_lossy().into_owned(),
        };
        self.append_journal(&self.events_path, &record)
    }

    pub(crate) fn log_tool_finish(&self, event: ToolFinishEvent<'_>) -> Result<(), SError> {
        #[derive(Serialize)]
        struct ToolFinishRecord {
            seq: u64,
            ts: String,
            kind: &'static str,
            tool_seq: u64,
            request_id: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            status: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            exit_code: Option<i32>,
            success: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            result_ok: Option<bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            output_len: Option<usize>,
            #[serde(skip_serializing_if = "Option::is_none")]
            error: Option<String>,
            scratch_preserved: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            scratch_dir: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            cleanup_error: Option<String>,
        }

        let record = ToolFinishRecord {
            seq: self.next_event_seq(),
            ts: now_created_at(),
            kind: "tool_finish",
            tool_seq: event.tool_seq,
            request_id: event.request_id.to_string(),
            status: event.status.map(str::to_string),
            exit_code: event.exit_code,
            success: event.success,
            result_ok: event.result_ok,
            output_len: event.output_len,
            error: event.error.map(str::to_string),
            scratch_preserved: event.scratch_preserved,
            scratch_dir: event
                .scratch_dir
                .map(|path| path.to_string_lossy().into_owned()),
            cleanup_error: event.cleanup_error.map(str::to_string),
        };
        self.append_journal(&self.events_path, &record)
    }

    fn log_session_start(&self) -> Result<(), SError> {
        #[derive(Serialize)]
        struct SessionStartRecord<'a> {
            seq: u64,
            ts: String,
            kind: &'static str,
            session_id: &'a str,
        }

        let record = SessionStartRecord {
            seq: self.next_event_seq(),
            ts: now_created_at(),
            kind: "session_start",
            session_id: &self.id,
        };
        self.append_journal(&self.events_path, &record)
    }

    fn log_session_resume(&self) -> Result<(), SError> {
        #[derive(Serialize)]
        struct SessionResumeRecord<'a> {
            seq: u64,
            ts: String,
            kind: &'static str,
            session_id: &'a str,
        }

        let record = SessionResumeRecord {
            seq: self.next_event_seq(),
            ts: now_created_at(),
            kind: "session_resume",
            session_id: &self.id,
        };
        self.append_journal(&self.events_path, &record)
    }

    fn log_api_payload(
        &self,
        kind: &'static str,
        api_seq: u64,
        value: &impl Serialize,
    ) -> Result<(), SError> {
        #[derive(Serialize)]
        struct ApiRecord<'a, T: Serialize + ?Sized> {
            seq: u64,
            ts: String,
            kind: &'static str,
            api_seq: u64,
            payload: &'a T,
        }

        let record = ApiRecord {
            seq: self.counters.next_api_entry(),
            ts: now_created_at(),
            kind,
            api_seq,
            payload: value,
        };
        self.append_journal(&self.api_path, &record)
    }

    fn write_metadata(&self, created_at: &str, created_unix_micros: i128) -> Result<(), SError> {
        #[derive(Serialize)]
        struct Metadata<'a> {
            id: &'a str,
            created_at: &'a str,
            created_unix_micros: i128,
            uuid: Option<&'a str>,
            pid: u32,
            sessions_root: String,
            session_dir: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            compacted_from: Option<&'a CompactionProvenance>,
        }

        let metadata = Metadata {
            id: &self.id,
            created_at,
            created_unix_micros,
            uuid: None,
            pid: std::process::id(),
            sessions_root: self.sessions_root.to_string_lossy().into_owned(),
            session_dir: self.root.to_string_lossy().into_owned(),
            compacted_from: self.compaction_provenance.as_ref(),
        };
        self.write_json(self.root.join(SESSION_METADATA_FILE), &metadata)
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

    fn append_journal(&self, path: &StdPath, value: &impl Serialize) -> Result<(), SError> {
        let _guard = self.journal_lock.lock().map_err(|_| {
            session_error("lock_poisoned", "session journal lock was poisoned")
                .with_string_field("path", path.to_string_lossy().as_ref())
        })?;
        append_jsonl(path, value)
    }

    fn next_event_seq(&self) -> u64 {
        self.counters.next_event_entry()
    }
}

impl Drop for SidSession {
    fn drop(&mut self) {
        if keep_any_tool_scratch() {
            return;
        }
        let _ = fs::remove_dir_all(&self.tmp_dir);
    }
}

impl ToolStreamJournal {
    pub(crate) fn append(&self, tool_seq: u64, stream: &str, bytes: &[u8]) -> Result<(), SError> {
        if bytes.is_empty() {
            return Ok(());
        }

        #[derive(Serialize)]
        struct ToolStreamRecord<'a> {
            tool_seq: u64,
            ts: String,
            stream: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            text: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            data_b64: Option<String>,
        }

        let (text, data_b64) = match std::str::from_utf8(bytes) {
            Ok(text) => (Some(text), None),
            Err(_) => (None, Some(BASE64_STANDARD.encode(bytes))),
        };
        let record = ToolStreamRecord {
            tool_seq,
            ts: now_created_at(),
            stream,
            text,
            data_b64,
        };

        let _guard = self.lock.lock().map_err(|_| {
            session_error("lock_poisoned", "session tool stream lock was poisoned")
                .with_string_field("path", self.path.to_string_lossy().as_ref())
        })?;
        append_jsonl(&self.path, &record)
    }
}

pub(crate) fn should_keep_tool_scratch(failed: bool) -> bool {
    env_truthy(SID_KEEP_TOOL_SCRATCH_ENV)
        || (failed && env_truthy(SID_KEEP_FAILED_TOOL_SCRATCH_ENV))
}

fn keep_any_tool_scratch() -> bool {
    env_truthy(SID_KEEP_TOOL_SCRATCH_ENV) || env_truthy(SID_KEEP_FAILED_TOOL_SCRATCH_ENV)
}

fn env_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

/// Return the transcript file path for a given session directory.
pub fn transcript_path_for_session_dir(root: &StdPath) -> PathBuf {
    root.join(TRANSCRIPT_FILE)
}

/// Read compaction provenance from a session directory's metadata file.
///
/// Returns `Ok(None)` when the session was not created via compaction.
///
/// # Errors
///
/// Returns an error when the metadata file cannot be read or parsed.
pub fn read_compaction_provenance_from_dir(
    root: &StdPath,
) -> Result<Option<CompactionProvenance>, SError> {
    Ok(read_session_metadata(root)?.compacted_from)
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

fn resolve_existing_session_root(sessions_root: &StdPath, spec: &str) -> Result<PathBuf, SError> {
    let direct = PathBuf::from(spec);
    if direct.exists() {
        return normalize_existing_session_root(direct, spec);
    }

    normalize_existing_session_root(sessions_root.join(spec), spec)
}

fn normalize_existing_session_root(candidate: PathBuf, spec: &str) -> Result<PathBuf, SError> {
    let root = if candidate.is_file() {
        if candidate.file_name().and_then(|name| name.to_str()) == Some(SESSION_METADATA_FILE) {
            candidate
                .parent()
                .map(StdPath::to_path_buf)
                .ok_or_else(|| {
                    session_error(
                        "invalid_session_path",
                        "session metadata has no parent directory",
                    )
                    .with_string_field("path", candidate.to_string_lossy().as_ref())
                })?
        } else {
            return Err(session_error(
                "invalid_session_path",
                "resume target must be a session directory",
            )
            .with_string_field("path", candidate.to_string_lossy().as_ref()));
        }
    } else {
        candidate
    };

    if root.is_dir() {
        Ok(root)
    } else {
        Err(
            session_error("session_not_found", "failed to locate session to resume")
                .with_string_field("session", spec),
        )
    }
}

fn read_session_metadata(root: &StdPath) -> Result<SessionMetadata, SError> {
    let path = root.join(SESSION_METADATA_FILE);
    let payload = fs::read_to_string(&path).map_err(|err| {
        session_error("io_error", "failed to read session metadata")
            .with_string_field("path", path.to_string_lossy().as_ref())
            .with_string_field("cause", &err.to_string())
    })?;
    serde_json::from_str(&payload).map_err(|err| {
        session_error("json_parse_error", "failed to parse session metadata")
            .with_string_field("path", path.to_string_lossy().as_ref())
            .with_string_field("cause", &err.to_string())
    })
}

fn load_session_counter_state(
    root: &StdPath,
    tmp_dir: &StdPath,
) -> Result<SessionCounterState, SError> {
    let events = inspect_event_journal(&root.join(EVENTS_JOURNAL_FILE))?;
    let api = inspect_api_journal(&root.join(API_JOURNAL_FILE))?;
    let tmp_tool_seq = max_tool_sequence_in_tmp(tmp_dir)?;
    Ok(SessionCounterState {
        event_entry: events.max_seq,
        api_entry: api.max_seq,
        api_call: api.max_api_seq,
        tool_invocation: events.max_tool_seq.max(tmp_tool_seq),
    })
}

fn inspect_event_journal(path: &StdPath) -> Result<EventJournalState, SError> {
    let mut state = EventJournalState::default();
    scan_jsonl(path, |entry| {
        state.max_seq = state.max_seq.max(u64_field(entry, "seq")?);
        if let Some(tool_seq) = optional_u64_field(entry, "tool_seq")? {
            state.max_tool_seq = state.max_tool_seq.max(tool_seq);
        }
        Ok(())
    })?;
    Ok(state)
}

fn inspect_api_journal(path: &StdPath) -> Result<ApiJournalState, SError> {
    let mut state = ApiJournalState::default();
    scan_jsonl(path, |entry| {
        state.max_seq = state.max_seq.max(u64_field(entry, "seq")?);
        if let Some(api_seq) = optional_u64_field(entry, "api_seq")? {
            state.max_api_seq = state.max_api_seq.max(api_seq);
        }
        Ok(())
    })?;
    Ok(state)
}

fn scan_jsonl(
    path: &StdPath,
    mut visit: impl FnMut(&serde_json::Value) -> Result<(), SError>,
) -> Result<(), SError> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(session_error("io_error", "failed to open session journal")
                .with_string_field("path", path.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string()));
        }
    };
    let reader = BufReader::new(file);
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|err| {
            session_error("io_error", "failed to read session journal")
                .with_string_field("path", path.to_string_lossy().as_ref())
                .with_string_field("line", &(line_no + 1).to_string())
                .with_string_field("cause", &err.to_string())
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let entry = serde_json::from_str::<serde_json::Value>(&line).map_err(|err| {
            session_error("json_parse_error", "failed to parse session journal entry")
                .with_string_field("path", path.to_string_lossy().as_ref())
                .with_string_field("line", &(line_no + 1).to_string())
                .with_string_field("cause", &err.to_string())
        })?;
        visit(&entry)?;
    }
    Ok(())
}

fn u64_field(entry: &serde_json::Value, field: &str) -> Result<u64, SError> {
    entry
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            session_error("invalid_journal_entry", "missing numeric journal field")
                .with_string_field("field", field)
        })
}

fn optional_u64_field(entry: &serde_json::Value, field: &str) -> Result<Option<u64>, SError> {
    match entry.get(field) {
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            session_error("invalid_journal_entry", "journal field must be numeric")
                .with_string_field("field", field)
        }),
        None => Ok(None),
    }
}

fn max_tool_sequence_in_tmp(tmp_dir: &StdPath) -> Result<u64, SError> {
    let mut max_seq = 0;
    let entries = match fs::read_dir(tmp_dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(err) => {
            return Err(
                session_error("io_error", "failed to read session tmp directory")
                    .with_string_field("path", tmp_dir.to_string_lossy().as_ref())
                    .with_string_field("cause", &err.to_string()),
            );
        }
    };
    for entry in entries {
        let entry = entry.map_err(|err| {
            session_error("io_error", "failed to read session tmp directory entry")
                .with_string_field("path", tmp_dir.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())
        })?;
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Some(suffix) = name.strip_prefix("tool-") else {
            continue;
        };
        let Ok(sequence) = suffix.parse::<u64>() else {
            continue;
        };
        max_seq = max_seq.max(sequence);
    }
    Ok(max_seq)
}

fn append_jsonl(path: &StdPath, value: &impl Serialize) -> Result<(), SError> {
    let mut payload = serde_json::to_vec(value).map_err(|err| {
        session_error(
            "json_serialize_error",
            "failed to serialize session journal entry",
        )
        .with_string_field("path", path.to_string_lossy().as_ref())
        .with_string_field("cause", &err.to_string())
    })?;
    payload.push(b'\n');

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| {
            session_error("io_error", "failed to open session journal")
                .with_string_field("path", path.to_string_lossy().as_ref())
                .with_string_field("cause", &err.to_string())
        })?;
    file.write_all(&payload).map_err(|err| {
        session_error("io_error", "failed to append session journal")
            .with_string_field("path", path.to_string_lossy().as_ref())
            .with_string_field("cause", &err.to_string())
    })
}

fn now_session_timestamp() -> SessionTimestamp {
    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    format_session_timestamp(now)
}

fn now_created_at() -> String {
    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    format_session_timestamp(now).created_at
}

fn format_session_timestamp(now: OffsetDateTime) -> SessionTimestamp {
    let offset_seconds = now.offset().whole_seconds();
    let offset_sign = if offset_seconds < 0 { '-' } else { '+' };
    let offset_abs = offset_seconds.abs();
    let offset_hours = offset_abs / 3600;
    let offset_minutes = (offset_abs % 3600) / 60;
    let micros = now.microsecond();
    let month = u8::from(now.month());

    let id = format!(
        "{:04}-{:02}-{:02}T{:02}-{:02}-{:02}.{:06}{}{:02}{:02}",
        now.year(),
        month,
        now.day(),
        now.hour(),
        now.minute(),
        now.second(),
        micros,
        offset_sign,
        offset_hours,
        offset_minutes,
    );
    let created_at = format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}{}{:02}:{:02}",
        now.year(),
        month,
        now.day(),
        now.hour(),
        now.minute(),
        now.second(),
        micros,
        offset_sign,
        offset_hours,
        offset_minutes,
    );
    let created_unix_micros = now.unix_timestamp_nanos() / 1_000;
    SessionTimestamp {
        id,
        created_at,
        created_unix_micros,
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
    use serde_json::json;

    #[test]
    fn session_create_allocates_timestamp_directory() {
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let session = SidSession::create_in(sessions_root.clone()).unwrap();

        assert_timestamp_session_id(session.id());
        assert_eq!(session.root(), &sessions_root.join(session.id()));
        assert!(session.root().join(SESSION_METADATA_FILE).is_file());
        assert!(session.root().join(EVENTS_JOURNAL_FILE).is_file());
        assert!(!session.root().join("api").exists());
        assert!(!session.root().join("tools").exists());
        assert!(session.root().join("tmp/bash").is_dir());

        let metadata: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(session.root().join(SESSION_METADATA_FILE)).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["id"], json!(session.id()));
        assert_timestamp_created_at(metadata["created_at"].as_str().unwrap());
        assert!(metadata["created_unix_micros"].as_i64().unwrap() > 0);
        assert_eq!(metadata["uuid"], serde_json::Value::Null);
        assert_eq!(metadata["pid"], json!(std::process::id()));
        assert!(metadata.get("compacted_from").is_none());

        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn compacted_session_persists_provenance_and_reload() {
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let parent = SidSession::create_in(sessions_root.clone()).unwrap();
        let provenance = CompactionProvenance {
            session_id: parent.id().to_string(),
            session_dir: parent.root().to_string_lossy().into_owned(),
            expert: CompactionExpertConfig {
                agent_id: Some("compact".to_string()),
                model: "claude-sonnet-4-5".to_string(),
                system_prompt: Some("Summarize carefully.".to_string()),
            },
        };

        let child =
            SidSession::create_compacted_in(sessions_root.clone(), provenance.clone()).unwrap();
        assert_eq!(child.compaction_provenance(), Some(&provenance));

        let metadata: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(child.root().join(SESSION_METADATA_FILE)).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["compacted_from"]["session_id"], json!(parent.id()));
        assert_eq!(
            metadata["compacted_from"]["session_dir"],
            json!(parent.root().to_string_lossy().as_ref())
        );
        assert_eq!(
            metadata["compacted_from"]["expert"]["agent_id"],
            json!("compact")
        );

        let resumed = SidSession::resume_in(sessions_root.clone(), child.id()).unwrap();
        assert_eq!(resumed.compaction_provenance(), Some(&provenance));
        assert_eq!(
            read_compaction_provenance_from_dir(child.root()).unwrap(),
            Some(provenance.clone())
        );
        assert_eq!(
            transcript_path_for_session_dir(child.root()),
            child.root().join(TRANSCRIPT_FILE)
        );

        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn api_logging_uses_single_jsonl_journal() {
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let session = SidSession::create_in(sessions_root.clone()).unwrap();

        session
            .log_api_request(&json!({ "messages": ["hello"] }))
            .unwrap();
        session
            .log_api_response(&json!({ "id": "msg_123" }))
            .unwrap();
        session
            .log_api_request(&json!({ "messages": ["retry"] }))
            .unwrap();
        session
            .log_api_response(&json!({ "id": "msg_456" }))
            .unwrap();

        let lines = fs::read_to_string(session.root().join(API_JOURNAL_FILE)).unwrap();
        let entries = lines
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0]["seq"], json!(1));
        assert_eq!(entries[0]["kind"], json!("request"));
        assert_eq!(entries[0]["api_seq"], json!(1));
        assert_eq!(entries[0]["payload"], json!({ "messages": ["hello"] }));
        assert_eq!(entries[1]["seq"], json!(2));
        assert_eq!(entries[1]["kind"], json!("response"));
        assert_eq!(entries[1]["api_seq"], json!(1));
        assert_eq!(entries[1]["payload"], json!({ "id": "msg_123" }));
        assert_eq!(entries[2]["seq"], json!(3));
        assert_eq!(entries[2]["kind"], json!("request"));
        assert_eq!(entries[2]["api_seq"], json!(2));
        assert_eq!(entries[2]["payload"], json!({ "messages": ["retry"] }));
        assert_eq!(entries[3]["seq"], json!(4));
        assert_eq!(entries[3]["kind"], json!("response"));
        assert_eq!(entries[3]["api_seq"], json!(2));
        assert_eq!(entries[3]["payload"], json!({ "id": "msg_456" }));

        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn tool_invocation_dirs_are_ordered_under_session_tmp() {
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let session = SidSession::create_in(sessions_root.clone()).unwrap();

        let first = session.create_tool_invocation_dirs("sidreq_first").unwrap();
        let second = session
            .create_tool_invocation_dirs("sidreq_second")
            .unwrap();

        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert!(first.root.ends_with("tmp/tool-000001"));
        assert!(second.root.ends_with("tmp/tool-000002"));
        assert_eq!(first.scratch_dir, first.root);
        assert_eq!(first.temp_dir, first.root.join("tmp"));

        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn tool_stream_journal_preserves_utf8_and_binary_chunks() {
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let session = SidSession::create_in(sessions_root.clone()).unwrap();
        let journal = session.tool_stream_journal();

        journal.append(1, "stdout", b"hello\n").unwrap();
        journal.append(1, "stderr", &[0, 159, 146, 150]).unwrap();

        let lines = fs::read_to_string(session.root().join(TOOL_STREAMS_JOURNAL_FILE)).unwrap();
        let entries = lines
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(entries[0]["tool_seq"], json!(1));
        assert_eq!(entries[0]["stream"], json!("stdout"));
        assert_eq!(entries[0]["text"], json!("hello\n"));
        assert_eq!(entries[1]["stream"], json!("stderr"));
        assert_eq!(entries[1]["data_b64"], json!("AJ+Slg=="));

        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn session_resume_reuses_session_directory_and_continues_counters() {
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let session = SidSession::create_in(sessions_root.clone()).unwrap();

        session
            .log_api_request(&json!({ "messages": ["hello"] }))
            .unwrap();
        session.log_api_response(&json!({ "id": "msg_1" })).unwrap();
        session.create_tool_invocation_dirs("sidreq_first").unwrap();
        session.write_bash_state("export FOO=bar\n").unwrap();

        let resumed = SidSession::resume_in(sessions_root.clone(), session.id()).unwrap();
        assert_eq!(resumed.root(), session.root());
        assert_eq!(
            resumed.read_bash_state().unwrap(),
            Some("export FOO=bar\n".to_string())
        );

        resumed
            .log_api_request(&json!({ "messages": ["again"] }))
            .unwrap();
        resumed.log_api_response(&json!({ "id": "msg_2" })).unwrap();
        let next_tool = resumed
            .create_tool_invocation_dirs("sidreq_second")
            .unwrap();
        assert_eq!(next_tool.sequence, 2);

        let events = fs::read_to_string(resumed.root().join(EVENTS_JOURNAL_FILE)).unwrap();
        let events = events
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            vec![
                json!({
                    "seq": 1,
                    "kind": "session_start",
                    "session_id": session.id(),
                    "ts": events[0]["ts"].clone(),
                }),
                json!({
                    "seq": 2,
                    "kind": "session_resume",
                    "session_id": session.id(),
                    "ts": events[1]["ts"].clone(),
                }),
            ]
        );

        let api = fs::read_to_string(resumed.root().join(API_JOURNAL_FILE)).unwrap();
        let api = api
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(api[0]["api_seq"], json!(1));
        assert_eq!(api[1]["api_seq"], json!(1));
        assert_eq!(api[2]["api_seq"], json!(2));
        assert_eq!(api[3]["api_seq"], json!(2));
        assert_eq!(api[2]["seq"], json!(3));
        assert_eq!(api[3]["seq"], json!(4));

        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn session_resume_advances_past_stale_tool_directories() {
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let session = SidSession::create_in(sessions_root.clone()).unwrap();

        let first = session.create_tool_invocation_dirs("sidreq_first").unwrap();
        assert_eq!(first.sequence, 1);

        let resumed = SidSession::resume_in(sessions_root.clone(), session.id()).unwrap();
        let second = resumed
            .create_tool_invocation_dirs("sidreq_second")
            .unwrap();
        assert_eq!(second.sequence, 2);

        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn clear_bash_state_removes_snapshot_file() {
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let session = SidSession::create_in(sessions_root.clone()).unwrap();

        session.write_bash_state("export FOO=bar\n").unwrap();
        assert_eq!(
            session.read_bash_state().unwrap(),
            Some("export FOO=bar\n".to_string())
        );

        session.clear_bash_state().unwrap();
        assert_eq!(session.read_bash_state().unwrap(), None);

        fs::remove_dir_all(sessions_root).unwrap();
    }

    fn assert_timestamp_session_id(id: &str) {
        assert_eq!(id.len(), 31);
        assert_eq!(&id[4..5], "-");
        assert_eq!(&id[7..8], "-");
        assert_eq!(&id[10..11], "T");
        assert_eq!(&id[13..14], "-");
        assert_eq!(&id[16..17], "-");
        assert_eq!(&id[19..20], ".");
        assert!(matches!(&id[26..27], "+" | "-"));
        assert!(id[..4].chars().all(|ch| ch.is_ascii_digit()));
        assert!(id[27..31].chars().all(|ch| ch.is_ascii_digit()));
    }

    fn assert_timestamp_created_at(value: &str) {
        assert_eq!(value.len(), 32);
        assert_eq!(&value[4..5], "-");
        assert_eq!(&value[7..8], "-");
        assert_eq!(&value[10..11], "T");
        assert_eq!(&value[13..14], ":");
        assert_eq!(&value[16..17], ":");
        assert_eq!(&value[19..20], ".");
        assert!(matches!(&value[26..27], "+" | "-"));
        assert_eq!(&value[29..30], ":");
    }
}
