//! Per-run journal: steps.jsonl, the suggestions ledger, stdin caps, and the
//! parent-transcript linkage.
//!
//! State lives in files, git, and journals — never in any single context
//! window.  Every agent and judge step appends one record to
//! `${RUN_DIR}/steps.jsonl`; `--resume` replays those records to the last
//! completed step and re-executes the in-flight one.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::verdict::Finding;

/// Head bytes preserved when capping piped context.
pub const STDIN_CAP_HEAD: usize = 64 * 1024;
/// Tail bytes preserved when capping piped context.
pub const STDIN_CAP_TAIL: usize = 16 * 1024;

/// One record per loop step in `steps.jsonl`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StepRecord {
    /// Marks the start (or resumption) of a run.
    RunStart {
        /// The run id.
        run_id: String,
        /// SHA-256-free cheap fingerprint of the script text (length + first line).
        script_fingerprint: String,
    },
    /// A fresh agent invocation.
    Agent {
        /// Monotonic step number within the run.
        step: u64,
        /// agents.conf service name.
        service: String,
        /// Exit code per the protocol (§3).
        exit: i32,
        /// Tokens consumed by the child session.
        tokens: u64,
        /// Child session id, when one was created.
        session: Option<String>,
        /// Checkpoint ref recorded before the invocation.
        checkpoint: Option<String>,
    },
    /// A judge invocation (after jury/soak gating).
    Judge {
        /// Monotonic step number within the run.
        step: u64,
        /// agents.conf service name.
        service: String,
        /// Exit code per the protocol (§3).
        exit: i32,
        /// Tokens consumed across samples.
        tokens: u64,
        /// The gated verdict boolean, when a verdict was rendered.
        sufficient: Option<bool>,
        /// Soak counter after this step.
        soak: u32,
        /// The verdict summary, when a verdict was rendered.
        summary: Option<String>,
        /// The rendered markdown work order (stdout of the builtin).
        rendered: String,
    },
}

impl StepRecord {
    /// The step number, when the record represents an agent/judge step.
    pub fn step(&self) -> Option<u64> {
        match self {
            StepRecord::RunStart { .. } => None,
            StepRecord::Agent { step, .. } | StepRecord::Judge { step, .. } => Some(*step),
        }
    }

    /// The exit code, when the record represents an agent/judge step.
    pub fn exit(&self) -> Option<i32> {
        match self {
            StepRecord::RunStart { .. } => None,
            StepRecord::Agent { exit, .. } | StepRecord::Judge { exit, .. } => Some(*exit),
        }
    }
}

/// Append-only steps journal backed by `${RUN_DIR}/steps.jsonl`.
#[derive(Debug)]
pub struct StepsJournal {
    path: PathBuf,
}

impl StepsJournal {
    /// A journal stored at `run_dir/steps.jsonl`.
    pub fn new(run_dir: &Path) -> StepsJournal {
        StepsJournal {
            path: run_dir.join("steps.jsonl"),
        }
    }

    /// The journal's path on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one record, fsync-free but flushed.
    pub fn append(&self, record: &StepRecord) -> Result<(), String> {
        let line = serde_json::to_string(record)
            .map_err(|err| format!("failed to serialize step record: {err}"))?;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|err| format!("failed to open {}: {err}", self.path.display()))?;
        writeln!(file, "{line}").map_err(|err| format!("failed to append step record: {err}"))?;
        Ok(())
    }

    /// Load every parseable record.  A trailing partial line (a step that was
    /// in flight at interrupt) is ignored, which is exactly the resume
    /// semantics: replay to the last completed step.
    pub fn load(&self) -> Result<Vec<StepRecord>, String> {
        let text = match fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(format!("failed to read {}: {err}", self.path.display())),
        };
        let mut records = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<StepRecord>(line) {
                Ok(record) => records.push(record),
                Err(_) => break,
            }
        }
        Ok(records)
    }
}

/// The per-run suggestions ledger at `${RUN_DIR}/suggestions.md`.
///
/// Suggestion-severity findings on a passing verdict accrue here and are
/// injected into the judge's next-pass context.  The ledger is scoped to the
/// run and dies with it.
#[derive(Debug)]
pub struct SuggestionsLedger {
    path: PathBuf,
}

impl SuggestionsLedger {
    /// A ledger stored at `run_dir/suggestions.md`.
    pub fn new(run_dir: &Path) -> SuggestionsLedger {
        SuggestionsLedger {
            path: run_dir.join("suggestions.md"),
        }
    }

    /// The ledger's path on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append suggestions from a passing verdict, attributed to a step.
    pub fn append(&self, step: u64, suggestions: &[&Finding]) -> Result<(), String> {
        if suggestions.is_empty() {
            return Ok(());
        }
        let mut entry = String::new();
        for finding in suggestions {
            entry.push_str(&format!(
                "- (step {step}) {} — {} ({})\n",
                finding.where_, finding.what, finding.why
            ));
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|err| format!("failed to open {}: {err}", self.path.display()))?;
        file.write_all(entry.as_bytes())
            .map_err(|err| format!("failed to append suggestions: {err}"))?;
        Ok(())
    }

    /// The ledger's contents, or empty when nothing has accrued.
    pub fn read(&self) -> String {
        fs::read_to_string(&self.path).unwrap_or_default()
    }

    /// Number of ledger entries (lines).
    pub fn entry_count(&self) -> usize {
        self.read().lines().filter(|l| !l.trim().is_empty()).count()
    }
}

/// The result of capping piped context.
#[derive(Clone, Debug, PartialEq)]
pub struct CappedContext {
    /// The capped text handed to the agent.
    pub text: String,
    /// Whether the input was truncated.
    pub truncated: bool,
}

/// Cap piped context at 64 KiB head + 16 KiB tail with a truncation marker
/// pointing at the full log on disk.  Splits respect UTF-8 boundaries.
pub fn cap_context(input: &str, full_log_path: &str) -> CappedContext {
    if input.len() <= STDIN_CAP_HEAD + STDIN_CAP_TAIL {
        return CappedContext {
            text: input.to_string(),
            truncated: false,
        };
    }
    let head_end = floor_char_boundary(input, STDIN_CAP_HEAD);
    let tail_start = ceil_char_boundary(input, input.len() - STDIN_CAP_TAIL);
    let text = format!(
        "{}\n[truncated: full log at {}]\n{}",
        &input[..head_end],
        full_log_path,
        &input[tail_start..]
    );
    CappedContext {
        text,
        truncated: true,
    }
}

fn floor_char_boundary(s: &str, mut index: usize) -> usize {
    while index > 0 && !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn ceil_char_boundary(s: &str, mut index: usize) -> usize {
    while index < s.len() && !s.is_char_boundary(index) {
        index += 1;
    }
    index
}

/// Generate a run id in the `2026-06-10T14-22-07` shape, with a numeric
/// suffix on collision within the runs root.
pub fn generate_run_id(runs_root: &Path) -> String {
    let now = time::OffsetDateTime::now_utc();
    let base = format!(
        "{:04}-{:02}-{:02}T{:02}-{:02}-{:02}",
        now.year(),
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    );
    let mut candidate = base.clone();
    let mut suffix = 1;
    while runs_root.join(&candidate).exists() {
        candidate = format!("{base}.{suffix}");
        suffix += 1;
    }
    candidate
}

/// Aggregate statistics for the parent-linkage report.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunReport {
    /// The run id.
    pub run_id: String,
    /// The run's journal directory.
    pub run_dir: PathBuf,
    /// Run-level exit status.
    pub exit: i32,
    /// Total agent invocations (the loop's iterations).
    pub iterations: u64,
    /// Per-service agent invocation counts, in first-seen order.
    pub agent_counts: Vec<(String, u64)>,
    /// The final verdict's summary, when a judge rendered one.
    pub final_verdict_summary: Option<String>,
    /// Soak progress of the final judge step, when soaking: (passes, target).
    pub final_soak: Option<(u32, u32)>,
    /// Number of suggestions-ledger entries at run end.
    pub suggestions_entries: usize,
    /// Whether the run was interrupted by SIGINT.
    pub interrupted: bool,
}

impl RunReport {
    /// Render the single synthetic paste-style turn injected into the parent
    /// transcript (§5).
    pub fn parent_summary(&self, script: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!("Ran /run {script} (run {}).\n", self.run_id));
        let counts = if self.agent_counts.is_empty() {
            "no agent invocations".to_string()
        } else {
            self.agent_counts
                .iter()
                .map(|(service, count)| format!("{count} {service}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut status_line = format!(
            "Exit {} after {} iterations ({counts})",
            self.exit, self.iterations
        );
        if let Some((passes, target)) = self.final_soak {
            status_line.push_str(&format!(", judge passed soak {passes}/{target}"));
        }
        if self.interrupted {
            status_line.push_str(", interrupted by SIGINT");
        }
        status_line.push_str(".\n");
        out.push_str(&status_line);
        if let Some(summary) = self.final_verdict_summary.as_ref() {
            out.push_str(&format!("Final verdict: {summary:?}\n"));
        }
        out.push_str(&format!("Journal: {}/\n", self.run_dir.display()));
        out.push_str(&format!(
            "Suggestions ledger: {} entries.\n",
            self.suggestions_entries
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::verdict::Severity;
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sid-ralph-journal-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn journal_round_trips_records() {
        let dir = temp_dir("roundtrip");
        let journal = StepsJournal::new(&dir);
        let records = vec![
            StepRecord::RunStart {
                run_id: "r1".to_string(),
                script_fingerprint: "f".to_string(),
            },
            StepRecord::Agent {
                step: 1,
                service: "fix".to_string(),
                exit: 0,
                tokens: 100,
                session: Some("s1".to_string()),
                checkpoint: Some("refs/sid/ralph/r1/1".to_string()),
            },
            StepRecord::Judge {
                step: 2,
                service: "judge".to_string(),
                exit: 1,
                tokens: 50,
                sufficient: Some(false),
                soak: 0,
                summary: Some("not done".to_string()),
                rendered: "# Verdict: insufficient\n".to_string(),
            },
        ];
        for record in &records {
            journal.append(record).unwrap();
        }
        assert_eq!(journal.load().unwrap(), records);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn journal_ignores_trailing_partial_line() {
        let dir = temp_dir("partial");
        let journal = StepsJournal::new(&dir);
        journal
            .append(&StepRecord::Agent {
                step: 1,
                service: "fix".to_string(),
                exit: 0,
                tokens: 0,
                session: None,
                checkpoint: None,
            })
            .unwrap();
        // Simulate an in-flight step at interrupt: a torn write.
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(journal.path())
            .unwrap();
        file.write_all(b"{\"kind\":\"judge\",\"step\":2,").unwrap();
        drop(file);
        let records = journal.load().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].step(), Some(1));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn journal_load_missing_file_is_empty() {
        let dir = temp_dir("missing");
        let journal = StepsJournal::new(&dir);
        assert!(journal.load().unwrap().is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ledger_appends_and_counts() {
        let dir = temp_dir("ledger");
        let ledger = SuggestionsLedger::new(&dir);
        assert_eq!(ledger.entry_count(), 0);
        let finding = Finding {
            severity: Severity::Suggestion,
            where_: "README.md".to_string(),
            what: "Mention the soak flag".to_string(),
            why: "Operators will want it".to_string(),
        };
        ledger.append(3, &[&finding]).unwrap();
        ledger.append(5, &[&finding]).unwrap();
        ledger.append(6, &[]).unwrap();
        assert_eq!(ledger.entry_count(), 2);
        let text = ledger.read();
        assert!(text.contains("(step 3) README.md — Mention the soak flag"));
        assert!(text.contains("(step 5)"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cap_context_passes_small_input_through() {
        let capped = cap_context("hello", "/run/ci-001.log");
        assert_eq!(capped.text, "hello");
        assert!(!capped.truncated);
    }

    #[test]
    fn cap_context_truncates_large_input() {
        let big = "a".repeat(10 * 1024 * 1024);
        let capped = cap_context(&big, "/run/ci-001.log");
        assert!(capped.truncated);
        assert!(
            capped
                .text
                .contains("[truncated: full log at /run/ci-001.log]")
        );
        // head + tail + marker, give or take the marker line.
        assert!(capped.text.len() < STDIN_CAP_HEAD + STDIN_CAP_TAIL + 128);
        assert!(capped.text.starts_with(&"a".repeat(100)));
        assert!(capped.text.ends_with(&"a".repeat(100)));
    }

    #[test]
    fn cap_context_respects_utf8_boundaries() {
        let big = "é".repeat((STDIN_CAP_HEAD + STDIN_CAP_TAIL) / 2 + 1024);
        let capped = cap_context(&big, "/run/ci-002.log");
        assert!(capped.truncated);
        // Must not panic and must remain valid UTF-8 (guaranteed by String).
        assert!(
            capped
                .text
                .contains("[truncated: full log at /run/ci-002.log]")
        );
    }

    #[test]
    fn run_id_shape_and_collisions() {
        let dir = temp_dir("runid");
        let id = generate_run_id(&dir);
        assert_eq!(id.len(), "2026-06-10T14-22-07".len());
        assert!(id.contains('T'));
        fs::create_dir_all(dir.join(&id)).unwrap();
        let second = generate_run_id(&dir);
        assert_ne!(id, second);
        assert!(second.starts_with(&id[..11]));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn parent_summary_matches_plan_shape() {
        let report = RunReport {
            run_id: "2026-06-10T14-22-07".to_string(),
            run_dir: PathBuf::from("/sessions/abc/runs/2026-06-10T14-22-07"),
            exit: 0,
            iterations: 4,
            agent_counts: vec![("fix".to_string(), 2), ("task".to_string(), 2)],
            final_verdict_summary: Some("The plan is complete.".to_string()),
            final_soak: Some((5, 5)),
            suggestions_entries: 3,
            interrupted: false,
        };
        let text = report.parent_summary("ralph.sid");
        assert!(text.starts_with("Ran /run ralph.sid (run 2026-06-10T14-22-07).\n"));
        assert!(
            text.contains("Exit 0 after 4 iterations (2 fix, 2 task), judge passed soak 5/5.\n")
        );
        assert!(text.contains("Final verdict: \"The plan is complete.\"\n"));
        assert!(text.contains("Journal: /sessions/abc/runs/2026-06-10T14-22-07/\n"));
        assert!(text.contains("Suggestions ledger: 3 entries.\n"));
    }

    #[test]
    fn parent_summary_notes_interrupt() {
        let report = RunReport {
            run_id: "r".to_string(),
            run_dir: PathBuf::from("/r"),
            exit: 130,
            iterations: 1,
            agent_counts: vec![("fix".to_string(), 1)],
            final_verdict_summary: None,
            final_soak: None,
            suggestions_entries: 0,
            interrupted: true,
        };
        let text = report.parent_summary("ralph.sid");
        assert!(text.contains("interrupted by SIGINT"));
        assert!(!text.contains("Final verdict"));
    }
}
