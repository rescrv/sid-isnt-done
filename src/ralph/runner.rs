//! The ralph runner: an embedded mxsh script with `agent` and `judge` bound,
//! a control socket the builtins call back into, and the per-run journal.
//!
//! `agent` and `judge` appear to the script as ordinary commands (so pipes
//! and redirections behave exactly like POSIX), implemented by a tiny shim
//! (`sid-ralph-shim`) that forwards argv plus stdin over a unix socket to the
//! in-process [`RunnerCore`].  The shim prints whatever the core says and
//! exits with the protocol's exit code.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::args::{AgentArgs, JudgeArgs, JudgeMode};
use super::checkpoint::create_checkpoint;
use super::journal::{RunReport, StepRecord, StepsJournal, SuggestionsLedger, cap_context};
use super::verdict::Verdict;
use super::{EXIT_ESCALATED, EXIT_INSUFFICIENT, EXIT_OK, EXIT_SIGINT, EXIT_TRANSPORT};

/// Environment variable carrying the control spool directory to the shim.
pub const CONTROL_DIR_ENV: &str = "RALPH_CONTROL_DIR";
/// Environment variable carrying the run directory into the script.
pub const RUN_DIR_ENV: &str = "RUN_DIR";
/// Environment variable carrying the run id into the script.
pub const RUN_ID_ENV: &str = "RALPH_RUN_ID";
/// Environment variable overriding the shim binary location.
pub const SHIM_PATH_ENV: &str = "SID_RALPH_SHIM";

/// Request sent by the shim over the control socket.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ShimRequest {
    /// The builtin name: `agent` or `judge` (the shim's argv[0] basename).
    pub name: String,
    /// argv[1..] of the builtin.
    pub args: Vec<String>,
    /// Everything piped to the builtin's stdin (lossy UTF-8).
    pub context: String,
}

/// Response returned to the shim.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ShimResponse {
    /// Exit code per the protocol (§3).
    pub exit: i32,
    /// Bytes for the shim's stdout (the rendered verdict for `judge`).
    pub stdout: String,
    /// Bytes for the shim's stderr (short runner-level diagnostics).
    pub stderr: String,
}

impl ShimResponse {
    fn err(exit: i32, message: impl Into<String>) -> ShimResponse {
        ShimResponse {
            exit,
            stdout: String::new(),
            stderr: message.into(),
        }
    }
}

/// What a fresh agent call produced.
#[derive(Clone, Debug, PartialEq)]
pub enum AgentOutcome {
    /// The child session ran to completion (says nothing about task success).
    Completed,
    /// The agent called `escalate(reason)` — it wants a human.
    Escalated(String),
    /// API error, config error, or another transport-class failure.
    Transport(String),
    /// The run was interrupted while the call was in flight.
    Interrupted,
}

/// The result of one fresh agent invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentCallResult {
    /// What happened.
    pub outcome: AgentOutcome,
    /// Tokens consumed by the child session.
    pub tokens: u64,
    /// Child session id, when one was created.
    pub session: Option<String>,
}

/// What one judge sample produced.
#[derive(Clone, Debug, PartialEq)]
pub enum JudgeOutcome {
    /// A validated verdict.
    Verdict(Verdict),
    /// The judge called `escalate(reason)`.
    Escalated(String),
    /// Transport/config/malformed-verdict failure — never conflated with a
    /// verdict.
    Transport(String),
    /// The run was interrupted while the sample was in flight.
    Interrupted,
}

/// The result of one judge sample.
#[derive(Clone, Debug, PartialEq)]
pub struct JudgeCallResult {
    /// What happened.
    pub outcome: JudgeOutcome,
    /// Tokens consumed by the sample.
    pub tokens: u64,
}

/// One fresh agent invocation, as handed to the host.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentInvocation {
    /// agents.conf service name.
    pub service: String,
    /// The instruction from argv (empty when none given).
    pub instruction: String,
    /// Capped piped context.
    pub context: String,
    /// The step number for journaling.
    pub step: u64,
}

/// One judge sample, as handed to the host.
#[derive(Clone, Debug, PartialEq)]
pub struct JudgeInvocation {
    /// agents.conf service name.
    pub service: String,
    /// The fully assembled prompt for this sample.
    pub prompt: String,
    /// Truncate the pinned transcript to its seed before this sample.
    pub goldfish: bool,
    /// Seeding policy (relevant to the first sample only).
    pub seed: super::args::SeedMode,
    /// The step number for journaling.
    pub step: u64,
}

/// The two LLM roles, as seen by the runner.  Production hosts spawn sid
/// child sessions; tests use scripted stubs.  No control flow lives here:
/// the runner owns gating, journaling, checkpoints, and exit codes.
pub trait RalphHost: Send {
    /// Fail fast (exit ≥ 4, before any API call) when `service` is not
    /// configured to render verdicts (its `_TOOLS` lacks `verdict`).
    fn validate_judge(&mut self, service: &str) -> Result<(), String>;

    /// Run a fresh agent: new child session, dies at call end.
    fn run_agent(&mut self, invocation: &AgentInvocation) -> AgentCallResult;

    /// Run one sample against the pinned judge session, creating and seeding
    /// it on first use.
    fn judge_sample(&mut self, invocation: &JudgeInvocation) -> JudgeCallResult;
}

/// Observer for stdout/stderr emitted by the embedded mxsh script itself.
pub trait ScriptOutputSink: Send + Sync {
    /// Forward one chunk from `stream`, either `"stdout"` or `"stderr"`.
    fn on_script_output(&self, stream: &str, data: &[u8]);
}

/// Options governing a run.
#[derive(Clone, Debug)]
pub struct RunnerOptions {
    /// The run id (journal directory name).
    pub run_id: String,
    /// The run's journal directory.
    pub run_dir: PathBuf,
    /// Workspace root for git checkpoints; `None` disables checkpointing.
    pub workspace_root: Option<PathBuf>,
    /// Cap on agent invocations (loop iterations).
    pub max_iters: Option<u64>,
    /// Cap on total tokens across all child sessions.
    pub budget_tokens: Option<u64>,
    /// Replay the journal to the last completed step before going live.
    pub resume: bool,
}

/// The runner's mutable state, shared between the socket server and the
/// report assembly.  All gating decisions (§2, §3) happen here.
pub struct RunnerCore {
    host: Box<dyn RalphHost>,
    options: RunnerOptions,
    journal: StepsJournal,
    ledger: SuggestionsLedger,
    interrupted: Arc<AtomicBool>,
    replay: VecDeque<StepRecord>,
    step: u64,
    iterations: u64,
    agent_counts: Vec<(String, u64)>,
    tokens_used: u64,
    soak_counters: HashMap<String, u32>,
    validated_judges: HashSet<String>,
    last_checkpoint: Option<String>,
    final_verdict_summary: Option<String>,
    final_soak: Option<(u32, u32)>,
}

impl RunnerCore {
    /// Build a core over a host.  When `options.resume` is set, the journal
    /// is loaded and completed steps will be replayed instead of re-invoked.
    pub fn new(
        host: Box<dyn RalphHost>,
        options: RunnerOptions,
        interrupted: Arc<AtomicBool>,
    ) -> Result<RunnerCore, String> {
        fs::create_dir_all(&options.run_dir)
            .map_err(|err| format!("failed to create run dir: {err}"))?;
        let journal = StepsJournal::new(&options.run_dir);
        let ledger = SuggestionsLedger::new(&options.run_dir);
        let mut replay = VecDeque::new();
        if options.resume {
            for record in journal.load()? {
                if !matches!(record, StepRecord::RunStart { .. }) {
                    replay.push_back(record);
                }
            }
        }
        Ok(RunnerCore {
            host,
            options,
            journal,
            ledger,
            interrupted,
            replay,
            step: 0,
            iterations: 0,
            agent_counts: Vec::new(),
            tokens_used: 0,
            soak_counters: HashMap::new(),
            validated_judges: HashSet::new(),
            last_checkpoint: None,
            final_verdict_summary: None,
            final_soak: None,
        })
    }

    /// Record the start of a run in the journal.
    pub fn record_run_start(&mut self, script_text: &str) -> Result<(), String> {
        let first_line = script_text.lines().next().unwrap_or("");
        self.journal.append(&StepRecord::RunStart {
            run_id: self.options.run_id.clone(),
            script_fingerprint: format!("len={};first={first_line}", script_text.len()),
        })
    }

    /// Dispatch one shim request.
    pub fn handle(&mut self, request: &ShimRequest) -> ShimResponse {
        match request.name.as_str() {
            "agent" => self.handle_agent(&request.args, &request.context),
            "judge" => self.handle_judge(&request.args, &request.context),
            other => ShimResponse::err(EXIT_TRANSPORT, format!("ralph: unknown builtin {other:?}")),
        }
    }

    fn handle_agent(&mut self, args: &[String], context: &str) -> ShimResponse {
        let parsed = match AgentArgs::parse(args) {
            Ok(parsed) => parsed,
            Err(err) => return ShimResponse::err(EXIT_TRANSPORT, format!("ralph: {err}")),
        };

        if let Some(record) = self.try_replay_agent(&parsed.service) {
            return record;
        }

        if self.interrupted.load(Ordering::Relaxed) {
            return ShimResponse::err(EXIT_SIGINT, "ralph: interrupted");
        }
        if let Some(max) = self.options.max_iters
            && self.iterations >= max
        {
            return ShimResponse::err(
                EXIT_TRANSPORT,
                format!("ralph: --max-iters {max} exhausted"),
            );
        }
        if let Some(budget) = self.options.budget_tokens
            && self.tokens_used >= budget
        {
            return ShimResponse::err(
                EXIT_TRANSPORT,
                format!("ralph: --budget {budget} tokens exhausted"),
            );
        }

        self.step += 1;
        let step = self.step;
        let context = self.capped_context(context, step);
        let checkpoint = self.take_checkpoint(step);

        let invocation = AgentInvocation {
            service: parsed.service.clone(),
            instruction: parsed.instruction.clone().unwrap_or_default(),
            context,
            step,
        };
        let result = self.host.run_agent(&invocation);
        self.iterations += 1;
        self.bump_agent_count(&parsed.service);
        self.tokens_used = self.tokens_used.saturating_add(result.tokens);

        let (exit, stderr) = match &result.outcome {
            AgentOutcome::Completed => (EXIT_OK, String::new()),
            AgentOutcome::Escalated(reason) => {
                (EXIT_ESCALATED, format!("ralph: agent escalated: {reason}"))
            }
            AgentOutcome::Transport(reason) => (
                EXIT_TRANSPORT,
                format!("ralph: transport failure: {reason}"),
            ),
            AgentOutcome::Interrupted => (EXIT_SIGINT, "ralph: interrupted".to_string()),
        };

        let record = StepRecord::Agent {
            step,
            service: parsed.service.clone(),
            exit,
            tokens: result.tokens,
            session: result.session.clone(),
            checkpoint,
        };
        if let Err(err) = self.journal.append(&record) {
            return ShimResponse::err(EXIT_TRANSPORT, format!("ralph: journal failure: {err}"));
        }
        ShimResponse {
            exit,
            stdout: String::new(),
            stderr,
        }
    }

    fn handle_judge(&mut self, args: &[String], context: &str) -> ShimResponse {
        let parsed = match JudgeArgs::parse(args) {
            Ok(parsed) => parsed,
            Err(err) => return ShimResponse::err(EXIT_TRANSPORT, format!("ralph: {err}")),
        };

        if let Some(record) = self.try_replay_judge(&parsed) {
            return record;
        }

        if self.interrupted.load(Ordering::Relaxed) {
            return ShimResponse::err(EXIT_SIGINT, "ralph: interrupted");
        }

        // Config error before any API call.
        if !self.validated_judges.contains(&parsed.service) {
            if let Err(err) = self.host.validate_judge(&parsed.service) {
                return ShimResponse::err(EXIT_TRANSPORT, format!("ralph: {err}"));
            }
            self.validated_judges.insert(parsed.service.clone());
        }
        if let Some(budget) = self.options.budget_tokens
            && self.tokens_used >= budget
        {
            return ShimResponse::err(
                EXIT_TRANSPORT,
                format!("ralph: --budget {budget} tokens exhausted"),
            );
        }

        self.step += 1;
        let step = self.step;
        let context = self.capped_context(context, step);
        let soak_state = match parsed.mode {
            JudgeMode::Soak(target) => Some((
                *self.soak_counters.get(&parsed.service).unwrap_or(&0),
                target,
            )),
            _ => None,
        };
        let prompt = assemble_judge_prompt(
            parsed.instruction.as_deref(),
            &context,
            soak_state,
            self.last_checkpoint.as_deref(),
            &self.ledger.read(),
        );

        let (samples, goldfish) = match parsed.mode {
            JudgeMode::Jury(n) => (n, true),
            _ => (1, parsed.goldfish),
        };

        let mut verdicts: Vec<Verdict> = Vec::new();
        let mut tokens = 0u64;
        let mut failure: Option<(i32, String)> = None;
        for _ in 0..samples {
            let invocation = JudgeInvocation {
                service: parsed.service.clone(),
                prompt: prompt.clone(),
                goldfish,
                seed: parsed.seed,
                step,
            };
            let result = self.host.judge_sample(&invocation);
            tokens = tokens.saturating_add(result.tokens);
            match result.outcome {
                JudgeOutcome::Verdict(verdict) => {
                    if let Err(err) = verdict.validate() {
                        failure = Some((EXIT_TRANSPORT, format!("ralph: {err}")));
                        break;
                    }
                    verdicts.push(verdict);
                }
                JudgeOutcome::Escalated(reason) => {
                    failure = Some((EXIT_ESCALATED, format!("ralph: judge escalated: {reason}")));
                    break;
                }
                JudgeOutcome::Transport(reason) => {
                    failure = Some((
                        EXIT_TRANSPORT,
                        format!("ralph: judge malfunction: {reason}"),
                    ));
                    break;
                }
                JudgeOutcome::Interrupted => {
                    failure = Some((EXIT_SIGINT, "ralph: interrupted".to_string()));
                    break;
                }
            }
        }
        self.tokens_used = self.tokens_used.saturating_add(tokens);

        if let Some((exit, stderr)) = failure {
            let record = StepRecord::Judge {
                step,
                service: parsed.service.clone(),
                exit,
                tokens,
                sufficient: None,
                soak: *self.soak_counters.get(&parsed.service).unwrap_or(&0),
                summary: None,
                rendered: String::new(),
            };
            if let Err(err) = self.journal.append(&record) {
                return ShimResponse::err(EXIT_TRANSPORT, format!("ralph: journal failure: {err}"));
            }
            return ShimResponse::err(exit, stderr);
        }

        let all_pass = verdicts
            .iter()
            .all(|v| v.effective_sufficient(parsed.pedantic));

        // Passing verdicts shed their suggestions into the per-run ledger.
        for verdict in &verdicts {
            if verdict.effective_sufficient(parsed.pedantic) {
                let suggestions = verdict.suggestions();
                if let Err(err) = self.ledger.append(step, &suggestions) {
                    return ShimResponse::err(
                        EXIT_TRANSPORT,
                        format!("ralph: ledger failure: {err}"),
                    );
                }
            }
        }

        let mut soak_now = 0u32;
        let exit = match parsed.mode {
            JudgeMode::Single => {
                if all_pass {
                    EXIT_OK
                } else {
                    EXIT_INSUFFICIENT
                }
            }
            JudgeMode::Jury(_) => {
                if all_pass {
                    EXIT_OK
                } else {
                    EXIT_INSUFFICIENT
                }
            }
            JudgeMode::Soak(target) => {
                let counter = self
                    .soak_counters
                    .entry(parsed.service.clone())
                    .or_insert(0);
                if all_pass {
                    *counter += 1;
                } else {
                    *counter = 0;
                }
                soak_now = *counter;
                self.final_soak = Some((soak_now, target));
                if soak_now >= target {
                    EXIT_OK
                } else {
                    EXIT_INSUFFICIENT
                }
            }
        };

        let mut rendered = String::new();
        for (i, verdict) in verdicts.iter().enumerate() {
            if verdicts.len() > 1 {
                rendered.push_str(&format!("<!-- juror {}/{} -->\n", i + 1, verdicts.len()));
            }
            rendered.push_str(&verdict.render_markdown());
            if i + 1 < verdicts.len() {
                rendered.push('\n');
            }
        }
        match parsed.mode {
            JudgeMode::Soak(target) => {
                rendered.push_str(&format!(
                    "\nSoak: {soak_now}/{target} consecutive passes.\n"
                ));
            }
            JudgeMode::Jury(n) => {
                let passed = verdicts
                    .iter()
                    .filter(|v| v.effective_sufficient(parsed.pedantic))
                    .count();
                rendered.push_str(&format!("\nJury: {passed}/{n} jurors passed.\n"));
            }
            JudgeMode::Single => {}
        }

        let summary = verdicts.last().map(|v| v.summary.clone());
        self.final_verdict_summary = summary.clone();

        let record = StepRecord::Judge {
            step,
            service: parsed.service.clone(),
            exit,
            tokens,
            sufficient: Some(all_pass),
            soak: soak_now,
            summary,
            rendered: rendered.clone(),
        };
        if let Err(err) = self.journal.append(&record) {
            return ShimResponse::err(EXIT_TRANSPORT, format!("ralph: journal failure: {err}"));
        }

        ShimResponse {
            exit,
            stdout: rendered,
            stderr: String::new(),
        }
    }

    fn try_replay_agent(&mut self, service: &str) -> Option<ShimResponse> {
        match self.replay.front() {
            Some(StepRecord::Agent {
                service: recorded, ..
            }) if recorded == service => {}
            Some(_) => {
                // The script diverged from the journal: go live from here.
                self.replay.clear();
                return None;
            }
            None => return None,
        }
        let Some(StepRecord::Agent {
            step,
            service,
            exit,
            checkpoint,
            ..
        }) = self.replay.pop_front()
        else {
            unreachable!("front was just matched as an agent record");
        };
        self.step = step;
        self.iterations += 1;
        self.bump_agent_count(&service);
        if checkpoint.is_some() {
            self.last_checkpoint = checkpoint;
        }
        Some(ShimResponse {
            exit,
            stdout: String::new(),
            stderr: format!("ralph: replayed step {step} (agent {service})"),
        })
    }

    fn try_replay_judge(&mut self, parsed: &JudgeArgs) -> Option<ShimResponse> {
        match self.replay.front() {
            Some(StepRecord::Judge {
                service: recorded, ..
            }) if *recorded == parsed.service => {}
            Some(_) => {
                self.replay.clear();
                return None;
            }
            None => return None,
        }
        let Some(StepRecord::Judge {
            step,
            service,
            exit,
            soak,
            summary,
            rendered,
            ..
        }) = self.replay.pop_front()
        else {
            unreachable!("front was just matched as a judge record");
        };
        self.step = step;
        self.soak_counters.insert(service.clone(), soak);
        if let JudgeMode::Soak(target) = parsed.mode {
            self.final_soak = Some((soak, target));
        }
        if summary.is_some() {
            self.final_verdict_summary = summary;
        }
        Some(ShimResponse {
            exit,
            stdout: rendered,
            stderr: format!("ralph: replayed step {step} (judge {service})"),
        })
    }

    fn capped_context(&self, context: &str, step: u64) -> String {
        let full_log = self.options.run_dir.join(format!("ci-{step:03}.log"));
        let capped = cap_context(context, &full_log.to_string_lossy());
        if capped.truncated {
            // Best effort: the capped marker points here.
            let _ = fs::write(&full_log, context);
        }
        capped.text
    }

    fn take_checkpoint(&mut self, step: u64) -> Option<String> {
        let workspace_root = self.options.workspace_root.as_ref()?;
        match create_checkpoint(
            workspace_root,
            &self.options.run_dir,
            &self.options.run_id,
            step,
        ) {
            Ok(reference) => {
                if reference.is_some() {
                    self.last_checkpoint = reference.clone();
                }
                reference
            }
            Err(err) => {
                eprintln!("ralph: checkpoint failed (continuing): {err}");
                None
            }
        }
    }

    fn bump_agent_count(&mut self, service: &str) {
        for (existing, count) in &mut self.agent_counts {
            if existing == service {
                *count += 1;
                return;
            }
        }
        self.agent_counts.push((service.to_string(), 1));
    }

    /// Assemble the final report once the script has exited.
    pub fn report(&self, exit: i32) -> RunReport {
        let interrupted = self.interrupted.load(Ordering::Relaxed);
        RunReport {
            run_id: self.options.run_id.clone(),
            run_dir: self.options.run_dir.clone(),
            exit: if interrupted { EXIT_SIGINT } else { exit },
            iterations: self.iterations,
            agent_counts: self.agent_counts.clone(),
            final_verdict_summary: self.final_verdict_summary.clone(),
            final_soak: self.final_soak,
            suggestions_entries: self.ledger.entry_count(),
            interrupted,
        }
    }
}

/// Assemble the judge prompt: instruction, capped context, soak note,
/// previous-checkpoint ref, suggestions ledger, and the verdict mandate.
pub fn assemble_judge_prompt(
    instruction: Option<&str>,
    context: &str,
    soak: Option<(u32, u32)>,
    prior_checkpoint: Option<&str>,
    suggestions: &str,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(instruction.unwrap_or("Render your verdict on the current state of the work."));
    prompt.push('\n');
    if let Some((passes, target)) = soak {
        prompt.push_str(&format!(
            "\nYou are on soak pass {} of {target}; {passes} consecutive passes so far. \
             Vary your angle of scrutiny on each pass — revisit a different aspect of the \
             design each time rather than re-running the same checks.\n",
            passes + 1,
        ));
    }
    if let Some(reference) = prior_checkpoint {
        prompt.push_str(&format!(
            "\nThe tree state when an agent last ran is checkpointed at `{reference}`. \
             Diff against it (e.g. git_diff with base `{reference}`) to see what changed \
             since you last looked instead of re-reading the world.\n",
        ));
    }
    if !suggestions.trim().is_empty() {
        prompt.push_str(&format!(
            "\n## Suggestions ledger (this run)\n\n{}\n\
             If one of your own suggestions keeps recurring, you may promote it to a \
             required finding.\n",
            suggestions.trim_end(),
        ));
    }
    if !context.trim().is_empty() {
        prompt.push_str(&format!("\n## Piped context\n\n{context}\n"));
    }
    prompt.push_str("\nEnd your turn by calling the `verdict` tool.\n");
    prompt
}

/// Marker file the runner drops when the control spool shuts down, so a
/// straggling shim fails fast instead of polling forever.
const CONTROL_CLOSED_MARKER: &str = "closed";
/// Spool poll interval.  Agent calls take seconds to minutes; a few
/// milliseconds of latency is noise.
const CONTROL_POLL: Duration = Duration::from_millis(5);

/// Serve shim requests spooled into `control_dir` until `stop` is set.
///
/// The transport is files plus rename (no sockets, no fifos): the shim
/// writes `req-<id>.json` atomically, the runner answers with
/// `resp-<id>.json` atomically and removes the request.  One request at a
/// time: agent invocations within a run are serialized by design (v1).
pub fn serve_control_dir(control_dir: &Path, core: Arc<Mutex<RunnerCore>>, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::Relaxed) {
        let mut requests: Vec<PathBuf> = match fs::read_dir(control_dir) {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("req-") && name.ends_with(".json"))
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        requests.sort();
        for request_path in requests {
            let response = match fs::read_to_string(&request_path)
                .map_err(|err| format!("failed to read shim request: {err}"))
                .and_then(|text| {
                    serde_json::from_str::<ShimRequest>(&text)
                        .map_err(|err| format!("malformed shim request: {err}"))
                }) {
                Ok(request) => {
                    let mut core = core.lock().expect("runner core poisoned");
                    core.handle(&request)
                }
                Err(err) => ShimResponse::err(EXIT_TRANSPORT, format!("ralph: {err}")),
            };
            let request_name = request_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("req-unknown.json")
                .to_string();
            let response_name = format!("resp-{}", &request_name["req-".len()..]);
            let payload = serde_json::to_vec(&response).unwrap_or_default();
            let tmp = control_dir.join(format!("{response_name}.tmp"));
            let fin = control_dir.join(&response_name);
            let _ = fs::write(&tmp, &payload);
            let _ = fs::rename(&tmp, &fin);
            let _ = fs::remove_file(&request_path);
        }
        std::thread::sleep(CONTROL_POLL);
    }
    let _ = fs::write(control_dir.join(CONTROL_CLOSED_MARKER), b"closed\n");
}

/// Client side of the control spool, used by `sid-ralph-shim`.
pub fn call_control_dir(control_dir: &Path, request: &ShimRequest) -> Result<ShimResponse, String> {
    let id = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let payload =
        serde_json::to_vec(request).map_err(|err| format!("failed to encode request: {err}"))?;
    let tmp = control_dir.join(format!("req-{id}.json.tmp"));
    let fin = control_dir.join(format!("req-{id}.json"));
    fs::write(&tmp, &payload).map_err(|err| format!("failed to spool request: {err}"))?;
    fs::rename(&tmp, &fin).map_err(|err| format!("failed to commit request: {err}"))?;
    let response_path = control_dir.join(format!("resp-{id}.json"));
    let closed_marker = control_dir.join(CONTROL_CLOSED_MARKER);
    loop {
        if let Ok(text) = fs::read_to_string(&response_path) {
            let _ = fs::remove_file(&response_path);
            return serde_json::from_str(&text).map_err(|err| format!("malformed response: {err}"));
        }
        if closed_marker.exists() {
            return Err("the ralph runner shut down without answering".to_string());
        }
        std::thread::sleep(CONTROL_POLL);
    }
}

/// Locate the `sid-ralph-shim` binary: `$SID_RALPH_SHIM` override first, then
/// a sibling of the current executable.
pub fn locate_shim() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var(SHIM_PATH_ENV) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "{SHIM_PATH_ENV} points at {} which does not exist",
            path.display()
        ));
    }
    let current = std::env::current_exe()
        .map_err(|err| format!("failed to locate current executable: {err}"))?;
    let sibling = current
        .parent()
        .map(|dir| dir.join("sid-ralph-shim"))
        .filter(|path| path.is_file());
    sibling.ok_or_else(|| {
        "sid-ralph-shim not found next to the sid executable; set SID_RALPH_SHIM".to_string()
    })
}

/// The final result of driving a script.
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptOutcome {
    /// The script's exit status.
    pub status: i32,
}

/// Run `script_text` under embedded mxsh with `agent`/`judge` wired to
/// `core`.  Returns the script's exit status.
pub fn run_script(
    core: Arc<Mutex<RunnerCore>>,
    script_text: &str,
    extra_env: &[(String, String)],
) -> Result<ScriptOutcome, String> {
    run_script_with_output(core, script_text, extra_env, None)
}

/// Run `script_text` under embedded mxsh, optionally forwarding script
/// stdout/stderr through `output_sink`.
pub fn run_script_with_output(
    core: Arc<Mutex<RunnerCore>>,
    script_text: &str,
    extra_env: &[(String, String)],
    output_sink: Option<Arc<dyn ScriptOutputSink>>,
) -> Result<ScriptOutcome, String> {
    let (run_dir, run_id, workspace_root) = {
        let core = core.lock().expect("runner core poisoned");
        (
            core.options.run_dir.clone(),
            core.options.run_id.clone(),
            core.options.workspace_root.clone(),
        )
    };

    // Wire the shim commands into PATH.
    let bin_dir = run_dir.join("bin");
    fs::create_dir_all(&bin_dir).map_err(|err| format!("failed to create bin dir: {err}"))?;
    let shim = locate_shim()?;
    for name in ["agent", "judge"] {
        let link = bin_dir.join(name);
        if !link.exists() {
            std::os::unix::fs::symlink(&shim, &link)
                .map_err(|err| format!("failed to link {name}: {err}"))?;
        }
    }

    // The control spool lives inside the run dir.
    let control_dir = run_dir.join("ctl");
    let _ = fs::remove_dir_all(&control_dir);
    fs::create_dir_all(&control_dir)
        .map_err(|err| format!("failed to create control dir: {err}"))?;
    let control_server = ControlServerGuard::start(control_dir.clone(), Arc::clone(&core));

    let path = format!(
        "{}:{}",
        bin_dir.to_string_lossy(),
        std::env::var("PATH").unwrap_or_default()
    );
    let devnull =
        fs::File::open("/dev/null").map_err(|err| format!("failed to open /dev/null: {err}"))?;

    use std::os::fd::AsRawFd as _;
    let mut output_handles = ScriptOutputHandles::new(output_sink.as_ref().map(Arc::clone))?;
    let mut builder = mxsh::ShellBuilder::new()
        .shell_name("mxsh")
        .env(
            RUN_DIR_ENV,
            run_dir.to_string_lossy(),
            mxsh::embed::VariableAttributes::EXPORT,
        )
        .env(RUN_ID_ENV, &run_id, mxsh::embed::VariableAttributes::EXPORT)
        .env(
            CONTROL_DIR_ENV,
            control_dir.to_string_lossy(),
            mxsh::embed::VariableAttributes::EXPORT,
        )
        .env("PATH", path, mxsh::embed::VariableAttributes::EXPORT)
        .stdio(mxsh::embed::StdioConfig {
            stdin: mxsh::runtime::fd::FileDescriptor::new(devnull.as_raw_fd()),
            stdout: output_handles.stdout_fd(),
            stderr: output_handles.stderr_fd(),
        });
    for (key, value) in extra_env {
        builder = builder.env(key, value, mxsh::embed::VariableAttributes::EXPORT);
    }
    if let Some(workspace_root) = workspace_root.as_ref() {
        // Scripts run from the workspace root: `./ci` means the workspace's ci.
        builder = builder.current_dir(workspace_root);
    }
    let mut shell = builder
        .build(mxsh::runtime::unix::UnixRuntime::new())
        .map_err(|err| format!("failed to build shell: {err}"))?;

    let outcome = shell.run(script_text);
    drop(shell);
    drop(devnull);
    output_handles.close();

    control_server.stop();

    let status = outcome.exit_code.unwrap_or(outcome.status);
    Ok(ScriptOutcome { status })
}

struct ControlServerGuard {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ControlServerGuard {
    fn start(control_dir: PathBuf, core: Arc<Mutex<RunnerCore>>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let handle = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || serve_control_dir(&control_dir, core, stop))
        };
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn stop(mut self) {
        self.stop_inner();
    }

    fn stop_inner(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ControlServerGuard {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

struct ScriptOutputHandles {
    stdout_fd: mxsh::runtime::fd::FileDescriptor,
    stderr_fd: mxsh::runtime::fd::FileDescriptor,
    forwarders: Vec<std::thread::JoinHandle<()>>,
    enabled: bool,
}

impl ScriptOutputHandles {
    fn new(sink: Option<Arc<dyn ScriptOutputSink>>) -> Result<Self, String> {
        let Some(sink) = sink else {
            return Ok(Self {
                stdout_fd: mxsh::runtime::fd::FileDescriptor::STDOUT,
                stderr_fd: mxsh::runtime::fd::FileDescriptor::STDERR,
                forwarders: Vec::new(),
                enabled: false,
            });
        };

        let stdout_pipe = mxsh::runtime::fd::OsPipe::new()
            .map_err(|err| format!("failed to create ralph stdout pipe: {err}"))?;
        let stderr_pipe = match mxsh::runtime::fd::OsPipe::new() {
            Ok(pipe) => pipe,
            Err(err) => {
                stdout_pipe.read_fd.close();
                stdout_pipe.write_fd.close();
                return Err(format!("failed to create ralph stderr pipe: {err}"));
            }
        };
        let forwarders = vec![
            spawn_script_output_forwarder(stdout_pipe.read_fd, "stdout", Arc::clone(&sink)),
            spawn_script_output_forwarder(stderr_pipe.read_fd, "stderr", sink),
        ];
        Ok(Self {
            stdout_fd: stdout_pipe.write_fd,
            stderr_fd: stderr_pipe.write_fd,
            forwarders,
            enabled: true,
        })
    }

    fn stdout_fd(&self) -> mxsh::runtime::fd::FileDescriptor {
        self.stdout_fd
    }

    fn stderr_fd(&self) -> mxsh::runtime::fd::FileDescriptor {
        self.stderr_fd
    }

    fn close(&mut self) {
        if !self.enabled {
            return;
        }
        self.enabled = false;
        self.stdout_fd.close();
        self.stderr_fd.close();
        for forwarder in self.forwarders.drain(..) {
            let _ = forwarder.join();
        }
    }
}

impl Drop for ScriptOutputHandles {
    fn drop(&mut self) {
        self.close();
    }
}

fn spawn_script_output_forwarder(
    fd: mxsh::runtime::fd::FileDescriptor,
    stream: &'static str,
    sink: Arc<dyn ScriptOutputSink>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        loop {
            let n = unsafe {
                libc::read(
                    fd.into_raw_fd(),
                    chunk.as_mut_ptr() as *mut libc::c_void,
                    chunk.len(),
                )
            };
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                break;
            }
            if n == 0 {
                break;
            }
            sink.on_script_output(stream, &chunk[..n as usize]);
        }
        fd.close();
    })
}

/// Drive a full run: journal start, run the script, assemble the report.
pub fn run_ralph(
    host: Box<dyn RalphHost>,
    options: RunnerOptions,
    script_text: &str,
    extra_env: &[(String, String)],
    interrupted: Arc<AtomicBool>,
) -> Result<RunReport, String> {
    run_ralph_with_output(host, options, script_text, extra_env, interrupted, None)
}

/// Drive a full run and optionally forward mxsh stdout/stderr.
pub fn run_ralph_with_output(
    host: Box<dyn RalphHost>,
    options: RunnerOptions,
    script_text: &str,
    extra_env: &[(String, String)],
    interrupted: Arc<AtomicBool>,
    output_sink: Option<Arc<dyn ScriptOutputSink>>,
) -> Result<RunReport, String> {
    let core = RunnerCore::new(host, options, interrupted)?;
    let core = Arc::new(Mutex::new(core));
    {
        let mut core = core.lock().expect("runner core poisoned");
        core.record_run_start(script_text)?;
    }
    let outcome = run_script_with_output(Arc::clone(&core), script_text, extra_env, output_sink)?;
    let core = core.lock().expect("runner core poisoned");
    Ok(core.report(outcome.status))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::super::verdict::{Finding, Severity};
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sid-ralph-runner-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn options(run_dir: &Path) -> RunnerOptions {
        RunnerOptions {
            run_id: "test-run".to_string(),
            run_dir: run_dir.to_path_buf(),
            workspace_root: None,
            max_iters: None,
            budget_tokens: None,
            resume: false,
        }
    }

    /// A scripted host: pops pre-arranged outcomes.
    #[derive(Default)]
    struct StubHost {
        agent_results: VecDeque<AgentCallResult>,
        judge_results: VecDeque<JudgeCallResult>,
        agent_invocations: Arc<Mutex<Vec<AgentInvocation>>>,
        judge_invocations: Arc<Mutex<Vec<JudgeInvocation>>>,
        invalid_judges: HashSet<String>,
    }

    impl RalphHost for StubHost {
        fn validate_judge(&mut self, service: &str) -> Result<(), String> {
            if self.invalid_judges.contains(service) {
                Err(format!(
                    "agent {service:?} is not configured with the verdict tool"
                ))
            } else {
                Ok(())
            }
        }

        fn run_agent(&mut self, invocation: &AgentInvocation) -> AgentCallResult {
            self.agent_invocations
                .lock()
                .unwrap()
                .push(invocation.clone());
            self.agent_results.pop_front().unwrap_or(AgentCallResult {
                outcome: AgentOutcome::Completed,
                tokens: 10,
                session: Some("stub-session".to_string()),
            })
        }

        fn judge_sample(&mut self, invocation: &JudgeInvocation) -> JudgeCallResult {
            self.judge_invocations
                .lock()
                .unwrap()
                .push(invocation.clone());
            self.judge_results.pop_front().unwrap_or(JudgeCallResult {
                outcome: JudgeOutcome::Verdict(passing_verdict("fallback pass")),
                tokens: 5,
            })
        }
    }

    fn passing_verdict(summary: &str) -> Verdict {
        Verdict {
            sufficient: true,
            summary: summary.to_string(),
            findings: Vec::new(),
            acceptance: Vec::new(),
        }
    }

    fn failing_verdict(summary: &str) -> Verdict {
        Verdict {
            sufficient: false,
            summary: summary.to_string(),
            findings: vec![Finding {
                severity: Severity::Required,
                where_: "src/lib.rs:1".to_string(),
                what: "Do the thing".to_string(),
                why: "The plan".to_string(),
            }],
            acceptance: vec!["the thing is done".to_string()],
        }
    }

    fn verdict_result(verdict: Verdict) -> JudgeCallResult {
        JudgeCallResult {
            outcome: JudgeOutcome::Verdict(verdict),
            tokens: 5,
        }
    }

    #[derive(Default)]
    struct RecordingScriptOutputSink {
        chunks: Mutex<Vec<(String, Vec<u8>)>>,
    }

    impl RecordingScriptOutputSink {
        fn text(&self, stream: &str) -> String {
            let chunks = self.chunks.lock().unwrap();
            String::from_utf8_lossy(
                &chunks
                    .iter()
                    .filter(|(candidate, _)| candidate == stream)
                    .flat_map(|(_, data)| data.clone())
                    .collect::<Vec<_>>(),
            )
            .into_owned()
        }
    }

    impl ScriptOutputSink for RecordingScriptOutputSink {
        fn on_script_output(&self, stream: &str, data: &[u8]) {
            self.chunks
                .lock()
                .unwrap()
                .push((stream.to_string(), data.to_vec()));
        }
    }

    fn request(name: &str, args: &[&str], context: &str) -> ShimRequest {
        ShimRequest {
            name: name.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            context: context.to_string(),
        }
    }

    fn core_with(host: StubHost, options: RunnerOptions) -> RunnerCore {
        RunnerCore::new(Box::new(host), options, Arc::new(AtomicBool::new(false))).unwrap()
    }

    #[test]
    fn run_ralph_with_output_captures_script_stdout_and_stderr() {
        static ENV_LOCK: Mutex<()> = Mutex::new(());

        let _guard = ENV_LOCK.lock().unwrap();
        let dir = temp_dir("script-output");
        let shim = dir.join("sid-ralph-shim");
        fs::write(&shim, "#!/bin/sh\nexit 99\n").unwrap();
        let previous_shim = std::env::var_os(SHIM_PATH_ENV);
        // SAFETY: this test serializes all mutations of this process-global
        // environment variable with ENV_LOCK and restores it before releasing.
        unsafe {
            std::env::set_var(SHIM_PATH_ENV, &shim);
        }

        let sink = Arc::new(RecordingScriptOutputSink::default());
        let output_sink: Arc<dyn ScriptOutputSink> = sink.clone();
        let report = run_ralph_with_output(
            Box::new(StubHost::default()),
            options(&dir),
            "echo stdout-line; echo stderr-line >&2; /bin/sh -c 'printf external-out; printf external-err >&2'",
            &[],
            Arc::new(AtomicBool::new(false)),
            Some(output_sink),
        )
        .unwrap();

        assert_eq!(report.exit, EXIT_OK);
        assert!(sink.text("stdout").contains("stdout-line"));
        assert!(sink.text("stderr").contains("stderr-line"));
        assert!(sink.text("stdout").contains("external-out"));
        assert!(sink.text("stderr").contains("external-err"));

        match previous_shim {
            Some(value) => unsafe {
                std::env::set_var(SHIM_PATH_ENV, value);
            },
            None => unsafe {
                std::env::remove_var(SHIM_PATH_ENV);
            },
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn agent_completion_exits_zero_and_journals() {
        let dir = temp_dir("agent-zero");
        let host = StubHost::default();
        let invocations = Arc::clone(&host.agent_invocations);
        let mut core = core_with(host, options(&dir));
        let response = core.handle(&request("agent", &["fix", "Make CI pass."], "the log"));
        assert_eq!(response.exit, EXIT_OK);
        let seen = invocations.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].service, "fix");
        assert_eq!(seen[0].instruction, "Make CI pass.");
        assert_eq!(seen[0].context, "the log");
        let records = StepsJournal::new(&dir).load().unwrap();
        assert_eq!(records.len(), 1);
        assert!(matches!(
            &records[0],
            StepRecord::Agent { service, exit: 0, .. } if service == "fix"
        ));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn agent_escalation_exits_three() {
        let dir = temp_dir("agent-three");
        let mut host = StubHost::default();
        host.agent_results.push_back(AgentCallResult {
            outcome: AgentOutcome::Escalated("environment broken".to_string()),
            tokens: 7,
            session: None,
        });
        let mut core = core_with(host, options(&dir));
        let response = core.handle(&request("agent", &["fix"], ""));
        assert_eq!(response.exit, EXIT_ESCALATED);
        assert!(response.stderr.contains("environment broken"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn agent_transport_failure_exits_four() {
        let dir = temp_dir("agent-four");
        let mut host = StubHost::default();
        host.agent_results.push_back(AgentCallResult {
            outcome: AgentOutcome::Transport("api 500".to_string()),
            tokens: 0,
            session: None,
        });
        let mut core = core_with(host, options(&dir));
        let response = core.handle(&request("agent", &["fix"], ""));
        assert_eq!(response.exit, EXIT_TRANSPORT);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn agent_usage_error_exits_transport_class() {
        let dir = temp_dir("agent-usage");
        let mut core = core_with(StubHost::default(), options(&dir));
        let response = core.handle(&request("agent", &[], ""));
        assert_eq!(response.exit, EXIT_TRANSPORT);
        assert!(response.stderr.contains("usage"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn max_iters_exhaustion_exits_four() {
        let dir = temp_dir("max-iters");
        let mut opts = options(&dir);
        opts.max_iters = Some(1);
        let mut core = core_with(StubHost::default(), opts);
        assert_eq!(core.handle(&request("agent", &["fix"], "")).exit, EXIT_OK);
        let response = core.handle(&request("agent", &["task"], ""));
        assert_eq!(response.exit, EXIT_TRANSPORT);
        assert!(response.stderr.contains("--max-iters"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn budget_exhaustion_exits_four() {
        let dir = temp_dir("budget");
        let mut opts = options(&dir);
        opts.budget_tokens = Some(5);
        // The stub charges 10 tokens per agent call.
        let mut core = core_with(StubHost::default(), opts);
        assert_eq!(core.handle(&request("agent", &["fix"], "")).exit, EXIT_OK);
        let response = core.handle(&request("agent", &["fix"], ""));
        assert_eq!(response.exit, EXIT_TRANSPORT);
        assert!(response.stderr.contains("--budget"));
        let response = core.handle(&request("judge", &["judge"], ""));
        assert_eq!(response.exit, EXIT_TRANSPORT);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn interrupted_flag_exits_130() {
        let dir = temp_dir("sigint");
        let interrupted = Arc::new(AtomicBool::new(true));
        let mut core = RunnerCore::new(
            Box::new(StubHost::default()),
            options(&dir),
            Arc::clone(&interrupted),
        )
        .unwrap();
        assert_eq!(
            core.handle(&request("agent", &["fix"], "")).exit,
            EXIT_SIGINT
        );
        assert_eq!(
            core.handle(&request("judge", &["judge"], "")).exit,
            EXIT_SIGINT
        );
        let report = core.report(0);
        assert_eq!(report.exit, EXIT_SIGINT);
        assert!(report.interrupted);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn judge_without_verdict_tool_is_a_config_error_before_any_call() {
        let dir = temp_dir("judge-config");
        let mut host = StubHost::default();
        host.invalid_judges.insert("build".to_string());
        let samples = Arc::clone(&host.judge_invocations);
        let mut core = core_with(host, options(&dir));
        let response = core.handle(&request("judge", &["build"], ""));
        assert_eq!(response.exit, EXIT_TRANSPORT);
        assert!(response.stderr.contains("verdict"));
        assert!(samples.lock().unwrap().is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn judge_pass_and_fail_exit_codes() {
        let dir = temp_dir("judge-codes");
        let mut host = StubHost::default();
        host.judge_results
            .push_back(verdict_result(failing_verdict("not yet")));
        host.judge_results
            .push_back(verdict_result(passing_verdict("done")));
        let mut core = core_with(host, options(&dir));
        let response = core.handle(&request("judge", &["judge"], ""));
        assert_eq!(response.exit, EXIT_INSUFFICIENT);
        assert!(response.stdout.contains("# Verdict: insufficient"));
        assert!(response.stdout.contains("Do the thing"));
        let response = core.handle(&request("judge", &["judge"], ""));
        assert_eq!(response.exit, EXIT_OK);
        assert!(response.stdout.contains("# Verdict: sufficient"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn judge_escalation_exits_three() {
        let dir = temp_dir("judge-escalate");
        let mut host = StubHost::default();
        host.judge_results.push_back(JudgeCallResult {
            outcome: JudgeOutcome::Escalated("ambiguous plan".to_string()),
            tokens: 3,
        });
        let mut core = core_with(host, options(&dir));
        let response = core.handle(&request("judge", &["judge"], ""));
        assert_eq!(response.exit, EXIT_ESCALATED);
        assert!(response.stderr.contains("ambiguous plan"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn soak_requires_consecutive_passes_and_resets() {
        let dir = temp_dir("soak");
        let mut host = StubHost::default();
        host.judge_results
            .push_back(verdict_result(passing_verdict("pass 1")));
        host.judge_results
            .push_back(verdict_result(failing_verdict("regression")));
        host.judge_results
            .push_back(verdict_result(passing_verdict("pass 1 again")));
        host.judge_results
            .push_back(verdict_result(passing_verdict("pass 2")));
        let mut core = core_with(host, options(&dir));
        let soak_args = ["judge", "--soak", "2"];
        let r1 = core.handle(&request("judge", &soak_args, ""));
        assert_eq!(r1.exit, EXIT_INSUFFICIENT);
        assert!(r1.stdout.contains("Soak: 1/2"));
        let r2 = core.handle(&request("judge", &soak_args, ""));
        assert_eq!(r2.exit, EXIT_INSUFFICIENT);
        assert!(r2.stdout.contains("Soak: 0/2"));
        let r3 = core.handle(&request("judge", &soak_args, ""));
        assert_eq!(r3.exit, EXIT_INSUFFICIENT);
        assert!(r3.stdout.contains("Soak: 1/2"));
        let r4 = core.handle(&request("judge", &soak_args, ""));
        assert_eq!(r4.exit, EXIT_OK);
        assert!(r4.stdout.contains("Soak: 2/2"));
        // The soak counter is visible in steps.jsonl.
        let records = StepsJournal::new(&dir).load().unwrap();
        let soaks: Vec<u32> = records
            .iter()
            .filter_map(|r| match r {
                StepRecord::Judge { soak, .. } => Some(*soak),
                _ => None,
            })
            .collect();
        assert_eq!(soaks, vec![1, 0, 1, 2]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn jury_samples_are_goldfish_and_all_must_pass() {
        let dir = temp_dir("jury");
        let mut host = StubHost::default();
        host.judge_results
            .push_back(verdict_result(passing_verdict("juror 1 pass")));
        host.judge_results
            .push_back(verdict_result(failing_verdict("juror 2 fail")));
        host.judge_results
            .push_back(verdict_result(passing_verdict("juror 3 pass")));
        let samples = Arc::clone(&host.judge_invocations);
        let mut core = core_with(host, options(&dir));
        let response = core.handle(&request("judge", &["judge", "--jury", "3"], ""));
        assert_eq!(response.exit, EXIT_INSUFFICIENT);
        assert!(response.stdout.contains("Jury: 2/3 jurors passed."));
        assert!(response.stdout.contains("juror 2 fail"));
        let seen = samples.lock().unwrap();
        assert_eq!(seen.len(), 3);
        assert!(seen.iter().all(|s| s.goldfish));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn jury_all_pass_exits_zero() {
        let dir = temp_dir("jury-pass");
        let mut host = StubHost::default();
        for i in 0..3 {
            host.judge_results
                .push_back(verdict_result(passing_verdict(&format!("juror {i}"))));
        }
        let mut core = core_with(host, options(&dir));
        let response = core.handle(&request("judge", &["judge", "--jury", "3"], ""));
        assert_eq!(response.exit, EXIT_OK);
        assert!(response.stdout.contains("Jury: 3/3 jurors passed."));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pedantic_turns_suggestions_into_rejection() {
        let dir = temp_dir("pedantic");
        let mut verdict = passing_verdict("pass with notes");
        verdict.findings.push(Finding {
            severity: Severity::Suggestion,
            where_: "README.md".to_string(),
            what: "Mention the flag".to_string(),
            why: "nice to have".to_string(),
        });
        let mut host = StubHost::default();
        host.judge_results
            .push_back(verdict_result(verdict.clone()));
        host.judge_results.push_back(verdict_result(verdict));
        let mut core = core_with(host, options(&dir));
        let response = core.handle(&request("judge", &["judge", "--pedantic"], ""));
        assert_eq!(response.exit, EXIT_INSUFFICIENT);
        let response = core.handle(&request("judge", &["judge"], ""));
        assert_eq!(response.exit, EXIT_OK);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn passing_suggestions_land_in_ledger_and_next_prompt() {
        let dir = temp_dir("ledger-flow");
        let mut verdict = passing_verdict("pass with notes");
        verdict.findings.push(Finding {
            severity: Severity::Suggestion,
            where_: "README.md".to_string(),
            what: "Mention the soak flag".to_string(),
            why: "operators will want it".to_string(),
        });
        let mut host = StubHost::default();
        host.judge_results.push_back(verdict_result(verdict));
        host.judge_results
            .push_back(verdict_result(passing_verdict("second pass")));
        let samples = Arc::clone(&host.judge_invocations);
        let mut core = core_with(host, options(&dir));
        assert_eq!(core.handle(&request("judge", &["judge"], "")).exit, EXIT_OK);
        let ledger = SuggestionsLedger::new(&dir);
        assert_eq!(ledger.entry_count(), 1);
        assert!(ledger.read().contains("Mention the soak flag"));
        // The next judge pass sees the ledger.
        assert_eq!(core.handle(&request("judge", &["judge"], "")).exit, EXIT_OK);
        let seen = samples.lock().unwrap();
        assert!(seen[1].prompt.contains("Suggestions ledger"));
        assert!(seen[1].prompt.contains("Mention the soak flag"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn failing_verdict_suggestions_stay_out_of_the_ledger() {
        let dir = temp_dir("ledger-fail");
        let mut verdict = failing_verdict("not done");
        verdict.findings.push(Finding {
            severity: Severity::Suggestion,
            where_: "README.md".to_string(),
            what: "Polish prose".to_string(),
            why: "style".to_string(),
        });
        let mut host = StubHost::default();
        host.judge_results.push_back(verdict_result(verdict));
        let mut core = core_with(host, options(&dir));
        assert_eq!(
            core.handle(&request("judge", &["judge"], "")).exit,
            EXIT_INSUFFICIENT
        );
        assert_eq!(SuggestionsLedger::new(&dir).entry_count(), 0);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn malformed_verdict_from_host_is_transport_class() {
        let dir = temp_dir("malformed");
        let mut host = StubHost::default();
        host.judge_results.push_back(JudgeCallResult {
            outcome: JudgeOutcome::Verdict(Verdict {
                sufficient: false,
                summary: "no".to_string(),
                findings: Vec::new(),
                acceptance: Vec::new(),
            }),
            tokens: 1,
        });
        let mut core = core_with(host, options(&dir));
        let response = core.handle(&request("judge", &["judge"], ""));
        assert_eq!(response.exit, EXIT_TRANSPORT);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn large_context_is_capped_and_logged() {
        let dir = temp_dir("cap");
        let host = StubHost::default();
        let invocations = Arc::clone(&host.agent_invocations);
        let mut core = core_with(host, options(&dir));
        let big = "x".repeat(10 * 1024 * 1024);
        let response = core.handle(&request("agent", &["fix"], &big));
        assert_eq!(response.exit, EXIT_OK);
        let seen = invocations.lock().unwrap();
        assert!(seen[0].context.contains("[truncated: full log at"));
        assert!(seen[0].context.len() < big.len());
        let log = dir.join("ci-001.log");
        assert_eq!(fs::read_to_string(&log).unwrap().len(), big.len());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn judge_prompt_carries_soak_note_and_checkpoint() {
        let prompt = assemble_judge_prompt(
            Some("Is PLAN.md complete?"),
            "ci output",
            Some((2, 5)),
            Some("refs/sid/ralph/run-1/7"),
            "- (step 3) README.md — note\n",
        );
        assert!(prompt.starts_with("Is PLAN.md complete?"));
        assert!(prompt.contains("soak pass 3 of 5"));
        assert!(prompt.contains("2 consecutive passes"));
        assert!(prompt.contains("refs/sid/ralph/run-1/7"));
        assert!(prompt.contains("Suggestions ledger"));
        assert!(prompt.contains("## Piped context"));
        assert!(prompt.contains("ci output"));
        assert!(prompt.contains("End your turn by calling the `verdict` tool."));
    }

    #[test]
    fn judge_prompt_omits_empty_sections() {
        let prompt = assemble_judge_prompt(None, "", None, None, "");
        assert!(prompt.contains("Render your verdict"));
        assert!(!prompt.contains("soak pass"));
        assert!(!prompt.contains("Suggestions ledger"));
        assert!(!prompt.contains("## Piped context"));
    }

    #[test]
    fn resume_replays_completed_steps_without_invoking_the_host() {
        let dir = temp_dir("resume");
        // First run: one agent step, one failing judge step at soak 1.
        {
            let mut host = StubHost::default();
            host.judge_results
                .push_back(verdict_result(passing_verdict("pass 1")));
            let mut core = core_with(host, options(&dir));
            assert_eq!(
                core.handle(&request("agent", &["fix"], "log")).exit,
                EXIT_OK
            );
            assert_eq!(
                core.handle(&request("judge", &["judge", "--soak", "2"], ""))
                    .exit,
                EXIT_INSUFFICIENT
            );
        }
        // Resume: the same script shape replays both steps, then goes live.
        let mut host = StubHost::default();
        host.judge_results
            .push_back(verdict_result(passing_verdict("pass 2")));
        let agent_invocations = Arc::clone(&host.agent_invocations);
        let judge_invocations = Arc::clone(&host.judge_invocations);
        let mut opts = options(&dir);
        opts.resume = true;
        let mut core = core_with(host, opts);
        let r1 = core.handle(&request("agent", &["fix"], "log"));
        assert_eq!(r1.exit, EXIT_OK);
        assert!(r1.stderr.contains("replayed"));
        let r2 = core.handle(&request("judge", &["judge", "--soak", "2"], ""));
        assert_eq!(r2.exit, EXIT_INSUFFICIENT);
        assert!(r2.stderr.contains("replayed"));
        assert!(agent_invocations.lock().unwrap().is_empty());
        assert!(judge_invocations.lock().unwrap().is_empty());
        // Live again: the soak counter was restored, so one more pass converges.
        let r3 = core.handle(&request("judge", &["judge", "--soak", "2"], ""));
        assert_eq!(r3.exit, EXIT_OK);
        assert!(r3.stdout.contains("Soak: 2/2"));
        assert_eq!(judge_invocations.lock().unwrap().len(), 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn resume_divergence_falls_back_to_live_execution() {
        let dir = temp_dir("resume-diverge");
        {
            let mut core = core_with(StubHost::default(), options(&dir));
            assert_eq!(core.handle(&request("agent", &["fix"], "")).exit, EXIT_OK);
        }
        let host = StubHost::default();
        let invocations = Arc::clone(&host.agent_invocations);
        let mut opts = options(&dir);
        opts.resume = true;
        let mut core = core_with(host, opts);
        // The script now asks for a judge first: the journal diverges.
        let response = core.handle(&request("judge", &["judge"], ""));
        assert_eq!(response.exit, EXIT_OK);
        assert!(!response.stderr.contains("replayed"));
        // And subsequent agent calls are live too.
        assert_eq!(core.handle(&request("agent", &["fix"], "")).exit, EXIT_OK);
        assert_eq!(invocations.lock().unwrap().len(), 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn report_counts_iterations_per_service() {
        let dir = temp_dir("report");
        let mut host = StubHost::default();
        host.judge_results
            .push_back(verdict_result(failing_verdict("more work")));
        host.judge_results
            .push_back(verdict_result(passing_verdict("The plan is complete.")));
        let mut core = core_with(host, options(&dir));
        core.handle(&request("agent", &["fix"], ""));
        core.handle(&request("judge", &["judge"], ""));
        core.handle(&request("agent", &["task"], ""));
        core.handle(&request("agent", &["task"], ""));
        core.handle(&request("judge", &["judge"], ""));
        let report = core.report(0);
        assert_eq!(report.exit, 0);
        assert_eq!(report.iterations, 3);
        assert_eq!(
            report.agent_counts,
            vec![("fix".to_string(), 1), ("task".to_string(), 2)]
        );
        assert_eq!(
            report.final_verdict_summary.as_deref(),
            Some("The plan is complete.")
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unknown_builtin_name_is_rejected() {
        let dir = temp_dir("unknown");
        let mut core = core_with(StubHost::default(), options(&dir));
        let response = core.handle(&request("jury", &["judge"], ""));
        assert_eq!(response.exit, EXIT_TRANSPORT);
        fs::remove_dir_all(&dir).unwrap();
    }
}
