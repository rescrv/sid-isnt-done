//! End-to-end tests for the ralph runner: real embedded mxsh, the real
//! `sid-ralph-shim` on PATH, real pipes — and a scripted stub host instead of
//! LLM inference.

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, Once};

use sid_isnt_done::ralph::journal::{StepRecord, StepsJournal, SuggestionsLedger};
use sid_isnt_done::ralph::runner::{
    AgentCallResult, AgentInvocation, AgentOutcome, JudgeCallResult, JudgeInvocation, JudgeOutcome,
    RalphHost, RunnerOptions, SHIM_PATH_ENV, run_ralph,
};
use sid_isnt_done::ralph::verdict::{Finding, Severity, Verdict};
use sid_isnt_done::ralph::{EXIT_ESCALATED, EXIT_OK, EXIT_TRANSPORT};

fn ensure_shim_env() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Safety: called exactly once before any threads depend on the value;
        // every test that reads it goes through this function first.
        unsafe {
            std::env::set_var(SHIM_PATH_ENV, env!("CARGO_BIN_EXE_sid-ralph-shim"));
        }
    });
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sid-ralph-e2e-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn options(name: &str, workspace: &Path) -> (RunnerOptions, PathBuf) {
    let run_dir = temp_dir(&format!("{name}-run"));
    (
        RunnerOptions {
            run_id: format!("{name}-run"),
            run_dir: run_dir.clone(),
            workspace_root: Some(workspace.to_path_buf()),
            max_iters: None,
            budget_tokens: None,
            resume: false,
        },
        run_dir,
    )
}

/// A scripted host with externally visible call logs.
#[derive(Default)]
struct StubHost {
    agent_log: Arc<Mutex<Vec<AgentInvocation>>>,
    judge_log: Arc<Mutex<Vec<JudgeInvocation>>>,
    agent_results: Arc<Mutex<VecDeque<AgentCallResult>>>,
    judge_results: Arc<Mutex<VecDeque<JudgeCallResult>>>,
    /// When set, the first `fix` invocation drops a `fixed` marker file here,
    /// simulating a repair that makes `./ci` pass.
    fix_touches: Option<PathBuf>,
}

impl RalphHost for StubHost {
    fn validate_judge(&mut self, _service: &str) -> Result<(), String> {
        Ok(())
    }

    fn run_agent(&mut self, invocation: &AgentInvocation) -> AgentCallResult {
        self.agent_log.lock().unwrap().push(invocation.clone());
        if invocation.service == "fix"
            && let Some(marker) = self.fix_touches.as_ref()
        {
            fs::write(marker, "fixed\n").unwrap();
        }
        self.agent_results
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(AgentCallResult {
                outcome: AgentOutcome::Completed,
                tokens: 10,
                session: Some("stub".to_string()),
            })
    }

    fn judge_sample(&mut self, invocation: &JudgeInvocation) -> JudgeCallResult {
        self.judge_log.lock().unwrap().push(invocation.clone());
        self.judge_results
            .lock()
            .unwrap()
            .pop_front()
            .expect("test must script every judge sample")
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
            where_: "PLAN.md:1".to_string(),
            what: "Implement the missing half".to_string(),
            why: "the plan demands it".to_string(),
        }],
        acceptance: vec!["the missing half exists".to_string()],
    }
}

fn verdict_result(verdict: Verdict) -> JudgeCallResult {
    JudgeCallResult {
        outcome: JudgeOutcome::Verdict(verdict),
        tokens: 5,
    }
}

#[test]
fn inline_script_sees_run_dir_and_environment_passthrough() {
    ensure_shim_env();
    let workspace = temp_dir("inline-ws");
    let (opts, run_dir) = options("inline", &workspace);
    let report = run_ralph(
        Box::new(StubHost::default()),
        opts,
        "echo \"hello from $PLAN in $RUN_DIR\" > \"$RUN_DIR/out.txt\"",
        &[("PLAN".to_string(), "PLAN.md".to_string())],
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_eq!(report.exit, EXIT_OK);
    let out = fs::read_to_string(run_dir.join("out.txt")).unwrap();
    assert_eq!(
        out,
        format!("hello from PLAN.md in {}\n", run_dir.display())
    );
    fs::remove_dir_all(&workspace).unwrap();
    fs::remove_dir_all(&run_dir).unwrap();
}

#[test]
fn pipes_carry_context_into_the_agent() {
    ensure_shim_env();
    let workspace = temp_dir("pipes-ws");
    let (opts, run_dir) = options("pipes", &workspace);
    let host = StubHost::default();
    let agent_log = Arc::clone(&host.agent_log);
    let report = run_ralph(
        Box::new(host),
        opts,
        "printf '%s' 'the failing log' | agent fix 'Make CI pass.'",
        &[],
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_eq!(report.exit, EXIT_OK);
    let log = agent_log.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].service, "fix");
    assert_eq!(log[0].instruction, "Make CI pass.");
    assert_eq!(log[0].context, "the failing log");
    fs::remove_dir_all(&workspace).unwrap();
    fs::remove_dir_all(&run_dir).unwrap();
}

#[test]
fn judge_stdout_is_the_work_order_and_exit_codes_steer_the_shell() {
    ensure_shim_env();
    let workspace = temp_dir("steer-ws");
    let (opts, run_dir) = options("steer", &workspace);
    let host = StubHost::default();
    let agent_log = Arc::clone(&host.agent_log);
    host.judge_results
        .lock()
        .unwrap()
        .push_back(verdict_result(failing_verdict("not done yet")));
    // The shell pipes the rendered verdict into the task agent only on exit 1.
    let script = r#"
findings=$(judge judge "Is it done?")
case $? in
  0) echo unexpected-pass > "$RUN_DIR/path.txt" ;;
  1) printf '%s' "$findings" | agent task "Execute this work order." ;;
  *) echo unexpected-failure > "$RUN_DIR/path.txt" ;;
esac
"#;
    let report = run_ralph(
        Box::new(host),
        opts,
        script,
        &[],
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_eq!(report.exit, EXIT_OK);
    assert!(!run_dir.join("path.txt").exists());
    let log = agent_log.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].service, "task");
    assert!(log[0].context.contains("# Verdict: insufficient"));
    assert!(log[0].context.contains("Implement the missing half"));
    fs::remove_dir_all(&workspace).unwrap();
    fs::remove_dir_all(&run_dir).unwrap();
}

#[test]
fn the_reference_ralph_script_converges() {
    ensure_shim_env();
    let workspace = temp_dir("ralph-ws");
    // A fake ./ci that fails until the `fixed` marker exists.
    let ci = workspace.join("ci");
    fs::write(
        &ci,
        "#!/bin/sh\nif test -f fixed; then echo ci ok; else echo 'ci output: not fixed'; exit 1; fi\n",
    )
    .unwrap();
    let mut perms = fs::metadata(&ci).unwrap().permissions();
    use std::os::unix::fs::PermissionsExt as _;
    perms.set_mode(0o755);
    fs::set_permissions(&ci, perms).unwrap();

    let (opts, run_dir) = options("ralph", &workspace);
    let host = StubHost {
        fix_touches: Some(workspace.join("fixed")),
        ..Default::default()
    };
    let agent_log = Arc::clone(&host.agent_log);
    let judge_log = Arc::clone(&host.judge_log);
    {
        let mut judge_results = host.judge_results.lock().unwrap();
        // First pass: a work order.  Second pass: sufficient, with a
        // suggestion that must land in the ledger and get swept.
        judge_results.push_back(verdict_result(failing_verdict("half done")));
        let mut pass = passing_verdict("The plan is complete.");
        pass.findings.push(Finding {
            severity: Severity::Suggestion,
            where_: "README.md".to_string(),
            what: "Mention the soak flag".to_string(),
            why: "operators will want it".to_string(),
        });
        judge_results.push_back(verdict_result(pass));
    }

    let script = include_str!("../init/ralph.sid");
    let report = run_ralph(
        Box::new(host),
        opts,
        script,
        &[("SOAK".to_string(), "1".to_string())],
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();

    assert_eq!(report.exit, EXIT_OK, "report: {report:?}");
    // 1 fix + 1 task (work order) + 1 task (suggestions sweep).
    assert_eq!(report.iterations, 3);
    assert_eq!(
        report.agent_counts,
        vec![("fix".to_string(), 1), ("task".to_string(), 2)]
    );
    assert_eq!(
        report.final_verdict_summary.as_deref(),
        Some("The plan is complete.")
    );
    assert_eq!(report.final_soak, Some((1, 1)));
    assert_eq!(report.suggestions_entries, 1);

    let log = agent_log.lock().unwrap();
    // The fix agent got the piped CI log.
    assert_eq!(log[0].service, "fix");
    assert!(log[0].context.contains("ci output: not fixed"));
    // The task agent got the rendered work order.
    assert_eq!(log[1].service, "task");
    assert!(log[1].context.contains("Implement the missing half"));
    // The sweep got the suggestions ledger on stdin.
    assert_eq!(log[2].service, "task");
    assert!(log[2].context.contains("Mention the soak flag"));

    // The judge's prompts: both passes mention the plan; the second pass is
    // mid-soak with the previous prompt's soak framing.
    let judges = judge_log.lock().unwrap();
    assert_eq!(judges.len(), 2);
    assert!(judges[0].prompt.contains("Is PLAN.md complete"));
    assert!(judges[0].prompt.contains("soak pass 1 of 1"));

    // The journal narrates the whole run.
    let records = StepsJournal::new(&run_dir).load().unwrap();
    let kinds: Vec<&str> = records
        .iter()
        .map(|record| match record {
            StepRecord::RunStart { .. } => "start",
            StepRecord::Agent { .. } => "agent",
            StepRecord::Judge { .. } => "judge",
        })
        .collect();
    assert_eq!(
        kinds,
        vec!["start", "agent", "judge", "agent", "judge", "agent"]
    );
    assert!(
        SuggestionsLedger::new(&run_dir)
            .read()
            .contains("Mention the soak flag")
    );

    fs::remove_dir_all(&workspace).unwrap();
    fs::remove_dir_all(&run_dir).unwrap();
}

#[test]
fn escalation_stops_the_reference_loop_with_exit_three() {
    ensure_shim_env();
    let workspace = temp_dir("escalate-ws");
    let ci = workspace.join("ci");
    fs::write(&ci, "#!/bin/sh\nexit 1\n").unwrap();
    let mut perms = fs::metadata(&ci).unwrap().permissions();
    use std::os::unix::fs::PermissionsExt as _;
    perms.set_mode(0o755);
    fs::set_permissions(&ci, perms).unwrap();

    let (opts, run_dir) = options("escalate", &workspace);
    let host = StubHost::default();
    host.agent_results
        .lock()
        .unwrap()
        .push_back(AgentCallResult {
            outcome: AgentOutcome::Escalated("ci needs credentials".to_string()),
            tokens: 1,
            session: None,
        });
    let script = include_str!("../init/ralph.sid");
    let report = run_ralph(
        Box::new(host),
        opts,
        script,
        &[],
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_eq!(report.exit, EXIT_ESCALATED);
    fs::remove_dir_all(&workspace).unwrap();
    fs::remove_dir_all(&run_dir).unwrap();
}

#[test]
fn max_iters_is_a_transport_class_stop() {
    ensure_shim_env();
    let workspace = temp_dir("iters-ws");
    let (mut opts, run_dir) = options("iters", &workspace);
    opts.max_iters = Some(2);
    let host = StubHost::default();
    let script = r#"
agent fix || exit $?
agent fix || exit $?
agent fix || exit $?
exit 0
"#;
    let report = run_ralph(
        Box::new(host),
        opts,
        script,
        &[],
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_eq!(report.exit, EXIT_TRANSPORT);
    assert_eq!(report.iterations, 2);
    fs::remove_dir_all(&workspace).unwrap();
    fs::remove_dir_all(&run_dir).unwrap();
}

#[test]
fn debug_sweep_snippet() {
    ensure_shim_env();
    let workspace = temp_dir("dbg-ws");
    let (opts, run_dir) = options("dbg", &workspace);
    fs::write(run_dir.join("suggestions.md"), "- a suggestion\n").unwrap();
    let host = StubHost::default();
    let agent_log = Arc::clone(&host.agent_log);
    let script = r#"
test -s "$RUN_DIR/suggestions.md" &&
  agent task "Triage:" < "$RUN_DIR/suggestions.md"

exit 0
"#;
    let report = run_ralph(
        Box::new(host),
        opts,
        script,
        &[],
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    eprintln!("report: {report:?}");
    eprintln!("agents: {:?}", agent_log.lock().unwrap().len());
    assert_eq!(report.exit, EXIT_OK);
    assert_eq!(agent_log.lock().unwrap().len(), 1);
    fs::remove_dir_all(&workspace).unwrap();
    fs::remove_dir_all(&run_dir).unwrap();
}
