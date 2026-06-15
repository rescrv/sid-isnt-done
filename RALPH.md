# PLAN: ralph.sid — a fixpoint loop for sid

Ralph is a verified fixpoint loop: a mostly-POSIX shell (mxsh) embedded in sid
as the orchestrator, with sid agents as subroutines. The shell provides control
flow and the exit-code protocol; the LLM appears in exactly two roles —
**repairer** (when `./ci` fails) and **judge** (when it passes). The judge
never self-reports prose; it emits a structured verdict through a mandated
tool, and that verdict is the work order for the next agent. State lives in
files, git, and journals — never in any single context window.

The name is a reference to the Ralph Wiggum technique (Huntley): a bash loop
that re-feeds an agent a prompt until done. Ralph-in-sid is Ralph who learnded:
same naive persistence, but `./ci` is the verifier and a persistent judge with
read-only tools replaces the self-graded "completion promise" that makes the
original overbake.

---

## 1. Scope

Implement, in sid:

1. **Embedded mxsh** invocable as `! <script text>` (inline) and
   `/run <file>.sid` (file) from an interactive sid session.
2. **Two builtins** exposed into mxsh — `agent` and `judge` — which are one
   implementation differing only in *lifecycle* and *verdict mapping*.
3. **The `verdict` tool**, its schema, its rendering, and the forced-tool-choice
   harness behavior.
4. **Run infrastructure**: per-run journal directory, suggestions ledger,
   checkpoint refs, stdin caps, SIGINT handling, resumable step journal,
   parent-transcript injection on exit.
5. **`init/` starter config additions**: a `judge` agent service, the `verdict`
   and `escalate` tool definitions, prompt files, and the reference `ralph.sid`.

Out of scope for v1: parallel agent invocations within one run; cross-run
suggestion persistence; non-Anthropic providers.

---

## 2. Unified agent species, two lifecycles

There is no judge type. Any service in `agents.conf` can serve in either role.
The builtins differ only in lifecycle:

| builtin | lifecycle | session | exit code source |
|---|---|---|---|
| `agent NAME [INSTR]` | fresh per call | new child session, dies at call end | transport + escalate |
| `judge NAME [INSTR]` | pinned per `/run` | born at first call, dies at script exit | `verdict` tool + transport + escalate |

- **stdin is context, argv is instruction.** `printf '%s' "$out" | agent fix
  "Make CI pass."` The agent lands fresh: instruction + piped context + its
  normal agents.conf configuration (tools, skills, AGENTS.md). Nothing else.
- **Judge seeding.** At first `judge` call in a run, the judge's transcript is
  seeded from the *launching interactive thread* — verbatim if under a size
  cap, else via the existing `/compact`-style summarizer. Flag:
  `--seed=full|compact|none` (default: full-if-fits, else compact). A re-run of
  `/run ralph.sid` starts a **fresh judge**, re-seeded from the launch thread
  as it stands at launch time. This is the mechanism by which "grill me about
  this feature, then stop before implementation" loads the judge with the
  operator's design decisions.
- **Calling `judge NAME` on a service whose `_TOOLS` lacks `verdict` is a
  config error** (exit ≥ 4, before any API call).

### Judge flags

```
judge NAME [--jury N | --soak N] [--goldfish] [--pedantic] [INSTR]
```

- `--goldfish` — truncate the judge's transcript to its seed before each call.
- `--jury N` — one call samples N independent goldfish verdicts; all N must
  pass for exit 0. Consensus against single-sample flakiness.
- `--soak N` — counter persists across loop iterations in run state. Pass
  increments; fail resets to 0; exit 0 only at N *consecutive* passes. The
  prompt (not the schema) tells the judge which soak iteration it is on and
  encourages — does not force — fresh scrutiny angles per pass.
- `--pedantic` — treat `suggestion` findings as `required` (any finding ⇒
  exit 1). **Not** the default; see §4.
- `--jury` and `--soak` are mutually exclusive.

---

## 3. Exit-code protocol

The loop's epistemology: `./ci` is the verifier; agents do not self-report
success. The only honest self-report is escalation.

```
agent NAME [INSTR]
  0   child session ran to completion (says nothing about task success)
  3   agent called escalate(reason) — wants a human
  ≥4  transport: API error, config error, budget exhausted

judge NAME [...] [INSTR]
  0   verdict.sufficient == true   (after jury/soak gating)
  1   verdict.sufficient == false
  3   judge called escalate(reason)
  ≥4  transport / config / malformed-verdict failure
      — NEVER conflated with a verdict
130  SIGINT (run level)
```

`judge && break || agent task` therefore behaves correctly only because
verdict (0/1) and malfunction (≥3) are disjoint; scripts that want to
distinguish must use `case $?`. The reference script does.

---

## 4. The verdict tool

### 4.1 Forcing

The judge must call `verdict` to end its turn. If it end-turns without doing
so, the harness re-sends the conversation with
`tool_choice: {"type": "tool", "name": "verdict"}` — a mandated tool
selection, never regex over prose. A verdict with `sufficient=false` and zero
`blocker`/`required` findings is malformed: bounce back once with an error
message; a second malformed verdict is exit ≥ 4.

### 4.2 Schema

```json
{
  "name": "verdict",
  "description": "Render your verdict. The next agent has NO context except what you write here. Write a work order, not a judgment.",
  "input_schema": {
    "type": "object",
    "required": ["sufficient", "summary", "findings"],
    "properties": {
      "sufficient": {"type": "boolean"},
      "summary": {
        "type": "string",
        "description": "One paragraph. The next agent has no other context about your reasoning."
      },
      "findings": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["severity", "where", "what", "why"],
          "properties": {
            "severity": {"enum": ["blocker", "required", "suggestion"]},
            "where": {"type": "string", "description": "file:line or component"},
            "what":  {"type": "string", "description": "Imperative. 'Add X', not 'X is missing'."},
            "why":   {"type": "string", "description": "Tie to the plan or the design thread."}
          }
        }
      },
      "acceptance": {
        "type": "array", "items": {"type": "string"},
        "description": "What you will verify on the next pass. Specific enough that passing is checkable."
      }
    }
  }
}
```

### 4.3 Channels

- Judge transcript streams to **stderr** with normal sid rendering — the
  operator watches it go, as if prompting manually.
- The rendered verdict (markdown, not JSON) goes to **stdout**.
- The boolean goes to the **exit code**.

This makes the pass pure shell plumbing:

```sh
findings=$(judge judge) || printf '%s' "$findings" | agent task "Address the judge's findings."
```

### 4.4 Suggestions ledger (per-run)

`suggestion`-severity findings on a **passing** verdict do not flip the exit
code and are not dropped. They are appended to
`${RUN_DIR}/suggestions.md` and injected into the judge's next-pass context.
A judge that keeps seeing its own suggestion recur may promote it to
`required` — the judge changing its mind with memory, not the harness
overruling the taxonomy. The ledger is scoped to the `/run` and dies with it;
the reference script sweeps it once after the loop.

---

## 5. Run infrastructure

```
/run SCRIPT.sid [--max-iters N] [--budget TOKENS] [--resume RUN_ID]
```

- **Journal**: `${SID_SESSION_DIR}/runs/<run-id>/` containing `steps.jsonl`
  (one record per loop step: deterministic shell op | agent child-session id |
  judge verdict + soak counter), `suggestions.md`, `ci-latest.log`, capped
  context spill files like `ci-NNN.log`, and the judge's pinned transcript.
- **Resume**: `--resume` replays `steps.jsonl` to the last completed step:
  shell state restored, soak counter restored, judge transcript reloaded;
  in-flight step at interrupt is re-executed.
- **Checkpoints**: before each `agent` invocation, commit the tree state to
  `refs/sid/ralph/<run-id>/<step>`. The judge prompt includes the ref of its
  previous verdict's checkpoint so it can diff *what changed since it last
  looked* rather than re-reading the world — this is what makes soak passes
  cheap.
- **stdin cap**: piped context is capped at 64 KiB head + 16 KiB tail with a
  `[truncated: full log at ${RUN_DIR}/ci-NNN.log]` marker; the agent can
  `read` deeper.
- **Budgets**: `--max-iters` (judge-started fixpoint iterations; the run starts
  in implicit iteration 0, so pre-judge repair agents are grouped there) and
  `--budget` (total tokens across all child sessions, riding `_SESSION_BUDGET`
  machinery). Exhaustion is exit 4.
- **SIGINT**: kill the in-flight child session cleanly, journal intact,
  exit 130, return to the sid prompt.
- **Parent linkage on exit**: inject one synthetic user-style turn into the
  parent transcript — as if the operator pasted it — containing run id, exit
  status, iteration counts, the final verdict's `summary`, and the journal
  path. Nothing else. The parent agent answers deeper questions by `read`-ing
  the journal; caching stays intact; the operator can `less` the same files.

```
Ran /run ralph.sid (run 2026-06-10T14-22-07).
Exit 0 after 2 iterations (2 fix, 2 task), judge passed soak 5/5.
Final verdict: "<summary field>"
Journal: <session>/runs/2026-06-10T14-22-07/
Suggestions ledger: 3 entries (triaged by task agent, see steps.jsonl).
```

---

## 6. Starter configuration (deliverables in `init/`)

### 6.1 `agents.conf` additions

```
fix_ENABLED="YES"
fix_TOOLS="bash edit read search git_status git_diff escalate"
fix_DESC="Repairer: lands fresh with a failing CI log on stdin."

task_ENABLED="YES"
task_TOOLS="bash edit read search git_status git_diff escalate"
task_DESC="Worker: lands fresh with the judge's work order on stdin."

judge_ENABLED="YES"
judge_TOOLS="read search git_diff verdict escalate"
judge_MODEL="claude-opus-4-8"
judge_DESC="Examiner: read-only tools plus the mandated verdict."
```

### 6.2 Prompt files

- `agents/fix.md` — you receive a failing CI log on stdin; make `./ci` pass;
  do not expand scope; escalate if the failure is environmental or impossible.
- `agents/task.md` — you receive a judge's work order on stdin and a plan file
  path in your instruction; execute the findings in severity order; the judge
  will re-examine your work, so `acceptance` items are your checklist.
- `agents/judge.md` — you inherited the operator's design thread; you have
  read-only tools; you must end every turn with `verdict`; you will be told
  the soak iteration; vary your angle of scrutiny across passes (e.g., revisit
  each question you asked the operator when the design was grilled); a
  rejection without blocker/required findings is malformed; suggestions are
  for observations you do not stand behind as mandates.

### 6.3 The reference script: `ralph.sid`

```sh
#!/usr/bin/env mxsh
# ralph.sid — verified fixpoint: ./ci is the verifier, the judge is the examiner.
# Invoke:  /run ralph.sid [--max-iters N]
# Vars:    PLAN (default PLAN.md), SOAK (default 3)

PLAN=${PLAN:-PLAN.md}
SOAK=${SOAK:-3}

while :; do
  # Stream CI live while preserving combined output for the fix agent.
  ci_log="$RUN_DIR/ci-latest.log"
  ci_status="$RUN_DIR/ci-latest.status"
  rm -f "$ci_log" "$ci_status"
  { ./ci 2>&1; echo $? > "$ci_status"; } | tee "$ci_log"
  ci_rc=$(cat "$ci_status")
  rm -f "$ci_status"

  if test "x$ci_rc" != x0; then
    agent fix "CI is failing. Make ./ci pass. Do not expand scope." < "$ci_log"
    case $? in
      0) continue ;;
      3) echo "ralph: fix agent escalated" >&2; exit 3 ;;
      *) echo "ralph: fix agent transport failure" >&2; exit 4 ;;
    esac
  fi

  findings=$(judge judge --soak "$SOAK" \
    "CI passes. Is $PLAN complete given the current tree? You are mid-soak; vary your scrutiny.")
  case $? in
    0) break ;;
    1) printf '%s' "$findings" | agent task "Execute this work order against $PLAN." || exit $? ;;
    3) echo "ralph: judge escalated" >&2; exit 3 ;;
    *) echo "ralph: judge malfunction" >&2; exit 4 ;;
  esac
done

# Sweep the per-run suggestions ledger once, after the fixpoint converges.
test -s "$RUN_DIR/suggestions.md" &&
  agent task "Plan is complete. Triage these non-blocking suggestions; act only on the worthwhile ones:" \
    < "$RUN_DIR/suggestions.md"

exit 0
```

---

## 7. Acceptance criteria (what the judge checks)

1. `! echo hello` and `/run` execute mxsh with `agent`/`judge` builtins bound;
   `$RUN_DIR`, `$PLAN`-style environment passthrough works.
2. `agent fix` with stdin spawns a fresh child session, journaled under the
   run dir, streaming to the terminal with normal sid rendering; exit codes
   conform to §3 (verified with a stub that escalates, and a forced API error).
3. `judge judge` pins one session across loop iterations; transcript persists;
   `--goldfish` truncates to seed; re-`/run` produces a fresh judge seeded
   from the launch thread (verified by asking the judge to recall a design
   decision stated only in the launch thread).
4. End-turn without `verdict` triggers exactly one forced-tool-choice
   re-send; malformed rejection (no blocker/required) bounces once then
   exits ≥ 4.
5. `--jury 3` issues three independent goldfish samples; any failure ⇒ exit 1.
   `--soak 3` requires three consecutive passes; an interleaved failure resets
   the counter (visible in `steps.jsonl`).
6. Passing-verdict suggestions land in `suggestions.md`, appear in the judge's
   next-pass context, and the reference script's post-loop sweep feeds them to
   `agent task`.
7. `refs/sid/ralph/<run>/<step>` exists before each agent call; the judge's
   prompt contains the prior checkpoint ref; `git diff <ref>` from inside the
   judge's `git_diff` tool works.
8. SIGINT mid-agent: child dies, `steps.jsonl` is valid, `--resume` continues
   from the last completed step with soak counter and judge transcript intact.
9. A 10 MiB synthetic CI log is capped per §5 with the truncation marker and a
   readable full log on disk.
10. On run exit, the parent transcript contains exactly one injected
    paste-style turn matching §5's shape, and asking the parent "why did the
    judge reject pass 2" causes it to read the journal rather than hallucinate.

---

## Appendix A: README section (embed in repo README under TOOLS or its own RALPH(1) heading)

> ### RALPH(1) — verified fixpoint loops
>
> `ralph.sid` runs a plan to completion: a shell loop runs `./ci`, hands
> failures to a fresh `fix` agent, and when CI passes asks a persistent
> `judge` — seeded with the conversation that launched it — whether the plan
> is done. The judge must answer through a mandated `verdict` tool; its
> structured findings become the work order for a fresh `task` agent. Exit
> codes carry the protocol: 0 done, 1 keep working, 3 a human is wanted,
> ≥4 the machinery broke. `--soak N` demands N consecutive passing verdicts;
> `--jury N` demands N independent ones. Suggestions that don't block accrue
> in a per-run ledger and are triaged once at the end. Everything streams to
> your terminal as if you were prompting by hand, and everything is journaled
> so `--resume` picks up where SIGINT left off.
>
>     $ sid
>     > Grill me about this feature, then stop before implementation.
>     ...design conversation...
>     > /run ralph.sid --max-iters 25
