//! ralph — a verified fixpoint loop for sid.
//!
//! A mostly-POSIX shell (mxsh) embedded in sid is the orchestrator, with sid
//! agents as subroutines.  The shell provides control flow and the exit-code
//! protocol; the LLM appears in exactly two roles — repairer (when `./ci`
//! fails) and judge (when it passes).  The judge never self-reports prose; it
//! emits a structured verdict through a mandated tool, and that verdict is
//! the work order for the next agent.  State lives in files, git, and
//! journals — never in any single context window.
//!
//! Exit-code protocol (§3 of the plan):
//!
//! ```text
//! agent NAME [INSTR]
//!   0   child session ran to completion (says nothing about task success)
//!   3   agent called escalate(reason) — wants a human
//!   ≥4  transport: API error, config error, budget exhausted
//!
//! judge NAME [...] [INSTR]
//!   0   verdict.sufficient == true   (after jury/soak gating)
//!   1   verdict.sufficient == false
//!   3   judge called escalate(reason)
//!   ≥4  transport / config / malformed-verdict failure
//! 130   SIGINT (run level)
//! ```

pub mod args;
pub mod checkpoint;
pub mod host;
pub mod journal;
pub mod runner;
pub mod verdict;

/// The verdict said sufficient (or the agent completed).
pub const EXIT_OK: i32 = 0;
/// The verdict said insufficient.
pub const EXIT_INSUFFICIENT: i32 = 1;
/// The agent or judge called `escalate(reason)`.
pub const EXIT_ESCALATED: i32 = 3;
/// Transport-class failure: API error, config error, budget exhausted,
/// malformed verdict.  Never conflated with a verdict.
pub const EXIT_TRANSPORT: i32 = 4;
/// SIGINT at run level.
pub const EXIT_SIGINT: i32 = 130;
