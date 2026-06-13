//! The `verdict` tool: schema, parsing, validation, and rendering.
//!
//! The judge must end its turn by calling `verdict`.  The harness extracts the
//! structured verdict from the transcript, validates it, and renders it as a
//! markdown work order for the next agent.  The boolean lands in the exit
//! code; the rendering lands on stdout; the judge's reasoning streams to
//! stderr.

use claudius::{ContentBlock, MessageParam, MessageParamContent};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Name of the mandated verdict tool.
pub const VERDICT_TOOL_NAME: &str = "verdict";
/// Name of the escalation tool shared by agents and judges.
pub const ESCALATE_TOOL_NAME: &str = "escalate";

/// Severity taxonomy for verdict findings.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// The plan cannot be considered done until this is fixed.
    Blocker,
    /// Must be addressed before the verdict can flip to sufficient.
    Required,
    /// An observation the judge does not stand behind as a mandate.
    Suggestion,
}

impl Severity {
    /// True for severities that justify an insufficient verdict.
    pub fn is_mandate(self) -> bool {
        matches!(self, Severity::Blocker | Severity::Required)
    }

    /// Lowercase label used in rendering.
    pub fn label(self) -> &'static str {
        match self {
            Severity::Blocker => "blocker",
            Severity::Required => "required",
            Severity::Suggestion => "suggestion",
        }
    }
}

/// A single finding in a verdict.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Finding {
    /// Severity of the finding.
    pub severity: Severity,
    /// file:line or component the finding refers to.
    #[serde(rename = "where")]
    pub where_: String,
    /// Imperative description of the work ("Add X", not "X is missing").
    pub what: String,
    /// Why this matters, tied to the plan or design thread.
    pub why: String,
}

/// The structured verdict emitted through the mandated tool.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Verdict {
    /// Whether the work under examination is sufficient.
    pub sufficient: bool,
    /// One-paragraph summary; the next agent has no other context.
    pub summary: String,
    /// Structured findings; the work order for the next agent.
    #[serde(default)]
    pub findings: Vec<Finding>,
    /// What the judge will verify on the next pass.
    #[serde(default)]
    pub acceptance: Vec<String>,
}

impl Verdict {
    /// Parse a verdict from the raw `tool_use` input JSON.
    pub fn from_input(input: &Value) -> Result<Verdict, String> {
        serde_json::from_value(input.clone()).map_err(|err| format!("malformed verdict: {err}"))
    }

    /// A rejection without a mandate is malformed: the next agent would have
    /// nothing actionable to execute.
    pub fn validate(&self) -> Result<(), String> {
        if !self.sufficient && !self.findings.iter().any(|f| f.severity.is_mandate()) {
            return Err(
                "malformed verdict: sufficient=false requires at least one blocker or \
                 required finding; emit a work order, not a bare rejection"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// The verdict boolean after `--pedantic` gating: any finding at all
    /// (including suggestions) flips the verdict to insufficient.
    pub fn effective_sufficient(&self, pedantic: bool) -> bool {
        if pedantic {
            self.sufficient && self.findings.is_empty()
        } else {
            self.sufficient
        }
    }

    /// Suggestion-severity findings, which accrue in the per-run ledger.
    pub fn suggestions(&self) -> Vec<&Finding> {
        self.findings
            .iter()
            .filter(|f| matches!(f.severity, Severity::Suggestion))
            .collect()
    }

    /// Render the verdict as a markdown work order for the next agent.
    pub fn render_markdown(&self) -> String {
        let mut out = String::new();
        let status = if self.sufficient {
            "sufficient"
        } else {
            "insufficient"
        };
        out.push_str(&format!("# Verdict: {status}\n\n"));
        out.push_str(self.summary.trim());
        out.push_str("\n\n## Findings\n\n");
        if self.findings.is_empty() {
            out.push_str("No findings.\n");
        } else {
            for (i, finding) in self.findings.iter().enumerate() {
                out.push_str(&format!(
                    "{}. **{}** — {}\n   - What: {}\n   - Why: {}\n",
                    i + 1,
                    finding.severity.label(),
                    finding.where_,
                    finding.what,
                    finding.why,
                ));
            }
        }
        if !self.acceptance.is_empty() {
            out.push_str("\n## Acceptance\n\n");
            for item in &self.acceptance {
                out.push_str(&format!("- {item}\n"));
            }
        }
        out
    }
}

/// The JSON Schema for the `verdict` tool input, exactly as mandated by the
/// plan (§4.2).
pub fn verdict_input_schema() -> Value {
    json!({
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
    })
}

/// The full tool description for the mandated verdict tool.
pub fn verdict_tool_description() -> &'static str {
    "Render your verdict. The next agent has NO context except what you write here. \
     Write a work order, not a judgment."
}

/// Governs the judge exchange: at most one forced-tool-choice re-send when the
/// judge end-turns without calling `verdict`, and at most one bounce when the
/// verdict is malformed.  A second occurrence of either is a harness failure
/// (exit ≥ 4), never conflated with a verdict.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExchangeGuard {
    forced: bool,
    bounced: bool,
}

/// What the harness observed after a judge turn.
#[derive(Clone, Debug, PartialEq)]
pub enum ExchangeEvent {
    /// The judge ended its turn without calling `verdict`.
    NoVerdict,
    /// The judge produced a verdict that failed validation.
    Malformed(String),
    /// The judge produced a well-formed verdict.
    Ok,
}

/// What the harness should do next.
#[derive(Clone, Debug, PartialEq)]
pub enum ExchangeAction {
    /// Re-send the conversation with `tool_choice: {"type":"tool","name":"verdict"}`.
    ForceVerdict,
    /// Bounce the malformed verdict back with the given error message.
    Bounce(String),
    /// Accept the verdict.
    Accept,
    /// Give up: transport/config-class failure (exit ≥ 4).
    Fail(String),
}

impl ExchangeGuard {
    /// Advance the guard with an observed event and learn what to do next.
    pub fn observe(&mut self, event: ExchangeEvent) -> ExchangeAction {
        match event {
            ExchangeEvent::Ok => ExchangeAction::Accept,
            ExchangeEvent::NoVerdict => {
                if self.forced {
                    ExchangeAction::Fail(
                        "judge ended its turn without calling verdict even under forced \
                         tool choice"
                            .to_string(),
                    )
                } else {
                    self.forced = true;
                    ExchangeAction::ForceVerdict
                }
            }
            ExchangeEvent::Malformed(reason) => {
                if self.bounced {
                    ExchangeAction::Fail(format!("second malformed verdict: {reason}"))
                } else {
                    self.bounced = true;
                    ExchangeAction::Bounce(reason)
                }
            }
        }
    }
}

/// Collect the inputs of every `tool_use` block named `tool` in
/// `messages[from..]`, in transcript order.
pub fn tool_use_inputs_since(messages: &[MessageParam], from: usize, tool: &str) -> Vec<Value> {
    let mut inputs = Vec::new();
    for message in messages.iter().skip(from) {
        let MessageParamContent::Array(blocks) = &message.content else {
            continue;
        };
        for block in blocks {
            if let ContentBlock::ToolUse(tool_use) = block
                && tool_use.name == tool
            {
                inputs.push(tool_use.input.clone());
            }
        }
    }
    inputs
}

/// The input of the last `verdict` call in `messages[from..]`, if any.
pub fn last_verdict_input(messages: &[MessageParam], from: usize) -> Option<Value> {
    tool_use_inputs_since(messages, from, VERDICT_TOOL_NAME)
        .into_iter()
        .next_back()
}

/// The reason from the first `escalate` call in `messages[from..]`, if any.
pub fn find_escalation(messages: &[MessageParam], from: usize) -> Option<String> {
    tool_use_inputs_since(messages, from, ESCALATE_TOOL_NAME)
        .into_iter()
        .next()
        .map(|input| {
            input
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("escalation requested without a reason")
                .to_string()
        })
}

#[cfg(test)]
mod tests {
    use claudius::{MessageRole, ToolUseBlock};

    use super::*;

    fn verdict(sufficient: bool, findings: Vec<Finding>) -> Verdict {
        Verdict {
            sufficient,
            summary: "summary".to_string(),
            findings,
            acceptance: Vec::new(),
        }
    }

    fn finding(severity: Severity) -> Finding {
        Finding {
            severity,
            where_: "src/lib.rs:1".to_string(),
            what: "Add the thing".to_string(),
            why: "The plan demands it".to_string(),
        }
    }

    #[test]
    fn schema_matches_plan() {
        let schema = verdict_input_schema();
        assert_eq!(
            schema["required"],
            json!(["sufficient", "summary", "findings"])
        );
        assert_eq!(
            schema["properties"]["findings"]["items"]["required"],
            json!(["severity", "where", "what", "why"])
        );
        assert_eq!(
            schema["properties"]["findings"]["items"]["properties"]["severity"]["enum"],
            json!(["blocker", "required", "suggestion"])
        );
        assert_eq!(schema["properties"]["acceptance"]["type"], json!("array"));
    }

    #[test]
    fn shipped_tool_manifest_matches_the_mandated_schema() {
        let manifest: Value =
            serde_json::from_str(include_str!("../../init/tools/verdict.json")).unwrap();
        assert_eq!(manifest["input_schema"], verdict_input_schema());
        assert_eq!(
            manifest["description"].as_str().unwrap(),
            verdict_tool_description()
        );
    }

    #[test]
    fn parses_verdict_with_where_keyword() {
        let input = json!({
            "sufficient": false,
            "summary": "Not done.",
            "findings": [
                {"severity": "blocker", "where": "ci", "what": "Fix CI", "why": "It fails"}
            ],
            "acceptance": ["ci passes"]
        });
        let verdict = Verdict::from_input(&input).unwrap();
        assert!(!verdict.sufficient);
        assert_eq!(verdict.findings[0].where_, "ci");
        assert_eq!(verdict.acceptance, vec!["ci passes".to_string()]);
    }

    #[test]
    fn missing_required_fields_is_an_error() {
        let input = json!({"sufficient": true});
        assert!(Verdict::from_input(&input).is_err());
    }

    #[test]
    fn findings_and_acceptance_default_to_empty() {
        let input = json!({"sufficient": true, "summary": "Done.", "findings": []});
        let verdict = Verdict::from_input(&input).unwrap();
        assert!(verdict.findings.is_empty());
        assert!(verdict.acceptance.is_empty());
    }

    #[test]
    fn rejection_without_mandate_is_malformed() {
        let bare = verdict(false, vec![]);
        assert!(bare.validate().is_err());
        let suggestion_only = verdict(false, vec![finding(Severity::Suggestion)]);
        assert!(suggestion_only.validate().is_err());
        let with_required = verdict(false, vec![finding(Severity::Required)]);
        assert!(with_required.validate().is_ok());
        let with_blocker = verdict(false, vec![finding(Severity::Blocker)]);
        assert!(with_blocker.validate().is_ok());
        let passing = verdict(true, vec![]);
        assert!(passing.validate().is_ok());
    }

    #[test]
    fn pedantic_promotes_suggestions() {
        let v = verdict(true, vec![finding(Severity::Suggestion)]);
        assert!(v.effective_sufficient(false));
        assert!(!v.effective_sufficient(true));
        let clean = verdict(true, vec![]);
        assert!(clean.effective_sufficient(true));
    }

    #[test]
    fn render_markdown_is_a_work_order() {
        let v = Verdict {
            sufficient: false,
            summary: "CI passes but the plan is incomplete.".to_string(),
            findings: vec![finding(Severity::Blocker)],
            acceptance: vec!["the thing exists".to_string()],
        };
        let md = v.render_markdown();
        assert!(md.starts_with("# Verdict: insufficient\n"));
        assert!(md.contains("CI passes but the plan is incomplete."));
        assert!(md.contains("1. **blocker** — src/lib.rs:1"));
        assert!(md.contains("- What: Add the thing"));
        assert!(md.contains("- Why: The plan demands it"));
        assert!(md.contains("## Acceptance"));
        assert!(md.contains("- the thing exists"));
    }

    #[test]
    fn render_markdown_without_findings() {
        let v = verdict(true, vec![]);
        let md = v.render_markdown();
        assert!(md.starts_with("# Verdict: sufficient\n"));
        assert!(md.contains("No findings."));
        assert!(!md.contains("## Acceptance"));
    }

    #[test]
    fn exchange_guard_forces_exactly_once() {
        let mut guard = ExchangeGuard::default();
        assert_eq!(
            guard.observe(ExchangeEvent::NoVerdict),
            ExchangeAction::ForceVerdict
        );
        match guard.observe(ExchangeEvent::NoVerdict) {
            ExchangeAction::Fail(_) => {}
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    #[test]
    fn exchange_guard_bounces_exactly_once() {
        let mut guard = ExchangeGuard::default();
        assert_eq!(
            guard.observe(ExchangeEvent::Malformed("no mandate".to_string())),
            ExchangeAction::Bounce("no mandate".to_string())
        );
        match guard.observe(ExchangeEvent::Malformed("still no mandate".to_string())) {
            ExchangeAction::Fail(_) => {}
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    #[test]
    fn exchange_guard_accepts_after_force_then_bounce() {
        let mut guard = ExchangeGuard::default();
        assert_eq!(
            guard.observe(ExchangeEvent::NoVerdict),
            ExchangeAction::ForceVerdict
        );
        assert_eq!(
            guard.observe(ExchangeEvent::Malformed("bad".to_string())),
            ExchangeAction::Bounce("bad".to_string())
        );
        assert_eq!(guard.observe(ExchangeEvent::Ok), ExchangeAction::Accept);
    }

    fn tool_use_message(name: &str, input: Value) -> MessageParam {
        MessageParam::new_with_blocks(
            vec![ContentBlock::ToolUse(ToolUseBlock {
                id: "toolu_1".to_string(),
                input,
                name: name.to_string(),
                cache_control: None,
            })],
            MessageRole::Assistant,
        )
    }

    #[test]
    fn extracts_last_verdict_after_index() {
        let messages = vec![
            MessageParam::user("hello"),
            tool_use_message(VERDICT_TOOL_NAME, json!({"sufficient": false})),
            MessageParam::user("next pass"),
            tool_use_message(
                VERDICT_TOOL_NAME,
                json!({"sufficient": true, "summary": "ok", "findings": []}),
            ),
        ];
        let input = last_verdict_input(&messages, 2).unwrap();
        assert_eq!(input["sufficient"], json!(true));
        assert!(last_verdict_input(&messages, 4).is_none());
    }

    #[test]
    fn finds_escalation_reason() {
        let messages = vec![
            MessageParam::user("go"),
            tool_use_message(ESCALATE_TOOL_NAME, json!({"reason": "need a human"})),
        ];
        assert_eq!(
            find_escalation(&messages, 0).as_deref(),
            Some("need a human")
        );
        assert!(find_escalation(&messages, 2).is_none());
        let no_reason = vec![tool_use_message(ESCALATE_TOOL_NAME, json!({}))];
        assert_eq!(
            find_escalation(&no_reason, 0).as_deref(),
            Some("escalation requested without a reason")
        );
    }
}
