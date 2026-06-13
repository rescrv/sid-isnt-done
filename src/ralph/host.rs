//! The production [`RalphHost`]: sid child sessions for `agent`, one pinned
//! session for `judge`.
//!
//! Both roles are the same agent species; the builtins differ only in
//! lifecycle and verdict mapping.  Fresh agents land with instruction +
//! piped context + their agents.conf configuration; nothing else.  The judge
//! is seeded once per run from the launching interactive thread and must end
//! every turn through the mandated `verdict` tool: if it end-turns without
//! doing so, the conversation is re-sent once with a forced tool choice; a
//! malformed verdict bounces back once; a second failure of either kind is a
//! transport-class error, never conflated with a verdict.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use claudius::chat::{ChatConfig, ChatSession};
use claudius::{Anthropic, MessageParam, Renderer, ToolChoice};
use utf8path::Path;

use crate::config::Config;
use crate::render::PlainTextRenderer;
use crate::session::SidSession;
use crate::{
    COMPACTION_REQUEST_PROMPT, SidAgent, compacted_transcript, extract_last_assistant_text,
    load_transcript_messages, sanitize_transcript_messages,
};

use super::args::SeedMode;
use super::runner::{
    AgentCallResult, AgentInvocation, AgentOutcome, JudgeCallResult, JudgeInvocation, JudgeOutcome,
    RalphHost,
};
use super::verdict::{
    ExchangeAction, ExchangeEvent, ExchangeGuard, VERDICT_TOOL_NAME, Verdict, find_escalation,
    last_verdict_input,
};

/// Serialized-seed size above which the default seeding falls back to the
/// compaction summarizer.
pub const SEED_FULL_CAP_BYTES: usize = 256 * 1024;

/// How the judge's transcript gets seeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeedDecision {
    /// Seed verbatim from the launch thread.
    Full,
    /// Seed from a compacted summary of the launch thread.
    Compact,
    /// Start cold.
    Empty,
}

/// Pure seeding policy: full-if-fits, else compact (§2).
pub fn choose_seed(mode: SeedMode, serialized_len: usize, cap: usize) -> SeedDecision {
    match mode {
        SeedMode::Full => SeedDecision::Full,
        SeedMode::Compact => SeedDecision::Compact,
        SeedMode::None => SeedDecision::Empty,
        SeedMode::Auto => {
            if serialized_len <= cap {
                SeedDecision::Full
            } else {
                SeedDecision::Compact
            }
        }
    }
}

/// Check that `service` exists in the loaded configuration and is wired with
/// the `verdict` tool.  Calling `judge NAME` on a service whose `_TOOLS`
/// lacks `verdict` is a config error (exit ≥ 4, before any API call).
pub fn validate_judge_config(config: &Config, service: &str) -> Result<(), String> {
    let Some(agent) = config.agents.get(service) else {
        return Err(format!(
            "config error: agent {service:?} is not defined in agents.conf"
        ));
    };
    if !agent.tools.iter().any(|tool| tool == VERDICT_TOOL_NAME) {
        return Err(format!(
            "config error: agent {service:?} cannot serve as a judge: its _TOOLS lacks \
             the `verdict` tool"
        ));
    }
    Ok(())
}

/// Factory for renderers used by ralph child sessions.
pub trait RalphRendererFactory: Send + Sync {
    /// Build a renderer for a child stream labeled with `label`.
    fn renderer(
        &self,
        label: &str,
        use_color: bool,
        interrupted: Arc<AtomicBool>,
    ) -> Box<dyn Renderer + Send + 'static>;
}

struct PlainTextRalphRendererFactory;

impl RalphRendererFactory for PlainTextRalphRendererFactory {
    fn renderer(
        &self,
        _label: &str,
        use_color: bool,
        interrupted: Arc<AtomicBool>,
    ) -> Box<dyn Renderer + Send + 'static> {
        Box::new(PlainTextRenderer::stderr_with_color_and_interrupt(
            use_color,
            interrupted,
        ))
    }
}

struct PinnedJudge {
    service: String,
    chat: ChatSession<SidAgent>,
    seed_len: usize,
    one_shot_tool_choice: Arc<Mutex<Option<ToolChoice>>>,
}

/// The production host: child sid sessions journaled under the run dir.
pub struct SidRalphHost {
    workspace_root: Path<'static>,
    config_root: Path<'static>,
    fallback: ChatConfig,
    run_dir: PathBuf,
    seed_messages: Vec<MessageParam>,
    parent_session_id: String,
    handle: tokio::runtime::Handle,
    interrupted: Arc<AtomicBool>,
    renderer_factory: Arc<dyn RalphRendererFactory>,
    judge: Option<PinnedJudge>,
    seed_cap_bytes: usize,
}

impl SidRalphHost {
    /// Build a host.  `seed_messages` is the launching interactive thread at
    /// launch time; a re-`/run` re-seeds a fresh judge from the thread as it
    /// stands then.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        workspace_root: Path<'static>,
        config_root: Path<'static>,
        fallback: ChatConfig,
        run_dir: PathBuf,
        seed_messages: Vec<MessageParam>,
        parent_session_id: String,
        handle: tokio::runtime::Handle,
        interrupted: Arc<AtomicBool>,
    ) -> SidRalphHost {
        SidRalphHost {
            workspace_root,
            config_root,
            fallback,
            run_dir,
            seed_messages,
            parent_session_id,
            handle,
            interrupted,
            renderer_factory: Arc::new(PlainTextRalphRendererFactory),
            judge: None,
            seed_cap_bytes: SEED_FULL_CAP_BYTES,
        }
    }

    /// Replace the renderer factory used for child sessions.
    pub fn with_renderer_factory(
        mut self,
        renderer_factory: Arc<dyn RalphRendererFactory>,
    ) -> Self {
        self.renderer_factory = renderer_factory;
        self
    }

    fn interrupted(&self) -> bool {
        self.interrupted.load(Ordering::Relaxed)
    }

    fn child_fallback(&self) -> ChatConfig {
        let mut fallback = self.fallback.clone();
        fallback.set_system_prompt(None);
        fallback.transcript_path = None;
        fallback
    }

    fn build_client(agent: &SidAgent) -> Result<Anthropic, String> {
        let config = agent.anthropic_config();
        let client = Anthropic::new(config.api_key.clone())
            .map_err(|err| format!("failed to initialize the Anthropic client: {err}"))?;
        Ok(match config.base_url.as_ref() {
            Some(base_url) => client.with_base_url(base_url.clone()),
            None => client,
        })
    }

    fn create_child_session(&self) -> Result<Arc<SidSession>, String> {
        SidSession::create_in(self.run_dir.join("sessions"))
            .map(Arc::new)
            .map_err(|err| format!("failed to create child session: {err}"))
    }

    /// Build a chat for `service`, returning the chat together with the
    /// agent's one-shot tool-choice cell (captured before the chat takes
    /// ownership of the agent).
    #[allow(clippy::type_complexity)]
    fn build_chat(
        &self,
        service: &str,
        session: Arc<SidSession>,
    ) -> Result<(ChatSession<SidAgent>, Arc<Mutex<Option<ToolChoice>>>), String> {
        let agent = SidAgent::from_workspace_agent_with_config_root(
            &self.workspace_root,
            &self.config_root,
            service,
            self.child_fallback(),
        )
        .map_err(|err| format!("failed to load agent {service:?}: {err}"))?
        .with_session(session);
        let cell = agent.one_shot_tool_choice_cell();
        let client = Self::build_client(&agent)?;
        Ok((ChatSession::with_agent(client, agent), cell))
    }

    fn judge_transcript_path(&self) -> PathBuf {
        self.run_dir.join("judge-transcript.json")
    }

    fn judge_seed_len_path(&self) -> PathBuf {
        self.run_dir.join("judge-seed-len")
    }

    fn seed_judge_messages(&self, mode: SeedMode) -> Result<Vec<MessageParam>, String> {
        let serialized_len = serde_json::to_string(&self.seed_messages)
            .map(|s| s.len())
            .unwrap_or(usize::MAX);
        Ok(
            match choose_seed(mode, serialized_len, self.seed_cap_bytes) {
                SeedDecision::Empty => Vec::new(),
                SeedDecision::Full => {
                    let mut messages = self.seed_messages.clone();
                    sanitize_transcript_messages(&mut messages);
                    messages
                }
                SeedDecision::Compact => {
                    let summary = self.compact_seed()?;
                    compacted_transcript(&self.parent_session_id, &summary)
                }
            },
        )
    }

    fn compact_seed(&self) -> Result<String, String> {
        let compactor = SidAgent::from_workspace_compactor_with_config_root(
            &self.workspace_root,
            &self.config_root,
            self.child_fallback(),
        )
        .map_err(|err| format!("failed to load compaction agent: {err}"))?;
        let prompt = compactor
            .named_prompt_markdown(crate::config::COMPACTION_PROMPT_ID)
            .unwrap_or(COMPACTION_REQUEST_PROMPT)
            .to_string();
        let client = Self::build_client(&compactor)?;
        let mut chat = ChatSession::with_agent(client, compactor);
        let mut messages = self.seed_messages.clone();
        sanitize_transcript_messages(&mut messages);
        chat.replace_messages(messages);
        let mut renderer = self.renderer_factory.renderer(
            "seed-compact",
            chat.config().use_color,
            Arc::clone(&self.interrupted),
        );
        self.handle
            .block_on(chat.send_message(MessageParam::user(prompt), renderer.as_mut()))
            .map_err(|err| format!("seed compaction failed: {err}"))?;
        extract_last_assistant_text(&chat.clone_messages())
            .ok_or_else(|| "seed compaction produced no summary".to_string())
    }

    fn ensure_judge(&mut self, invocation: &JudgeInvocation) -> Result<(), String> {
        if let Some(judge) = self.judge.as_ref() {
            if judge.service == invocation.service {
                return Ok(());
            }
            return Err(format!(
                "judge {:?} is already pinned for this run; cannot also pin {:?}",
                judge.service, invocation.service
            ));
        }
        let session = self.create_child_session()?;
        let (mut chat, one_shot_tool_choice) = self.build_chat(&invocation.service, session)?;
        let transcript_path = self.judge_transcript_path();
        let seed_len;
        if transcript_path.is_file() {
            // Resume: reload the pinned transcript and its recorded seed length.
            let mut messages = load_transcript_messages(&transcript_path)
                .map_err(|err| format!("failed to reload judge transcript: {err}"))?;
            sanitize_transcript_messages(&mut messages);
            seed_len = fs::read_to_string(self.judge_seed_len_path())
                .ok()
                .and_then(|text| text.trim().parse().ok())
                .unwrap_or(messages.len());
            chat.replace_messages(messages);
        } else {
            let messages = self.seed_judge_messages(invocation.seed)?;
            seed_len = messages.len();
            chat.replace_messages(messages);
            let _ = fs::write(self.judge_seed_len_path(), format!("{seed_len}\n"));
        }
        self.judge = Some(PinnedJudge {
            service: invocation.service.clone(),
            chat,
            seed_len,
            one_shot_tool_choice,
        });
        Ok(())
    }

    fn persist_judge_transcript(&self) {
        let Some(judge) = self.judge.as_ref() else {
            return;
        };
        let _ = judge.chat.save_transcript_to(self.judge_transcript_path());
    }

    fn tokens_delta(
        before: &claudius::chat::SessionStats,
        after: &claudius::chat::SessionStats,
    ) -> u64 {
        let before = before
            .total_input_tokens
            .saturating_add(before.total_output_tokens);
        let after = after
            .total_input_tokens
            .saturating_add(after.total_output_tokens);
        after.saturating_sub(before)
    }
}

impl RalphHost for SidRalphHost {
    fn validate_judge(&mut self, service: &str) -> Result<(), String> {
        let config = Config::load(&self.config_root)
            .map_err(|err| format!("config error: failed to load configuration: {err}"))?;
        validate_judge_config(&config, service)
    }

    fn run_agent(&mut self, invocation: &AgentInvocation) -> AgentCallResult {
        if self.interrupted() {
            return AgentCallResult {
                outcome: AgentOutcome::Interrupted,
                tokens: 0,
                session: None,
            };
        }
        let session = match self.create_child_session() {
            Ok(session) => session,
            Err(err) => {
                return AgentCallResult {
                    outcome: AgentOutcome::Transport(err),
                    tokens: 0,
                    session: None,
                };
            }
        };
        let session_id = session.id().to_string();
        let transcript_path = session.transcript_path();
        let (mut chat, _cell) = match self.build_chat(&invocation.service, session) {
            Ok(chat) => chat,
            Err(err) => {
                return AgentCallResult {
                    outcome: AgentOutcome::Transport(err),
                    tokens: 0,
                    session: Some(session_id),
                };
            }
        };

        let mut text = if invocation.instruction.is_empty() {
            "Proceed with your configured role.".to_string()
        } else {
            invocation.instruction.clone()
        };
        if !invocation.context.is_empty() {
            text.push_str("\n\n## Context (piped)\n\n");
            text.push_str(&invocation.context);
        }

        let stats_before = chat.stats();
        let mut renderer = self.renderer_factory.renderer(
            &invocation.service,
            chat.config().use_color,
            Arc::clone(&self.interrupted),
        );
        let result = self
            .handle
            .block_on(chat.send_message(MessageParam::user(text), renderer.as_mut()));
        let _ = chat.save_transcript_to(&transcript_path);
        let tokens = Self::tokens_delta(&stats_before, &chat.stats());

        let outcome = match result {
            Err(err) => AgentOutcome::Transport(format!("API failure: {err}")),
            Ok(()) => {
                if let Some(reason) = find_escalation(&chat.clone_messages(), 0) {
                    AgentOutcome::Escalated(reason)
                } else if self.interrupted() {
                    AgentOutcome::Interrupted
                } else {
                    AgentOutcome::Completed
                }
            }
        };
        AgentCallResult {
            outcome,
            tokens,
            session: Some(session_id),
        }
    }

    fn judge_sample(&mut self, invocation: &JudgeInvocation) -> JudgeCallResult {
        if self.interrupted() {
            return JudgeCallResult {
                outcome: JudgeOutcome::Interrupted,
                tokens: 0,
            };
        }
        if let Err(err) = self.ensure_judge(invocation) {
            return JudgeCallResult {
                outcome: JudgeOutcome::Transport(err),
                tokens: 0,
            };
        }

        // --goldfish: truncate the pinned transcript to its seed.
        if invocation.goldfish {
            let judge = self.judge.as_mut().expect("judge was just ensured");
            let mut messages = judge.chat.clone_messages();
            messages.truncate(judge.seed_len);
            judge.chat.replace_messages(messages);
        }

        let interrupted = Arc::clone(&self.interrupted);
        let label = invocation.service.clone();
        let handle = self.handle.clone();
        let renderer_factory = Arc::clone(&self.renderer_factory);
        let judge = self.judge.as_mut().expect("judge was just ensured");
        let stats_before = judge.chat.stats();

        let mut guard = ExchangeGuard::default();
        let mut next_message = MessageParam::user(invocation.prompt.clone());
        let mut force = false;
        let outcome = loop {
            if interrupted.load(Ordering::Relaxed) {
                break JudgeOutcome::Interrupted;
            }
            let before_len = judge.chat.clone_messages().len();
            if force {
                *judge
                    .one_shot_tool_choice
                    .lock()
                    .expect("tool choice cell poisoned") = Some(ToolChoice::Tool {
                    name: VERDICT_TOOL_NAME.to_string(),
                    disable_parallel_tool_use: None,
                });
                force = false;
            }
            let mut renderer = renderer_factory.renderer(
                &label,
                judge.chat.config().use_color,
                Arc::clone(&interrupted),
            );
            let result = handle.block_on(judge.chat.send_message(next_message, renderer.as_mut()));
            if let Err(err) = result {
                break JudgeOutcome::Transport(format!("API failure: {err}"));
            }
            let messages = judge.chat.clone_messages();
            if let Some(reason) = find_escalation(&messages, before_len) {
                break JudgeOutcome::Escalated(reason);
            }
            let mut parsed: Option<Verdict> = None;
            let event = match last_verdict_input(&messages, before_len) {
                None => ExchangeEvent::NoVerdict,
                Some(input) => match Verdict::from_input(&input) {
                    Err(reason) => ExchangeEvent::Malformed(reason),
                    Ok(verdict) => match verdict.validate() {
                        Err(reason) => ExchangeEvent::Malformed(reason),
                        Ok(()) => {
                            parsed = Some(verdict);
                            ExchangeEvent::Ok
                        }
                    },
                },
            };
            match guard.observe(event) {
                ExchangeAction::Accept => {
                    break JudgeOutcome::Verdict(parsed.expect("Ok implies a parsed verdict"));
                }
                ExchangeAction::ForceVerdict => {
                    force = true;
                    next_message = MessageParam::user(
                        "You ended your turn without rendering a verdict. \
                         Call the `verdict` tool now.",
                    );
                }
                ExchangeAction::Bounce(reason) => {
                    next_message = MessageParam::user(format!(
                        "{reason}\n\nRe-render your verdict through the `verdict` tool."
                    ));
                }
                ExchangeAction::Fail(reason) => {
                    break JudgeOutcome::Transport(reason);
                }
            }
        };

        let tokens = {
            let judge = self.judge.as_ref().expect("judge was just ensured");
            Self::tokens_delta(&stats_before, &judge.chat.stats())
        };
        self.persist_judge_transcript();
        JudgeCallResult { outcome, tokens }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_decision_full_if_fits_else_compact() {
        assert_eq!(choose_seed(SeedMode::Auto, 100, 1000), SeedDecision::Full);
        assert_eq!(
            choose_seed(SeedMode::Auto, 1001, 1000),
            SeedDecision::Compact
        );
        assert_eq!(
            choose_seed(SeedMode::Full, 1_000_000, 10),
            SeedDecision::Full
        );
        assert_eq!(
            choose_seed(SeedMode::Compact, 1, 1000),
            SeedDecision::Compact
        );
        assert_eq!(choose_seed(SeedMode::None, 1, 1000), SeedDecision::Empty);
    }

    #[test]
    fn judge_validation_demands_the_verdict_tool() {
        // The shipped starter configuration must validate: judge carries
        // verdict; build does not.
        let init_root = utf8path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/init"));
        let config = Config::load(&init_root).expect("the shipped init/ config must load");
        validate_judge_config(&config, "judge").expect("judge must carry the verdict tool");
        let err = validate_judge_config(&config, "build").unwrap_err();
        assert!(err.contains("_TOOLS lacks"), "{err}");
        let err = validate_judge_config(&config, "no-such-agent").unwrap_err();
        assert!(err.contains("not defined"), "{err}");
    }
}
