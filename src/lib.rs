//! A UNIX-inspired coding agent for Anthropic-compatible APIs.
//!
//! `sid-isnt-done` combines an rc-conf configuration system with the Anthropic
//! messages API to produce an interactive, tool-using coding agent.  Agents,
//! tools, and skills are defined through plain configuration files rather than
//! hard-coded tool lists, and the runtime manages sessions, sandboxing, and
//! tool invocation lifecycle.
//!
//! The primary entry point is [`SidAgent`], which assembles a chat
//! configuration, tool bindings, filesystem mounts, and sandbox policy into a
//! single [`claudius::Agent`] implementation.

#![deny(missing_docs)]

/// Built-in tool implementations (editor, read-only viewer).
pub mod builtin_tools;
/// Workspace configuration loading and types.
pub mod config;
mod filesystem;
mod retry;
/// macOS Seatbelt sandbox integration.
pub mod seatbelt;
/// Session lifecycle: creation, resumption, journalling, and transcripts.
pub mod session;
/// Semantic diff rendering with syntax-aware annotations.
pub mod sidiff;
/// Skill-reference injection into user messages.
pub mod skill_inject;
#[cfg(test)]
pub(crate) mod test_support;
mod tool_protocol;
mod tool_runtime;
mod user_instructions;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use claudius::chat::{ChatAgent, ChatConfig};
use claudius::{
    Agent, AgentStreamContext, Anthropic, BashPtyConfig, BashPtyResult, BashPtySession, Budget,
    Content, ContentBlock, Error, FileSystem, IntermediateToolResult, Message, MessageParam,
    MessageParamContent, MessageRole, Metadata, Model, MountHierarchy, OperatorLine, Renderer,
    StopReason, StreamContext, SystemPrompt, TextBlock, ThinkingConfig, TokenRates, Tool,
    ToolBash20250124, ToolCallback, ToolChoice, ToolParam, ToolResult, ToolResultBlock,
    ToolResultBlockContent, ToolTextEditor20250728, ToolUnionParam, ToolUseBlock, TurnOutcome,
    Usage,
};
use handled::SError;
use rc_conf::SwitchPosition;
use tokio::sync::Mutex;
use utf8path::Path;

use crate::config::{
    AGENTS_CONF_FILE, AgentConfig, Config, MEMORY_EXPERT_PROMPT_ID, SYSTEM_PROMPT_ID, SkillConfig,
    TOOLS_CONF_FILE, ToolConfig, is_valid_anthropic_tool_name, resolve_canonical_tool_id,
};
use crate::filesystem::{build_agent_filesystem, build_default_filesystem, resolve_agent_skills};
use crate::seatbelt::WritableRoots;
use crate::session::{CompactionExpertConfig, CompactionProvenance, SidSession};
use crate::tool_runtime::ToolRuntimeContext;
use crate::user_instructions::{
    UserInstructionRuntimeContext, UserInstructionSettings, append_agents_md_to_system_prompt,
    append_user_instruction_block, build_user_instruction_block,
    disabled_user_instruction_settings, resolve_user_instruction_settings,
};

const DEFAULT_AGENT_ID: &str = "sid";
const DEFAULT_COMPACTOR_AGENT_ID: &str = "compact";
const USER_CANCELLED_ACTION: &str = "user cancelled action";
const ASK_AN_EXPERT_TOOL_NAME: &str = "ask_an_expert";
const MAX_ASK_AN_EXPERT_DEPTH: usize = 8;
const BASH_STATE_CAPTURE_COMMAND: &str = concat!(
    "builtin printf 'builtin cd -- %q\\n' \"$PWD\"\n",
    "builtin set +o\n",
    "builtin shopt -p\n",
    "builtin export -p\n",
    "builtin alias -p\n",
    "builtin declare -pf\n",
);
const DEFAULT_COMPACTOR_SYSTEM_PROMPT: &str = concat!(
    "You are the sid compaction agent.\n",
    "Turn the conversation history into a compact, high-signal handoff summary for a future session.\n",
    "Preserve objectives, constraints, decisions, unfinished work, notable files, commands, errors, and next steps.\n",
    "Prefer concrete facts over narration.\n",
    "State uncertainty explicitly.\n",
    "Do not ask follow-up questions.\n",
);
/// User-facing prompt sent to the compactor agent to trigger session compaction.
pub const COMPACTION_REQUEST_PROMPT: &str = concat!(
    "Compact this conversation into a standalone handoff summary for the next session.\n",
    "Write only the summary.\n",
);
const MEMORY_EXPERT_PROMPT_ADDENDUM: &str = concat!(
    "\n\n# Expert follow-up mode\n",
    "You previously wrote the summary for a compacted sid session.\n",
    "Answer the latest user question from contextual memory only.\n",
    "If the answer is not supported by the conversation context you have, reply exactly: I don't know\n",
    "You may use ask_an_expert only to consult the earlier summary writer when available.\n",
);
const COMPACTED_SESSION_CONTEXT_TEMPLATE: &str = concat!(
    "This session is a compacted continuation of session {session_id}.\n",
    "The next assistant message is the handoff summary written by the previous owner of the project.\n",
    "Use that summary as working context. If you need details that are not in it, use ask_an_expert.\n",
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SidToolScope {
    Normal,
    MemoryOnly,
}

impl SidToolScope {
    fn exposes_standard_tools(self) -> bool {
        matches!(self, Self::Normal)
    }
}

/// Top-level agent combining chat configuration, tool bindings, and sandbox policy.
pub struct SidAgent {
    id: String,
    enabled: SwitchPosition,
    config: ChatConfig,
    named_prompts: BTreeMap<String, String>,
    tools: Vec<Arc<dyn Tool<Self>>>,
    builtin_bindings: BuiltinToolBindings,
    skills: Vec<SkillConfig>,
    config_root: Path<'static>,
    workspace_root: Path<'static>,
    user_instructions: UserInstructionSettings,
    writable_roots: WritableRoots,
    filesystem: MountHierarchy,
    session: Option<Arc<SidSession>>,
    memory_source: Option<CompactionProvenance>,
    tool_scope: SidToolScope,
    memory_depth: usize,
    auto_compact_tokens: Option<u64>,
    bash_session: Mutex<Option<BashPtySession>>,
    tool_cancellation_pending: AtomicBool,
    token_usage_totals: StdMutex<TokenUsageTotals>,
}

impl SidAgent {
    /// Create a new agent with default settings rooted at `workspace_root`.
    ///
    /// The workspace root is used as both the configuration root and the
    /// filesystem root.  No rc.conf files are loaded; the agent operates
    /// with the supplied [`ChatConfig`] only.
    pub fn new(config: ChatConfig, workspace_root: Path<'static>) -> Self {
        Self::new_with_roots(config, workspace_root.clone(), workspace_root)
    }

    fn new_with_roots(
        config: ChatConfig,
        config_root: Path<'static>,
        workspace_root: Path<'static>,
    ) -> Self {
        Self::new_custom(
            DEFAULT_AGENT_ID.to_string(),
            config,
            config_root,
            workspace_root,
            UserInstructionSettings::default(),
        )
    }

    fn new_custom(
        id: String,
        config: ChatConfig,
        config_root: Path<'static>,
        workspace_root: Path<'static>,
        user_instructions: UserInstructionSettings,
    ) -> Self {
        let filesystem = build_default_filesystem(&workspace_root);
        let writable_roots = default_writable_roots(&workspace_root);
        Self::with_parts(
            id,
            SwitchPosition::Yes,
            config,
            BTreeMap::new(),
            vec![],
            BuiltinToolBindings::default(),
            Vec::new(),
            config_root,
            workspace_root,
            user_instructions,
            writable_roots,
            filesystem,
            None,
            None,
            SidToolScope::Normal,
            0,
            None,
        )
    }

    /// Load the default agent from a workspace's configuration files.
    ///
    /// When no configuration exists at `root`, the `fallback` chat config is
    /// used with default settings.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration exists but is malformed, or
    /// when the default agent cannot be resolved.
    pub fn from_workspace(root: &Path, fallback: ChatConfig) -> Result<Self, SError> {
        Self::from_workspace_with_config_root(root, root, fallback)
    }

    /// Load the default agent using separate workspace and configuration roots.
    ///
    /// The `config_root` is the directory containing `agents.conf` and
    /// `tools.conf`, while `workspace_root` is the directory exposed to tool
    /// invocations and sandbox policies.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration exists but is malformed, or
    /// when the default agent cannot be resolved.
    pub fn from_workspace_with_config_root(
        workspace_root: &Path,
        config_root: &Path,
        fallback: ChatConfig,
    ) -> Result<Self, SError> {
        if !workspace_has_config(config_root) {
            let mut agent = Self::new_with_roots(
                fallback,
                config_root.clone().into_owned(),
                workspace_root.clone().into_owned(),
            );
            agent.append_agents_md_to_system_prompt()?;
            return Ok(agent);
        }
        let config = Config::load(config_root)?;
        let agent = default_agent_id(&config)?;
        Self::from_loaded_config(
            &config,
            &agent,
            Some(&fallback),
            workspace_root.clone().into_owned(),
        )
    }

    /// Load a named agent from a workspace's configuration files.
    ///
    /// # Errors
    ///
    /// Returns an error when the agent does not exist in the configuration or
    /// is disabled.
    pub fn from_workspace_agent(
        root: &Path,
        agent: &str,
        fallback: ChatConfig,
    ) -> Result<Self, SError> {
        Self::from_workspace_agent_with_config_root(root, root, agent, fallback)
    }

    /// Load a named agent using separate workspace and configuration roots.
    ///
    /// # Errors
    ///
    /// Returns an error when the agent does not exist in the configuration or
    /// is disabled.
    pub fn from_workspace_agent_with_config_root(
        workspace_root: &Path,
        config_root: &Path,
        agent: &str,
        fallback: ChatConfig,
    ) -> Result<Self, SError> {
        if !workspace_has_config(config_root) {
            return if agent == DEFAULT_AGENT_ID {
                let mut agent = Self::new_with_roots(
                    fallback,
                    config_root.clone().into_owned(),
                    workspace_root.clone().into_owned(),
                );
                agent.append_agents_md_to_system_prompt()?;
                Ok(agent)
            } else {
                Err(missing_agent_error(agent))
            };
        }
        let config = Config::load(config_root)?;
        Self::from_loaded_config(
            &config,
            agent,
            Some(&fallback),
            workspace_root.clone().into_owned(),
        )
    }

    /// Load the compaction agent using separate workspace and configuration roots.
    ///
    /// If a `compact` agent is declared in the configuration it is used;
    /// otherwise a default compactor is built from `fallback` with the
    /// built-in compaction system prompt.  The returned agent exposes
    /// memory-only tools.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration exists but is malformed.
    pub fn from_workspace_compactor_with_config_root(
        workspace_root: &Path,
        config_root: &Path,
        fallback: ChatConfig,
    ) -> Result<Self, SError> {
        if workspace_has_config(config_root) {
            let config = Config::load(config_root)?;
            if config.agents.contains_key(DEFAULT_COMPACTOR_AGENT_ID) {
                return Self::from_loaded_config_inner(
                    &config,
                    DEFAULT_COMPACTOR_AGENT_ID,
                    Some(&fallback),
                    workspace_root.clone().into_owned(),
                    true,
                )
                .map(|agent| agent.with_tool_scope(SidToolScope::MemoryOnly));
            }
        }

        let mut config = fallback;
        config.set_system_prompt(Some(DEFAULT_COMPACTOR_SYSTEM_PROMPT.to_string()));
        Ok(Self::new_custom(
            DEFAULT_COMPACTOR_AGENT_ID.to_string(),
            config,
            config_root.clone().into_owned(),
            workspace_root.clone().into_owned(),
            disabled_user_instruction_settings(),
        )
        .with_tool_scope(SidToolScope::MemoryOnly))
    }

    /// Build an agent directly from a pre-loaded [`Config`].
    ///
    /// # Errors
    ///
    /// Returns an error when the agent does not exist in the configuration or
    /// is disabled.
    pub fn from_config(
        config: &Config,
        agent: &str,
        filesystem: Path<'static>,
    ) -> Result<Self, SError> {
        Self::from_loaded_config(config, agent, None, filesystem)
    }

    /// Return the agent's identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Return `true` when the agent's enablement switch is [`SwitchPosition::Manual`].
    pub fn requires_confirmation(&self) -> bool {
        self.enabled == SwitchPosition::Manual
    }

    /// Return the auto-compact threshold in output tokens, if configured.
    pub fn auto_compact_tokens(&self) -> Option<u64> {
        self.auto_compact_tokens
    }

    fn append_agents_md_to_system_prompt(&mut self) -> Result<(), SError> {
        append_agents_md_to_system_prompt(
            &mut self.config,
            &self.user_instructions,
            &self.workspace_root,
        )
    }

    fn from_loaded_config(
        config: &Config,
        agent: &str,
        fallback: Option<&ChatConfig>,
        workspace_root: Path<'static>,
    ) -> Result<Self, SError> {
        Self::from_loaded_config_inner(config, agent, fallback, workspace_root, false)
    }

    fn from_loaded_config_inner(
        config: &Config,
        agent: &str,
        fallback: Option<&ChatConfig>,
        workspace_root: Path<'static>,
        allow_disabled: bool,
    ) -> Result<Self, SError> {
        let agent_config = config
            .agents
            .get(agent)
            .ok_or_else(|| missing_agent_error(agent))?;
        if !allow_disabled && !agent_config.enabled.can_be_started() {
            return Err(disabled_agent_error(agent, agent_config.enabled));
        }

        let built_tools = build_tools(config, agent_config)?;
        let mut chat_config = merged_chat_config(agent_config, fallback);
        let named_prompts = agent_config
            .prompts
            .iter()
            .map(|(id, prompt)| (id.clone(), prompt.markdown.clone()))
            .collect();
        let skills = resolve_agent_skills(config, agent_config)?;
        let agent_skills = skills.iter().map(|skill| (*skill).clone()).collect();
        let filesystem = build_agent_filesystem(&workspace_root, config, agent_config)?;
        let user_instructions = resolve_user_instruction_settings(config, agent_config)?;
        let writable_roots = default_writable_roots(&workspace_root);
        append_system_description(&mut chat_config, &workspace_root);
        if !skills.is_empty() {
            append_skill_index_to_system_prompt(&mut chat_config, &skills);
        }
        append_agents_md_to_system_prompt(&mut chat_config, &user_instructions, &workspace_root)?;
        Ok(Self::with_parts(
            agent.to_string(),
            agent_config.enabled,
            chat_config,
            named_prompts,
            built_tools.tools,
            built_tools.builtin_bindings,
            agent_skills,
            config.root.clone(),
            workspace_root,
            user_instructions,
            writable_roots,
            filesystem,
            None,
            None,
            SidToolScope::Normal,
            0,
            agent_config.auto_compact_tokens,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn with_parts(
        id: String,
        enabled: SwitchPosition,
        config: ChatConfig,
        named_prompts: BTreeMap<String, String>,
        tools: Vec<Arc<dyn Tool<Self>>>,
        builtin_bindings: BuiltinToolBindings,
        skills: Vec<SkillConfig>,
        config_root: Path<'static>,
        workspace_root: Path<'static>,
        user_instructions: UserInstructionSettings,
        writable_roots: WritableRoots,
        filesystem: MountHierarchy,
        session: Option<Arc<SidSession>>,
        memory_source: Option<CompactionProvenance>,
        tool_scope: SidToolScope,
        memory_depth: usize,
        auto_compact_tokens: Option<u64>,
    ) -> Self {
        Self {
            id,
            enabled,
            config,
            named_prompts,
            tools,
            builtin_bindings,
            skills,
            config_root,
            workspace_root,
            user_instructions,
            writable_roots,
            filesystem,
            session,
            memory_source,
            tool_scope,
            memory_depth,
            auto_compact_tokens,
            bash_session: Mutex::new(None),
            tool_cancellation_pending: AtomicBool::new(false),
            token_usage_totals: StdMutex::new(TokenUsageTotals::default()),
        }
    }

    /// Attach a session to this agent, making the session root writable.
    pub fn with_session(mut self, session: Arc<SidSession>) -> Self {
        append_writable_root(&mut self.writable_roots, session.root());
        if self.memory_source.is_none() {
            self.memory_source = session.compaction_provenance().cloned();
        }
        self.session = Some(session);
        self
    }

    /// Override the compaction provenance used for ask-an-expert memory chains.
    pub fn with_memory_source(mut self, memory_source: Option<CompactionProvenance>) -> Self {
        self.memory_source = memory_source;
        self
    }

    fn with_tool_scope(mut self, tool_scope: SidToolScope) -> Self {
        self.tool_scope = tool_scope;
        if !tool_scope.exposes_standard_tools() {
            self.tools.clear();
            self.builtin_bindings = BuiltinToolBindings::default();
        }
        self
    }

    fn with_memory_depth(mut self, memory_depth: usize) -> Self {
        self.memory_depth = memory_depth;
        self
    }

    /// Snapshot the agent's identity and model for use as a compaction expert.
    pub fn compaction_snapshot(&self) -> CompactionExpertConfig {
        CompactionExpertConfig {
            agent_id: Some(self.id.clone()),
            model: self.config.model().to_string(),
            system_prompt: self.config.system_prompt_text().map(str::to_string),
            memory_expert_prompt: Some(
                self.named_prompt_markdown(MEMORY_EXPERT_PROMPT_ID)
                    .unwrap_or(MEMORY_EXPERT_PROMPT_ADDENDUM)
                    .to_string(),
            ),
        }
    }

    async fn run_bash_command(
        &self,
        command: &str,
        restart: bool,
    ) -> Result<String, std::io::Error> {
        self.run_bash_command_with_renderer(command, restart, None)
            .await
    }

    /// Execute a bash command in the agent's PTY session, streaming output to `renderer`.
    ///
    /// When `restart` is `true` the existing PTY is torn down and a fresh
    /// session is started before executing `command`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the PTY cannot be created or the command
    /// fails to execute.
    pub async fn bash_with_renderer(
        &self,
        command: &str,
        restart: bool,
        renderer: &mut dyn Renderer,
    ) -> Result<String, std::io::Error> {
        self.run_bash_command_with_renderer(command, restart, Some(renderer))
            .await
    }

    async fn run_bash_command_with_renderer(
        &self,
        command: &str,
        restart: bool,
        renderer: Option<&mut dyn Renderer>,
    ) -> Result<String, std::io::Error> {
        let Some(binding) = self.builtin_bindings.bash.as_ref() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "bash is not supported",
            ));
        };

        match binding.enabled {
            SwitchPosition::Yes => {}
            SwitchPosition::No => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "bash is disabled",
                ));
            }
            SwitchPosition::Manual => {
                let input_value = serde_json::json!({
                    "command": command,
                    "restart": restart,
                    "cwd": self.workspace_root.as_str(),
                    "writable_roots": self.writable_roots.as_slice(),
                });
                match confirm_manual_tool_call("bash", &input_value, None, renderer) {
                    Ok(ManualToolConfirmation::Allow) => {}
                    Ok(ManualToolConfirmation::Deny) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            "bash call denied by operator",
                        ));
                    }
                    Ok(ManualToolConfirmation::Cancel) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            USER_CANCELLED_ACTION,
                        ));
                    }
                    Err(err) => {
                        return Err(std::io::Error::other(err));
                    }
                }
            }
        }

        if restart {
            self.clear_persisted_bash_state()?;
        }

        let mut session = self.bash_session.lock().await;
        if session.is_none() {
            *session = Some(self.spawn_bash_session(!restart).await?);
        }

        let result = {
            let session = session
                .as_mut()
                .expect("bash PTY session should be initialized");
            session.run(command, restart).await
        };
        match result {
            Ok(result) => {
                let bash_session = session
                    .as_mut()
                    .expect("bash PTY session should still be initialized");
                self.persist_bash_state(bash_session).await?;
                render_bash_pty_result(result)
            }
            Err(err) => {
                let _ = self.clear_persisted_bash_state();
                *session = None;
                Err(err)
            }
        }
    }

    async fn spawn_bash_session(
        &self,
        restore_state: bool,
    ) -> Result<BashPtySession, std::io::Error> {
        let mut session = BashPtySession::new(self.bash_pty_config()).await?;
        if restore_state {
            self.restore_bash_state(&mut session).await?;
        }
        Ok(session)
    }

    fn bash_pty_config(&self) -> BashPtyConfig {
        let mut env: BTreeMap<String, String> = std::env::vars().collect();
        env.insert(
            "SID_WORKSPACE_ROOT".to_string(),
            self.workspace_root.as_str().to_string(),
        );
        if let Some(session) = self.session.as_ref() {
            env.insert(
                session::SID_SESSION_ID_ENV.to_string(),
                session.id().to_string(),
            );
            env.insert(
                session::SID_SESSION_DIR_ENV.to_string(),
                session.root().to_string_lossy().into_owned(),
            );
            env.insert(
                session::SID_SESSIONS_ENV.to_string(),
                session.sessions_root().to_string_lossy().into_owned(),
            );
            env.insert(
                "TMPDIR".to_string(),
                session.bash_tmp_dir().to_string_lossy().into_owned(),
            );
        }
        env.insert("PAGER".to_string(), "cat".to_string());
        env.insert("HISTFILE".to_string(), "/dev/null".to_string());
        env.insert("INPUTRC".to_string(), "/dev/null".to_string());
        BashPtyConfig {
            cwd: PathBuf::from(self.workspace_root.as_str()),
            env,
            shell_wrapper: seatbelt::shell_wrapper(&self.writable_roots),
            ..BashPtyConfig::default()
        }
    }

    fn clear_persisted_bash_state(&self) -> Result<(), std::io::Error> {
        if let Some(session) = self.session.as_ref() {
            session
                .clear_bash_state()
                .map_err(|err| std::io::Error::other(err.to_string()))?;
        }
        Ok(())
    }

    async fn restore_bash_state(
        &self,
        bash_session: &mut BashPtySession,
    ) -> Result<(), std::io::Error> {
        let Some(sid_session) = self.session.as_ref() else {
            return Ok(());
        };
        let Some(_) = sid_session
            .read_bash_state()
            .map_err(|err| std::io::Error::other(err.to_string()))?
        else {
            return Ok(());
        };

        let restore_command = format!(
            "builtin source {}",
            shvar::quote_string(sid_session.bash_state_path().to_string_lossy().as_ref())
        );
        let result = bash_session.run(&restore_command, false).await?;
        if result.status.success() {
            Ok(())
        } else {
            Err(bash_state_io_error("failed to restore bash state", &result))
        }
    }

    async fn persist_bash_state(
        &self,
        bash_session: &mut BashPtySession,
    ) -> Result<(), std::io::Error> {
        let Some(sid_session) = self.session.as_ref() else {
            return Ok(());
        };
        if !bash_session.is_alive()? {
            sid_session
                .clear_bash_state()
                .map_err(|err| std::io::Error::other(err.to_string()))?;
            return Ok(());
        }

        let snapshot = bash_session.run(BASH_STATE_CAPTURE_COMMAND, false).await?;
        if !snapshot.status.success() {
            return Err(bash_state_io_error(
                "failed to capture bash state",
                &snapshot,
            ));
        }
        sid_session
            .write_bash_state(&snapshot.output)
            .map_err(|err| std::io::Error::other(err.to_string()))
    }

    fn memory_source(&self) -> Option<&CompactionProvenance> {
        self.memory_source.as_ref()
    }

    async fn ask_an_expert(
        &self,
        client: &Anthropic,
        question: &str,
    ) -> Result<String, std::io::Error> {
        self.ask_an_expert_with_renderer(client, question, None)
            .await
    }

    async fn ask_an_expert_with_renderer(
        &self,
        client: &Anthropic,
        question: &str,
        renderer: Option<&mut dyn Renderer>,
    ) -> Result<String, std::io::Error> {
        let Some(source) = self.memory_source() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "ask_an_expert is not available in this session",
            ));
        };
        if self.memory_depth >= MAX_ASK_AN_EXPERT_DEPTH {
            return Ok("I don't know".to_string());
        }

        let parent_root = std::path::Path::new(&source.session_dir);
        let parent_messages = load_transcript_messages(
            session::transcript_path_for_session_dir(parent_root).as_path(),
        )
        .map_err(|err| std::io::Error::other(err.to_string()))?;
        let parent_memory_source = session::read_compaction_provenance_from_dir(parent_root)
            .map_err(|err| std::io::Error::other(err.to_string()))?;

        let expert = Self::memory_expert_from_snapshot(
            &source.expert,
            &self.config_root,
            &self.workspace_root,
            parent_memory_source,
            self.memory_depth + 1,
        );
        let mut chat = claudius::chat::ChatSession::with_agent(client.clone(), expert);
        chat.replace_messages(parent_messages);

        match renderer {
            Some(renderer) => chat
                .send_message(MessageParam::user(question), renderer)
                .await
                .map_err(|err| std::io::Error::other(err.to_string()))?,
            None => {
                let mut renderer = NullRenderer;
                chat.send_message(MessageParam::user(question), &mut renderer)
                    .await
                    .map_err(|err| std::io::Error::other(err.to_string()))?;
            }
        }

        extract_last_assistant_text(&chat.clone_messages()).ok_or_else(|| {
            std::io::Error::other("expert conversation produced no assistant response")
        })
    }

    fn memory_expert_from_snapshot(
        snapshot: &CompactionExpertConfig,
        config_root: &Path,
        workspace_root: &Path,
        memory_source: Option<CompactionProvenance>,
        memory_depth: usize,
    ) -> Self {
        let mut config = ChatConfig::new();
        let model = snapshot
            .model
            .parse()
            .unwrap_or_else(|_| Model::Custom(snapshot.model.clone()));
        config.set_model(model);
        let memory_expert_prompt = snapshot
            .memory_expert_prompt
            .as_deref()
            .unwrap_or(MEMORY_EXPERT_PROMPT_ADDENDUM);
        config.set_system_prompt(Some(format!(
            "{}{}",
            snapshot.system_prompt.as_deref().unwrap_or_default(),
            memory_expert_prompt
        )));

        Self::new_custom(
            snapshot
                .agent_id
                .clone()
                .unwrap_or_else(|| DEFAULT_COMPACTOR_AGENT_ID.to_string()),
            config,
            config_root.clone().into_owned(),
            workspace_root.clone().into_owned(),
            disabled_user_instruction_settings(),
        )
        .with_tool_scope(SidToolScope::MemoryOnly)
        .with_memory_source(memory_source)
        .with_memory_depth(memory_depth)
    }

    fn tool_runtime_context(&self) -> ToolRuntimeContext<'_> {
        ToolRuntimeContext {
            agent_id: &self.id,
            config_root: &self.config_root,
            workspace_root: &self.workspace_root,
            writable_roots: &self.writable_roots,
            session: self.session.as_deref(),
        }
    }

    async fn inject_user_instructions_for_turn(
        &self,
        messages: &mut [MessageParam],
    ) -> Result<(), Error> {
        let latest_user_message = latest_user_message_text(messages);
        let context = UserInstructionRuntimeContext {
            agent_id: &self.id,
            config_root: &self.config_root,
            workspace_root: &self.workspace_root,
            session: self.session.as_deref(),
            latest_user_message: latest_user_message.as_deref(),
            skills: &self.skills,
        };
        let Some(instructions) = build_user_instruction_block(&self.user_instructions, &context)
            .await
            .map_err(|err| Error::unknown(format!("failed to build user instructions: {err}")))?
        else {
            return Ok(());
        };
        append_user_instruction_block(messages, instructions);
        Ok(())
    }

    fn prepare_rc_tool(
        &self,
        display_name: &str,
        binding: &RcToolBinding,
        tool_use_id: &str,
        input: serde_json::Map<String, serde_json::Value>,
    ) -> Result<tool_runtime::PreparedRcToolInvocation, String> {
        let context = self.tool_runtime_context();
        tool_runtime::prepare_rc_tool_invocation(
            display_name,
            &binding.service_name,
            &binding.canonical_id,
            &binding.executable_path,
            &context,
            tool_use_id,
            input,
            binding.timeout,
        )
    }

    /// Return the markdown for a resolved named prompt.
    pub fn named_prompt_markdown(&self, prompt_id: &str) -> Option<&str> {
        if prompt_id == SYSTEM_PROMPT_ID {
            return self.config.system_prompt_text();
        }
        self.named_prompts.get(prompt_id).map(String::as_str)
    }

    async fn invoke_rc_tool(
        &self,
        binding: &RcToolBinding,
        tool_use_id: &str,
        input: serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, std::io::Error> {
        let context = self.tool_runtime_context();
        tool_runtime::invoke_rc_tool_text(
            &binding.service_name,
            &binding.service_name,
            &binding.canonical_id,
            &binding.executable_path,
            &context,
            tool_use_id,
            input,
            binding.timeout,
        )
        .await
        .map_err(std::io::Error::other)
    }

    async fn run_text_editor_tool(
        &self,
        tool_use: ToolUseBlock,
        renderer: Option<&mut dyn Renderer>,
    ) -> Result<String, std::io::Error> {
        #[derive(serde::Deserialize)]
        struct Command {
            command: String,
        }
        let cmd: Command = serde_json::from_value(tool_use.input.clone())?;
        let Some(binding) = self.builtin_bindings.edit.as_ref() else {
            return self
                .default_text_editor_command(cmd.command.as_str(), tool_use)
                .await;
        };
        match binding.enabled {
            SwitchPosition::Yes => {
                let input = tool_use.input.as_object().cloned().ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "text editor input must be a JSON object",
                    )
                })?;
                return self.invoke_rc_tool(binding, &tool_use.id, input).await;
            }
            SwitchPosition::No => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "edit is disabled",
                ));
            }
            SwitchPosition::Manual => {}
        }
        let input = tool_use.input.as_object().cloned().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "text editor input must be a JSON object",
            )
        })?;
        let prepared = self
            .prepare_rc_tool("edit", binding, &tool_use.id, input)
            .map_err(std::io::Error::other)?;
        match confirm_manual_prepared_tool_call(
            "edit",
            &tool_use.input,
            binding.confirm_preview,
            &prepared,
            self.session.as_deref(),
            renderer,
        )
        .await
        {
            Ok(ManualToolConfirmation::Allow) => {}
            Ok(ManualToolConfirmation::Deny) => {
                let _ = tool_runtime::cleanup_prepared_rc_tool(&prepared, false);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "edit call denied by operator",
                ));
            }
            Ok(ManualToolConfirmation::Cancel) => {
                let _ = tool_runtime::cleanup_prepared_rc_tool(&prepared, false);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    USER_CANCELLED_ACTION,
                ));
            }
            Err(err) => {
                let _ = tool_runtime::cleanup_prepared_rc_tool(&prepared, true);
                return Err(std::io::Error::other(err));
            }
        }
        tool_runtime::run_prepared_rc_tool_text(
            &prepared,
            &self.writable_roots,
            self.session.as_deref(),
        )
        .await
        .map_err(std::io::Error::other)
    }

    async fn default_text_editor_command(
        &self,
        command: &str,
        tool_use: ToolUseBlock,
    ) -> Result<String, std::io::Error> {
        match command {
            "view" => {
                #[derive(serde::Deserialize)]
                struct ViewTool {
                    path: String,
                    view_range: Option<(u32, u32)>,
                }
                let args: ViewTool = serde_json::from_value(tool_use.input)?;
                self.view(&args.path, args.view_range).await
            }
            "str_replace" => {
                #[derive(serde::Deserialize)]
                struct StrReplaceTool {
                    path: String,
                    old_str: String,
                    new_str: Option<String>,
                }
                let args: StrReplaceTool = serde_json::from_value(tool_use.input)?;
                let new_str = args.new_str.as_deref().unwrap_or("");
                self.str_replace(&args.path, &args.old_str, new_str).await
            }
            "insert" => {
                #[derive(serde::Deserialize)]
                struct InsertTool {
                    path: String,
                    insert_line: u32,
                    insert_text: Option<String>,
                    new_str: Option<String>,
                }
                let args: InsertTool = serde_json::from_value(tool_use.input)?;
                let text = args.insert_text.or(args.new_str).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "missing insert_text field",
                    )
                })?;
                self.insert(&args.path, args.insert_line, &text).await
            }
            "create" => {
                #[derive(serde::Deserialize)]
                struct CreateTool {
                    path: String,
                    file_text: String,
                }
                let args: CreateTool = serde_json::from_value(tool_use.input)?;
                self.create(&args.path, &args.file_text).await
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!("{} is not a supported tool", tool_use.name),
            )),
        }
    }
}

#[async_trait::async_trait]
impl Agent for SidAgent {
    fn stream_label(&self) -> String {
        self.id.clone()
    }

    async fn take_turn(
        &mut self,
        client: &Anthropic,
        messages: &mut Vec<MessageParam>,
        budget: &Arc<Budget>,
    ) -> Result<TurnOutcome, Error> {
        self.tool_cancellation_pending
            .store(false, Ordering::Relaxed);
        top_up_cancelled_tool_results(messages);
        self.inject_user_instructions_for_turn(messages).await?;
        let Some(mut tokens_rem) = budget.allocate(self.max_tokens().await) else {
            let stop_reason = self.handle_max_tokens().await?;
            top_up_cancelled_tool_results(messages);
            return Ok(TurnOutcome {
                stop_reason,
                usage: Usage::new(0, 0),
                request_count: 0,
            });
        };

        let mut usage_total = Usage::new(0, 0);
        let mut request_count: u64 = 0;
        while tokens_rem.remaining_tokens()
            > self
                .thinking()
                .await
                .map(|thinking| thinking.num_tokens())
                .unwrap_or(0)
        {
            let retry_policy = retry::ApiRetryPolicy::default();
            let retry_backoff = retry_policy.backoff();
            let mut retry_count = 0usize;
            let step = loop {
                let step = self
                    .step_default_turn(client, messages, &mut tokens_rem)
                    .await;
                let Some(delay) = (match &step {
                    ControlFlow::Break(Err(err)) => {
                        retry_policy.retry_delay(&retry_backoff, retry_count, err)
                    }
                    _ => None,
                }) else {
                    break step;
                };
                retry_count += 1;
                tokio::time::sleep(delay).await;
            };
            match step {
                ControlFlow::Continue(step) => {
                    usage_total = usage_total + step.usage;
                    request_count = request_count.saturating_add(step.request_count);
                    if self
                        .tool_cancellation_pending
                        .swap(false, Ordering::Relaxed)
                    {
                        top_up_cancelled_tool_results(messages);
                        return Ok(TurnOutcome {
                            stop_reason: StopReason::EndTurn,
                            usage: usage_total,
                            request_count,
                        });
                    }
                }
                ControlFlow::Break(res) => {
                    return match res {
                        Ok(mut outcome) => {
                            outcome.usage = outcome.usage + usage_total;
                            outcome.request_count =
                                outcome.request_count.saturating_add(request_count);
                            top_up_cancelled_tool_results(messages);
                            Ok(outcome)
                        }
                        Err(err) => {
                            eprintln!("API error (conversation preserved): {err}");
                            top_up_cancelled_tool_results(messages);
                            Ok(TurnOutcome {
                                stop_reason: StopReason::EndTurn,
                                usage: usage_total,
                                request_count,
                            })
                        }
                    };
                }
            }
        }

        let stop_reason = self.handle_max_tokens().await?;
        top_up_cancelled_tool_results(messages);
        Ok(TurnOutcome {
            stop_reason,
            usage: usage_total,
            request_count,
        })
    }

    async fn take_turn_streaming(
        &mut self,
        client: &Anthropic,
        messages: &mut Vec<MessageParam>,
        budget: &Arc<Budget>,
        renderer: &mut dyn Renderer,
        context: AgentStreamContext,
    ) -> Result<TurnOutcome, Error> {
        self.tool_cancellation_pending
            .store(false, Ordering::Relaxed);
        top_up_cancelled_tool_results(messages);
        self.inject_user_instructions_for_turn(messages).await?;
        renderer.start_agent(&context);
        let Some(mut tokens_rem) = budget.allocate(self.max_tokens().await) else {
            let stop_reason = self.handle_max_tokens().await?;
            top_up_cancelled_tool_results(messages);
            renderer.finish_agent(&context, Some(&stop_reason));
            return Ok(TurnOutcome {
                stop_reason,
                usage: Usage::new(0, 0),
                request_count: 0,
            });
        };

        let mut usage_total = Usage::new(0, 0);
        let mut request_count: u64 = 0;
        while tokens_rem.remaining_tokens()
            > self
                .thinking()
                .await
                .map(|thinking| thinking.num_tokens())
                .unwrap_or(0)
        {
            let retry_policy = retry::ApiRetryPolicy::default();
            let retry_backoff = retry_policy.backoff();
            let mut retry_count = 0usize;
            let step = loop {
                let step = self
                    .step_default_turn_streaming(
                        client,
                        messages,
                        &mut tokens_rem,
                        renderer,
                        &context,
                    )
                    .await;
                let Some(delay) = (match &step {
                    ControlFlow::Break(Err(err)) => {
                        retry_policy.retry_delay(&retry_backoff, retry_count, err)
                    }
                    _ => None,
                }) else {
                    break step;
                };
                retry_count += 1;
                renderer.print_info(
                    &context,
                    &format!(
                        "Transient API failure; retrying in {} (retry {}/{})",
                        retry::format_delay(delay),
                        retry_count,
                        retry_policy.max_retries(),
                    ),
                );
                tokio::time::sleep(delay).await;
            };
            match step {
                ControlFlow::Continue(step) => {
                    usage_total = usage_total + step.usage;
                    request_count = request_count.saturating_add(step.request_count);
                    if self
                        .tool_cancellation_pending
                        .swap(false, Ordering::Relaxed)
                    {
                        let stop_reason = StopReason::EndTurn;
                        top_up_cancelled_tool_results(messages);
                        renderer.finish_agent(&context, Some(&stop_reason));
                        return Ok(TurnOutcome {
                            stop_reason,
                            usage: usage_total,
                            request_count,
                        });
                    }
                }
                ControlFlow::Break(res) => match res {
                    Ok(mut outcome) => {
                        outcome.usage = outcome.usage + usage_total;
                        outcome.request_count = outcome.request_count.saturating_add(request_count);
                        top_up_cancelled_tool_results(messages);
                        renderer.finish_agent(&context, Some(&outcome.stop_reason));
                        return Ok(outcome);
                    }
                    Err(err) => {
                        renderer.print_error(
                            &context,
                            &format!("API error (conversation preserved): {err}"),
                        );
                        top_up_cancelled_tool_results(messages);
                        let stop_reason = StopReason::EndTurn;
                        renderer.finish_agent(&context, Some(&stop_reason));
                        return Ok(TurnOutcome {
                            stop_reason,
                            usage: usage_total,
                            request_count,
                        });
                    }
                },
            }
        }

        let stop_reason = self.handle_max_tokens().await?;
        top_up_cancelled_tool_results(messages);
        renderer.finish_agent(&context, Some(&stop_reason));
        Ok(TurnOutcome {
            stop_reason,
            usage: usage_total,
            request_count,
        })
    }

    async fn max_tokens(&self) -> u32 {
        self.config.max_tokens()
    }

    async fn model(&self) -> Model {
        self.config.model()
    }

    async fn metadata(&self) -> Option<Metadata> {
        self.config.template.metadata.clone()
    }

    async fn stop_sequences(&self) -> Option<Vec<String>> {
        let sequences = self.config.stop_sequences();
        if sequences.is_empty() {
            None
        } else {
            Some(sequences.to_vec())
        }
    }

    async fn system(&self) -> Option<SystemPrompt> {
        let prompt = self.config.template.system.as_ref()?;
        Some(prompt.clone())
    }

    fn caching_enabled(&self) -> bool {
        self.config.caching_enabled
    }

    async fn temperature(&self) -> Option<f32> {
        self.config.template.temperature
    }

    async fn thinking(&self) -> Option<ThinkingConfig> {
        self.config.template.thinking
    }

    async fn tool_choice(&self) -> Option<ToolChoice> {
        self.config.template.tool_choice.clone()
    }

    async fn tools(&self) -> Vec<Arc<dyn Tool<Self>>> {
        let mut tools = if self.tool_scope.exposes_standard_tools() {
            self.tools.clone()
        } else {
            Vec::new()
        };
        if self.memory_source().is_some() {
            tools.push(Arc::new(AskAnExpertTool) as Arc<dyn Tool<Self>>);
        }
        tools
    }

    async fn top_k(&self) -> Option<u32> {
        self.config.template.top_k
    }

    async fn top_p(&self) -> Option<f32> {
        self.config.template.top_p
    }

    async fn filesystem(&self) -> Option<&dyn FileSystem> {
        Some(&self.filesystem)
    }

    async fn hook_message(&self, resp: &Message) -> Result<(), Error> {
        if let Some(session) = self.session.as_ref() {
            session
                .log_api_response(resp)
                .map_err(|err| Error::unknown(format!("failed to log API response: {err}")))?;
        }
        let totals = self.record_token_usage(resp.usage);
        let rates = self.token_rates();
        let cost_suffix = match rates {
            Some(rates) => {
                let turn_cost = compute_cost_micro_cents(resp.usage, rates);
                format!(" turn={} total={}", format_cost(turn_cost), format_cost(totals.cost_micro_cents))
            }
            None => String::new(),
        };
        println!(
            "[tokens: input={} cached_input={} output={}{}]",
            totals.input, totals.cached_input, totals.output, cost_suffix
        );
        println!("[usage: {:?}]", resp.usage);
        Ok(())
    }

    async fn hook_message_create_params(
        &self,
        req: &claudius::MessageCreateParams,
    ) -> Result<(), Error> {
        if let Some(session) = self.session.as_ref() {
            session
                .log_api_request(req)
                .map_err(|err| Error::unknown(format!("failed to log API request: {err}")))?;
        }
        Ok(())
    }

    async fn text_editor(&self, tool_use: ToolUseBlock) -> Result<String, std::io::Error> {
        self.run_text_editor_tool(tool_use, None).await
    }

    async fn bash(&self, command: &str, restart: bool) -> Result<String, std::io::Error> {
        self.run_bash_command(command, restart).await
    }

    async fn handle_tool_use(
        &mut self,
        client: &Anthropic,
        resp: &Message,
    ) -> ControlFlow<Result<StopReason, Error>, Vec<ContentBlock>> {
        let requested_tools = collect_requested_tool_calls(self, resp).await;
        let mut tool_results = Vec::new();
        for (index, requested) in requested_tools.iter().enumerate() {
            let result = match requested.tool.as_ref() {
                Some(tool) => {
                    let callback = tool.callback();
                    let tool_use = requested.tool_use.clone();
                    let this = &*self;
                    let intermediate = callback.compute_tool_result(client, this, &tool_use).await;
                    match callback
                        .apply_tool_result(client, self, &tool_use, intermediate)
                        .await
                    {
                        ControlFlow::Continue(result) => result,
                        ControlFlow::Break(err) => return ControlFlow::Break(Err(err)),
                    }
                }
                None => missing_tool_result(&requested.tool_use),
            };
            let cancelled = tool_result_is_user_cancelled(&result);
            push_tool_result(&mut tool_results, None, result);
            if cancelled {
                self.tool_cancellation_pending
                    .store(true, Ordering::Relaxed);
                cancel_remaining_tool_calls(
                    &mut tool_results,
                    requested_tools.iter().skip(index + 1),
                );
                break;
            }
        }
        ControlFlow::Continue(tool_results)
    }

    async fn handle_tool_use_streaming(
        &mut self,
        client: &Anthropic,
        resp: &Message,
        renderer: &mut dyn Renderer,
        context: &AgentStreamContext,
    ) -> ControlFlow<Result<StopReason, Error>, Vec<ContentBlock>> {
        let requested_tools = collect_requested_tool_calls(self, resp).await;
        let mut tool_results = Vec::new();
        for (index, requested) in requested_tools.iter().enumerate() {
            let tool_context = context.child(format!("tool:{}", requested.tool_use.name));
            let result = match requested.tool.as_ref() {
                Some(tool) => {
                    let callback = tool.callback();
                    let this = &*self;
                    let intermediate = callback
                        .compute_tool_result_streaming(
                            client,
                            this,
                            &requested.tool_use,
                            renderer,
                            &tool_context,
                        )
                        .await;
                    let render_result = should_render_tool_result(intermediate.as_ref());
                    match callback
                        .apply_tool_result(client, self, &requested.tool_use, intermediate)
                        .await
                    {
                        ControlFlow::Continue(result) => {
                            let cancelled = tool_result_is_user_cancelled(&result);
                            if render_result {
                                push_tool_result(
                                    &mut tool_results,
                                    Some((renderer, &tool_context as &dyn StreamContext)),
                                    result,
                                );
                            } else {
                                push_tool_result(&mut tool_results, None, result);
                            }
                            if cancelled {
                                self.tool_cancellation_pending
                                    .store(true, Ordering::Relaxed);
                                cancel_remaining_tool_calls_streaming(
                                    &mut tool_results,
                                    renderer,
                                    context,
                                    requested_tools.iter().skip(index + 1),
                                );
                                break;
                            }
                            continue;
                        }
                        ControlFlow::Break(err) => return ControlFlow::Break(Err(err)),
                    }
                }
                None => missing_tool_result(&requested.tool_use),
            };
            let cancelled = tool_result_is_user_cancelled(&result);
            push_tool_result(&mut tool_results, Some((renderer, &tool_context)), result);
            if cancelled {
                self.tool_cancellation_pending
                    .store(true, Ordering::Relaxed);
                cancel_remaining_tool_calls_streaming(
                    &mut tool_results,
                    renderer,
                    context,
                    requested_tools.iter().skip(index + 1),
                );
                break;
            }
        }
        ControlFlow::Continue(tool_results)
    }
}

impl ChatAgent for SidAgent {
    fn config(&self) -> &ChatConfig {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ChatConfig {
        &mut self.config
    }
}

impl SidAgent {
    fn record_token_usage(&self, usage: Usage) -> TokenUsageTotals {
        let rates = self.token_rates();
        let mut totals = self
            .token_usage_totals
            .lock()
            .expect("token usage totals lock poisoned");
        totals.add(usage, rates);
        *totals
    }

    fn token_rates(&self) -> Option<TokenRates> {
        match self.config.model() {
            Model::Known(km) => Some(km.token_rates()),
            Model::Custom(_) => None,
        }
    }
}

struct RequestedToolCall {
    tool_use: ToolUseBlock,
    tool: Option<Arc<dyn Tool<SidAgent>>>,
}

async fn collect_requested_tool_calls(agent: &SidAgent, resp: &Message) -> Vec<RequestedToolCall> {
    let tools = agent.tools().await;
    resp.content
        .iter()
        .filter_map(|block| {
            let ContentBlock::ToolUse(tool_use) = block else {
                return None;
            };
            let tool = tools
                .iter()
                .find(|tool| tool.name() == tool_use.name)
                .cloned();
            Some(RequestedToolCall {
                tool_use: tool_use.clone(),
                tool,
            })
        })
        .collect()
}

fn top_up_cancelled_tool_results(messages: &mut Vec<MessageParam>) -> usize {
    let mut cancelled = 0;
    let mut index = 0;
    while index < messages.len() {
        let tool_use_ids = assistant_tool_use_ids(&messages[index]);
        if tool_use_ids.is_empty() {
            index += 1;
            continue;
        }

        let next_index = index + 1;
        if next_index < messages.len()
            && messages[next_index].role == MessageRole::User
            && user_message_has_required_tool_result(&messages[next_index], &tool_use_ids)
        {
            cancelled +=
                top_up_user_tool_results(&mut messages[next_index], tool_use_ids.as_slice());
        } else {
            let blocks = tool_use_ids
                .iter()
                .map(|tool_use_id| user_cancelled_tool_result_block(tool_use_id).into())
                .collect::<Vec<ContentBlock>>();
            cancelled += blocks.len();
            messages.insert(
                next_index,
                MessageParam::new(MessageParamContent::Array(blocks), MessageRole::User),
            );
        }
        index += 2;
    }
    cancelled
}

fn assistant_tool_use_ids(message: &MessageParam) -> Vec<String> {
    if message.role != MessageRole::Assistant {
        return Vec::new();
    }
    let MessageParamContent::Array(blocks) = &message.content else {
        return Vec::new();
    };

    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for block in blocks {
        let Some(tool_use) = block.as_tool_use() else {
            continue;
        };
        if seen.insert(tool_use.id.clone()) {
            ids.push(tool_use.id.clone());
        }
    }
    ids
}

fn user_message_has_required_tool_result(message: &MessageParam, tool_use_ids: &[String]) -> bool {
    let required = tool_use_ids.iter().collect::<BTreeSet<_>>();
    let MessageParamContent::Array(blocks) = &message.content else {
        return false;
    };
    blocks.iter().any(|block| {
        block
            .as_tool_result()
            .is_some_and(|tool_result| required.contains(&tool_result.tool_use_id))
    })
}

fn top_up_user_tool_results(message: &mut MessageParam, tool_use_ids: &[String]) -> usize {
    let required = tool_use_ids.iter().cloned().collect::<BTreeSet<_>>();
    let content = std::mem::replace(&mut message.content, MessageParamContent::Array(Vec::new()));
    let mut existing_results = BTreeMap::new();
    let mut tail = Vec::new();

    match content {
        MessageParamContent::String(text) => {
            tail.push(ContentBlock::Text(TextBlock::new(text)));
        }
        MessageParamContent::Array(blocks) => {
            for block in blocks {
                if let ContentBlock::ToolResult(tool_result) = &block
                    && required.contains(&tool_result.tool_use_id)
                    && !existing_results.contains_key(&tool_result.tool_use_id)
                {
                    existing_results.insert(tool_result.tool_use_id.clone(), block);
                    continue;
                }
                tail.push(block);
            }
        }
    }

    let mut cancelled = 0;
    let mut blocks = Vec::with_capacity(tool_use_ids.len() + tail.len());
    for tool_use_id in tool_use_ids {
        if let Some(block) = existing_results.remove(tool_use_id) {
            blocks.push(block);
        } else {
            blocks.push(user_cancelled_tool_result_block(tool_use_id).into());
            cancelled += 1;
        }
    }
    blocks.extend(tail);
    message.content = MessageParamContent::Array(blocks);
    cancelled
}

fn missing_tool_result(tool_use: &ToolUseBlock) -> Result<ToolResultBlock, ToolResultBlock> {
    Err(ToolResultBlock::new(tool_use.id.clone())
        .with_string_content(format!("{} not found", tool_use.name))
        .with_error(true))
}

fn cancel_remaining_tool_calls<'a>(
    tool_results: &mut Vec<ContentBlock>,
    remaining: impl Iterator<Item = &'a RequestedToolCall>,
) {
    for requested in remaining {
        let result = user_cancelled_tool_result(&requested.tool_use.id);
        push_tool_result(tool_results, None, result);
    }
}

fn cancel_remaining_tool_calls_streaming<'a>(
    tool_results: &mut Vec<ContentBlock>,
    renderer: &mut dyn Renderer,
    context: &AgentStreamContext,
    remaining: impl Iterator<Item = &'a RequestedToolCall>,
) {
    for requested in remaining {
        let result = user_cancelled_tool_result(&requested.tool_use.id);
        let tool_context = context.child(format!("tool:{}", requested.tool_use.name));
        push_tool_result(tool_results, Some((renderer, &tool_context)), result);
    }
}

fn user_cancelled_tool_result(tool_use_id: &str) -> Result<ToolResultBlock, ToolResultBlock> {
    Err(user_cancelled_tool_result_block(tool_use_id))
}

fn user_cancelled_tool_result_block(tool_use_id: &str) -> ToolResultBlock {
    ToolResultBlock::new(tool_use_id.to_string())
        .with_string_content(USER_CANCELLED_ACTION.to_string())
        .with_error(true)
}

fn tool_result_is_user_cancelled(result: &Result<ToolResultBlock, ToolResultBlock>) -> bool {
    let block = match result {
        Ok(block) | Err(block) => block,
    };
    block.is_error.unwrap_or(false)
        && matches!(
            block.content.as_ref(),
            Some(ToolResultBlockContent::String(text)) if text == USER_CANCELLED_ACTION
        )
}

fn push_tool_result(
    tool_results: &mut Vec<ContentBlock>,
    renderer: Option<(&mut dyn Renderer, &dyn StreamContext)>,
    result: Result<ToolResultBlock, ToolResultBlock>,
) {
    let block = match result {
        Ok(block) => block,
        Err(block) => block.with_error(true),
    };
    if let Some((renderer, context)) = renderer {
        render_tool_result_block(renderer, context, &block);
    }
    tool_results.push(block.into());
}

fn render_tool_result_block(
    renderer: &mut dyn Renderer,
    context: &dyn StreamContext,
    block: &ToolResultBlock,
) {
    renderer.start_tool_result(context, &block.tool_use_id, block.is_error.unwrap_or(false));
    if let Some(content) = &block.content {
        render_tool_result_content(renderer, context, content);
    }
    renderer.finish_tool_result(context);
}

fn render_tool_result_content(
    renderer: &mut dyn Renderer,
    context: &dyn StreamContext,
    content: &ToolResultBlockContent,
) {
    match content {
        ToolResultBlockContent::String(text) => renderer.print_tool_result_text(context, text),
        ToolResultBlockContent::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    renderer.print_tool_result_text(context, "\n");
                }
                match item {
                    Content::Text(text) => renderer.print_tool_result_text(context, &text.text),
                    Content::Image(_) => renderer.print_tool_result_text(context, "[image]"),
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct TokenUsageTotals {
    input: u64,
    cached_input: u64,
    output: u64,
    cost_micro_cents: u64,
}

impl TokenUsageTotals {
    fn add(&mut self, usage: Usage, rates: Option<TokenRates>) {
        self.input = self.input.saturating_add(tokens_to_u64(usage.input_tokens));
        self.cached_input = self
            .cached_input
            .saturating_add(optional_tokens_to_u64(usage.cache_read_input_tokens));
        self.output = self
            .output
            .saturating_add(tokens_to_u64(usage.output_tokens));
        if let Some(rates) = rates {
            self.cost_micro_cents = self
                .cost_micro_cents
                .saturating_add(compute_cost_micro_cents(usage, rates));
        }
    }
}

/// Compute the cost in micro-cents for a single Usage at the given TokenRates.
fn compute_cost_micro_cents(usage: Usage, rates: TokenRates) -> u64 {
    let input = tokens_to_u64(usage.input_tokens).saturating_mul(rates.input);
    let output = tokens_to_u64(usage.output_tokens).saturating_mul(rates.output);
    let cache_creation =
        optional_tokens_to_u64(usage.cache_creation_input_tokens).saturating_mul(rates.cache_creation);
    let cache_read =
        optional_tokens_to_u64(usage.cache_read_input_tokens).saturating_mul(rates.cache_read);
    input
        .saturating_add(output)
        .saturating_add(cache_creation)
        .saturating_add(cache_read)
}

/// Format a cost in micro-cents as a human-readable dollar string.
///
/// Micro-cents are 1/1,000,000 of a cent, so 100,000,000 micro-cents = $1.00.
fn format_cost(micro_cents: u64) -> String {
    // Convert to cents (integer division, then show fractional cents).
    // micro_cents / 100_000_000 = dollars
    let dollars = micro_cents / 100_000_000;
    let remainder = micro_cents % 100_000_000;
    // Convert remainder to fractional dollars with 4 decimal places.
    // remainder / 10_000 gives ten-thousandths of a dollar.
    let frac = remainder / 10_000;
    format!("${}.{:04}", dollars, frac)
}

fn tokens_to_u64(tokens: i32) -> u64 {
    tokens.max(0) as u64
}

fn optional_tokens_to_u64(tokens: Option<i32>) -> u64 {
    tokens.map(tokens_to_u64).unwrap_or(0)
}

#[derive(Clone, Debug, Default)]
struct BuiltinToolBindings {
    bash: Option<BuiltinBashBinding>,
    edit: Option<RcToolBinding>,
}

#[derive(Clone, Debug)]
struct BuiltinBashBinding {
    enabled: SwitchPosition,
}

#[derive(Clone, Debug)]
struct RcToolBinding {
    service_name: String,
    canonical_id: String,
    enabled: SwitchPosition,
    confirm_preview: bool,
    executable_path: Path<'static>,
    timeout: Option<Duration>,
}

struct BuiltTools {
    tools: Vec<Arc<dyn Tool<SidAgent>>>,
    builtin_bindings: BuiltinToolBindings,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BuiltinToolKind {
    Bash,
    Edit,
}

impl BuiltinToolKind {
    fn from_canonical_id(canonical_id: &str) -> Option<Self> {
        match canonical_id {
            "bash" => Some(Self::Bash),
            "edit" => Some(Self::Edit),
            _ => None,
        }
    }

    fn tool(self) -> Arc<dyn Tool<SidAgent>> {
        match self {
            Self::Bash => Arc::new(SidBashTool::new()) as Arc<dyn Tool<SidAgent>>,
            Self::Edit => Arc::new(SidTextEditorTool::new()) as Arc<dyn Tool<SidAgent>>,
        }
    }
}

#[derive(Clone, Debug)]
struct SidBashTool {
    param: ToolBash20250124,
}

impl SidBashTool {
    fn new() -> Self {
        Self {
            param: ToolBash20250124::new(),
        }
    }
}

impl Tool<SidAgent> for SidBashTool {
    fn name(&self) -> String {
        self.param.name.clone()
    }

    fn callback(&self) -> Box<dyn ToolCallback<SidAgent> + '_> {
        Box::new(SidBashCallback)
    }

    fn to_param(&self) -> ToolUnionParam {
        ToolUnionParam::Bash20250124(self.param.clone())
    }
}

struct SidBashCallback;

impl SidBashCallback {
    async fn compute(
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
        renderer: Option<&mut dyn Renderer>,
    ) -> ToolResult {
        #[derive(serde::Deserialize)]
        struct BashInput {
            command: String,
            #[serde(default)]
            restart: bool,
        }

        let bash: BashInput = match serde_json::from_value(tool_use.input.clone()) {
            Ok(input) => input,
            Err(err) => return tool_error_result(&tool_use.id, err.to_string()),
        };

        match agent
            .run_bash_command_with_renderer(&bash.command, bash.restart, renderer)
            .await
        {
            Ok(output) => tool_success_result(&tool_use.id, output),
            Err(err) => tool_error_result(&tool_use.id, err.to_string()),
        }
    }
}

#[async_trait::async_trait]
impl ToolCallback<SidAgent> for SidBashCallback {
    async fn compute_tool_result(
        &self,
        _client: &Anthropic,
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
    ) -> Box<dyn IntermediateToolResult> {
        Box::new(Self::compute(agent, tool_use, None).await)
    }

    async fn compute_tool_result_streaming(
        &self,
        _client: &Anthropic,
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
        renderer: &mut dyn Renderer,
        _context: &AgentStreamContext,
    ) -> Box<dyn IntermediateToolResult> {
        Box::new(Self::compute(agent, tool_use, Some(renderer)).await)
    }

    async fn apply_tool_result(
        &self,
        _client: &Anthropic,
        _agent: &mut SidAgent,
        _tool_use: &ToolUseBlock,
        intermediate: Box<dyn IntermediateToolResult>,
    ) -> ToolResult {
        apply_computed_tool_result(intermediate)
    }
}

#[derive(Clone, Debug)]
struct SidTextEditorTool {
    param: ToolTextEditor20250728,
}

impl SidTextEditorTool {
    fn new() -> Self {
        Self {
            param: ToolTextEditor20250728::new(),
        }
    }
}

impl Tool<SidAgent> for SidTextEditorTool {
    fn name(&self) -> String {
        self.param.name.clone()
    }

    fn callback(&self) -> Box<dyn ToolCallback<SidAgent> + '_> {
        Box::new(SidTextEditorCallback)
    }

    fn to_param(&self) -> ToolUnionParam {
        ToolUnionParam::TextEditor20250728(self.param.clone())
    }
}

struct SidTextEditorCallback;

impl SidTextEditorCallback {
    async fn compute(
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
        renderer: Option<&mut dyn Renderer>,
    ) -> ToolResult {
        match agent.run_text_editor_tool(tool_use.clone(), renderer).await {
            Ok(output) => tool_success_result(&tool_use.id, output),
            Err(err) => tool_error_result(&tool_use.id, err.to_string()),
        }
    }
}

#[async_trait::async_trait]
impl ToolCallback<SidAgent> for SidTextEditorCallback {
    async fn compute_tool_result(
        &self,
        _client: &Anthropic,
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
    ) -> Box<dyn IntermediateToolResult> {
        Box::new(Self::compute(agent, tool_use, None).await)
    }

    async fn compute_tool_result_streaming(
        &self,
        _client: &Anthropic,
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
        renderer: &mut dyn Renderer,
        _context: &AgentStreamContext,
    ) -> Box<dyn IntermediateToolResult> {
        Box::new(Self::compute(agent, tool_use, Some(renderer)).await)
    }

    async fn apply_tool_result(
        &self,
        _client: &Anthropic,
        _agent: &mut SidAgent,
        _tool_use: &ToolUseBlock,
        intermediate: Box<dyn IntermediateToolResult>,
    ) -> ToolResult {
        apply_computed_tool_result(intermediate)
    }
}

fn apply_computed_tool_result(intermediate: Box<dyn IntermediateToolResult>) -> ToolResult {
    if let Some(intermediate) = intermediate.as_any().downcast_ref::<ToolResult>() {
        return intermediate.clone();
    }
    if let Some(intermediate) = intermediate.as_any().downcast_ref::<ComputedToolResult>() {
        return intermediate.result.clone();
    }
    ControlFlow::Break(Error::unknown(
        "intermediate tool result fails to deserialize",
    ))
}

#[derive(Clone, Debug)]
struct ComputedToolResult {
    result: ToolResult,
    render_result: bool,
}

impl ComputedToolResult {
    fn new(result: ToolResult) -> Self {
        Self {
            result,
            render_result: true,
        }
    }

    fn with_render_result(result: ToolResult, render_result: bool) -> Self {
        Self {
            result,
            render_result,
        }
    }
}

impl IntermediateToolResult for ComputedToolResult {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn should_render_tool_result(intermediate: &dyn IntermediateToolResult) -> bool {
    intermediate
        .as_any()
        .downcast_ref::<ComputedToolResult>()
        .map(|intermediate| intermediate.render_result)
        .unwrap_or(true)
}

#[derive(Clone, Debug)]
struct AskAnExpertTool;

impl Tool<SidAgent> for AskAnExpertTool {
    fn name(&self) -> String {
        ASK_AN_EXPERT_TOOL_NAME.to_string()
    }

    fn callback(&self) -> Box<dyn ToolCallback<SidAgent> + '_> {
        Box::new(AskAnExpertCallback)
    }

    fn to_param(&self) -> ToolUnionParam {
        ToolUnionParam::CustomTool(
            ToolParam::new(
                ASK_AN_EXPERT_TOOL_NAME.to_string(),
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The direct question to ask the previous summary writer."
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": false
                }),
            )
            .with_description("Ask a direct question of the person who wrote the summary for this compacted session. They are the previous owner of the project and only available for consultation by contextual memory.".to_string()),
        )
    }
}

struct AskAnExpertCallback;

#[async_trait::async_trait]
impl ToolCallback<SidAgent> for AskAnExpertCallback {
    async fn compute_tool_result(
        &self,
        client: &Anthropic,
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
    ) -> Box<dyn IntermediateToolResult> {
        Box::new(ask_an_expert_result(client, agent, tool_use).await)
    }

    async fn compute_tool_result_streaming(
        &self,
        client: &Anthropic,
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
        renderer: &mut dyn Renderer,
        context: &AgentStreamContext,
    ) -> Box<dyn IntermediateToolResult> {
        Box::new(ask_an_expert_result_streaming(client, agent, tool_use, renderer, context).await)
    }

    async fn apply_tool_result(
        &self,
        _client: &Anthropic,
        _agent: &mut SidAgent,
        _tool_use: &ToolUseBlock,
        intermediate: Box<dyn IntermediateToolResult>,
    ) -> ToolResult {
        apply_computed_tool_result(intermediate)
    }
}

async fn ask_an_expert_result(
    client: &Anthropic,
    agent: &SidAgent,
    tool_use: &ToolUseBlock,
) -> ToolResult {
    let question = match ask_an_expert_question(tool_use) {
        Ok(question) => question,
        Err(result) => return result,
    };

    match agent.ask_an_expert(client, &question).await {
        Ok(output) => tool_success_result(&tool_use.id, output),
        Err(err) => tool_error_result(&tool_use.id, err.to_string()),
    }
}

fn ask_an_expert_question(tool_use: &ToolUseBlock) -> Result<String, ToolResult> {
    #[derive(serde::Deserialize)]
    struct AskAnExpertInput {
        question: String,
    }

    let input: AskAnExpertInput = match serde_json::from_value(tool_use.input.clone()) {
        Ok(input) => input,
        Err(err) => return Err(tool_error_result(&tool_use.id, err.to_string())),
    };
    let question = input.question.trim();
    if question.is_empty() {
        return Err(tool_error_result(
            &tool_use.id,
            "question must not be empty".to_string(),
        ));
    }

    Ok(question.to_string())
}

async fn ask_an_expert_result_streaming(
    client: &Anthropic,
    agent: &SidAgent,
    tool_use: &ToolUseBlock,
    renderer: &mut dyn Renderer,
    context: &AgentStreamContext,
) -> ComputedToolResult {
    let question = match ask_an_expert_question(tool_use) {
        Ok(question) => question,
        Err(result) => return ComputedToolResult::new(result),
    };
    let mut renderer = AskAnExpertStreamRenderer::new(renderer, context, &question);
    let result = match agent
        .ask_an_expert_with_renderer(client, &question, Some(&mut renderer))
        .await
    {
        Ok(output) => ComputedToolResult::with_render_result(
            tool_success_result(&tool_use.id, output),
            !renderer.started(),
        ),
        Err(err) => ComputedToolResult::new(tool_error_result(&tool_use.id, err.to_string())),
    };
    renderer.finish_if_open(None);
    result
}

struct AskAnExpertStreamRenderer<'a> {
    parent: &'a mut dyn Renderer,
    context: AgentStreamContext,
    question: &'a str,
    started: bool,
    finished: bool,
}

impl<'a> AskAnExpertStreamRenderer<'a> {
    fn new(
        parent: &'a mut dyn Renderer,
        parent_context: &AgentStreamContext,
        question: &'a str,
    ) -> Self {
        Self {
            parent,
            context: parent_context.child("expert"),
            question,
            started: false,
            finished: false,
        }
    }

    fn ensure_started(&mut self) {
        if self.started {
            return;
        }
        self.parent.start_agent(&self.context);
        self.parent.print_info(&self.context, "question:");
        self.parent.print_text(&self.context, self.question);
        if !self.question.ends_with('\n') {
            self.parent.print_text(&self.context, "\n");
        }
        self.parent.print_text(&self.context, "\n");
        self.started = true;
    }

    fn finish_if_open(&mut self, stop_reason: Option<&StopReason>) {
        if self.started && !self.finished {
            self.parent.finish_agent(&self.context, stop_reason);
            self.finished = true;
        }
    }

    fn started(&self) -> bool {
        self.started
    }
}

impl Renderer for AskAnExpertStreamRenderer<'_> {
    fn start_agent(&mut self, _context: &dyn StreamContext) {
        self.ensure_started();
    }

    fn finish_agent(&mut self, _context: &dyn StreamContext, stop_reason: Option<&StopReason>) {
        self.finish_if_open(stop_reason);
    }

    fn print_text(&mut self, _context: &dyn StreamContext, text: &str) {
        self.ensure_started();
        self.parent.print_text(&self.context, text);
    }

    fn print_thinking(&mut self, _context: &dyn StreamContext, text: &str) {
        self.ensure_started();
        self.parent.print_thinking(&self.context, text);
    }

    fn print_error(&mut self, _context: &dyn StreamContext, error: &str) {
        self.ensure_started();
        self.parent.print_error(&self.context, error);
    }

    fn print_info(&mut self, _context: &dyn StreamContext, info: &str) {
        self.ensure_started();
        self.parent.print_info(&self.context, info);
    }

    fn start_tool_use(&mut self, _context: &dyn StreamContext, name: &str, id: &str) {
        self.ensure_started();
        self.parent.start_tool_use(&self.context, name, id);
    }

    fn print_tool_input(&mut self, _context: &dyn StreamContext, partial_json: &str) {
        self.ensure_started();
        self.parent.print_tool_input(&self.context, partial_json);
    }

    fn finish_tool_use(&mut self, _context: &dyn StreamContext) {
        self.ensure_started();
        self.parent.finish_tool_use(&self.context);
    }

    fn start_tool_result(
        &mut self,
        _context: &dyn StreamContext,
        tool_use_id: &str,
        is_error: bool,
    ) {
        self.ensure_started();
        self.parent
            .start_tool_result(&self.context, tool_use_id, is_error);
    }

    fn print_tool_result_text(&mut self, _context: &dyn StreamContext, text: &str) {
        self.ensure_started();
        self.parent.print_tool_result_text(&self.context, text);
    }

    fn finish_tool_result(&mut self, _context: &dyn StreamContext) {
        self.ensure_started();
        self.parent.finish_tool_result(&self.context);
    }

    fn finish_response(&mut self, _context: &dyn StreamContext) {
        self.ensure_started();
        self.parent.finish_response(&self.context);
    }

    fn print_interrupted(&mut self, _context: &dyn StreamContext) {
        self.ensure_started();
        self.parent.print_interrupted(&self.context);
    }

    fn should_interrupt(&self) -> bool {
        self.parent.should_interrupt()
    }

    fn read_operator_line(&mut self, prompt: &str) -> std::io::Result<Option<OperatorLine>> {
        self.parent.read_operator_line(prompt)
    }
}

#[derive(Clone, Debug)]
struct ExternalTool {
    name: String,
    canonical_id: String,
    enabled: SwitchPosition,
    confirm_preview: bool,
    executable_path: Path<'static>,
    description: String,
    input_schema: serde_json::Value,
    timeout: Option<Duration>,
}

impl ExternalTool {
    fn from_config(name: String, canonical_id: String, tool: &ToolConfig) -> Self {
        let manifest = tool
            .manifest
            .as_ref()
            .expect("external tools require a manifest");
        let executable_path = tool
            .executable_path
            .clone()
            .expect("external tools require an executable path");
        Self {
            name,
            canonical_id,
            enabled: tool.enabled,
            confirm_preview: tool.confirm_preview,
            executable_path,
            description: manifest.description.clone(),
            input_schema: manifest.input_schema.clone(),
            timeout: tool.timeout,
        }
    }
}

impl Tool<SidAgent> for ExternalTool {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn callback(&self) -> Box<dyn ToolCallback<SidAgent> + '_> {
        Box::new(ExternalToolCallback { tool: self.clone() })
    }

    fn to_param(&self) -> ToolUnionParam {
        ToolUnionParam::CustomTool(
            ToolParam::new(self.name.clone(), self.input_schema.clone())
                .with_description(self.description.clone()),
        )
    }
}

#[derive(Clone, Debug)]
struct ExternalToolCallback {
    tool: ExternalTool,
}

#[async_trait::async_trait]
impl ToolCallback<SidAgent> for ExternalToolCallback {
    async fn compute_tool_result(
        &self,
        _client: &Anthropic,
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
    ) -> Box<dyn IntermediateToolResult> {
        Box::new(invoke_external_tool(&self.tool, agent, tool_use, None).await)
    }

    async fn compute_tool_result_streaming(
        &self,
        _client: &Anthropic,
        agent: &SidAgent,
        tool_use: &ToolUseBlock,
        renderer: &mut dyn Renderer,
        _context: &AgentStreamContext,
    ) -> Box<dyn IntermediateToolResult> {
        Box::new(invoke_external_tool(&self.tool, agent, tool_use, Some(renderer)).await)
    }

    async fn apply_tool_result(
        &self,
        _client: &Anthropic,
        _agent: &mut SidAgent,
        _tool_use: &ToolUseBlock,
        intermediate: Box<dyn IntermediateToolResult>,
    ) -> ToolResult {
        apply_computed_tool_result(intermediate)
    }
}

async fn invoke_external_tool(
    tool: &ExternalTool,
    agent: &SidAgent,
    tool_use: &ToolUseBlock,
    renderer: Option<&mut dyn Renderer>,
) -> ToolResult {
    if tool.enabled == SwitchPosition::No {
        return tool_error_result(&tool_use.id, format!("tool '{}' is disabled", tool.name));
    }

    let input = match tool_use.input.as_object() {
        Some(input) => input.clone(),
        None => {
            return tool_error_result(
                &tool_use.id,
                format!(
                    "tool '{}' protocol error: tool input must be a JSON object",
                    tool.name
                ),
            );
        }
    };

    match tool.enabled {
        SwitchPosition::Yes => {}
        SwitchPosition::No => unreachable!("disabled tools return before input validation"),
        SwitchPosition::Manual => {
            let context = agent.tool_runtime_context();
            let prepared = match tool_runtime::prepare_rc_tool_invocation(
                &tool.name,
                &tool.name,
                &tool.canonical_id,
                &tool.executable_path,
                &context,
                &tool_use.id,
                input,
                tool.timeout,
            ) {
                Ok(prepared) => prepared,
                Err(message) => return tool_error_result(&tool_use.id, message),
            };
            match confirm_manual_prepared_tool_call(
                &tool.name,
                &tool_use.input,
                tool.confirm_preview,
                &prepared,
                agent.session.as_deref(),
                renderer,
            )
            .await
            {
                Ok(ManualToolConfirmation::Allow) => {}
                Ok(ManualToolConfirmation::Deny) => {
                    let _ = tool_runtime::cleanup_prepared_rc_tool(&prepared, false);
                    return tool_error_result(
                        &tool_use.id,
                        format!("tool '{}' call denied by operator", tool.name),
                    );
                }
                Ok(ManualToolConfirmation::Cancel) => {
                    let _ = tool_runtime::cleanup_prepared_rc_tool(&prepared, false);
                    return tool_user_cancelled_result(&tool_use.id);
                }
                Err(err) => {
                    let _ = tool_runtime::cleanup_prepared_rc_tool(&prepared, true);
                    return tool_error_result(&tool_use.id, err);
                }
            }
            return match tool_runtime::run_prepared_rc_tool_text(
                &prepared,
                &agent.writable_roots,
                agent.session.as_deref(),
            )
            .await
            {
                Ok(text) => tool_success_result(&tool_use.id, text),
                Err(message) => tool_error_result(&tool_use.id, message),
            };
        }
    }

    let context = ToolRuntimeContext {
        agent_id: &agent.id,
        config_root: &agent.config_root,
        workspace_root: &agent.workspace_root,
        writable_roots: &agent.writable_roots,
        session: agent.session.as_deref(),
    };
    match tool_runtime::invoke_rc_tool_text(
        &tool.name,
        &tool.name,
        &tool.canonical_id,
        &tool.executable_path,
        &context,
        &tool_use.id,
        input,
        tool.timeout,
    )
    .await
    {
        Ok(text) => tool_success_result(&tool_use.id, text),
        Err(message) => tool_error_result(&tool_use.id, message),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ManualToolConfirmation {
    Allow,
    Deny,
    Cancel,
}

#[derive(Debug, Eq, PartialEq)]
enum ManualToolInput {
    Line(String),
    Cancel,
}

/// Prompt the operator to confirm a manual tool call via stdin/stdout.
///
/// Returns `Allow` when the operator approves, `Deny` when the operator answers
/// no, `Cancel` when the prompt is interrupted or reaches EOF, and `Err` on I/O
/// failure.
async fn confirm_manual_prepared_tool_call(
    tool_name: &str,
    input: &serde_json::Value,
    confirm_preview: bool,
    prepared: &tool_runtime::PreparedRcToolInvocation,
    session: Option<&SidSession>,
    renderer: Option<&mut dyn Renderer>,
) -> Result<ManualToolConfirmation, String> {
    let preview = if confirm_preview {
        tool_runtime::render_rc_tool_confirmation_preview(prepared, session)
            .await
            .ok()
    } else {
        None
    };
    confirm_manual_tool_call(tool_name, input, preview.as_deref(), renderer)
}

fn confirm_manual_tool_call(
    tool_name: &str,
    input: &serde_json::Value,
    preview: Option<&str>,
    mut renderer: Option<&mut dyn Renderer>,
) -> Result<ManualToolConfirmation, String> {
    let input_display = preview.map(str::to_string).unwrap_or_else(|| {
        serde_json::to_string_pretty(input).unwrap_or_else(|_| format!("{input:?}"))
    });

    loop {
        let prompt =
            format!("Tool '{tool_name}' is MANUAL.\n{input_display}\nAllow this call? [yes/no]: ");
        let input = read_operator_line(&mut renderer, &prompt)?;
        let ManualToolInput::Line(buf) = input else {
            return Ok(ManualToolConfirmation::Cancel);
        };

        match parse_tool_confirmation(&buf) {
            Some(true) => return Ok(ManualToolConfirmation::Allow),
            Some(false) => return Ok(ManualToolConfirmation::Deny),
            None => println!("Please answer yes or no."),
        }
    }
}

fn read_operator_line(
    renderer: &mut Option<&mut dyn Renderer>,
    prompt: &str,
) -> Result<ManualToolInput, String> {
    use std::io::{self, Write};

    if let Some(renderer) = renderer.as_mut()
        && let Some(line) = (*renderer)
            .read_operator_line(prompt)
            .map_err(|err| format!("failed to read manual-tool confirmation input: {err}"))?
    {
        return match line {
            OperatorLine::Line(line) => Ok(ManualToolInput::Line(line)),
            OperatorLine::Eof | OperatorLine::Interrupted => {
                println!();
                Ok(ManualToolInput::Cancel)
            }
        };
    }

    print!("{prompt}");
    io::stdout()
        .flush()
        .map_err(|err| format!("failed to flush manual-tool confirmation prompt: {err}"))?;

    let mut buf = String::new();
    match io::stdin().read_line(&mut buf) {
        Ok(0) => {
            println!();
            Ok(ManualToolInput::Cancel)
        }
        Ok(_) => Ok(ManualToolInput::Line(buf)),
        Err(err) if err.kind() == io::ErrorKind::Interrupted => {
            println!();
            Ok(ManualToolInput::Cancel)
        }
        Err(err) => Err(format!(
            "failed to read manual-tool confirmation input: {err}"
        )),
    }
}

fn parse_tool_confirmation(input: &str) -> Option<bool> {
    match input.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Some(true),
        "n" | "no" => Some(false),
        _ => None,
    }
}

fn render_bash_pty_result(result: BashPtyResult) -> Result<String, std::io::Error> {
    let mut rendered = strip_ansi_escapes(&result.output);
    if result.status.success() {
        if rendered.is_empty() {
            rendered.push_str("success\n");
        }
        Ok(rendered)
    } else {
        if !rendered.is_empty() && !rendered.ends_with('\n') {
            rendered.push('\n');
        }
        rendered.push_str(&format!("{}\n", result.status));
        Err(std::io::Error::other(rendered))
    }
}

fn bash_state_io_error(context: &str, result: &BashPtyResult) -> std::io::Error {
    let mut rendered = strip_ansi_escapes(&result.output);
    if !rendered.is_empty() && !rendered.ends_with('\n') {
        rendered.push('\n');
    }
    rendered.push_str(&format!("{}", result.status));
    std::io::Error::other(format!("{context}: {rendered}"))
}

/// Strip ANSI escape sequences from terminal output.
///
/// Removes CSI sequences (`ESC[…X`), OSC sequences (`ESC]…ST`), and simple
/// two-character escape pairs (`ESC X`).  This ensures the model never sees
/// raw color or cursor-control codes in bash tool output.
fn strip_ansi_escapes(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch != '\x1b' {
            out.push(ch);
            continue;
        }
        // We saw ESC.  Peek at the next character to decide the sequence type.
        match chars.next() {
            // CSI sequence: ESC [ <params> <intermediate> <final byte>
            Some('[') => {
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() || c == '@' || c == '~' {
                        break;
                    }
                }
            }
            // OSC sequence: ESC ] … terminated by BEL or ST (ESC \)
            Some(']') => {
                let mut prev = '\0';
                for c in chars.by_ref() {
                    if c == '\x07' {
                        break;
                    }
                    if prev == '\x1b' && c == '\\' {
                        break;
                    }
                    prev = c;
                }
            }
            // Two-character escape (e.g. ESC M, ESC 7, ESC 8) — consume and
            // discard the second byte.
            Some(_) => {}
            // Trailing bare ESC at end of input — discard it.
            None => {}
        }
    }
    out
}

#[derive(serde::Deserialize)]
struct TranscriptSnapshot {
    version: u8,
    messages: Vec<MessageParam>,
}

struct NullRenderer;

impl Renderer for NullRenderer {
    fn print_text(&mut self, _context: &dyn StreamContext, _text: &str) {}

    fn print_thinking(&mut self, _context: &dyn StreamContext, _text: &str) {}

    fn print_error(&mut self, _context: &dyn StreamContext, _error: &str) {}

    fn print_info(&mut self, _context: &dyn StreamContext, _info: &str) {}

    fn start_tool_use(&mut self, _context: &dyn StreamContext, _name: &str, _id: &str) {}

    fn print_tool_input(&mut self, _context: &dyn StreamContext, _partial_json: &str) {}

    fn finish_tool_use(&mut self, _context: &dyn StreamContext) {}

    fn start_tool_result(
        &mut self,
        _context: &dyn StreamContext,
        _tool_use_id: &str,
        _is_error: bool,
    ) {
    }

    fn print_tool_result_text(&mut self, _context: &dyn StreamContext, _text: &str) {}

    fn finish_tool_result(&mut self, _context: &dyn StreamContext) {}

    fn finish_response(&mut self, _context: &dyn StreamContext) {}
}

/// Deserialize a transcript file into a sequence of message parameters.
///
/// The file must be a version-1 transcript snapshot written by the session
/// runtime.
///
/// # Errors
///
/// Returns an error when the file cannot be read, is not valid JSON, or has
/// an unsupported transcript version.
pub fn load_transcript_messages(path: &std::path::Path) -> Result<Vec<MessageParam>, SError> {
    let payload = std::fs::read(path).map_err(|err| {
        SError::new("sid-transcript")
            .with_code("transcript_read_failed")
            .with_message("failed to read transcript")
            .with_string_field("path", path.to_string_lossy().as_ref())
            .with_string_field("cause", &err.to_string())
    })?;
    let snapshot: TranscriptSnapshot = serde_json::from_slice(&payload).map_err(|err| {
        SError::new("sid-transcript")
            .with_code("transcript_parse_failed")
            .with_message("failed to parse transcript")
            .with_string_field("path", path.to_string_lossy().as_ref())
            .with_string_field("cause", &err.to_string())
    })?;
    if snapshot.version != 1 {
        return Err(SError::new("sid-transcript")
            .with_code("unsupported_transcript_version")
            .with_message("unsupported transcript version")
            .with_string_field("path", path.to_string_lossy().as_ref())
            .with_string_field("version", &snapshot.version.to_string()));
    }
    Ok(snapshot.messages)
}

/// Build a two-message transcript representing a compacted session.
///
/// The first message is a user message containing the compaction context
/// template, and the second is an assistant message carrying the handoff
/// `summary`.
pub fn compacted_transcript(parent_session_id: &str, summary: &str) -> Vec<MessageParam> {
    vec![
        MessageParam::user(
            COMPACTED_SESSION_CONTEXT_TEMPLATE.replace("{session_id}", parent_session_id),
        ),
        MessageParam::assistant(summary),
    ]
}

fn tool_success_result(tool_use_id: &str, message: String) -> ToolResult {
    ControlFlow::Continue(Ok(
        ToolResultBlock::new(tool_use_id.to_string()).with_string_content(message)
    ))
}

fn tool_error_result(tool_use_id: &str, message: String) -> ToolResult {
    ControlFlow::Continue(Err(ToolResultBlock::new(tool_use_id.to_string())
        .with_string_content(message)
        .with_error(true)))
}

fn tool_user_cancelled_result(tool_use_id: &str) -> ToolResult {
    ControlFlow::Continue(user_cancelled_tool_result(tool_use_id))
}

/// Build the default writable roots for a workspace.
///
/// Includes the workspace root itself and the system temp directory.  Paths are
/// canonicalized so that the sandbox policy matches the kernel-resolved paths
/// (e.g. `/var` -> `/private/var` on macOS).
fn default_writable_roots(workspace_root: &Path) -> WritableRoots {
    let mut roots = WritableRoots::default();
    if let Ok(canonical) = std::fs::canonicalize(workspace_root.as_str()) {
        if let Some(s) = canonical.to_str() {
            roots.push(s.to_string());
        }
    } else {
        roots.push(workspace_root.as_str().to_string());
    }
    let temp_dir = std::env::temp_dir();
    if let Ok(canonical) = std::fs::canonicalize(&temp_dir) {
        if let Some(s) = canonical.to_str() {
            roots.push(s.to_string());
        }
    } else if let Some(s) = temp_dir.to_str() {
        roots.push(s.to_string());
    }
    roots
}

fn append_writable_root(roots: &mut WritableRoots, root: &std::path::Path) {
    if let Ok(canonical) = std::fs::canonicalize(root)
        && let Some(s) = canonical.to_str()
    {
        roots.push(s.to_string());
        return;
    }
    if let Some(s) = root.to_str() {
        roots.push(s.to_string());
    }
}

fn workspace_has_config(root: &Path) -> bool {
    root.join(AGENTS_CONF_FILE).is_file() || root.join(TOOLS_CONF_FILE).is_file()
}

/// Determine the default agent to use when none is explicitly requested.
///
/// Prefers the explicit `DEFAULT_AGENT` setting from agents.conf when present.
/// Falls back to the first enabled agent, then the first manual agent.
fn default_agent_id(config: &Config) -> Result<String, SError> {
    if let Some(default) = config.default_agent.as_ref() {
        let agent_config = config.agents.get(default).ok_or_else(|| {
            SError::new("sid-agent")
                .with_code("invalid_default_agent")
                .with_message("DEFAULT_AGENT names an undefined agent")
                .with_string_field("default_agent", default)
        })?;
        if !agent_config.enabled.can_be_started() {
            return Err(SError::new("sid-agent")
                .with_code("disabled_default_agent")
                .with_message("DEFAULT_AGENT names a disabled agent")
                .with_string_field("default_agent", default)
                .with_string_field("enabled", &format!("{:?}", agent_config.enabled)));
        }
        return Ok(default.clone());
    }

    if let Some(agent) = config
        .agents
        .values()
        .find(|agent| agent.enabled == SwitchPosition::Yes)
    {
        return Ok(agent.id.clone());
    }
    if let Some(agent) = config
        .agents
        .values()
        .find(|agent| agent.enabled == SwitchPosition::Manual)
    {
        return Ok(agent.id.clone());
    }

    Err(SError::new("sid-agent")
        .with_code("no_startable_agents")
        .with_message("workspace config defines no runnable agents"))
}

fn build_tools(config: &Config, agent_config: &AgentConfig) -> Result<BuiltTools, SError> {
    let mut tools = Vec::new();
    let mut seen = BTreeSet::new();
    let mut builtin_bindings = BuiltinToolBindings::default();
    for tool_name in &agent_config.tools {
        let tool_config = config.tools.get(tool_name).ok_or_else(|| {
            SError::new("sid-agent")
                .with_code("unknown_tool")
                .with_message("agent references an undefined tool")
                .with_string_field("agent", &agent_config.id)
                .with_string_field("tool", tool_name)
        })?;
        if !tool_config.enabled.can_be_started() {
            return Err(disabled_tool_error(
                &agent_config.id,
                tool_name,
                tool_config.enabled,
            ));
        }
        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, tool_name)?;
        let Some(builtin_kind) = BuiltinToolKind::from_canonical_id(&canonical_id) else {
            let tool = ExternalTool::from_config(
                exposed_tool_name(&agent_config.id, tool_name)?,
                canonical_id,
                tool_config,
            );
            if seen.insert(tool.name.clone()) {
                tools.push(Arc::new(tool) as Arc<dyn Tool<SidAgent>>);
            }
            continue;
        };
        let tool = builtin_kind.tool();
        let exposed_name = tool.name();
        if seen.insert(exposed_name) {
            match builtin_kind {
                BuiltinToolKind::Bash if builtin_bindings.bash.is_none() => {
                    builtin_bindings.bash = Some(BuiltinBashBinding {
                        enabled: tool_config.enabled,
                    });
                }
                BuiltinToolKind::Edit if builtin_bindings.edit.is_none() => {
                    builtin_bindings.edit = Some(RcToolBinding {
                        service_name: tool_name.clone(),
                        canonical_id,
                        enabled: tool_config.enabled,
                        confirm_preview: tool_config.confirm_preview,
                        executable_path: tool_config
                            .executable_path
                            .clone()
                            .expect("built-in edit tool requires an executable path"),
                        timeout: tool_config.timeout,
                    });
                }
                _ => {}
            }
            tools.push(tool);
        }
    }
    Ok(BuiltTools {
        tools,
        builtin_bindings,
    })
}

fn exposed_tool_name(agent: &str, tool_name: &str) -> Result<String, SError> {
    if is_valid_anthropic_tool_name(tool_name) {
        Ok(tool_name.to_string())
    } else {
        Err(SError::new("sid-agent")
            .with_code("invalid_tool_name")
            .with_message("tool name exposed to the model is not legal")
            .with_string_field("agent", agent)
            .with_string_field("tool", tool_name)
            .with_string_field("name", tool_name))
    }
}

fn merged_chat_config(agent_config: &AgentConfig, fallback: Option<&ChatConfig>) -> ChatConfig {
    let mut merged = fallback.cloned().unwrap_or_else(ChatConfig::new);
    let defaults = ChatConfig::new();
    let agent = &agent_config.chat_config;

    if agent.template.model != defaults.template.model {
        merged.template.model = agent.template.model.clone();
    }
    if agent.template.system.is_some() {
        merged.template.system = agent.template.system.clone();
    }
    if agent.template.max_tokens != defaults.template.max_tokens {
        merged.template.max_tokens = agent.template.max_tokens;
    }
    if agent.template.temperature != defaults.template.temperature {
        merged.template.temperature = agent.template.temperature;
    }
    if agent.template.top_p != defaults.template.top_p {
        merged.template.top_p = agent.template.top_p;
    }
    if agent.template.top_k != defaults.template.top_k {
        merged.template.top_k = agent.template.top_k;
    }
    if agent.template.stop_sequences != defaults.template.stop_sequences {
        merged.template.stop_sequences = agent.template.stop_sequences.clone();
    }
    if agent.template.thinking != defaults.template.thinking {
        merged.template.thinking = agent.template.thinking;
    }
    if agent.use_color != defaults.use_color {
        merged.use_color = agent.use_color;
    }
    if agent.session_budget.is_some() {
        merged.session_budget = agent.session_budget.clone();
    }
    if agent.caching_enabled != defaults.caching_enabled {
        merged.caching_enabled = agent.caching_enabled;
    }

    merged
}

fn append_system_description(chat_config: &mut ChatConfig, workspace_root: &Path) {
    let existing = chat_config.system_prompt_text().unwrap_or("").to_string();
    let addendum = format!(
        r#"# Environment

You are operating in the sid-isn't-done environment.  Tools:
- edit: The Anthropic Text Editor tool:
    - Uses sid's virtual filesystem.
    - The virtual filesystem maps / to the workspace root.
    - It is not an operating-system chroot.
    - Absolute editor paths are workspace-rooted; /foo means {workspace_root}/foo.
- bash: A genuine bash shell:
    - Connected via PTY.
    - Without support for cursor positioning.
    - With state persistence between invocations.
    - With PS0, PS1, PS2 and PROMPT_COMMAND set to readonly.
    - `restart: true` throws the session away and starts fresh.
    - Initial CWD is {workspace_root}.
    - Runs in the host filesystem namespace, not a chroot.
    - Host / remains visible subject to OS permissions and sandbox policy.
    - Bash cannot see sid's virtual /skills mount.
    - Use the index to browse skills if you need specialized knowledge.

CRITICAL — the edit tool and bash tool use different path namespaces:
- The edit tool's / is the workspace root ({workspace_root}).
  To edit a file at the workspace root, use /filename (e.g., /src/lib.rs).
  NEVER pass a full host path to the edit tool.
- Bash sees the real host filesystem.  `pwd` prints {workspace_root}, not /.
  To convert a bash path to an edit path, strip the {workspace_root} prefix.
  To convert an edit path to a bash path, prepend {workspace_root}.
  If bash `pwd` is a subdirectory, a relative path like ./foo.rs in bash
  corresponds to stripping the workspace prefix from the absolute bash path.
- Example: the bash path {workspace_root}/src/lib.rs is the edit path /src/lib.rs.
"#,
    );

    chat_config.set_system_prompt(Some(format!("{existing}{addendum}")));
}

/// Append a skill index to the system prompt so the model knows what skills are
/// available and where they are mounted.
fn append_skill_index_to_system_prompt(chat_config: &mut ChatConfig, skills: &[&SkillConfig]) {
    let mut index = String::from("\n\n# Available skills (mounted read-only under /skills/):\n");
    for skill in skills {
        index.push_str(&format!("  - /skills/{}/SKILL.md\n", skill.id));
    }
    let existing = chat_config.system_prompt_text().unwrap_or("").to_string();
    chat_config.set_system_prompt(Some(format!("{existing}{index}")));
}

fn latest_user_message_text(messages: &[MessageParam]) -> Option<String> {
    let message = messages.last()?;
    if message.role != MessageRole::User {
        return None;
    }
    match &message.content {
        claudius::MessageParamContent::String(text) => Some(text.clone()),
        claudius::MessageParamContent::Array(blocks) => {
            if blocks.iter().any(ContentBlock::is_tool_result) {
                return None;
            }
            let text = blocks
                .iter()
                .filter_map(ContentBlock::as_text)
                .map(|block| block.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            Some(text)
        }
    }
}

/// Extract the concatenated text blocks from the last assistant message.
///
/// Returns `None` when the slice contains no assistant messages or the last
/// assistant message has no text content.
pub fn extract_last_assistant_text(messages: &[MessageParam]) -> Option<String> {
    let message = messages
        .iter()
        .rev()
        .find(|message| message.role == MessageRole::Assistant)?;
    assistant_message_text(message)
}

fn assistant_message_text(message: &MessageParam) -> Option<String> {
    if message.role != MessageRole::Assistant {
        return None;
    }
    match &message.content {
        MessageParamContent::String(text) => Some(text.clone()),
        MessageParamContent::Array(blocks) => {
            let text = blocks
                .iter()
                .filter_map(ContentBlock::as_text)
                .map(|block| block.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            if text.is_empty() { None } else { Some(text) }
        }
    }
}

fn missing_agent_error(agent: &str) -> SError {
    SError::new("sid-agent")
        .with_code("unknown_agent")
        .with_message("requested agent is not defined")
        .with_string_field("agent", agent)
}

fn disabled_agent_error(agent: &str, enabled: SwitchPosition) -> SError {
    SError::new("sid-agent")
        .with_code("disabled_agent")
        .with_message("requested agent is disabled")
        .with_string_field("agent", agent)
        .with_string_field("enabled", &format!("{enabled:?}"))
}

fn disabled_tool_error(agent: &str, tool: &str, enabled: SwitchPosition) -> SError {
    SError::new("sid-agent")
        .with_code("disabled_tool")
        .with_message("agent references a disabled tool")
        .with_string_field("agent", agent)
        .with_string_field("tool", tool)
        .with_string_field("enabled", &format!("{enabled:?}"))
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::fs;

    use claudius::{
        KnownModel, MessageParamContent, ToolBash20250124, ToolTextEditor20250728, ToolUnionParam,
    };
    use serde_json::json;

    use super::*;
    use crate::config::{TOOL_PROTOCOL_VERSION, TOOLS_DIR};
    use crate::test_support::{
        make_executable, temp_config_root, unique_temp_dir, write_default_tool_manifest,
    };

    struct ScriptedRenderer {
        lines: VecDeque<OperatorLine>,
        prompts: Vec<String>,
    }

    impl ScriptedRenderer {
        fn new(lines: impl IntoIterator<Item = OperatorLine>) -> Self {
            Self {
                lines: lines.into_iter().collect(),
                prompts: Vec::new(),
            }
        }
    }

    impl Renderer for ScriptedRenderer {
        fn print_text(&mut self, _context: &dyn claudius::StreamContext, _text: &str) {}

        fn print_thinking(&mut self, _context: &dyn claudius::StreamContext, _text: &str) {}

        fn print_error(&mut self, _context: &dyn claudius::StreamContext, _error: &str) {}

        fn print_info(&mut self, _context: &dyn claudius::StreamContext, _info: &str) {}

        fn start_tool_use(
            &mut self,
            _context: &dyn claudius::StreamContext,
            _name: &str,
            _id: &str,
        ) {
        }

        fn print_tool_input(
            &mut self,
            _context: &dyn claudius::StreamContext,
            _partial_json: &str,
        ) {
        }

        fn finish_tool_use(&mut self, _context: &dyn claudius::StreamContext) {}

        fn start_tool_result(
            &mut self,
            _context: &dyn claudius::StreamContext,
            _tool_use_id: &str,
            _is_error: bool,
        ) {
        }

        fn print_tool_result_text(&mut self, _context: &dyn claudius::StreamContext, _text: &str) {}

        fn finish_tool_result(&mut self, _context: &dyn claudius::StreamContext) {}

        fn finish_response(&mut self, _context: &dyn claudius::StreamContext) {}

        fn read_operator_line(&mut self, prompt: &str) -> std::io::Result<Option<OperatorLine>> {
            self.prompts.push(prompt.to_string());
            Ok(self.lines.pop_front())
        }
    }

    #[derive(Default)]
    struct RecordingRenderer {
        output: String,
    }

    impl Renderer for RecordingRenderer {
        fn start_agent(&mut self, context: &dyn claudius::StreamContext) {
            self.output.push_str(&format!(
                "[start:{}:{}]\n",
                context.label().unwrap_or_default(),
                context.depth()
            ));
        }

        fn finish_agent(
            &mut self,
            context: &dyn claudius::StreamContext,
            stop_reason: Option<&StopReason>,
        ) {
            self.output.push_str(&format!(
                "[finish:{}:{}:{stop_reason:?}]\n",
                context.label().unwrap_or_default(),
                context.depth(),
            ));
        }

        fn print_text(&mut self, _context: &dyn claudius::StreamContext, text: &str) {
            self.output.push_str(text);
        }

        fn print_thinking(&mut self, _context: &dyn claudius::StreamContext, text: &str) {
            self.output.push_str(text);
        }

        fn print_error(&mut self, _context: &dyn claudius::StreamContext, error: &str) {
            self.output.push_str(error);
            self.output.push('\n');
        }

        fn print_info(&mut self, _context: &dyn claudius::StreamContext, info: &str) {
            self.output.push_str(info);
            self.output.push('\n');
        }

        fn start_tool_use(
            &mut self,
            _context: &dyn claudius::StreamContext,
            _name: &str,
            _id: &str,
        ) {
        }

        fn print_tool_input(&mut self, _context: &dyn claudius::StreamContext, partial_json: &str) {
            self.output.push_str(partial_json);
        }

        fn finish_tool_use(&mut self, _context: &dyn claudius::StreamContext) {}

        fn start_tool_result(
            &mut self,
            _context: &dyn claudius::StreamContext,
            _tool_use_id: &str,
            _is_error: bool,
        ) {
        }

        fn print_tool_result_text(&mut self, _context: &dyn claudius::StreamContext, text: &str) {
            self.output.push_str(text);
        }

        fn finish_tool_result(&mut self, _context: &dyn claudius::StreamContext) {}

        fn finish_response(&mut self, _context: &dyn claudius::StreamContext) {}
    }

    #[test]
    fn ask_an_expert_stream_renderer_streams_question_and_answer() {
        let parent_context = AgentStreamContext::root("build").child("tool:ask_an_expert");
        let mut parent = RecordingRenderer::default();
        {
            let mut renderer =
                AskAnExpertStreamRenderer::new(&mut parent, &parent_context, "Where is it?");
            renderer.start_agent(&());
            renderer.print_text(&(), "In src/lib.rs.");
            renderer.finish_agent(&(), Some(&StopReason::EndTurn));
        }

        assert!(parent.output.contains("[start:expert:2]"));
        assert!(parent.output.contains("question:\nWhere is it?\n\n"));
        assert!(parent.output.contains("In src/lib.rs."));
        assert!(parent.output.contains("[finish:expert:2:Some(EndTurn)]"));
    }

    #[test]
    fn computed_tool_result_can_suppress_terminal_rendering() {
        let result = ComputedToolResult::with_render_result(
            tool_success_result("toolu_test", "ok".to_string()),
            false,
        );

        assert!(!should_render_tool_result(&result));
        assert_eq!(
            unwrap_success_text(apply_computed_tool_result(Box::new(result))),
            "ok"
        );
    }

    #[test]
    fn from_config_uses_agent_prompt_and_tools() {
        let root = temp_config_root("agent");
        write_sample_config(&root);

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        assert_eq!(agent.id(), "build");
        assert_eq!(agent.stream_label(), "build".to_string());
        assert_eq!(
            agent.config.system_prompt_text(),
            Some(expected_build_system_prompt(&root).as_str())
        );

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tool_names = runtime
            .block_on(agent.tools())
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>();
        assert_eq!(tool_names, vec!["format".to_string(), "shell".to_string()]);
        assert!(!agent.requires_confirmation());

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_exposes_named_agent_prompts_and_snapshot_memory_prompt() {
        let root = temp_config_root("agent-prompts");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            r#"
build_ENABLED="YES"
build_PROMPT_COMPACTION='agents/compact-request.md'
build_PROMPT_MEMORY_EXPERT='agents/memory-expert.md'
"#,
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "").unwrap();
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(
            root.join("agents/compact-request.md").as_str(),
            "Write the summary as bullets.\n",
        )
        .unwrap();
        fs::write(
            root.join("agents/memory-expert.md").as_str(),
            "\n\n# Memory mode\n\nAnswer from the saved summary only.\n",
        )
        .unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        assert_eq!(
            agent.named_prompt_markdown("COMPACTION"),
            Some("Write the summary as bullets.\n")
        );
        assert_eq!(
            agent.named_prompt_markdown("MEMORY_EXPERT"),
            Some("\n\n# Memory mode\n\nAnswer from the saved summary only.\n")
        );
        assert_eq!(agent.named_prompt_markdown("MISSING"), None);

        let snapshot = agent.compaction_snapshot();
        assert_eq!(
            snapshot.memory_expert_prompt.as_deref(),
            Some("\n\n# Memory mode\n\nAnswer from the saved summary only.\n")
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_workspace_prefers_workspace_agent_and_fills_from_fallback() {
        let root = temp_config_root("agent");
        write_sample_config(&root);

        let fallback = ChatConfig::new()
            .with_model(Model::Known(KnownModel::ClaudeHaiku45))
            .with_system_prompt("fallback system".to_string())
            .with_max_tokens(2048);
        let agent = SidAgent::from_workspace(&root, fallback).unwrap();

        assert_eq!(agent.id(), "build");
        assert_eq!(
            agent.config.system_prompt_text(),
            Some(expected_build_system_prompt(&root).as_str())
        );
        assert_eq!(
            agent.config.model(),
            Model::Known(KnownModel::ClaudeHaiku45)
        );
        assert_eq!(agent.config.max_tokens(), 2048);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tool_names = runtime
            .block_on(agent.tools())
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>();
        assert_eq!(tool_names, vec!["format".to_string(), "shell".to_string()]);
        assert!(!agent.requires_confirmation());

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_workspace_with_config_root_loads_config_from_sid_home() {
        let config_root = temp_config_root("config-root");
        write_sample_config_with_fmt_script(
            &config_root,
            &capturing_tool_script(
                "format via sid_home",
                "sid-home-request-capture.json",
                "sid-home-env-capture.json",
            ),
        );
        let workspace_root = unique_temp_dir("workspace-root");
        fs::create_dir_all(workspace_root.as_str()).unwrap();

        let fallback = ChatConfig::new()
            .with_system_prompt("fallback system".to_string())
            .with_max_tokens(2048);
        let agent =
            SidAgent::from_workspace_with_config_root(&workspace_root, &config_root, fallback)
                .unwrap();
        assert_eq!(agent.id(), "build");
        assert_eq!(
            agent.config.system_prompt_text(),
            Some(expected_build_system_prompt(&workspace_root).as_str())
        );
        assert_eq!(agent.config.max_tokens(), 2048);

        let config = Config::load(&config_root).unwrap();
        let tool_config = config.tools.get("format").unwrap();
        let exposed_name = exposed_tool_name("build", "format").unwrap();
        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, "format").unwrap();
        let tool = ExternalTool::from_config(exposed_name.clone(), canonical_id, tool_config);
        let tool_use = ToolUseBlock::new(
            "toolu_sid_home_123",
            exposed_name,
            json!({ "paths": ["src/lib.rs"] }),
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(invoke_external_tool(&tool, &agent, &tool_use, None));
        assert_eq!(unwrap_success_text(result), "format via sid_home");

        let request: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(
                workspace_root
                    .join("sid-home-request-capture.json")
                    .as_str(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(request["workspace"]["root"], json!(workspace_root.as_str()));
        assert_eq!(request["workspace"]["cwd"], json!(workspace_root.as_str()));

        let env: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(workspace_root.join("sid-home-env-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(env["workspace_root"], json!(workspace_root.as_str()));
        assert_eq!(env["rc_d_path"], json!(config_root.join("tools").as_str()));
        assert!(
            env["rc_conf_path"]
                .as_str()
                .unwrap()
                .starts_with(config_root.join("tools.conf").as_str())
        );

        fs::remove_dir_all(config_root.as_str()).unwrap();
        fs::remove_dir_all(workspace_root.as_str()).unwrap();
    }

    #[test]
    fn from_workspace_with_home_config_root_runs_external_tools() {
        if !seatbelt::sandbox_available() {
            return;
        }

        let home = match std::env::var("HOME") {
            Ok(home) if !home.is_empty() => home,
            _ => return,
        };
        let config_leaf = std::path::Path::new(unique_temp_dir("home-tool-config").as_str())
            .file_name()
            .and_then(|name| name.to_str())
            .expect("temp dir should have a utf-8 leaf")
            .to_string();
        let config_root = Path::new(&home)
            .join(".sid-isnt-done-tests")
            .join(config_leaf)
            .into_owned();
        write_sample_config_with_fmt_script(
            &config_root,
            &capturing_tool_script(
                "format via home sid_root",
                "home-sid-root-request-capture.json",
                "home-sid-root-env-capture.json",
            ),
        );
        let workspace_root = unique_temp_dir("workspace-root");
        fs::create_dir_all(workspace_root.as_str()).unwrap();

        let fallback = ChatConfig::new()
            .with_system_prompt("fallback system".to_string())
            .with_max_tokens(2048);
        let agent =
            SidAgent::from_workspace_with_config_root(&workspace_root, &config_root, fallback)
                .unwrap();

        let config = Config::load(&config_root).unwrap();
        let tool_config = config.tools.get("format").unwrap();
        let exposed_name = exposed_tool_name("build", "format").unwrap();
        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, "format").unwrap();
        let tool = ExternalTool::from_config(exposed_name.clone(), canonical_id, tool_config);
        let tool_use = ToolUseBlock::new(
            "toolu_home_sid_root_123",
            exposed_name,
            json!({ "paths": ["src/lib.rs"] }),
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(invoke_external_tool(&tool, &agent, &tool_use, None));
        assert_eq!(unwrap_success_text(result), "format via home sid_root");

        fs::remove_dir_all(config_root.as_str()).unwrap();
        fs::remove_dir_all(workspace_root.as_str()).unwrap();
    }

    #[test]
    fn from_config_exposes_builtin_bash_and_edit_tools() {
        let root = temp_config_root("agent");
        write_builtin_config(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tool_names = runtime
            .block_on(agent.tools())
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>();
        assert_eq!(
            tool_names,
            vec![
                "bash".to_string(),
                "str_replace_based_edit_tool".to_string(),
            ]
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn builtin_tools_without_manifests_use_claudius_union_params() {
        let root = temp_config_root("agent");
        write_builtin_config_without_manifests(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tools = runtime.block_on(agent.tools());
        let params = tools
            .iter()
            .map(|tool| (tool.name(), tool.to_param()))
            .collect::<Vec<_>>();
        assert_eq!(
            params,
            vec![
                (
                    "bash".to_string(),
                    ToolUnionParam::Bash20250124(ToolBash20250124::new()),
                ),
                (
                    "str_replace_based_edit_tool".to_string(),
                    ToolUnionParam::TextEditor20250728(ToolTextEditor20250728::new()),
                ),
            ]
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_workspace_falls_back_without_config_files() {
        let root = unique_temp_dir("agent");
        fs::create_dir_all(root.as_str()).unwrap();

        let fallback = ChatConfig::new()
            .with_system_prompt("fallback system".to_string())
            .with_max_tokens(2048);
        let agent = SidAgent::from_workspace(&root, fallback).unwrap();

        assert_eq!(agent.id(), DEFAULT_AGENT_ID);
        assert_eq!(agent.config.system_prompt_text(), Some("fallback system"));
        assert_eq!(agent.config.max_tokens(), 2048);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        assert!(runtime.block_on(agent.tools()).is_empty());
        assert!(!agent.requires_confirmation());

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_workspace_compactor_falls_back_to_builtin_prompt() {
        let root = unique_temp_dir("compactor");
        fs::create_dir_all(root.as_str()).unwrap();

        let agent = SidAgent::from_workspace_compactor_with_config_root(
            &root,
            &root,
            ChatConfig::new().with_system_prompt("ignored".to_string()),
        )
        .unwrap();

        assert_eq!(agent.id(), DEFAULT_COMPACTOR_AGENT_ID);
        assert_eq!(
            agent.config.system_prompt_text(),
            Some(DEFAULT_COMPACTOR_SYSTEM_PROMPT)
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        assert!(runtime.block_on(agent.tools()).is_empty());

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn configured_compactor_uses_reserved_agent_even_if_disabled() {
        let root = temp_config_root("compact-agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            r#"
build_ENABLED="YES"
build_TOOLS='bash'
compact_ENABLED="NO"
compact_TOOLS='bash'
"#,
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(
            root.join("agents/compact.md").as_str(),
            "# Compact\n\nCustom compact prompt.\n",
        )
        .unwrap();

        let agent = SidAgent::from_workspace_compactor_with_config_root(
            &root,
            &root,
            ChatConfig::new().with_system_prompt("ignored".to_string()),
        )
        .unwrap();
        assert_eq!(agent.id(), DEFAULT_COMPACTOR_AGENT_ID);
        assert!(
            agent
                .config
                .system_prompt_text()
                .unwrap()
                .contains("Custom compact prompt.")
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        assert!(runtime.block_on(agent.tools()).is_empty());

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn compacted_session_exposes_ask_an_expert_tool() {
        let root = temp_config_root("compact-memory");
        write_builtin_config(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let parent = SidSession::create_in(sessions_root.clone()).unwrap();
        let child = SidSession::create_compacted_in(
            sessions_root.clone(),
            CompactionProvenance {
                session_id: parent.id().to_string(),
                session_dir: parent.root().to_string_lossy().into_owned(),
                expert: CompactionExpertConfig {
                    agent_id: Some(DEFAULT_COMPACTOR_AGENT_ID.to_string()),
                    model: "claude-sonnet-4-5".to_string(),
                    system_prompt: Some("Summarize carefully.".to_string()),
                    memory_expert_prompt: Some("Answer from memory only.".to_string()),
                },
            },
        )
        .unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone())
            .unwrap()
            .with_session(Arc::new(child));

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tool_names = runtime
            .block_on(agent.tools())
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>();
        assert_eq!(
            tool_names,
            vec![
                "bash".to_string(),
                "str_replace_based_edit_tool".to_string(),
                ASK_AN_EXPERT_TOOL_NAME.to_string(),
            ]
        );

        fs::remove_dir_all(root.as_str()).unwrap();
        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn from_workspace_without_config_appends_agents_md_to_system_prompt() {
        let root = unique_temp_dir("agent");
        fs::create_dir_all(root.as_str()).unwrap();
        fs::write(root.join("AGENTS.md").as_str(), "Use workspace rules.\n").unwrap();

        let fallback = ChatConfig::new()
            .with_system_prompt("fallback system".to_string())
            .with_max_tokens(2048);
        let agent = SidAgent::from_workspace(&root, fallback).unwrap();
        let prompt = agent.config.system_prompt_text().unwrap();

        assert!(prompt.starts_with("fallback system"));
        assert!(prompt.contains("# User instructions from AGENTS.md"));
        assert!(prompt.contains("Use workspace rules."));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn agent_create_request_forwards_chat_config_request_fields() {
        let root = unique_temp_dir("agent-request");
        fs::create_dir_all(root.as_str()).unwrap();

        let mut config = ChatConfig::new();
        config.template.metadata = Some(Metadata::with_user_id("opaque-user"));
        config.template.tool_choice = Some(ToolChoice::none());
        let agent = SidAgent::new(config, root.clone());

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let req = runtime.block_on(agent.create_request(123, vec![], false));

        assert!(req.cache_control.is_some());
        assert_eq!(req.metadata, Some(Metadata::with_user_id("opaque-user")));
        assert_eq!(req.tool_choice, Some(ToolChoice::none()));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn manual_agents_require_confirmation() {
        let root = temp_config_root("agent");
        write_sample_config(&root);

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "plan", root.clone()).unwrap();

        assert_eq!(agent.id(), "plan");
        assert!(agent.requires_confirmation());

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn parse_tool_confirmation_accepts_yes() {
        assert_eq!(parse_tool_confirmation("yes"), Some(true));
        assert_eq!(parse_tool_confirmation("YES"), Some(true));
        assert_eq!(parse_tool_confirmation("y"), Some(true));
        assert_eq!(parse_tool_confirmation("Y"), Some(true));
        assert_eq!(parse_tool_confirmation("  yes  \n"), Some(true));
    }

    #[test]
    fn parse_tool_confirmation_accepts_no() {
        assert_eq!(parse_tool_confirmation("no"), Some(false));
        assert_eq!(parse_tool_confirmation("NO"), Some(false));
        assert_eq!(parse_tool_confirmation("n"), Some(false));
        assert_eq!(parse_tool_confirmation("N"), Some(false));
        assert_eq!(parse_tool_confirmation("  no  \n"), Some(false));
    }

    #[test]
    fn parse_tool_confirmation_rejects_other() {
        assert_eq!(parse_tool_confirmation(""), None);
        assert_eq!(parse_tool_confirmation("maybe"), None);
        assert_eq!(parse_tool_confirmation("yep"), None);
        assert_eq!(parse_tool_confirmation("nope"), None);
    }

    #[test]
    fn manual_tool_confirmation_reads_from_renderer() {
        let input = json!({ "command": "pwd" });
        let mut renderer = ScriptedRenderer::new([
            OperatorLine::Line("maybe".to_string()),
            OperatorLine::Line("yes".to_string()),
        ]);

        assert_eq!(
            confirm_manual_tool_call("bash", &input, None, Some(&mut renderer)).unwrap(),
            ManualToolConfirmation::Allow
        );
        assert_eq!(renderer.prompts.len(), 2);
        assert!(renderer.prompts[0].contains("Tool 'bash' is MANUAL."));
    }

    #[test]
    fn manual_tool_confirmation_interrupted_renderer_cancels() {
        let input = json!({ "command": "pwd" });
        let mut renderer = ScriptedRenderer::new([OperatorLine::Interrupted]);

        assert_eq!(
            confirm_manual_tool_call("bash", &input, None, Some(&mut renderer)).unwrap(),
            ManualToolConfirmation::Cancel
        );
        assert_eq!(renderer.prompts.len(), 1);
    }

    #[test]
    fn interrupted_manual_tool_cancels_remaining_tool_calls() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash edit'\n",
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            "bash_ENABLED=MANUAL\nedit_ENABLED=MANUAL\n",
        )
        .unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        write_tool_runtime(&root, "edit", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let mut agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let client = Anthropic::new(Some("test-api-key".to_string())).unwrap();
        let response = Message::new(
            "msg_test".to_string(),
            vec![
                ToolUseBlock::new(
                    "toolu_bash",
                    "bash",
                    json!({
                        "command": "pwd"
                    }),
                )
                .into(),
                ToolUseBlock::new(
                    "toolu_edit",
                    "str_replace_based_edit_tool",
                    json!({
                        "command": "view",
                        "path": "/src/lib.rs"
                    }),
                )
                .into(),
            ],
            Model::Known(KnownModel::ClaudeHaiku45),
            Usage::new(0, 0),
        )
        .with_stop_reason(StopReason::ToolUse);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let mut renderer = ScriptedRenderer::new([OperatorLine::Interrupted]);
        let context = AgentStreamContext::root("build");

        let result = runtime.block_on(agent.handle_tool_use_streaming(
            &client,
            &response,
            &mut renderer,
            &context,
        ));
        let ControlFlow::Continue(blocks) = result else {
            panic!("expected tool result blocks, got {result:?}");
        };

        assert_eq!(renderer.prompts.len(), 1);
        assert_eq!(blocks.len(), 2);
        let first = blocks[0].as_tool_result().unwrap().clone();
        let second = blocks[1].as_tool_result().unwrap().clone();
        assert_eq!(first.tool_use_id, "toolu_bash");
        assert_eq!(second.tool_use_id, "toolu_edit");
        assert_eq!(first.is_error, Some(true));
        assert_eq!(second.is_error, Some(true));
        assert_eq!(tool_block_text(first), USER_CANCELLED_ACTION);
        assert_eq!(tool_block_text(second), USER_CANCELLED_ACTION);

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn top_up_cancelled_tool_results_appends_after_partial_assistant_tool_use() {
        let mut messages = vec![
            MessageParam::user("Do the work."),
            MessageParam::new(
                MessageParamContent::Array(vec![
                    ContentBlock::Text(TextBlock::new("I will check.".to_string())),
                    ToolUseBlock::new("toolu_partial", "bash", json!({"command": "pwd"})).into(),
                ]),
                MessageRole::Assistant,
            ),
        ];

        assert_eq!(top_up_cancelled_tool_results(&mut messages), 1);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2].role, MessageRole::User);
        let MessageParamContent::Array(blocks) = &messages[2].content else {
            panic!("expected synthesized tool result blocks");
        };
        assert_eq!(blocks.len(), 1);
        let result = blocks[0].as_tool_result().unwrap().clone();
        assert_eq!(result.tool_use_id, "toolu_partial");
        assert_eq!(result.is_error, Some(true));
        assert_eq!(tool_block_text(result), USER_CANCELLED_ACTION);
    }

    #[test]
    fn top_up_cancelled_tool_results_inserts_before_stale_user_message() {
        let mut messages = vec![
            MessageParam::new(
                MessageParamContent::Array(vec![
                    ToolUseBlock::new("toolu_stale", "bash", json!({"command": "pwd"})).into(),
                ]),
                MessageRole::Assistant,
            ),
            MessageParam::user("Try again."),
        ];

        assert_eq!(top_up_cancelled_tool_results(&mut messages), 1);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].role, MessageRole::User);
        assert_eq!(messages[2], MessageParam::user("Try again."));
        let MessageParamContent::Array(blocks) = &messages[1].content else {
            panic!("expected synthesized tool result blocks");
        };
        let result = blocks[0].as_tool_result().unwrap().clone();
        assert_eq!(result.tool_use_id, "toolu_stale");
        assert_eq!(result.is_error, Some(true));
        assert_eq!(tool_block_text(result), USER_CANCELLED_ACTION);
    }

    #[test]
    fn top_up_cancelled_tool_results_preserves_existing_results_and_cancels_missing() {
        let existing =
            ToolResultBlock::new("toolu_done".to_string()).with_string_content("done".to_string());
        let mut messages = vec![
            MessageParam::new(
                MessageParamContent::Array(vec![
                    ToolUseBlock::new("toolu_missing", "bash", json!({"command": "pwd"})).into(),
                    ToolUseBlock::new("toolu_done", "bash", json!({"command": "date"})).into(),
                ]),
                MessageRole::Assistant,
            ),
            MessageParam::new(
                MessageParamContent::Array(vec![
                    ContentBlock::Text(TextBlock::new("next request".to_string())),
                    existing.into(),
                ]),
                MessageRole::User,
            ),
        ];

        assert_eq!(top_up_cancelled_tool_results(&mut messages), 1);

        assert_eq!(messages.len(), 2);
        let MessageParamContent::Array(blocks) = &messages[1].content else {
            panic!("expected reordered user content blocks");
        };
        assert_eq!(blocks.len(), 3);
        let first = blocks[0].as_tool_result().unwrap().clone();
        let second = blocks[1].as_tool_result().unwrap().clone();
        assert_eq!(first.tool_use_id, "toolu_missing");
        assert_eq!(first.is_error, Some(true));
        assert_eq!(tool_block_text(first), USER_CANCELLED_ACTION);
        assert_eq!(second.tool_use_id, "toolu_done");
        assert_eq!(tool_block_text(second), "done");
        assert_eq!(
            blocks[2].as_text().map(|block| block.text.as_str()),
            Some("next request")
        );
    }

    #[test]
    fn token_usage_totals_accumulate_input_cached_input_and_output() {
        let mut totals = TokenUsageTotals::default();
        totals.add(Usage::new(10, 4).with_cache_read_input_tokens(6), None);
        totals.add(Usage::new(20, 8).with_cache_read_input_tokens(7), None);

        assert_eq!(
            totals,
            TokenUsageTotals {
                input: 30,
                cached_input: 13,
                output: 12,
                cost_micro_cents: 0,
            }
        );
    }

    #[test]
    fn token_usage_totals_treat_negative_counts_as_zero() {
        let mut totals = TokenUsageTotals::default();
        totals.add(Usage::new(-10, -4).with_cache_read_input_tokens(-6), None);

        assert_eq!(totals, TokenUsageTotals::default());
    }

    #[test]
    fn token_usage_totals_accumulate_cost_with_rates() {
        let rates = TokenRates {
            input: 300,
            output: 1500,
            cache_creation: 375,
            cache_read: 30,
        };
        let mut totals = TokenUsageTotals::default();
        totals.add(Usage::new(100, 50), Some(rates));

        // 100 input * 300 + 50 output * 1500 = 30_000 + 75_000 = 105_000
        assert_eq!(totals.cost_micro_cents, 105_000);
    }

    #[test]
    fn token_usage_totals_no_cost_without_rates() {
        let mut totals = TokenUsageTotals::default();
        totals.add(Usage::new(100, 50), None);

        assert_eq!(totals.cost_micro_cents, 0);
    }

    #[test]
    fn format_cost_renders_dollars_and_fractional_cents() {
        assert_eq!(format_cost(0), "$0.0000");
        assert_eq!(format_cost(100_000_000), "$1.0000");
        assert_eq!(format_cost(50_000_000), "$0.5000");
        assert_eq!(format_cost(1_230_000), "$0.0123");
        assert_eq!(format_cost(250_000_000), "$2.5000");
    }

    #[test]
    fn agents_md_instructions_are_appended_to_system_prompt() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(root.join("AGENTS.md").as_str(), "Use local rules.\n").unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let mut messages = vec![MessageParam::user("Do the work.")];
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(agent.inject_user_instructions_for_turn(&mut messages))
            .unwrap();

        assert_eq!(messages[0], MessageParam::user("Do the work."));
        let prompt = agent.config.system_prompt_text().unwrap();
        assert!(prompt.contains("# User instructions from AGENTS.md"));
        assert!(prompt.contains("Use local rules."));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn agents_md_path_concatenates_existing_files_in_order() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::create_dir_all(root.join("local").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_AGENTS_MD_PATH='global.md:missing.md:local/AGENTS.md'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(root.join("global.md").as_str(), "Global rules.\n").unwrap();
        fs::write(root.join("local/AGENTS.md").as_str(), "Local rules.\n").unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let prompt = agent.config.system_prompt_text().unwrap();
        let global = prompt.find("Global rules.").unwrap();
        let local = prompt.find("Local rules.").unwrap();
        assert!(global < local);
        assert!(!prompt.contains("missing.md"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn agents_md_injection_can_be_disabled() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_AGENTS_MD=NO\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(root.join("AGENTS.md").as_str(), "Use local rules.\n").unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let mut messages = vec![MessageParam::user("Do the work.")];
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(agent.inject_user_instructions_for_turn(&mut messages))
            .unwrap();

        assert_eq!(messages[0], MessageParam::user("Do the work."));
        assert!(
            !agent
                .config
                .system_prompt_text()
                .unwrap()
                .contains("Use local rules.")
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn user_instruction_hook_runs_with_rc_overlay() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_USER_INSTRUCTIONS_HOOK=context\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(root.join("AGENTS.md").as_str(), "Use local rules.\n").unwrap();
        write_agent_hook(
            &root,
            "context",
            r#"#!/bin/sh
set -eu
PREFIX=${RCVAR_ARGV0:?missing RCVAR_ARGV0}
case "${1:-}" in
rcvar)
    printf '%s\n' \
        "${PREFIX}_WORKSPACE_ROOT" \
        "${PREFIX}_CONFIG_ROOT" \
        "${PREFIX}_AGENT_ID" \
        "${PREFIX}_HOOK_NAME" \
        "${PREFIX}_AGENTS_MD_PATH" \
        "${PREFIX}_SCRATCH_DIR" \
        "${PREFIX}_TEMP_DIR" \
        "${PREFIX}_TMPDIR" \
        "${PREFIX}_RC_CONF_PATH" \
        "${PREFIX}_RC_D_PATH"
    ;;
run)
    printf 'workspace=%s\n' "$(printenv "${PREFIX}_WORKSPACE_ROOT")"
    printf 'config=%s\n' "$(printenv "${PREFIX}_CONFIG_ROOT")"
    printf 'agent=%s\n' "$(printenv "${PREFIX}_AGENT_ID")"
    printf 'hook=%s\n' "$(printenv "${PREFIX}_HOOK_NAME")"
    printf 'agents_md=%s\n' "$(printenv "${PREFIX}_AGENTS_MD_PATH")"
    printf 'scratch=%s\n' "$(printenv "${PREFIX}_SCRATCH_DIR")"
    printf 'temp=%s\n' "$(printenv "${PREFIX}_TEMP_DIR")"
    printf 'tmpdir=%s\n' "$(printenv "${PREFIX}_TMPDIR")"
    printf 'rc=%s\n' "$(printenv "${PREFIX}_RC_CONF_PATH")"
    printf 'rcd=%s\n' "$(printenv "${PREFIX}_RC_D_PATH")"
    ;;
*)
    exit 129
    ;;
esac
"#,
        );

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let mut messages = vec![MessageParam::user("Do the work.")];
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(agent.inject_user_instructions_for_turn(&mut messages))
            .unwrap();
        let injected = injected_user_instruction_text(&messages[0]);

        assert!(injected.contains("# User instructions from hook context"));
        assert!(!injected.contains("Use local rules."));
        assert!(injected.contains(&format!("workspace={}", root.as_str())));
        assert!(injected.contains(&format!("config={}", root.as_str())));
        assert!(injected.contains("agent=build"));
        assert!(injected.contains("hook=context"));
        assert!(injected.contains(&format!("agents_md={}", root.join("AGENTS.md").as_str())));
        assert!(injected.contains("scratch=/"));
        assert!(injected.contains("temp=/"));
        assert!(injected.contains("tmpdir=/"));
        assert!(injected.contains(root.join("agents.conf").as_str()));
        assert!(injected.contains(&format!("rcd={}", root.join("agents").as_str())));
        assert!(
            agent
                .config
                .system_prompt_text()
                .unwrap()
                .contains("Use local rules.")
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn user_instruction_hook_runs_from_home_config_root() {
        if !seatbelt::sandbox_available() {
            return;
        }

        let home = match std::env::var("HOME") {
            Ok(home) if !home.is_empty() => home,
            _ => return,
        };
        let config_leaf = std::path::Path::new(unique_temp_dir("home-config-root").as_str())
            .file_name()
            .and_then(|name| name.to_str())
            .expect("temp dir should have a utf-8 leaf")
            .to_string();
        let config_root = Path::new(&home)
            .join(".sid-isnt-done-tests")
            .join(config_leaf)
            .into_owned();
        let workspace_root = unique_temp_dir("workspace-root");

        fs::create_dir_all(config_root.join("agents").as_str()).unwrap();
        fs::write(
            config_root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_USER_INSTRUCTIONS_HOOK=context\n",
        )
        .unwrap();
        fs::write(
            config_root.join("tools.conf").as_str(),
            "bash_ENABLED=YES\n",
        )
        .unwrap();
        write_tool_runtime(&config_root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(config_root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(
            config_root.join("agents/context").as_str(),
            "#!/bin/sh\nexec \"$(dirname \"$0\")/context.impl\" \"$@\"\n",
        )
        .unwrap();
        make_executable(&config_root.join("agents/context"));
        fs::write(
            config_root.join("agents/context.impl").as_str(),
            r#"#!/bin/sh
case "${1:-}" in
rcvar)
    ;;
run)
    printf 'home hook ok\n'
    ;;
*)
    exit 129
    ;;
esac
"#,
        )
        .unwrap();
        make_executable(&config_root.join("agents/context.impl"));
        fs::create_dir_all(workspace_root.as_str()).unwrap();

        let config = Config::load(&config_root).unwrap();
        let agent = SidAgent::from_config(&config, "build", workspace_root.clone()).unwrap();
        let mut messages = vec![MessageParam::user("Do the work.")];
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(agent.inject_user_instructions_for_turn(&mut messages))
            .unwrap();

        let injected = injected_user_instruction_text(&messages[0]);
        assert!(injected.contains("# User instructions from hook context"));
        assert!(injected.contains("home hook ok"));

        fs::remove_dir_all(config_root.as_str()).unwrap();
        fs::remove_dir_all(workspace_root.as_str()).unwrap();
    }

    #[test]
    fn user_instruction_hook_receives_user_message_and_skill_manifest() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_SKILLS='rust python PATH'\nbuild_USER_INSTRUCTIONS_HOOK='context'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        write_agent_hook(
            &root,
            "context",
            r#"#!/bin/sh
set -eu
PREFIX=${RCVAR_ARGV0:?missing RCVAR_ARGV0}
case "${1:-}" in
rcvar)
    printf '%s\n' \
        "${PREFIX}_USER_MESSAGE_FILE" \
        "${PREFIX}_SKILLS_MANIFEST_FILE"
    ;;
run)
    USER_MESSAGE_FILE=$(printenv "${PREFIX}_USER_MESSAGE_FILE")
    SKILLS_MANIFEST_FILE=$(printenv "${PREFIX}_SKILLS_MANIFEST_FILE")
    printf 'message=%s\n' "$(cat "$USER_MESSAGE_FILE")"
    printf 'manifest='
    cat "$SKILLS_MANIFEST_FILE"
    ;;
*)
    exit 129
    ;;
esac
"#,
        );
        write_skill(&root, "rust", "# Rust Skill\n\nUse Rust idioms.\n");
        write_skill(&root, "python", "# Python Skill\n\nUse Python idioms.\n");
        write_skill(&root, "PATH", "# Path Skill\n\nShould not be injected.\n");

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let original = "Use $rust and $rust; ignore $python-extra and $PATH.";
        let mut messages = vec![MessageParam::user(original)];
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(agent.inject_user_instructions_for_turn(&mut messages))
            .unwrap();

        let injected = injected_user_instruction_text(&messages[0]);
        assert!(injected.contains("message=Use $rust and $rust; ignore $python-extra and $PATH."));
        assert!(injected.contains("rust\t/skills/rust/SKILL.md\t"));
        assert!(injected.contains("python\t/skills/python/SKILL.md\t"));
        assert!(!injected.contains("PATH\t/skills/PATH/SKILL.md\t"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn manual_external_tool_stores_enabled_state() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='fmt'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=MANUAL\n").unwrap();
        write_tool_contract(&root, "fmt", "Format files.", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let tool_config = config.tools.get("fmt").unwrap();
        assert_eq!(tool_config.enabled, SwitchPosition::Manual);

        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, "fmt").unwrap();
        let tool = ExternalTool::from_config("fmt".to_string(), canonical_id, tool_config);
        assert_eq!(tool.enabled, SwitchPosition::Manual);

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn manual_builtin_binding_stores_enabled_state() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash edit'\n",
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            "bash_ENABLED=MANUAL\nedit_ENABLED=MANUAL\n",
        )
        .unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        write_tool_runtime(&root, "edit", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        assert_eq!(
            agent.builtin_bindings.bash.as_ref().unwrap().enabled,
            SwitchPosition::Manual,
        );
        assert_eq!(
            agent.builtin_bindings.edit.as_ref().unwrap().enabled,
            SwitchPosition::Manual,
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_mounts_skills_as_read_only_files() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/rust").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/python").as_str()).unwrap();

        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_SKILLS='rust python'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(
            root.join("agents/build.md").as_str(),
            "# Build\n\nYou are a builder.\n",
        )
        .unwrap();
        fs::write(
            root.join("skills/rust/SKILL.md").as_str(),
            "# Rust Skill\n\nWrite safe Rust.\n",
        )
        .unwrap();
        fs::write(
            root.join("skills/python/SKILL.md").as_str(),
            "# Python Skill\n\nWrite clean Python.\n",
        )
        .unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let rust_content = runtime
            .block_on(agent.filesystem.view("/skills/rust/SKILL.md", None))
            .unwrap();
        assert!(
            rust_content.contains("Write safe Rust."),
            "skill content should be viewable: {rust_content}"
        );
        let python_content = runtime
            .block_on(agent.filesystem.view("/skills/python/SKILL.md", None))
            .unwrap();
        assert!(
            python_content.contains("Write clean Python."),
            "skill content should be viewable: {python_content}"
        );

        let replace_err = runtime
            .block_on(agent.filesystem.str_replace(
                "/skills/rust/SKILL.md",
                "Write safe Rust.",
                "Replaced.",
            ))
            .unwrap_err();
        assert_eq!(
            replace_err.kind(),
            std::io::ErrorKind::PermissionDenied,
            "skill files should be read-only: {replace_err}"
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_mounts_all_skills_with_wildcard() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/docs").as_str()).unwrap();

        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_SKILLS='*'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(root.join("skills/docs/SKILL.md").as_str(), "# Docs Skill\n").unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let docs_content = runtime
            .block_on(agent.filesystem.view("/skills/docs/SKILL.md", None))
            .unwrap();
        assert!(
            docs_content.contains("# Docs Skill"),
            "wildcard should mount all skills: {docs_content}"
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_no_skills_mounts_workspace_only() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();

        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let content = runtime
            .block_on(agent.filesystem.view("/agents/build.md", None))
            .unwrap();
        assert!(
            content.contains("# Build"),
            "workspace files should be viewable: {content}"
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_rejects_unknown_skill_references() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/rust").as_str()).unwrap();

        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_SKILLS='rust missing'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(root.join("skills/rust/SKILL.md").as_str(), "# Rust Skill\n").unwrap();

        let config = Config::load(&root).unwrap();
        let err = SidAgent::from_config(&config, "build", root.clone())
            .err()
            .expect("unknown skills should fail")
            .to_string();
        assert!(err.contains("unknown_skill"), "error: {err}");
        assert!(err.contains("missing"), "error: {err}");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn explicit_default_agent_is_preferred() {
        let root = temp_config_root("agent");
        write_sample_config(&root);
        fs::write(
            root.join("agents.conf").as_str(),
            concat!(
                "DEFAULT_AGENT=plan\n",
                "ROLE='principal engineer'\n",
                "build_ENABLED=\"YES\"\n",
                "plan_ENABLED=\"MANUAL\"\n",
                "evil_ENABLED=\"NO\"\n",
                "build_TOOLS='format shell'\n",
                "plan_MODEL=claude-sonnet-4-5\n",
                "plan_SYSTEM=\"You are ${ROLE}\"\n",
                "plan_MAX_TOKENS=8192\n",
            ),
        )
        .unwrap();

        let config = Config::load(&root).unwrap();
        let agent_id = default_agent_id(&config).unwrap();
        assert_eq!(agent_id, "plan");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn invalid_default_agent_is_rejected_at_config_load() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "DEFAULT_AGENT=nonexistent\nbuild_ENABLED=YES\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "").unwrap();
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let err = Config::load(&root)
            .expect_err("invalid DEFAULT_AGENT should fail")
            .to_string();
        assert!(err.contains("invalid_default_agent"), "error: {err}");
        assert!(err.contains("nonexistent"), "error: {err}");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn skills_are_advertised_in_system_prompt() {
        let root = temp_config_root("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/rust").as_str()).unwrap();
        fs::create_dir_all(root.join("skills/python").as_str()).unwrap();

        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='bash'\nbuild_SKILLS='rust python'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "bash_ENABLED=YES\n").unwrap();
        write_tool_runtime(&root, "bash", "#!/bin/sh\nexit 0\n");
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        fs::write(root.join("skills/rust/SKILL.md").as_str(), "# Rust Skill\n").unwrap();
        fs::write(
            root.join("skills/python/SKILL.md").as_str(),
            "# Python Skill\n",
        )
        .unwrap();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();

        let prompt = agent.config.system_prompt_text().unwrap();
        assert!(
            prompt.contains("/skills/rust/SKILL.md"),
            "system prompt should list rust skill: {prompt}"
        );
        assert!(
            prompt.contains("/skills/python/SKILL.md"),
            "system prompt should list python skill: {prompt}"
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_rejects_unknown_tools() {
        let root = unique_temp_dir("agent");
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            "build_ENABLED=YES\nbuild_TOOLS='missing'\n",
        )
        .unwrap();
        fs::write(root.join("tools.conf").as_str(), "fmt_ENABLED=YES\n").unwrap();
        write_tool_contract(
            &root,
            "fmt",
            "Format files in the workspace.",
            "#!/bin/sh\nexit 0\n",
        );
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let err = SidAgent::from_config(&config, "build", root.clone())
            .err()
            .expect("unknown tools should fail")
            .to_string();
        assert!(err.contains("unknown_tool"));
        assert!(err.contains("missing"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn from_config_rejects_invalid_tool_names() {
        let root = unique_temp_dir("agent");
        let invalid_name = "a".repeat(65);
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            format!("build_ENABLED=YES\nbuild_TOOLS='{invalid_name}'\n"),
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            format!("fmt_ENABLED=YES\n{invalid_name}_ALIASES=fmt\n"),
        )
        .unwrap();
        write_tool_contract(
            &root,
            "fmt",
            "Format files in the workspace.",
            "#!/bin/sh\nexit 0\n",
        );
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();

        let config = Config::load(&root).unwrap();
        let err = SidAgent::from_config(&config, "build", root.clone())
            .err()
            .expect("invalid tool names should fail")
            .to_string();
        assert!(err.contains("invalid_tool_name"));
        assert!(err.contains(&invalid_name));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn builtin_bash_persists_shell_state_across_calls() {
        let root = temp_config_root("agent");
        write_builtin_config(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        let canonical_agents = Path::try_from(
            fs::canonicalize(root.join("agents").as_str())
                .expect("canonicalize agents directory should succeed"),
        )
        .expect("canonical agents path should be valid UTF-8")
        .into_owned();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(agent.bash("export FOO=bar\nf() { printf hi; }\ncd agents", true))
            .unwrap();
        let result = runtime
            .block_on(agent.bash("printf '%s:%s:%s' \"$FOO\" \"$(f)\" \"$PWD\"", false))
            .unwrap();
        assert_eq!(
            result.trim_end(),
            format!("bar:hi:{}", canonical_agents.as_str()),
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn builtin_bash_restart_resets_shell_state() {
        let root = temp_config_root("agent");
        write_builtin_config(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        let canonical_root = Path::try_from(
            fs::canonicalize(root.as_str()).expect("canonicalize root directory should succeed"),
        )
        .expect("canonical root path should be valid UTF-8")
        .into_owned();
        let canonical_agents = Path::try_from(
            fs::canonicalize(root.join("agents").as_str())
                .expect("canonicalize agents directory should succeed"),
        )
        .expect("canonical agents path should be valid UTF-8")
        .into_owned();

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(agent.bash("export FOO=bar\ncd agents", true))
            .unwrap();
        let persisted = runtime
            .block_on(agent.bash("printf '%s:%s' \"$FOO\" \"$PWD\"", false))
            .unwrap();
        assert_eq!(
            persisted.trim_end(),
            format!("bar:{}", canonical_agents.as_str()),
        );

        let restarted = runtime
            .block_on(agent.bash("printf '%s:%s' \"${FOO-unset}\" \"$PWD\"", true))
            .unwrap();
        assert_eq!(
            restarted.trim_end(),
            format!("unset:{}", canonical_root.as_str())
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn builtin_bash_tool_restart_resets_shell_state() {
        let root = temp_config_root("agent");
        write_builtin_config(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        let canonical_root = Path::try_from(
            fs::canonicalize(root.as_str()).expect("canonicalize root directory should succeed"),
        )
        .expect("canonical root path should be valid UTF-8")
        .into_owned();
        let canonical_agents = Path::try_from(
            fs::canonicalize(root.join("agents").as_str())
                .expect("canonicalize agents directory should succeed"),
        )
        .expect("canonical agents path should be valid UTF-8")
        .into_owned();

        let config = Config::load(&root).unwrap();
        let mut agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let client = Anthropic::new(Some("test-api-key".to_string())).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let initialized = invoke_bash_tool(
            &runtime,
            &client,
            &mut agent,
            "toolu_bash_1",
            json!({
                "command": "export FOO=bar\ncd agents",
                "restart": true
            }),
        );
        assert_eq!(
            unwrap_success_block(initialized),
            ToolResultBlock {
                tool_use_id: "toolu_bash_1".to_string(),
                cache_control: None,
                content: Some(ToolResultBlockContent::String("success\n".to_string())),
                is_error: None,
            }
        );

        let persisted = invoke_bash_tool(
            &runtime,
            &client,
            &mut agent,
            "toolu_bash_2",
            json!({
                "command": "printf '%s:%s' \"$FOO\" \"$PWD\"",
                "restart": false
            }),
        );
        assert_eq!(
            unwrap_success_block(persisted),
            ToolResultBlock {
                tool_use_id: "toolu_bash_2".to_string(),
                cache_control: None,
                content: Some(ToolResultBlockContent::String(format!(
                    "bar:{}",
                    canonical_agents.as_str()
                ))),
                is_error: None,
            }
        );

        let restarted = invoke_bash_tool(
            &runtime,
            &client,
            &mut agent,
            "toolu_bash_3",
            json!({
                "command": "printf '%s:%s' \"${FOO-unset}\" \"$PWD\"",
                "restart": true
            }),
        );
        assert_eq!(
            unwrap_success_block(restarted),
            ToolResultBlock {
                tool_use_id: "toolu_bash_3".to_string(),
                cache_control: None,
                content: Some(ToolResultBlockContent::String(format!(
                    "unset:{}",
                    canonical_root.as_str()
                ))),
                is_error: None,
            }
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn resumed_session_restores_bash_state_snapshot() {
        let root = temp_config_root("agent");
        write_builtin_config(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        let canonical_agents = Path::try_from(
            fs::canonicalize(root.join("agents").as_str())
                .expect("canonicalize agents directory should succeed"),
        )
        .expect("canonical agents path should be valid UTF-8")
        .into_owned();
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());

        let config = Config::load(&root).unwrap();
        let sid_session = Arc::new(SidSession::create_in(sessions_root.clone()).unwrap());
        let agent = SidAgent::from_config(&config, "build", root.clone())
            .unwrap()
            .with_session(sid_session.clone());
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime
            .block_on(agent.bash(
                "export FOO=bar\nf() { printf hi; }\nalias ll='printf alias'\nset -o nounset\ncd agents",
                true,
            ))
            .unwrap();

        let resumed =
            Arc::new(SidSession::resume_in(sessions_root.clone(), sid_session.id()).unwrap());
        let resumed_agent = SidAgent::from_config(&config, "build", root.clone())
            .unwrap()
            .with_session(resumed);
        let restored = runtime
            .block_on(resumed_agent.bash(
                "if shopt -qo nounset; then nounset=on; else nounset=off; fi\nprintf '%s:%s:%s:%s:%s' \"$FOO\" \"$(f)\" \"$(ll)\" \"$PWD\" \"$nounset\"",
                false,
            ))
            .unwrap();

        assert_eq!(
            restored.trim_end(),
            format!("bar:hi:alias:{}:on", canonical_agents.as_str()),
        );

        fs::remove_dir_all(root.as_str()).unwrap();
        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn resumed_session_restart_discards_saved_bash_state() {
        let root = temp_config_root("agent");
        write_builtin_config(&root, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        let canonical_root = Path::try_from(
            fs::canonicalize(root.as_str()).expect("canonicalize root directory should succeed"),
        )
        .expect("canonical root path should be valid UTF-8")
        .into_owned();
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());

        let config = Config::load(&root).unwrap();
        let sid_session = Arc::new(SidSession::create_in(sessions_root.clone()).unwrap());
        let agent = SidAgent::from_config(&config, "build", root.clone())
            .unwrap()
            .with_session(sid_session.clone());
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime
            .block_on(agent.bash("export FOO=bar\ncd agents", true))
            .unwrap();

        let resumed =
            Arc::new(SidSession::resume_in(sessions_root.clone(), sid_session.id()).unwrap());
        let resumed_agent = SidAgent::from_config(&config, "build", root.clone())
            .unwrap()
            .with_session(resumed.clone());
        let restarted = runtime
            .block_on(resumed_agent.bash("printf '%s:%s' \"${FOO-unset}\" \"$PWD\"", true))
            .unwrap();
        assert_eq!(
            restarted.trim_end(),
            format!("unset:{}", canonical_root.as_str())
        );

        let resumed_again =
            Arc::new(SidSession::resume_in(sessions_root.clone(), sid_session.id()).unwrap());
        let resumed_again_agent = SidAgent::from_config(&config, "build", root.clone())
            .unwrap()
            .with_session(resumed_again);
        let restored = runtime
            .block_on(resumed_again_agent.bash("printf '%s:%s' \"${FOO-unset}\" \"$PWD\"", false))
            .unwrap();
        assert_eq!(
            restored.trim_end(),
            format!("unset:{}", canonical_root.as_str())
        );

        fs::remove_dir_all(root.as_str()).unwrap();
        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn two_agents_with_different_workspace_roots_have_no_crosstalk() {
        let root_a = temp_config_root("agent-a");
        let root_b = temp_config_root("agent-b");
        write_builtin_config(&root_a, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        write_builtin_config(&root_b, "#!/bin/sh\nexit 0\n", "#!/bin/sh\nexit 0\n");
        let canonical_a = Path::try_from(
            fs::canonicalize(root_a.as_str()).expect("canonicalize root_a should succeed"),
        )
        .expect("canonical root_a path should be valid UTF-8")
        .into_owned();
        let canonical_b = Path::try_from(
            fs::canonicalize(root_b.as_str()).expect("canonicalize root_b should succeed"),
        )
        .expect("canonical root_b path should be valid UTF-8")
        .into_owned();

        let config_a = Config::load(&root_a).unwrap();
        let agent_a = SidAgent::from_config(&config_a, "build", root_a.clone()).unwrap();
        let config_b = Config::load(&root_b).unwrap();
        let agent_b = SidAgent::from_config(&config_b, "build", root_b.clone()).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let pwd_a = runtime.block_on(agent_a.bash("pwd", true)).unwrap();
        let pwd_b = runtime.block_on(agent_b.bash("pwd", true)).unwrap();

        assert_eq!(pwd_a.trim_end(), canonical_a.as_str());
        assert_eq!(pwd_b.trim_end(), canonical_b.as_str());
        assert_ne!(
            canonical_a.as_str(),
            canonical_b.as_str(),
            "the two workspace roots must be distinct"
        );

        fs::remove_dir_all(root_a.as_str()).unwrap();
        fs::remove_dir_all(root_b.as_str()).unwrap();
    }

    #[test]
    fn builtin_text_editor_uses_rc_tool_runtime() {
        let root = temp_config_root("agent");
        write_builtin_config(
            &root,
            "#!/bin/sh\nexit 0\n",
            &capturing_tool_script(
                "edit via rc",
                "edit-request-capture.json",
                "edit-env-capture.json",
            ),
        );

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tool_use = ToolUseBlock::new(
            "toolu_edit_123",
            "str_replace_based_edit_tool",
            json!({
                "command": "str_replace",
                "path": "src/lib.rs",
                "old_str": "old",
                "new_str": "new"
            }),
        );
        let result = runtime.block_on(agent.text_editor(tool_use)).unwrap();
        assert_eq!(result, "edit via rc");

        let request: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("edit-request-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(request["tool"]["id"], json!("edit"));
        assert_eq!(
            request["invocation"]["tool_use_id"],
            json!("toolu_edit_123")
        );
        assert_eq!(
            request["invocation"]["input"],
            json!({
                "command": "str_replace",
                "path": "src/lib.rs",
                "old_str": "old",
                "new_str": "new"
            })
        );

        let env: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("edit-env-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(env["tool_id"], json!("edit"));
        assert_eq!(env["tool_name"], json!("edit"));
        assert_eq!(env["workspace_root"], json!(root.as_str()));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_writes_request_and_returns_success() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            &success_tool_script("Formatted 3 files.", true),
        );

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        assert_eq!(unwrap_success_text(result), "Formatted 3 files.");

        let request: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("request-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(request["protocol_version"], json!(TOOL_PROTOCOL_VERSION));
        assert_eq!(request["tool"]["id"], json!("fmt"));
        assert_eq!(request["invocation"]["tool_use_id"], json!("toolu_123"));
        assert_eq!(
            request["invocation"]["input"],
            json!({ "paths": ["src/lib.rs"] })
        );
        assert_eq!(request["agent"]["id"], json!("build"));
        assert_eq!(request["workspace"]["root"], json!(root.as_str()));
        assert_eq!(request["workspace"]["cwd"], json!(root.as_str()));
        let scratch_dir = request["files"]["scratch_dir"].as_str().unwrap();
        let result_file = request["files"]["result_file"].as_str().unwrap();
        assert!(scratch_dir.starts_with('/'));
        assert!(result_file.starts_with(scratch_dir));
        assert!(result_file.ends_with("/result.json"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_exposes_execution_env_and_ignores_terminal_output() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            &environment_capturing_tool_script("tool result only"),
        );

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        assert_eq!(unwrap_success_text(result), "tool result only");

        let env: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("env-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        assert_eq!(env["protocol"], json!(TOOL_PROTOCOL_VERSION.to_string()));
        assert_eq!(env["workspace_root"], json!(root.as_str()));
        assert_eq!(env["agent_id"], json!("build"));
        assert_eq!(env["tool_id"], json!("fmt"));
        assert_eq!(env["tool_name"], json!("format"));
        assert_eq!(env["rc_d_path"], json!(root.join("tools").as_str()));
        let scratch_dir = env["scratch_dir"].as_str().unwrap();
        assert!(scratch_dir.starts_with('/'));
        assert!(
            !PathBuf::from(scratch_dir).exists(),
            "tool scratch should be cleaned by default"
        );
        assert!(
            env["request_file"]
                .as_str()
                .unwrap()
                .ends_with("/request.json")
        );
        assert!(
            env["result_file"]
                .as_str()
                .unwrap()
                .ends_with("/result.json")
        );

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_uses_ordered_session_tmp_and_journals() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            &capturing_tool_script(
                "tool result only",
                "request-capture.json",
                "env-capture.json",
            ),
        );
        let sessions_root = PathBuf::from(unique_temp_dir("sessions").as_str());
        let sid_session = Arc::new(SidSession::create_in(sessions_root.clone()).unwrap());

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone())
            .unwrap()
            .with_session(sid_session.clone());
        let tool_config = config.tools.get("format").unwrap();
        let exposed_name = exposed_tool_name("build", "format").unwrap();
        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, "format").unwrap();
        let tool = ExternalTool::from_config(exposed_name.clone(), canonical_id, tool_config);
        let tool_use = ToolUseBlock::new(
            "toolu_123",
            exposed_name,
            json!({ "paths": ["src/lib.rs"] }),
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let result = runtime.block_on(invoke_external_tool(&tool, &agent, &tool_use, None));
        assert_eq!(unwrap_success_text(result), "tool result only");

        let request: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("request-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        let env: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("env-capture.json").as_str()).unwrap(),
        )
        .unwrap();
        let scratch_dir = PathBuf::from(request["files"]["scratch_dir"].as_str().unwrap());
        let temp_dir = PathBuf::from(request["files"]["temp_dir"].as_str().unwrap());
        let tool_root = sid_session.root().join("tmp").join("tool-000001");

        assert_eq!(scratch_dir, tool_root);
        assert_eq!(temp_dir, tool_root.join("tmp"));
        assert_eq!(
            request["files"]["result_file"],
            json!(scratch_dir.join("result.json").to_string_lossy())
        );
        assert_eq!(env["scratch_dir"], json!(scratch_dir.to_string_lossy()));
        assert_eq!(env["temp_dir"], json!(temp_dir.to_string_lossy()));
        assert_eq!(env["tmpdir"], json!(temp_dir.to_string_lossy()));
        assert_eq!(env["session_id"], json!(sid_session.id()));
        assert_eq!(
            env["session_dir"],
            json!(sid_session.root().to_string_lossy())
        );
        assert!(
            !tool_root.exists(),
            "session tool scratch should be cleaned by default"
        );

        let events = fs::read_to_string(sid_session.root().join("events.jsonl")).unwrap();
        let events = events
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(events[0]["kind"], json!("session_start"));
        assert_eq!(events[1]["kind"], json!("tool_start"));
        assert_eq!(events[1]["tool_seq"], json!(1));
        assert_eq!(events[1]["request_id"], request["request_id"]);
        assert_eq!(events[2]["kind"], json!("tool_finish"));
        assert_eq!(events[2]["tool_seq"], json!(1));
        assert_eq!(events[2]["success"], json!(true));
        assert_eq!(events[2]["scratch_preserved"], json!(false));

        fs::remove_dir_all(root.as_str()).unwrap();
        fs::remove_dir_all(sessions_root).unwrap();
    }

    #[test]
    fn tool_confirmation_preview_uses_prepared_invocation_env() {
        if sandbox_exec_refuses_children() {
            return;
        }

        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &confirmation_preview_tool_script());

        let config = Config::load(&root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone()).unwrap();
        let tool_config = config.tools.get("format").unwrap();
        let exposed_name = exposed_tool_name("build", "format").unwrap();
        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, "format").unwrap();
        let tool = ExternalTool::from_config(exposed_name.clone(), canonical_id, tool_config);
        let context = agent.tool_runtime_context();
        let prepared = tool_runtime::prepare_rc_tool_invocation(
            &tool.name,
            &tool.name,
            &tool.canonical_id,
            &tool.executable_path,
            &context,
            "toolu_confirm_123",
            serde_json::Map::from_iter([("paths".to_string(), json!(["src/lib.rs"]))]),
            tool.timeout,
        )
        .unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let preview = runtime
            .block_on(tool_runtime::render_rc_tool_confirmation_preview(
                &prepared, None,
            ))
            .unwrap();
        assert!(preview.contains("mode=confirm"));
        assert!(preview.contains("tool=format"));
        assert!(preview.contains("id=fmt"));
        assert!(preview.contains("request=request.json"));

        let text = runtime
            .block_on(tool_runtime::run_prepared_rc_tool_text(
                &prepared,
                &agent.writable_roots,
                None,
            ))
            .unwrap();
        assert_eq!(text, "format ran");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_handled_model_visible_error() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            &handled_error_tool_script("paths must not be empty"),
        );

        let result = invoke_configured_tool(&root, "format", json!({ "paths": [] }));
        assert_eq!(unwrap_error_text(result), "paths must not be empty");

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_process_error_when_tool_exits_nonzero() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &nonzero_exit_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("exited with status"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_result_protocol_version_is_unsupported() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &unsupported_protocol_version_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("unsupported result protocol version"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_output_kind_is_unsupported() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &unsupported_output_kind_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("unsupported output kind"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_success_output_is_missing() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &missing_success_output_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("missing success output"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_output_text_is_missing() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &missing_output_text_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("missing output.text"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_result_is_missing() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, "#!/bin/sh\nexit 0\n");

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("protocol error"));
        assert!(error.contains("failed to read tool result file"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_result_is_malformed() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            "#!/bin/sh\nprintf 'not json' >\"$RESULT_FILE\"\n",
        );

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("protocol error"));
        assert!(error.contains("failed to parse tool result file"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_error_object_is_missing() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &missing_error_object_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": [] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("missing error object"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_error_message_is_missing() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &missing_error_message_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": [] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("missing error.message"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_request_id_mismatches() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(&root, &mismatched_request_id_tool_script());

        let result = invoke_configured_tool(&root, "format", json!({ "paths": ["src/lib.rs"] }));
        let error = unwrap_error_text(result);
        assert!(error.contains("request_id mismatch"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    #[test]
    fn tool_invocation_returns_protocol_error_when_input_is_not_an_object() {
        let root = temp_config_root("agent");
        write_sample_config_with_fmt_script(
            &root,
            &success_tool_script("Formatted 3 files.", false),
        );

        let result = invoke_configured_tool(&root, "format", json!(["src/lib.rs"]));
        let error = unwrap_error_text(result);
        assert!(error.contains("tool input must be a JSON object"));

        fs::remove_dir_all(root.as_str()).unwrap();
    }

    fn expected_build_system_prompt(workspace_root: &Path) -> String {
        format!(
            r#"# Build

You are an expert builder.
# Environment

You are operating in the sid-isn't-done environment.  Tools:
- edit: The Anthropic Text Editor tool:
    - Uses sid's virtual filesystem.
    - The virtual filesystem maps / to the workspace root.
    - It is not an operating-system chroot.
    - Absolute editor paths are workspace-rooted; /foo means {}/foo.
- bash: A genuine bash shell:
    - Connected via PTY.
    - Without support for cursor positioning.
    - With state persistence between invocations.
    - With PS0, PS1, PS2 and PROMPT_COMMAND set to readonly.
    - `restart: true` throws the session away and starts fresh.
    - Initial CWD is {}.
    - Runs in the host filesystem namespace, not a chroot.
    - Host / remains visible subject to OS permissions and sandbox policy.
    - Bash cannot see sid's virtual /skills mount.
    - Use the index to browse skills if you need specialized knowledge.

CRITICAL — the edit tool and bash tool use different path namespaces:
- The edit tool's / is the workspace root ({}).
  To edit a file at the workspace root, use /filename (e.g., /src/lib.rs).
  NEVER pass a full host path to the edit tool.
- Bash sees the real host filesystem.  `pwd` prints {}, not /.
  To convert a bash path to an edit path, strip the {} prefix.
  To convert an edit path to a bash path, prepend {}.
  If bash `pwd` is a subdirectory, a relative path like ./foo.rs in bash
  corresponds to stripping the workspace prefix from the absolute bash path.
- Example: the bash path {}/src/lib.rs is the edit path /src/lib.rs.
"#,
            workspace_root,
            workspace_root,
            workspace_root,
            workspace_root,
            workspace_root,
            workspace_root,
            workspace_root
        )
    }

    fn write_sample_config(root: &Path) {
        write_sample_config_with_fmt_script(root, "#!/bin/sh\nexit 0\n");
    }

    fn write_builtin_config(root: &Path, bash_script: &str, edit_script: &str) {
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            r#"
build_ENABLED="YES"
build_TOOLS='bash edit'
"#,
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            r#"
bash_ENABLED="YES"
edit_ENABLED="YES"
"#,
        )
        .unwrap();
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        write_tool_contract(root, "bash", "Run a shell command.", bash_script);
        write_tool_contract(root, "edit", "Edit workspace files.", edit_script);
    }

    fn write_builtin_config_without_manifests(root: &Path, bash_script: &str, edit_script: &str) {
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            r#"
build_ENABLED="YES"
build_TOOLS='bash edit'
"#,
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            r#"
bash_ENABLED="YES"
edit_ENABLED="YES"
"#,
        )
        .unwrap();
        fs::write(root.join("agents/build.md").as_str(), "# Build\n").unwrap();
        write_tool_runtime(root, "bash", bash_script);
        write_tool_runtime(root, "edit", edit_script);
    }

    fn write_sample_config_with_fmt_script(root: &Path, fmt_script: &str) {
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        fs::write(
            root.join("agents.conf").as_str(),
            r#"
ROLE='principal engineer'
build_ENABLED="YES"
plan_ENABLED="MANUAL"
evil_ENABLED="NO"

build_NAME="Let's go ${ROLE}"
build_DESC="buildit"
build_TOOLS='format shell'

plan_MODEL=claude-sonnet-4-5
plan_SYSTEM="You are ${ROLE}"
plan_MAX_TOKENS=8192
"#,
        )
        .unwrap();
        fs::write(
            root.join("tools.conf").as_str(),
            r#"
fmt_ENABLED="YES"
shell_ENABLED="YES"

format_INHERIT="YES"
format_ALIASES="fmt"
"#,
        )
        .unwrap();
        fs::write(
            root.join("agents/build.md").as_str(),
            "# Build\n\nYou are an expert builder.\n",
        )
        .unwrap();
        write_tool_contract(root, "fmt", "Format files in the workspace.", fmt_script);
        write_tool_contract(root, "shell", "Run a shell command.", "#!/bin/sh\nexit 0\n");
    }

    fn invoke_configured_tool(
        root: &Path,
        tool_name: &str,
        input: serde_json::Value,
    ) -> ToolResult {
        let config = Config::load(root).unwrap();
        let agent = SidAgent::from_config(&config, "build", root.clone().into_owned()).unwrap();
        let tool_config = config.tools.get(tool_name).unwrap();
        let exposed_name = exposed_tool_name("build", tool_name).unwrap();
        let canonical_id = resolve_canonical_tool_id(&config.tools_rc_conf, tool_name).unwrap();
        let tool = ExternalTool::from_config(exposed_name.clone(), canonical_id, tool_config);
        let tool_use = ToolUseBlock::new("toolu_123", exposed_name, input);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(invoke_external_tool(&tool, &agent, &tool_use, None))
    }

    fn invoke_bash_tool(
        runtime: &tokio::runtime::Runtime,
        client: &Anthropic,
        agent: &mut SidAgent,
        tool_use_id: &str,
        input: serde_json::Value,
    ) -> ToolResult {
        let tool = runtime
            .block_on(agent.tools())
            .into_iter()
            .find(|tool| tool.name() == "bash")
            .expect("agent should expose the builtin bash tool");
        let tool_use = ToolUseBlock::new(tool_use_id, "bash", input);
        let callback = tool.callback();
        let intermediate = runtime.block_on(callback.compute_tool_result(client, agent, &tool_use));
        runtime.block_on(callback.apply_tool_result(client, agent, &tool_use, intermediate))
    }

    fn unwrap_success_text(result: ToolResult) -> String {
        match result {
            ControlFlow::Continue(Ok(block)) => tool_block_text(block),
            other => panic!("expected successful tool result, got {other:?}"),
        }
    }

    fn unwrap_success_block(result: ToolResult) -> ToolResultBlock {
        match result {
            ControlFlow::Continue(Ok(block)) => block,
            other => panic!("expected successful tool result, got {other:?}"),
        }
    }

    fn unwrap_error_text(result: ToolResult) -> String {
        match result {
            ControlFlow::Continue(Err(block)) => {
                assert_eq!(block.is_error, Some(true));
                tool_block_text(block)
            }
            other => panic!("expected errored tool result, got {other:?}"),
        }
    }

    fn tool_block_text(block: ToolResultBlock) -> String {
        match block.content.unwrap() {
            ToolResultBlockContent::String(text) => text,
            other => panic!("expected string tool result content, got {other:?}"),
        }
    }

    fn injected_user_instruction_text(message: &MessageParam) -> &str {
        let MessageParamContent::Array(blocks) = &message.content else {
            panic!("expected content blocks, got {:?}", message.content);
        };
        blocks
            .last()
            .and_then(ContentBlock::as_text)
            .map(|block| block.text.as_str())
            .expect("last block should be injected text")
    }

    fn success_tool_script(text: &str, capture_request: bool) -> String {
        let capture = if capture_request {
            "cp \"$REQUEST_FILE\" \"$WORKSPACE_ROOT/request-capture.json\"\n"
        } else {
            ""
        };
        format!(
            "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\n{capture}cat >\"$RESULT_FILE\" <<EOF\n{{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{{\"kind\":\"text\",\"text\":\"{text}\"}}}}\nEOF\n"
        )
    }

    fn handled_error_tool_script(message: &str) -> String {
        format!(
            "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":false,\"error\":{{\"code\":\"invalid_input\",\"message\":\"{message}\"}}}}\nEOF\n"
        )
    }

    fn environment_capturing_tool_script(text: &str) -> String {
        format!(
            "#!/bin/sh\nprintf 'stdout from tool\\n'\nprintf 'stderr from tool\\n' >&2\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$WORKSPACE_ROOT/env-capture.json\" <<EOF\n{{\"protocol\":\"$TOOL_PROTOCOL\",\"request_file\":\"$REQUEST_FILE\",\"result_file\":\"$RESULT_FILE\",\"scratch_dir\":\"$SCRATCH_DIR\",\"workspace_root\":\"$WORKSPACE_ROOT\",\"agent_id\":\"$AGENT_ID\",\"tool_id\":\"$TOOL_ID\",\"tool_name\":\"$TOOL_NAME\",\"rc_conf_path\":\"$RC_CONF_PATH\",\"rc_d_path\":\"$RC_D_PATH\"}}\nEOF\ncat >\"$RESULT_FILE\" <<EOF\n{{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{{\"kind\":\"text\",\"text\":\"{text}\"}}}}\nEOF\n"
        )
    }

    fn capturing_tool_script(text: &str, request_capture: &str, env_capture: &str) -> String {
        format!(
            "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncp \"$REQUEST_FILE\" \"$WORKSPACE_ROOT/{request_capture}\"\ncat >\"$WORKSPACE_ROOT/{env_capture}\" <<EOF\n{{\"protocol\":\"$TOOL_PROTOCOL\",\"request_file\":\"$REQUEST_FILE\",\"result_file\":\"$RESULT_FILE\",\"scratch_dir\":\"$SCRATCH_DIR\",\"temp_dir\":\"$TEMP_DIR\",\"tmpdir\":\"$TMPDIR\",\"workspace_root\":\"$WORKSPACE_ROOT\",\"agent_id\":\"$AGENT_ID\",\"session_id\":\"$SESSION_ID\",\"session_dir\":\"$SESSION_DIR\",\"tool_id\":\"$TOOL_ID\",\"tool_name\":\"$TOOL_NAME\",\"rc_conf_path\":\"$RC_CONF_PATH\",\"rc_d_path\":\"$RC_D_PATH\"}}\nEOF\ncat >\"$RESULT_FILE\" <<EOF\n{{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{{\"kind\":\"text\",\"text\":\"{text}\"}}}}\nEOF\n"
        )
    }

    fn confirmation_preview_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\nif [ \"$TOOL_MODE\" = confirm ]; then\n    printf 'mode=%s tool=%s id=%s request=%s\\n' \"$TOOL_MODE\" \"$TOOL_NAME\" \"$TOOL_ID\" \"$(basename \"$REQUEST_FILE\")\"\n    exit 0\nfi\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{\"kind\":\"text\",\"text\":\"format ran\"}}\nEOF\n".to_string()
    }

    fn sandbox_exec_refuses_children() -> bool {
        if !seatbelt::sandbox_available() {
            return false;
        }
        match std::process::Command::new("/usr/bin/sandbox-exec")
            .arg("-p")
            .arg("(version 1)\n(allow default)\n")
            .arg("/usr/bin/true")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
        {
            Ok(status) => !status.success(),
            Err(_) => true,
        }
    }

    fn nonzero_exit_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{\"kind\":\"text\",\"text\":\"ignored\"}}\nEOF\nexit 9\n".to_string()
    }

    fn mismatched_request_id_tool_script() -> String {
        "#!/bin/sh\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"wrong-request\",\"ok\":true,\"output\":{\"kind\":\"text\",\"text\":\"ignored\"}}\nEOF\n".to_string()
    }

    fn unsupported_protocol_version_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":2,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{\"kind\":\"text\",\"text\":\"ignored\"}}\nEOF\n".to_string()
    }

    fn unsupported_output_kind_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{\"kind\":\"json\",\"text\":\"ignored\"}}\nEOF\n".to_string()
    }

    fn missing_success_output_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true}\nEOF\n".to_string()
    }

    fn missing_output_text_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":true,\"output\":{\"kind\":\"text\"}}\nEOF\n".to_string()
    }

    fn missing_error_object_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":false}\nEOF\n".to_string()
    }

    fn missing_error_message_tool_script() -> String {
        "#!/bin/sh\nREQUEST_ID=$(sed -n 's/.*\"request_id\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' \"$REQUEST_FILE\")\ncat >\"$RESULT_FILE\" <<EOF\n{\"protocol_version\":1,\"request_id\":\"$REQUEST_ID\",\"ok\":false,\"error\":{\"code\":\"invalid_input\"}}\nEOF\n".to_string()
    }

    fn write_tool_contract(root: &Path, tool: &str, description: &str, body: &str) {
        write_tool_runtime(root, tool, body);
        write_default_tool_manifest(root, tool, description);
    }

    fn write_tool_runtime(root: &Path, tool: &str, body: &str) {
        fs::create_dir_all(root.join(TOOLS_DIR).as_str()).unwrap();
        let executable = root.join(format!("{TOOLS_DIR}/{tool}")).into_owned();
        let implementation = root.join(format!("{TOOLS_DIR}/{tool}.impl")).into_owned();
        fs::write(implementation.as_str(), body).unwrap();
        make_executable(&implementation);
        fs::write(
            executable.as_str(),
            format!(
                "#!/bin/sh\nset -eu\n\nlookup() {{\n    printenv \"$1\"\n}}\n\nPREFIX=${{RCVAR_ARGV0:?missing RCVAR_ARGV0}}\n\ncase \"${{1:-}}\" in\nrcvar)\n    printf '%s\\n' \\\n        \"${{PREFIX}}_REQUEST_FILE\" \\\n        \"${{PREFIX}}_RESULT_FILE\" \\\n        \"${{PREFIX}}_SCRATCH_DIR\" \\\n        \"${{PREFIX}}_TEMP_DIR\" \\\n        \"${{PREFIX}}_TMPDIR\" \\\n        \"${{PREFIX}}_WORKSPACE_ROOT\" \\\n        \"${{PREFIX}}_SESSION_ID\" \\\n        \"${{PREFIX}}_SESSION_DIR\" \\\n        \"${{PREFIX}}_AGENT_ID\" \\\n        \"${{PREFIX}}_TOOL_ID\" \\\n        \"${{PREFIX}}_TOOL_NAME\" \\\n        \"${{PREFIX}}_TOOL_PROTOCOL\" \\\n        \"${{PREFIX}}_RC_CONF_PATH\" \\\n        \"${{PREFIX}}_RC_D_PATH\"\n    ;;\nconfirm|run)\n    export TOOL_MODE=\"$1\"\n    shift\n    export REQUEST_FILE=\"$(lookup \"${{PREFIX}}_REQUEST_FILE\")\"\n    export RESULT_FILE=\"$(lookup \"${{PREFIX}}_RESULT_FILE\")\"\n    export SCRATCH_DIR=\"$(lookup \"${{PREFIX}}_SCRATCH_DIR\")\"\n    export TEMP_DIR=\"$(lookup \"${{PREFIX}}_TEMP_DIR\")\"\n    export TMPDIR=\"$(lookup \"${{PREFIX}}_TMPDIR\")\"\n    export WORKSPACE_ROOT=\"$(lookup \"${{PREFIX}}_WORKSPACE_ROOT\")\"\n    export SESSION_ID=\"$(lookup \"${{PREFIX}}_SESSION_ID\")\"\n    export SESSION_DIR=\"$(lookup \"${{PREFIX}}_SESSION_DIR\")\"\n    export AGENT_ID=\"$(lookup \"${{PREFIX}}_AGENT_ID\")\"\n    export TOOL_ID=\"$(lookup \"${{PREFIX}}_TOOL_ID\")\"\n    export TOOL_NAME=\"$(lookup \"${{PREFIX}}_TOOL_NAME\")\"\n    export TOOL_PROTOCOL=\"$(lookup \"${{PREFIX}}_TOOL_PROTOCOL\")\"\n    export RC_CONF_PATH=\"$(lookup \"${{PREFIX}}_RC_CONF_PATH\")\"\n    export RC_D_PATH=\"$(lookup \"${{PREFIX}}_RC_D_PATH\")\"\n    exec {} \"$@\"\n    ;;\n*)\n    echo \"usage: $0 [rcvar|confirm|run]\" >&2\n    exit 129\n    ;;\nesac\n",
                shvar::quote_string(implementation.as_str())
            ),
        )
        .unwrap();
        make_executable(&executable);
    }

    fn write_agent_hook(root: &Path, hook: &str, body: &str) {
        fs::create_dir_all(root.join("agents").as_str()).unwrap();
        let executable = root.join(format!("agents/{hook}")).into_owned();
        fs::write(executable.as_str(), body).unwrap();
        make_executable(&executable);
    }

    fn write_skill(root: &Path, skill: &str, body: &str) {
        let skill_dir = root.join(format!("skills/{skill}")).into_owned();
        fs::create_dir_all(skill_dir.as_str()).unwrap();
        fs::write(skill_dir.join("SKILL.md").as_str(), body).unwrap();
    }

    #[test]
    fn strip_ansi_escapes_plain_text() {
        assert_eq!(strip_ansi_escapes("hello world"), "hello world");
    }

    #[test]
    fn strip_ansi_escapes_empty() {
        assert_eq!(strip_ansi_escapes(""), "");
    }

    #[test]
    fn strip_ansi_escapes_sgr_color() {
        // Bold red "error" then reset.
        assert_eq!(strip_ansi_escapes("\x1b[1;31merror\x1b[0m"), "error");
    }

    #[test]
    fn strip_ansi_escapes_multiple_csi() {
        assert_eq!(
            strip_ansi_escapes("\x1b[32mok\x1b[0m \x1b[33mwarn\x1b[0m"),
            "ok warn"
        );
    }

    #[test]
    fn strip_ansi_escapes_256_color() {
        assert_eq!(strip_ansi_escapes("\x1b[38;5;196mred\x1b[0m"), "red");
    }

    #[test]
    fn strip_ansi_escapes_truecolor() {
        assert_eq!(strip_ansi_escapes("\x1b[38;2;255;0;0mred\x1b[0m"), "red");
    }

    #[test]
    fn strip_ansi_escapes_osc_bel() {
        // OSC title-set terminated by BEL.
        assert_eq!(strip_ansi_escapes("\x1b]0;my title\x07rest"), "rest");
    }

    #[test]
    fn strip_ansi_escapes_osc_st() {
        // OSC terminated by ST (ESC \).
        assert_eq!(strip_ansi_escapes("\x1b]0;my title\x1b\\rest"), "rest");
    }

    #[test]
    fn strip_ansi_escapes_two_char_escape() {
        // ESC M (reverse index) should be stripped.
        assert_eq!(strip_ansi_escapes("a\x1bMb"), "ab");
    }

    #[test]
    fn strip_ansi_escapes_trailing_esc() {
        assert_eq!(strip_ansi_escapes("text\x1b"), "text");
    }

    #[test]
    fn strip_ansi_escapes_cursor_csi() {
        // CSI sequences ending with ~ (e.g., key codes) or @ (insert).
        assert_eq!(strip_ansi_escapes("\x1b[2~x\x1b[1@y"), "xy");
    }

    #[test]
    fn strip_ansi_escapes_preserves_newlines() {
        assert_eq!(
            strip_ansi_escapes("\x1b[32mline1\x1b[0m\nline2\n"),
            "line1\nline2\n"
        );
    }
}
