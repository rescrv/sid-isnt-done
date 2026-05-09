//! JSONL wire protocol for `sid --raw`.
//!
//! The raw protocol is line-oriented UTF-8 JSON exchanged over stdin/stdout.
//! Clients send semantic requests instead of shell-style commands, and the
//! server emits typed events, prompts, and terminal results.

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::OnceLock;

use claudius::Effort;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Current `sid --raw` protocol version.
pub const RAW_PROTOCOL_VERSION: u32 = 1;

/// A client request envelope sent to `sid --raw`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawRequestEnvelope {
    /// Protocol version understood by the client.
    pub protocol_version: u32,
    /// Request identifier chosen by the client.
    pub request_id: String,
    /// Semantic request payload.
    #[serde(flatten)]
    pub request: RawRequest,
}

/// Semantic requests accepted by `sid --raw`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum RawRequest {
    /// Send a normal user turn to the agent.
    UserTurn {
        /// User-visible message text.
        text: String,
    },
    /// Reply to an outstanding server prompt.
    PromptResponse {
        /// Prompt identifier emitted by the server.
        prompt_id: String,
        /// Response text chosen by the client.
        response: String,
    },
    /// Show the current agent.
    ShowAgent,
    /// List configured agents.
    ListAgents,
    /// Switch to a different agent.
    SwitchAgent {
        /// Agent identifier to activate.
        agent: String,
    },
    /// Compact the current session into a child session.
    Compact,
    /// Clear the conversation transcript.
    Clear,
    /// Change the active model.
    SetModel {
        /// Model name to activate.
        model: String,
    },
    /// Change or clear the system prompt.
    SetSystemPrompt {
        /// New prompt text, or `null` to clear it.
        prompt: Option<String>,
    },
    /// Change the maximum response token limit.
    SetMaxTokens {
        /// Per-response token budget.
        max_tokens: u32,
    },
    /// Change or clear temperature.
    SetTemperature {
        /// Sampling temperature, or `null` to clear it.
        temperature: Option<f32>,
    },
    /// Change or clear top-p.
    SetTopP {
        /// Top-p value, or `null` to clear it.
        top_p: Option<f32>,
    },
    /// Change or clear top-k.
    SetTopK {
        /// Top-k value, or `null` to clear it.
        top_k: Option<u32>,
    },
    /// Add a stop sequence.
    AddStopSequence {
        /// Sequence to add.
        sequence: String,
    },
    /// Clear all stop sequences.
    ClearStopSequences,
    /// Return the active stop sequences.
    ListStopSequences,
    /// Change or clear the explicit thinking budget.
    SetThinkingBudget {
        /// Token budget, or `null` to disable explicit thinking.
        tokens: Option<u32>,
    },
    /// Enable adaptive thinking using the current effort setting.
    SetThinkingAdaptive,
    /// Change or clear the adaptive effort level.
    SetEffort {
        /// Effort level, or `null` to clear it.
        effort: Option<Effort>,
    },
    /// Change or clear the session spend limit.
    SetSpend {
        /// Spend limit in dollars, or `null` to clear it.
        dollars: Option<f64>,
    },
    /// Toggle prompt caching.
    SetCaching {
        /// Desired caching state.
        enabled: bool,
    },
    /// Save the transcript to an arbitrary path.
    SaveTranscript {
        /// Destination path.
        path: String,
    },
    /// Load a transcript from an arbitrary path.
    LoadTranscript {
        /// Source path.
        path: String,
    },
    /// Return session statistics.
    Stats,
    /// Return the current effective configuration.
    ShowConfig,
    /// Shut down the raw server.
    Shutdown,
}

/// A server-to-client JSONL message emitted by `sid --raw`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RawServerMessage {
    /// Initial capability and session announcement.
    Hello(RawHello),
    /// Incremental event associated with a request.
    Event(RawEventEnvelope),
    /// Interactive prompt that requires a `prompt_response`.
    Prompt(RawPrompt),
    /// Terminal request result.
    Result(RawResultEnvelope),
}

/// Initial session announcement for a raw server instance.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawHello {
    /// Protocol version emitted by the server.
    pub protocol_version: u32,
    /// Active session identifier.
    pub session_id: String,
    /// Active session directory path.
    pub session_dir: String,
    /// Workspace root visible to the agent.
    pub workspace_root: String,
    /// Active agent identifier.
    pub current_agent: String,
    /// Active model name.
    pub model: String,
    /// Whether the session was resumed from disk.
    pub resumed: bool,
    /// Whether the startup agent requires manual confirmation.
    pub startup_confirmation_required: bool,
    /// Whether sandboxing support is available.
    pub sandbox_available: bool,
}

/// Incremental event associated with a single request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawEventEnvelope {
    /// Protocol version emitted by the server.
    pub protocol_version: u32,
    /// Request identifier chosen by the client.
    pub request_id: String,
    /// Event payload.
    #[serde(flatten)]
    pub event: RawEvent,
}

/// Streaming events emitted while a request is in progress.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum RawEvent {
    /// Agent stream started.
    AgentStart {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
    },
    /// Agent stream finished.
    AgentFinish {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
        /// Optional stop reason string.
        stop_reason: Option<String>,
    },
    /// Assistant text delta.
    AssistantTextDelta {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
        /// Text chunk.
        text: String,
    },
    /// Thinking text delta.
    ThinkingDelta {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
        /// Text chunk.
        text: String,
    },
    /// Informational text.
    Info {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
        /// Message text.
        message: String,
    },
    /// Error text.
    Error {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
        /// Message text.
        message: String,
    },
    /// Tool use block started.
    ToolUseStart {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
        /// Tool display name.
        name: String,
        /// API-level tool-use identifier.
        tool_use_id: String,
    },
    /// Tool input delta.
    ToolInputDelta {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
        /// Partial JSON chunk.
        partial_json: String,
    },
    /// Tool use block finished.
    ToolUseEnd {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
    },
    /// Tool result block started.
    ToolResultStart {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
        /// API-level tool-use identifier.
        tool_use_id: String,
        /// Whether the result is an error.
        is_error: bool,
    },
    /// Tool result text delta.
    ToolResultTextDelta {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
        /// Text chunk.
        text: String,
    },
    /// Tool result block finished.
    ToolResultEnd {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
    },
    /// Response finished.
    ResponseFinish {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
    },
    /// Stream interrupted.
    Interrupted {
        /// Optional stream label.
        label: Option<String>,
        /// Stream nesting depth.
        depth: usize,
    },
    /// External tool stdout/stderr chunk.
    ToolOutput {
        /// Display name of the tool.
        tool_name: String,
        /// API-level tool-use identifier.
        tool_use_id: String,
        /// Output stream name such as `stdout` or `stderr`.
        stream: String,
        /// UTF-8 text chunk, when available.
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        /// Base64-encoded bytes for non-UTF-8 chunks.
        #[serde(skip_serializing_if = "Option::is_none")]
        data_b64: Option<String>,
    },
}

/// A server prompt waiting for a `prompt_response`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawPrompt {
    /// Protocol version emitted by the server.
    pub protocol_version: u32,
    /// Request identifier associated with the prompt.
    pub request_id: String,
    /// Stable prompt identifier used by the matching response.
    pub prompt_id: String,
    /// Prompt type.
    pub kind: String,
    /// Prompt text.
    pub message: String,
    /// Allowed textual choices.
    pub choices: Vec<String>,
}

/// Terminal response for a client request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawResultEnvelope {
    /// Protocol version emitted by the server.
    pub protocol_version: u32,
    /// Request identifier chosen by the client.
    pub request_id: String,
    /// `true` for success, `false` for error.
    pub ok: bool,
    /// Optional structured success payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    /// Optional structured error payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RawServerError>,
}

/// Structured error payload for a failed raw request.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RawServerError {
    /// Optional machine-readable error code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Human-readable error message.
    pub message: String,
}

/// Structured external-tool output emitted outside the model event stream.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolOutputEvent {
    /// Request identifier currently associated with the tool run.
    pub request_id: String,
    /// Display name of the tool.
    pub tool_name: String,
    /// API-level tool-use identifier.
    pub tool_use_id: String,
    /// Output stream name such as `stdout` or `stderr`.
    pub stream: String,
    /// UTF-8 text chunk, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64-encoded bytes for non-UTF-8 chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_b64: Option<String>,
}

/// Observer used by the raw runtime to receive external tool stdout/stderr.
pub trait ToolOutputObserver: Send + Sync {
    /// Called for each captured tool-output chunk.
    fn on_tool_output(&self, event: &ToolOutputEvent);
}

/// RAII handle that restores the previously installed observer on drop.
pub struct ToolOutputObserverRegistration {
    previous: Option<Arc<dyn ToolOutputObserver>>,
}

impl Drop for ToolOutputObserverRegistration {
    fn drop(&mut self) {
        set_active_tool_output_observer(self.previous.take());
    }
}

/// Install a process-local tool-output observer for the current request scope.
pub fn install_tool_output_observer(
    observer: Option<Arc<dyn ToolOutputObserver>>,
) -> ToolOutputObserverRegistration {
    let previous = set_active_tool_output_observer(observer);
    ToolOutputObserverRegistration { previous }
}

pub(crate) fn notify_tool_output_observer(event: &ToolOutputEvent) {
    let observer = active_tool_output_observer()
        .lock()
        .expect("tool output observer lock poisoned")
        .clone();
    if let Some(observer) = observer {
        observer.on_tool_output(event);
    }
}

pub(crate) fn has_active_tool_output_observer() -> bool {
    active_tool_output_observer()
        .lock()
        .expect("tool output observer lock poisoned")
        .is_some()
}

fn active_tool_output_observer() -> &'static StdMutex<Option<Arc<dyn ToolOutputObserver>>> {
    static ACTIVE: OnceLock<StdMutex<Option<Arc<dyn ToolOutputObserver>>>> = OnceLock::new();
    ACTIVE.get_or_init(|| StdMutex::new(None))
}

fn set_active_tool_output_observer(
    observer: Option<Arc<dyn ToolOutputObserver>>,
) -> Option<Arc<dyn ToolOutputObserver>> {
    let mut slot = active_tool_output_observer()
        .lock()
        .expect("tool output observer lock poisoned");
    std::mem::replace(&mut *slot, observer)
}
