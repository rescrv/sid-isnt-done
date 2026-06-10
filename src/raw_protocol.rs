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
pub const RAW_PROTOCOL_VERSION: u32 = 3;

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
    /// Insert a system message into the conversation transcript.
    InsertSystemMessage {
        /// System message text.
        text: String,
    },
    /// Reply to an outstanding server prompt.
    PromptResponse {
        /// Prompt identifier emitted by the server.
        prompt_id: String,
        /// Response text chosen by the client.
        response: String,
    },
    /// Interrupt the active streaming request.
    Interrupt,
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
    /// End marker for the history replay sent by reconnectable transports.
    ReplayComplete(RawReplayComplete),
    /// Accepted client request context that frontends can replay.
    Request(RawAcceptedRequest),
    /// Incremental event associated with a request.
    Event(RawEventEnvelope),
    /// Interactive prompt that requires a `prompt_response`.
    Prompt(RawPrompt),
    /// A previously emitted prompt was answered.
    PromptAck(RawPromptAck),
    /// Terminal request result.
    Result(RawResultEnvelope),
}

impl RawServerMessage {
    /// Return a copy of this message stamped with `sequence`.
    pub fn with_sequence(mut self, sequence: u64) -> Self {
        match &mut self {
            RawServerMessage::Hello(message) => message.sequence = sequence,
            RawServerMessage::ReplayComplete(message) => message.sequence = sequence,
            RawServerMessage::Request(message) => message.sequence = sequence,
            RawServerMessage::Event(message) => message.sequence = sequence,
            RawServerMessage::Prompt(message) => message.sequence = sequence,
            RawServerMessage::PromptAck(message) => message.sequence = sequence,
            RawServerMessage::Result(message) => message.sequence = sequence,
        }
        self
    }
}

/// Marker emitted by listening transports after replayed history has been sent.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawReplayComplete {
    /// Protocol version emitted by the server.
    pub protocol_version: u32,
    /// Monotonic server-message sequence number.
    ///
    /// Listening transports emit this marker outside the retained history so
    /// interactive clients can distinguish replayed messages from live output.
    #[serde(default)]
    pub sequence: u64,
}

/// Initial session announcement for a raw server instance.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawHello {
    /// Protocol version emitted by the server.
    pub protocol_version: u32,
    /// Monotonic server-message sequence number.
    ///
    /// Listening transports replay previously emitted messages to reconnecting
    /// clients.  Clients can use this value to de-duplicate replayed messages.
    #[serde(default)]
    pub sequence: u64,
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

/// Accepted request context replayed to raw frontends.
///
/// The listener retains this message in history before request-scoped events so
/// reconnecting frontends can reconstruct the user-visible transcript without
/// reading a filesystem-local saved transcript.  The server currently emits
/// this marker for accepted requests that affect transcript history without
/// being recoverable from streamed assistant output.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawAcceptedRequest {
    /// Protocol version emitted by the server.
    pub protocol_version: u32,
    /// Monotonic server-message sequence number.
    ///
    /// Listening transports replay previously emitted messages to reconnecting
    /// clients.  Clients can use this value to de-duplicate replayed messages.
    #[serde(default)]
    pub sequence: u64,
    /// Request identifier chosen by the client.
    pub request_id: String,
    /// Accepted request payload.
    #[serde(flatten)]
    pub request: RawRequest,
}

impl RawAcceptedRequest {
    /// Build a replayable request marker when the request affects transcript
    /// reconstruction.
    pub fn from_envelope(envelope: &RawRequestEnvelope) -> Option<Self> {
        match &envelope.request {
            RawRequest::UserTurn { .. } | RawRequest::InsertSystemMessage { .. } => Some(Self {
                protocol_version: RAW_PROTOCOL_VERSION,
                sequence: 0,
                request_id: envelope.request_id.clone(),
                request: envelope.request.clone(),
            }),
            _ => None,
        }
    }
}

/// Incremental event associated with a single request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawEventEnvelope {
    /// Protocol version emitted by the server.
    pub protocol_version: u32,
    /// Monotonic server-message sequence number.
    ///
    /// Listening transports replay previously emitted messages to reconnecting
    /// clients.  Clients can use this value to de-duplicate replayed messages.
    #[serde(default)]
    pub sequence: u64,
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
    /// Monotonic server-message sequence number.
    ///
    /// Listening transports replay previously emitted messages to reconnecting
    /// clients.  Clients can use this value to de-duplicate replayed messages.
    #[serde(default)]
    pub sequence: u64,
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

/// Prompt acknowledgement emitted when a `prompt_response` is accepted.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawPromptAck {
    /// Protocol version emitted by the server.
    pub protocol_version: u32,
    /// Monotonic server-message sequence number.
    ///
    /// Listening transports replay previously emitted messages to reconnecting
    /// clients.  Clients can use this value to de-duplicate replayed messages.
    #[serde(default)]
    pub sequence: u64,
    /// Request identifier associated with the original prompt.
    pub request_id: String,
    /// Request identifier of the accepted `prompt_response`.
    pub response_request_id: String,
    /// Prompt identifier that was acknowledged.
    pub prompt_id: String,
    /// Accepted response text.
    pub response: String,
}

/// Terminal response for a client request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawResultEnvelope {
    /// Protocol version emitted by the server.
    pub protocol_version: u32,
    /// Monotonic server-message sequence number.
    ///
    /// Listening transports replay previously emitted messages to reconnecting
    /// clients.  Clients can use this value to de-duplicate replayed messages.
    #[serde(default)]
    pub sequence: u64,
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

/// Human-readable API usage report emitted after a model response.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct UsageReportEvent {
    /// Token/cost summary line.
    pub token_line: String,
    /// Raw provider usage debug line.
    pub usage_line: String,
}

/// Observer used by the raw runtime to receive external tool stdout/stderr.
pub trait ToolOutputObserver: Send + Sync {
    /// Called for each captured tool-output chunk.
    fn on_tool_output(&self, event: &ToolOutputEvent);
}

/// Observer used by raw frontends to receive API usage reports.
pub trait UsageReportObserver: Send + Sync {
    /// Called after each model response usage report is available.
    fn on_usage_report(&self, event: &UsageReportEvent);
}

/// RAII handle that restores the previously installed observer on drop.
pub struct ToolOutputObserverRegistration {
    previous: Option<Arc<dyn ToolOutputObserver>>,
}

/// RAII handle that restores the previously installed usage observer on drop.
pub struct UsageReportObserverRegistration {
    previous: Option<Arc<dyn UsageReportObserver>>,
}

impl Drop for ToolOutputObserverRegistration {
    fn drop(&mut self) {
        set_active_tool_output_observer(self.previous.take());
    }
}

impl Drop for UsageReportObserverRegistration {
    fn drop(&mut self) {
        set_active_usage_report_observer(self.previous.take());
    }
}

/// Install a process-local tool-output observer for the current request scope.
pub fn install_tool_output_observer(
    observer: Option<Arc<dyn ToolOutputObserver>>,
) -> ToolOutputObserverRegistration {
    let previous = set_active_tool_output_observer(observer);
    ToolOutputObserverRegistration { previous }
}

/// Install a process-local API usage observer for the current request scope.
pub fn install_usage_report_observer(
    observer: Option<Arc<dyn UsageReportObserver>>,
) -> UsageReportObserverRegistration {
    let previous = set_active_usage_report_observer(observer);
    UsageReportObserverRegistration { previous }
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

pub(crate) fn notify_usage_report_observer(event: &UsageReportEvent) -> bool {
    let observer = active_usage_report_observer()
        .lock()
        .expect("usage report observer lock poisoned")
        .clone();
    if let Some(observer) = observer {
        observer.on_usage_report(event);
        true
    } else {
        false
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

fn active_usage_report_observer() -> &'static StdMutex<Option<Arc<dyn UsageReportObserver>>> {
    static ACTIVE: OnceLock<StdMutex<Option<Arc<dyn UsageReportObserver>>>> = OnceLock::new();
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

fn set_active_usage_report_observer(
    observer: Option<Arc<dyn UsageReportObserver>>,
) -> Option<Arc<dyn UsageReportObserver>> {
    let mut slot = active_usage_report_observer()
        .lock()
        .expect("usage report observer lock poisoned");
    std::mem::replace(&mut *slot, observer)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── RawRequest round-trip serialization ──────────────────────────────

    #[test]
    fn request_envelope_user_turn_roundtrip() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "r-1".to_string(),
            request: RawRequest::UserTurn {
                text: "hello".to_string(),
            },
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: RawRequestEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn request_envelope_prompt_response_roundtrip() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "r-2".to_string(),
            request: RawRequest::PromptResponse {
                prompt_id: "prompt-1".to_string(),
                response: "yes".to_string(),
            },
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: RawRequestEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn request_envelope_shutdown_roundtrip() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "r-3".to_string(),
            request: RawRequest::Shutdown,
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: RawRequestEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn request_set_model_roundtrip() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "r-4".to_string(),
            request: RawRequest::SetModel {
                model: "sonnet".to_string(),
            },
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: RawRequestEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn request_set_temperature_with_value_roundtrip() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "r-5".to_string(),
            request: RawRequest::SetTemperature {
                temperature: Some(0.7),
            },
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: RawRequestEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn request_set_temperature_null_roundtrip() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "r-6".to_string(),
            request: RawRequest::SetTemperature { temperature: None },
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: RawRequestEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn request_tag_uses_snake_case() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "r-7".to_string(),
            request: RawRequest::ShowAgent,
        };
        let json = serde_json::to_string(&envelope).unwrap();
        assert!(json.contains(r#""op":"show_agent""#), "json was: {json}");
    }

    #[test]
    fn request_save_transcript_roundtrip() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "r-8".to_string(),
            request: RawRequest::SaveTranscript {
                path: "/tmp/out.json".to_string(),
            },
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: RawRequestEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn request_set_spend_roundtrip() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "r-9".to_string(),
            request: RawRequest::SetSpend { dollars: Some(5.0) },
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: RawRequestEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, envelope);
    }

    // ── RawServerMessage round-trip serialization ────────────────────────

    #[test]
    fn server_hello_roundtrip() {
        let msg = RawServerMessage::Hello(RawHello {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 1,
            session_id: "sess-1".to_string(),
            session_dir: "/tmp/sess".to_string(),
            workspace_root: "/workspace".to_string(),
            current_agent: "default".to_string(),
            model: "sonnet-4".to_string(),
            resumed: false,
            startup_confirmation_required: false,
            sandbox_available: true,
        });
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: RawServerMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn server_replay_complete_roundtrip() {
        let msg = RawServerMessage::ReplayComplete(RawReplayComplete {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 9,
        });
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: RawServerMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn server_request_user_turn_roundtrip() {
        let msg = RawServerMessage::Request(RawAcceptedRequest {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 4,
            request_id: "r-0".to_string(),
            request: RawRequest::UserTurn {
                text: "hello".to_string(),
            },
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""type":"request""#), "got: {json}");
        assert!(json.contains(r#""op":"user_turn""#), "got: {json}");
        let decoded: RawServerMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn server_result_ok_roundtrip() {
        let msg = RawServerMessage::Result(RawResultEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 5,
            request_id: "r-1".to_string(),
            ok: true,
            data: Some(serde_json::json!({"agent": "default"})),
            error: None,
        });
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: RawServerMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn server_result_error_roundtrip() {
        let msg = RawServerMessage::Result(RawResultEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 6,
            request_id: "r-2".to_string(),
            ok: false,
            data: None,
            error: Some(RawServerError {
                code: Some("busy".to_string()),
                message: "server is busy".to_string(),
            }),
        });
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: RawServerMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn server_prompt_roundtrip() {
        let msg = RawServerMessage::Prompt(RawPrompt {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 7,
            request_id: "r-3".to_string(),
            prompt_id: "prompt-1".to_string(),
            kind: "confirmation".to_string(),
            message: "Continue?".to_string(),
            choices: vec!["yes".to_string(), "no".to_string()],
        });
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: RawServerMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn server_prompt_ack_roundtrip() {
        let msg = RawServerMessage::PromptAck(RawPromptAck {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 8,
            request_id: "r-3".to_string(),
            response_request_id: "r-4".to_string(),
            prompt_id: "prompt-1".to_string(),
            response: "yes".to_string(),
        });
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: RawServerMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn server_event_assistant_text_delta_roundtrip() {
        let msg = RawServerMessage::Event(RawEventEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 10,
            request_id: "r-5".to_string(),
            event: RawEvent::AssistantTextDelta {
                label: Some("main".to_string()),
                depth: 0,
                text: "Hello world".to_string(),
            },
        });
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: RawServerMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn server_event_tool_use_start_roundtrip() {
        let msg = RawServerMessage::Event(RawEventEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 11,
            request_id: "r-6".to_string(),
            event: RawEvent::ToolUseStart {
                label: None,
                depth: 1,
                name: "bash".to_string(),
                tool_use_id: "tu-1".to_string(),
            },
        });
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: RawServerMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn server_event_tool_output_omits_none_fields() {
        let msg = RawServerMessage::Event(RawEventEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 12,
            request_id: "r-7".to_string(),
            event: RawEvent::ToolOutput {
                tool_name: "bash".to_string(),
                tool_use_id: "tu-1".to_string(),
                stream: "stdout".to_string(),
                text: Some("output".to_string()),
                data_b64: None,
            },
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("data_b64"), "None field should be omitted");
        let decoded: RawServerMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn server_result_omits_none_error_and_data() {
        let msg = RawServerMessage::Result(RawResultEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 13,
            request_id: "r-8".to_string(),
            ok: true,
            data: None,
            error: None,
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("data"), "None data should be omitted");
        assert!(!json.contains("error"), "None error should be omitted");
    }

    // ── with_sequence ────────────────────────────────────────────────────

    #[test]
    fn with_sequence_stamps_hello() {
        let msg = RawServerMessage::Hello(RawHello {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
            session_id: "s".to_string(),
            session_dir: "/d".to_string(),
            workspace_root: "/w".to_string(),
            current_agent: "a".to_string(),
            model: "m".to_string(),
            resumed: false,
            startup_confirmation_required: false,
            sandbox_available: false,
        });
        let stamped = msg.with_sequence(42);
        match stamped {
            RawServerMessage::Hello(h) => assert_eq!(h.sequence, 42),
            _ => panic!("expected Hello"),
        }
    }

    #[test]
    fn with_sequence_stamps_event() {
        let msg = RawServerMessage::Event(RawEventEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
            request_id: "r".to_string(),
            event: RawEvent::Info {
                label: None,
                depth: 0,
                message: "info".to_string(),
            },
        });
        let stamped = msg.with_sequence(99);
        match stamped {
            RawServerMessage::Event(e) => assert_eq!(e.sequence, 99),
            _ => panic!("expected Event"),
        }
    }

    #[test]
    fn with_sequence_stamps_request() {
        let msg = RawServerMessage::Request(RawAcceptedRequest {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
            request_id: "r".to_string(),
            request: RawRequest::UserTurn {
                text: "hello".to_string(),
            },
        });
        let stamped = msg.with_sequence(6);
        match stamped {
            RawServerMessage::Request(r) => assert_eq!(r.sequence, 6),
            _ => panic!("expected Request"),
        }
    }

    #[test]
    fn with_sequence_stamps_replay_complete() {
        let msg = RawServerMessage::ReplayComplete(RawReplayComplete {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
        });
        let stamped = msg.with_sequence(4);
        match stamped {
            RawServerMessage::ReplayComplete(replay) => assert_eq!(replay.sequence, 4),
            _ => panic!("expected ReplayComplete"),
        }
    }

    #[test]
    fn with_sequence_stamps_prompt() {
        let msg = RawServerMessage::Prompt(RawPrompt {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
            request_id: "r".to_string(),
            prompt_id: "p".to_string(),
            kind: "confirmation".to_string(),
            message: "ok?".to_string(),
            choices: vec![],
        });
        let stamped = msg.with_sequence(7);
        match stamped {
            RawServerMessage::Prompt(p) => assert_eq!(p.sequence, 7),
            _ => panic!("expected Prompt"),
        }
    }

    #[test]
    fn with_sequence_stamps_prompt_ack() {
        let msg = RawServerMessage::PromptAck(RawPromptAck {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
            request_id: "r".to_string(),
            response_request_id: "rr".to_string(),
            prompt_id: "p".to_string(),
            response: "yes".to_string(),
        });
        let stamped = msg.with_sequence(8);
        match stamped {
            RawServerMessage::PromptAck(p) => assert_eq!(p.sequence, 8),
            _ => panic!("expected PromptAck"),
        }
    }

    #[test]
    fn with_sequence_stamps_result() {
        let msg = RawServerMessage::Result(RawResultEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
            request_id: "r".to_string(),
            ok: true,
            data: None,
            error: None,
        });
        let stamped = msg.with_sequence(3);
        match stamped {
            RawServerMessage::Result(r) => assert_eq!(r.sequence, 3),
            _ => panic!("expected Result"),
        }
    }

    // ── Server message type tags ─────────────────────────────────────────

    #[test]
    fn server_message_type_tags_are_snake_case() {
        let hello = serde_json::to_string(&RawServerMessage::Hello(RawHello {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
            session_id: "s".to_string(),
            session_dir: "/d".to_string(),
            workspace_root: "/w".to_string(),
            current_agent: "a".to_string(),
            model: "m".to_string(),
            resumed: false,
            startup_confirmation_required: false,
            sandbox_available: false,
        }))
        .unwrap();
        assert!(hello.contains(r#""type":"hello""#), "got: {hello}");

        let replay_complete =
            serde_json::to_string(&RawServerMessage::ReplayComplete(RawReplayComplete {
                protocol_version: RAW_PROTOCOL_VERSION,
                sequence: 0,
            }))
            .unwrap();
        assert!(
            replay_complete.contains(r#""type":"replay_complete""#),
            "got: {replay_complete}"
        );

        let request = serde_json::to_string(&RawServerMessage::Request(RawAcceptedRequest {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
            request_id: "r".to_string(),
            request: RawRequest::UserTurn {
                text: "hello".to_string(),
            },
        }))
        .unwrap();
        assert!(request.contains(r#""type":"request""#), "got: {request}");

        let prompt_ack = serde_json::to_string(&RawServerMessage::PromptAck(RawPromptAck {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
            request_id: "r".to_string(),
            response_request_id: "rr".to_string(),
            prompt_id: "p".to_string(),
            response: "yes".to_string(),
        }))
        .unwrap();
        assert!(
            prompt_ack.contains(r#""type":"prompt_ack""#),
            "got: {prompt_ack}"
        );
    }

    // ── Event variant tags ───────────────────────────────────────────────

    #[test]
    fn event_variant_tags_are_snake_case() {
        let event = serde_json::to_string(&RawEvent::AgentStart {
            label: None,
            depth: 0,
        })
        .unwrap();
        assert!(event.contains(r#""event":"agent_start""#), "got: {event}");

        let event = serde_json::to_string(&RawEvent::ToolResultTextDelta {
            label: None,
            depth: 0,
            text: "t".to_string(),
        })
        .unwrap();
        assert!(
            event.contains(r#""event":"tool_result_text_delta""#),
            "got: {event}"
        );
    }

    // ── RawHello sequence defaults to 0 ──────────────────────────────────

    #[test]
    fn hello_sequence_defaults_to_zero_when_absent() {
        let json = r#"{
            "type": "hello",
            "protocol_version": 2,
            "session_id": "s",
            "session_dir": "/d",
            "workspace_root": "/w",
            "current_agent": "a",
            "model": "m",
            "resumed": false,
            "startup_confirmation_required": false,
            "sandbox_available": false
        }"#;
        let msg: RawServerMessage = serde_json::from_str(json).unwrap();
        match msg {
            RawServerMessage::Hello(h) => assert_eq!(h.sequence, 0),
            _ => panic!("expected Hello"),
        }
    }

    #[test]
    fn replay_complete_sequence_defaults_to_zero_when_absent() {
        let json = r#"{
            "type": "replay_complete",
            "protocol_version": 2
        }"#;
        let msg: RawServerMessage = serde_json::from_str(json).unwrap();
        match msg {
            RawServerMessage::ReplayComplete(replay) => assert_eq!(replay.sequence, 0),
            _ => panic!("expected ReplayComplete"),
        }
    }

    #[test]
    fn request_sequence_defaults_to_zero_when_absent() {
        let json = r#"{
            "type": "request",
            "protocol_version": 2,
            "request_id": "r",
            "op": "user_turn",
            "text": "hello"
        }"#;
        let msg: RawServerMessage = serde_json::from_str(json).unwrap();
        match msg {
            RawServerMessage::Request(request) => assert_eq!(request.sequence, 0),
            _ => panic!("expected Request"),
        }
    }

    #[test]
    fn accepted_request_from_envelope_keeps_user_turn_text() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "turn-1".to_string(),
            request: RawRequest::UserTurn {
                text: "what did I ask?".to_string(),
            },
        };
        let request = RawAcceptedRequest::from_envelope(&envelope).unwrap();
        assert_eq!(request.request_id, "turn-1");
        assert_eq!(
            request.request,
            RawRequest::UserTurn {
                text: "what did I ask?".to_string(),
            }
        );
    }

    #[test]
    fn accepted_request_from_envelope_keeps_inserted_system_message_text() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "system-1".to_string(),
            request: RawRequest::InsertSystemMessage {
                text: "pin this instruction".to_string(),
            },
        };
        let request = RawAcceptedRequest::from_envelope(&envelope).unwrap();
        assert_eq!(request.request_id, "system-1");
        assert_eq!(
            request.request,
            RawRequest::InsertSystemMessage {
                text: "pin this instruction".to_string(),
            }
        );
    }

    #[test]
    fn accepted_request_from_envelope_skips_non_transcript_requests() {
        let envelope = RawRequestEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: "stats".to_string(),
            request: RawRequest::Stats,
        };
        assert!(RawAcceptedRequest::from_envelope(&envelope).is_none());
    }
}
