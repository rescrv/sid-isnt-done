//! Raw-mode transport utilities for `sid --raw`.
//!
//! This module owns the JSONL stdin/stdout transport, prompt handling, and
//! external-tool output forwarding used by raw-mode frontends.

use std::io::{self, BufRead, BufReader, Write};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use claudius::{OperatorLine, Renderer, StopReason, StreamContext};

use crate::raw_protocol::{
    RAW_PROTOCOL_VERSION, RawEvent, RawEventEnvelope, RawPrompt, RawRequest, RawRequestEnvelope,
    RawResultEnvelope, RawServerError, RawServerMessage, ToolOutputEvent, ToolOutputObserver,
};

/// Raw-protocol decode error that preserves any discovered request id.
#[derive(Debug)]
pub struct ParsedRequestError {
    /// Request id extracted from the partially decoded envelope, if any.
    pub request_id: String,
    /// Optional machine-readable error code.
    pub code: Option<String>,
    /// Human-readable error message.
    pub message: String,
}

/// Shared JSONL output sink used by the raw server and tool-output observer.
pub struct SharedOutput<W>
where
    W: Write + Send + 'static,
{
    writer: Arc<StdMutex<W>>,
}

impl<W> Clone for SharedOutput<W>
where
    W: Write + Send + 'static,
{
    fn clone(&self) -> Self {
        Self {
            writer: self.writer.clone(),
        }
    }
}

impl<W> SharedOutput<W>
where
    W: Write + Send + 'static,
{
    /// Create a new shared output sink.
    pub fn new(writer: W) -> Self {
        Self {
            writer: Arc::new(StdMutex::new(writer)),
        }
    }

    /// Write one JSONL server message.
    pub fn write_message(&self, message: &RawServerMessage) -> io::Result<()> {
        let payload = serde_json::to_vec(message)
            .map_err(|err| io::Error::other(format!("failed to serialize raw message: {err}")))?;
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| io::Error::other("raw output lock poisoned"))?;
        writer.write_all(&payload)?;
        writer.write_all(b"\n")?;
        writer.flush()
    }

    /// Execute `f` with a mutable borrow of the underlying writer.
    #[cfg(test)]
    pub(crate) fn with_writer<T>(&self, f: impl FnOnce(&W) -> T) -> T {
        let writer = self.writer.lock().expect("raw output lock poisoned");
        f(&writer)
    }
}

/// Line-oriented raw JSONL server that also implements `claudius::Renderer`.
pub struct RawServer<R, W>
where
    R: BufRead + Send,
    W: Write + Send + 'static,
{
    input: R,
    output: SharedOutput<W>,
    active_request_id: Option<String>,
    next_prompt_id: u64,
}

impl RawServer<BufReader<io::Stdin>, io::Stdout> {
    /// Create a raw server bound to process stdin/stdout.
    pub fn stdio() -> Self {
        Self::new(BufReader::new(io::stdin()), io::stdout())
    }
}

impl<R, W> RawServer<R, W>
where
    R: BufRead + Send,
    W: Write + Send + 'static,
{
    /// Create a raw server from arbitrary reader/writer handles.
    pub fn new(input: R, output: W) -> Self {
        Self {
            input,
            output: SharedOutput::new(output),
            active_request_id: None,
            next_prompt_id: 1,
        }
    }

    /// Clone the shared output sink.
    pub fn output(&self) -> SharedOutput<W> {
        self.output.clone()
    }

    /// Set the request id associated with future streamed events.
    pub fn set_request_id(&mut self, request_id: String) {
        self.active_request_id = Some(request_id);
    }

    /// Clear the request id associated with future streamed events.
    pub fn clear_request_id(&mut self) {
        self.active_request_id = None;
    }

    /// Write one JSONL server message.
    pub fn write_message(&self, message: &RawServerMessage) -> io::Result<()> {
        self.output.write_message(message)
    }

    /// Write a successful terminal result.
    pub fn write_ok_result(
        &self,
        request_id: &str,
        data: Option<serde_json::Value>,
    ) -> io::Result<()> {
        self.write_message(&RawServerMessage::Result(RawResultEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: request_id.to_string(),
            ok: true,
            data,
            error: None,
        }))
    }

    /// Write a failed terminal result.
    pub fn write_error_result(
        &self,
        request_id: &str,
        code: Option<&str>,
        message: &str,
    ) -> io::Result<()> {
        self.write_message(&RawServerMessage::Result(RawResultEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: request_id.to_string(),
            ok: false,
            data: None,
            error: Some(RawServerError {
                code: code.map(str::to_string),
                message: message.to_string(),
            }),
        }))
    }

    /// Read the next valid request, auto-reporting malformed request lines.
    pub fn read_request(&mut self) -> io::Result<Option<RawRequestEnvelope>> {
        loop {
            let mut line = String::new();
            let read = self.input.read_line(&mut line)?;
            if read == 0 {
                return Ok(None);
            }
            if line.trim().is_empty() {
                continue;
            }
            match parse_request_line(&line) {
                Ok(request) => return Ok(Some(request)),
                Err(err) => {
                    self.write_error_result(&err.request_id, err.code.as_deref(), &err.message)?;
                }
            }
        }
    }

    fn emit_event(&self, event: RawEvent) -> io::Result<()> {
        let request_id = self.active_request_id.as_ref().cloned().unwrap_or_default();
        self.write_message(&RawServerMessage::Event(RawEventEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id,
            event,
        }))
    }
}

impl<R, W> Renderer for RawServer<R, W>
where
    R: BufRead + Send,
    W: Write + Send + 'static,
{
    fn start_agent(&mut self, context: &dyn StreamContext) {
        let _ = self.emit_event(RawEvent::AgentStart {
            label: context.label().map(str::to_string),
            depth: context.depth(),
        });
    }

    fn finish_agent(&mut self, context: &dyn StreamContext, stop_reason: Option<&StopReason>) {
        let _ = self.emit_event(RawEvent::AgentFinish {
            label: context.label().map(str::to_string),
            depth: context.depth(),
            stop_reason: stop_reason.map(|reason| format!("{reason:?}")),
        });
    }

    fn print_text(&mut self, context: &dyn StreamContext, text: &str) {
        let _ = self.emit_event(RawEvent::AssistantTextDelta {
            label: context.label().map(str::to_string),
            depth: context.depth(),
            text: text.to_string(),
        });
    }

    fn print_thinking(&mut self, context: &dyn StreamContext, text: &str) {
        let _ = self.emit_event(RawEvent::ThinkingDelta {
            label: context.label().map(str::to_string),
            depth: context.depth(),
            text: text.to_string(),
        });
    }

    fn print_error(&mut self, context: &dyn StreamContext, error: &str) {
        let _ = self.emit_event(RawEvent::Error {
            label: context.label().map(str::to_string),
            depth: context.depth(),
            message: error.to_string(),
        });
    }

    fn print_info(&mut self, context: &dyn StreamContext, info: &str) {
        let _ = self.emit_event(RawEvent::Info {
            label: context.label().map(str::to_string),
            depth: context.depth(),
            message: info.to_string(),
        });
    }

    fn start_tool_use(&mut self, context: &dyn StreamContext, name: &str, id: &str) {
        let _ = self.emit_event(RawEvent::ToolUseStart {
            label: context.label().map(str::to_string),
            depth: context.depth(),
            name: name.to_string(),
            tool_use_id: id.to_string(),
        });
    }

    fn print_tool_input(&mut self, context: &dyn StreamContext, partial_json: &str) {
        let _ = self.emit_event(RawEvent::ToolInputDelta {
            label: context.label().map(str::to_string),
            depth: context.depth(),
            partial_json: partial_json.to_string(),
        });
    }

    fn finish_tool_use(&mut self, context: &dyn StreamContext) {
        let _ = self.emit_event(RawEvent::ToolUseEnd {
            label: context.label().map(str::to_string),
            depth: context.depth(),
        });
    }

    fn start_tool_result(
        &mut self,
        context: &dyn StreamContext,
        tool_use_id: &str,
        is_error: bool,
    ) {
        let _ = self.emit_event(RawEvent::ToolResultStart {
            label: context.label().map(str::to_string),
            depth: context.depth(),
            tool_use_id: tool_use_id.to_string(),
            is_error,
        });
    }

    fn print_tool_result_text(&mut self, context: &dyn StreamContext, text: &str) {
        let _ = self.emit_event(RawEvent::ToolResultTextDelta {
            label: context.label().map(str::to_string),
            depth: context.depth(),
            text: text.to_string(),
        });
    }

    fn finish_tool_result(&mut self, context: &dyn StreamContext) {
        let _ = self.emit_event(RawEvent::ToolResultEnd {
            label: context.label().map(str::to_string),
            depth: context.depth(),
        });
    }

    fn finish_response(&mut self, context: &dyn StreamContext) {
        let _ = self.emit_event(RawEvent::ResponseFinish {
            label: context.label().map(str::to_string),
            depth: context.depth(),
        });
    }

    fn print_interrupted(&mut self, context: &dyn StreamContext) {
        let _ = self.emit_event(RawEvent::Interrupted {
            label: context.label().map(str::to_string),
            depth: context.depth(),
        });
    }

    fn read_operator_line(&mut self, prompt: &str) -> io::Result<Option<OperatorLine>> {
        let request_id = self
            .active_request_id
            .as_ref()
            .cloned()
            .unwrap_or_else(|| "prompt".to_string());
        let prompt_id = format!("prompt-{}", self.next_prompt_id);
        self.next_prompt_id = self.next_prompt_id.saturating_add(1);
        self.write_message(&RawServerMessage::Prompt(RawPrompt {
            protocol_version: RAW_PROTOCOL_VERSION,
            request_id: request_id.clone(),
            prompt_id: prompt_id.clone(),
            kind: "confirmation".to_string(),
            message: prompt.to_string(),
            choices: vec!["yes".to_string(), "no".to_string()],
        }))?;
        loop {
            let Some(request) = self.read_request()? else {
                return Ok(Some(OperatorLine::Eof));
            };
            match request.request {
                RawRequest::PromptResponse {
                    prompt_id: response_prompt_id,
                    response,
                } if response_prompt_id == prompt_id => {
                    return Ok(Some(OperatorLine::Line(response)));
                }
                RawRequest::PromptResponse { .. } => {
                    self.write_error_result(
                        &request.request_id,
                        Some("stale_prompt"),
                        "prompt_response does not match the active prompt",
                    )?;
                }
                _ => {
                    self.write_error_result(
                        &request.request_id,
                        Some("busy"),
                        "server is waiting for prompt_response",
                    )?;
                }
            }
        }
    }
}

/// Tool-output observer that forwards chunks onto the raw JSONL stream.
pub struct RawToolOutputObserver<W>
where
    W: Write + Send + 'static,
{
    output: SharedOutput<W>,
}

impl<W> RawToolOutputObserver<W>
where
    W: Write + Send + 'static,
{
    /// Create a new observer that writes raw `tool_output` events.
    pub fn new(output: SharedOutput<W>) -> Self {
        Self { output }
    }
}

impl<W> ToolOutputObserver for RawToolOutputObserver<W>
where
    W: Write + Send + 'static,
{
    fn on_tool_output(&self, event: &ToolOutputEvent) {
        let _ = self
            .output
            .write_message(&RawServerMessage::Event(RawEventEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                request_id: event.request_id.clone(),
                event: RawEvent::ToolOutput {
                    tool_name: event.tool_name.clone(),
                    tool_use_id: event.tool_use_id.clone(),
                    stream: event.stream.clone(),
                    text: event.text.clone(),
                    data_b64: event.data_b64.clone(),
                },
            }));
    }
}

/// Parse a raw request line while preserving any extracted request id.
pub fn parse_request_line(line: &str) -> Result<RawRequestEnvelope, ParsedRequestError> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|err| ParsedRequestError {
            request_id: String::new(),
            code: Some("invalid_request_json".to_string()),
            message: format!("failed to parse raw request JSON: {err}"),
        })?;
    let request_id = value
        .get("request_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    serde_json::from_value(value).map_err(|err| ParsedRequestError {
        request_id,
        code: Some("invalid_request".to_string()),
        message: format!("failed to decode raw request: {err}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_request_line_extracts_request_id_on_decode_error() {
        let err = parse_request_line(r#"{"request_id":"abc","protocol_version":1}"#).unwrap_err();
        assert_eq!(err.request_id, "abc");
        assert_eq!(err.code.as_deref(), Some("invalid_request"));
    }

    #[test]
    fn prompt_wait_rejects_non_prompt_requests() {
        let input = BufReader::new(
            br#"{"protocol_version":1,"request_id":"bad","op":"stats"}
{"protocol_version":1,"request_id":"good","op":"prompt_response","prompt_id":"prompt-1","response":"yes"}
"#
            .as_slice(),
        );
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        server.set_request_id("req-1".to_string());
        let line = server.read_operator_line("Continue?").unwrap();
        assert_eq!(line, Some(OperatorLine::Line("yes".to_string())));

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        assert!(text.contains("\"type\":\"prompt\""));
        assert!(text.contains("\"request_id\":\"bad\""));
        assert!(text.contains("\"code\":\"busy\""));
    }
}
