//! Raw-mode transport utilities for `sid --raw`.
//!
//! This module owns the JSONL stdin/stdout transport, prompt handling, and
//! external-tool output forwarding used by raw-mode frontends.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use claudius::{OperatorLine, Renderer, StopReason, StreamContext};

use crate::ralph::runner::ScriptOutputSink;
use crate::raw_protocol::{
    RAW_PROTOCOL_VERSION, RawAcceptedRequest, RawEvent, RawEventEnvelope, RawPrompt, RawPromptAck,
    RawRequest, RawRequestEnvelope, RawResultEnvelope, RawServerError, RawServerMessage,
    ToolOutputEvent, ToolOutputObserver, UsageReportEvent, UsageReportObserver,
};

/// Raw request input with optional nonblocking line polling.
///
/// Plain stdio raw mode cannot be polled portably, so the default
/// implementation reports no available line.  Reconnectable transports
/// override this to let streamed renderers observe client interrupts.
pub trait RawInput: BufRead + Send {
    /// Try to read one complete request line without blocking.
    fn try_read_line(&mut self) -> io::Result<Option<String>> {
        Ok(None)
    }
}

impl<T> RawInput for BufReader<T> where T: Read + Send {}

impl<T> RawInput for Box<T>
where
    T: RawInput + ?Sized,
{
    fn try_read_line(&mut self) -> io::Result<Option<String>> {
        (**self).try_read_line()
    }
}

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
    state: Arc<StdMutex<SharedOutputState<W>>>,
}

struct SharedOutputState<W>
where
    W: Write + Send + 'static,
{
    writer: W,
    next_sequence: u64,
}

impl<W> Clone for SharedOutput<W>
where
    W: Write + Send + 'static,
{
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
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
            state: Arc::new(StdMutex::new(SharedOutputState {
                writer,
                next_sequence: 1,
            })),
        }
    }

    /// Write one JSONL server message.
    pub fn write_message(&self, message: &RawServerMessage) -> io::Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("raw output lock poisoned"))?;
        let message = message.clone().with_sequence(state.next_sequence);
        state.next_sequence = state.next_sequence.saturating_add(1);
        let payload = serde_json::to_vec(&message)
            .map_err(|err| io::Error::other(format!("failed to serialize raw message: {err}")))?;
        state.writer.write_all(&payload)?;
        state.writer.write_all(b"\n")?;
        state.writer.flush()
    }

    /// Execute `f` with a mutable borrow of the underlying writer.
    #[cfg(test)]
    pub(crate) fn with_writer<T>(&self, f: impl FnOnce(&W) -> T) -> T {
        let state = self.state.lock().expect("raw output lock poisoned");
        f(&state.writer)
    }
}

/// Line-oriented raw JSONL server that also implements `claudius::Renderer`.
pub struct RawServer<R, W>
where
    R: RawInput,
    W: Write + Send + 'static,
{
    input: StdMutex<R>,
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
    R: RawInput,
    W: Write + Send + 'static,
{
    /// Create a raw server from arbitrary reader/writer handles.
    pub fn new(input: R, output: W) -> Self {
        Self {
            input: StdMutex::new(input),
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

    /// Write replayable context for an accepted request when useful.
    pub fn write_accepted_request(&self, request: &RawRequestEnvelope) -> io::Result<()> {
        if let Some(request) = RawAcceptedRequest::from_envelope(request) {
            self.write_message(&RawServerMessage::Request(request))?;
        }
        Ok(())
    }

    /// Write a successful terminal result.
    pub fn write_ok_result(
        &self,
        request_id: &str,
        data: Option<serde_json::Value>,
    ) -> io::Result<()> {
        self.write_message(&RawServerMessage::Result(RawResultEnvelope {
            protocol_version: RAW_PROTOCOL_VERSION,
            sequence: 0,
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
            sequence: 0,
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
            let read = {
                let mut input = self
                    .input
                    .lock()
                    .map_err(|_| io::Error::other("raw input lock poisoned"))?;
                input.read_line(&mut line)?
            };
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
            sequence: 0,
            request_id,
            event,
        }))
    }

    /// Try to read one complete request without blocking.
    ///
    /// Active long-running requests use this to accept interrupts while
    /// rejecting unrelated requests as busy.  Plain stdio raw mode usually
    /// returns `Ok(None)` because its input cannot be polled portably.
    pub fn try_read_request(
        &self,
    ) -> io::Result<Option<Result<RawRequestEnvelope, ParsedRequestError>>> {
        loop {
            let line = {
                let mut input = self
                    .input
                    .lock()
                    .map_err(|_| io::Error::other("raw input lock poisoned"))?;
                input.try_read_line()?
            };
            let Some(line) = line else {
                return Ok(None);
            };
            if line.trim().is_empty() {
                continue;
            }
            return Ok(Some(parse_request_line(&line)));
        }
    }

    fn poll_interrupt(&self) -> io::Result<bool> {
        while let Some(request) = self.try_read_request()? {
            let request = match request {
                Ok(request) => request,
                Err(err) => {
                    self.write_error_result(&err.request_id, err.code.as_deref(), &err.message)?;
                    continue;
                }
            };
            if request.protocol_version != RAW_PROTOCOL_VERSION {
                self.write_error_result(
                    &request.request_id,
                    Some("unsupported_protocol_version"),
                    &format!(
                        "unsupported raw protocol version {}",
                        request.protocol_version
                    ),
                )?;
                continue;
            }
            match request.request {
                RawRequest::Interrupt => {
                    self.write_ok_result(&request.request_id, None)?;
                    return Ok(true);
                }
                _ => {
                    self.write_error_result(
                        &request.request_id,
                        Some("busy"),
                        "server is busy streaming another request",
                    )?;
                }
            }
        }
        Ok(false)
    }
}

impl<R, W> Renderer for RawServer<R, W>
where
    R: RawInput,
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

    fn should_interrupt(&self) -> bool {
        self.poll_interrupt().unwrap_or(false)
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
            sequence: 0,
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
                    self.write_message(&RawServerMessage::PromptAck(RawPromptAck {
                        protocol_version: RAW_PROTOCOL_VERSION,
                        sequence: 0,
                        request_id: request_id.clone(),
                        response_request_id: request.request_id,
                        prompt_id,
                        response: response.clone(),
                    }))?;
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

/// A renderer that writes model stream events to a raw JSONL output sink.
///
/// Unlike [`RawServer`], this renderer does not own request input.  It is used
/// by work running on helper threads, where a separate owner polls the raw
/// input and flips the shared interrupt flag.
pub struct RawEventRenderer<W>
where
    W: Write + Send + 'static,
{
    output: SharedOutput<W>,
    request_id: String,
    label: String,
    interrupted: Arc<std::sync::atomic::AtomicBool>,
}

impl<W> RawEventRenderer<W>
where
    W: Write + Send + 'static,
{
    /// Create a renderer for events associated with `request_id`.
    pub fn new(
        output: SharedOutput<W>,
        request_id: impl Into<String>,
        label: impl Into<String>,
        interrupted: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        Self {
            output,
            request_id: request_id.into(),
            label: label.into(),
            interrupted,
        }
    }

    fn emit_event(&self, event: RawEvent) {
        let _ = self
            .output
            .write_message(&RawServerMessage::Event(RawEventEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                sequence: 0,
                request_id: self.request_id.clone(),
                event,
            }));
    }

    fn label(&self, context: &dyn StreamContext) -> Option<String> {
        context
            .label()
            .map(str::to_string)
            .or_else(|| Some(self.label.clone()))
    }
}

impl<W> Renderer for RawEventRenderer<W>
where
    W: Write + Send + 'static,
{
    fn start_agent(&mut self, context: &dyn StreamContext) {
        self.emit_event(RawEvent::AgentStart {
            label: self.label(context),
            depth: context.depth(),
        });
    }

    fn finish_agent(&mut self, context: &dyn StreamContext, stop_reason: Option<&StopReason>) {
        self.emit_event(RawEvent::AgentFinish {
            label: self.label(context),
            depth: context.depth(),
            stop_reason: stop_reason.map(|reason| format!("{reason:?}")),
        });
    }

    fn print_text(&mut self, context: &dyn StreamContext, text: &str) {
        self.emit_event(RawEvent::AssistantTextDelta {
            label: self.label(context),
            depth: context.depth(),
            text: text.to_string(),
        });
    }

    fn print_thinking(&mut self, context: &dyn StreamContext, text: &str) {
        self.emit_event(RawEvent::ThinkingDelta {
            label: self.label(context),
            depth: context.depth(),
            text: text.to_string(),
        });
    }

    fn print_error(&mut self, context: &dyn StreamContext, error: &str) {
        self.emit_event(RawEvent::Error {
            label: self.label(context),
            depth: context.depth(),
            message: error.to_string(),
        });
    }

    fn print_info(&mut self, context: &dyn StreamContext, info: &str) {
        self.emit_event(RawEvent::Info {
            label: self.label(context),
            depth: context.depth(),
            message: info.to_string(),
        });
    }

    fn start_tool_use(&mut self, context: &dyn StreamContext, name: &str, id: &str) {
        self.emit_event(RawEvent::ToolUseStart {
            label: self.label(context),
            depth: context.depth(),
            name: name.to_string(),
            tool_use_id: id.to_string(),
        });
    }

    fn print_tool_input(&mut self, context: &dyn StreamContext, partial_json: &str) {
        self.emit_event(RawEvent::ToolInputDelta {
            label: self.label(context),
            depth: context.depth(),
            partial_json: partial_json.to_string(),
        });
    }

    fn finish_tool_use(&mut self, context: &dyn StreamContext) {
        self.emit_event(RawEvent::ToolUseEnd {
            label: self.label(context),
            depth: context.depth(),
        });
    }

    fn start_tool_result(
        &mut self,
        context: &dyn StreamContext,
        tool_use_id: &str,
        is_error: bool,
    ) {
        self.emit_event(RawEvent::ToolResultStart {
            label: self.label(context),
            depth: context.depth(),
            tool_use_id: tool_use_id.to_string(),
            is_error,
        });
    }

    fn print_tool_result_text(&mut self, context: &dyn StreamContext, text: &str) {
        self.emit_event(RawEvent::ToolResultTextDelta {
            label: self.label(context),
            depth: context.depth(),
            text: text.to_string(),
        });
    }

    fn finish_tool_result(&mut self, context: &dyn StreamContext) {
        self.emit_event(RawEvent::ToolResultEnd {
            label: self.label(context),
            depth: context.depth(),
        });
    }

    fn finish_response(&mut self, context: &dyn StreamContext) {
        self.emit_event(RawEvent::ResponseFinish {
            label: self.label(context),
            depth: context.depth(),
        });
    }

    fn print_interrupted(&mut self, context: &dyn StreamContext) {
        self.emit_event(RawEvent::Interrupted {
            label: self.label(context),
            depth: context.depth(),
        });
    }

    fn should_interrupt(&self) -> bool {
        self.interrupted.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn read_operator_line(&mut self, _prompt: &str) -> io::Result<Option<OperatorLine>> {
        // Ralph child-session renderers do not own operator input; the raw
        // request owner handles top-level prompts.
        Ok(None)
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
                sequence: 0,
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

/// Sink that forwards ralph shell stdout/stderr onto the raw JSONL stream.
pub struct RawScriptOutputSink<W>
where
    W: Write + Send + 'static,
{
    output: SharedOutput<W>,
    request_id: String,
}

impl<W> RawScriptOutputSink<W>
where
    W: Write + Send + 'static,
{
    /// Create a script-output sink associated with `request_id`.
    pub fn new(output: SharedOutput<W>, request_id: impl Into<String>) -> Self {
        Self {
            output,
            request_id: request_id.into(),
        }
    }
}

impl<W> ScriptOutputSink for RawScriptOutputSink<W>
where
    W: Write + Send + 'static,
{
    fn on_script_output(&self, stream: &str, data: &[u8]) {
        let (text, data_b64) = match std::str::from_utf8(data) {
            Ok(text) => (Some(text.to_string()), None),
            Err(_) => (None, Some(BASE64_STANDARD.encode(data))),
        };
        let _ = self
            .output
            .write_message(&RawServerMessage::Event(RawEventEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                sequence: 0,
                request_id: self.request_id.clone(),
                event: RawEvent::ToolOutput {
                    tool_name: "ralph".to_string(),
                    tool_use_id: "ralph-script".to_string(),
                    stream: stream.to_string(),
                    text,
                    data_b64,
                },
            }));
    }
}

/// Usage observer that forwards model usage reports onto the raw JSONL stream.
pub struct RawUsageReportObserver<W>
where
    W: Write + Send + 'static,
{
    output: SharedOutput<W>,
    request_id: String,
}

impl<W> RawUsageReportObserver<W>
where
    W: Write + Send + 'static,
{
    /// Create a new observer that writes usage reports as raw `info` events.
    pub fn new(output: SharedOutput<W>, request_id: impl Into<String>) -> Self {
        Self {
            output,
            request_id: request_id.into(),
        }
    }
}

impl<W> UsageReportObserver for RawUsageReportObserver<W>
where
    W: Write + Send + 'static,
{
    fn on_usage_report(&self, event: &UsageReportEvent) {
        for message in [&event.token_line, &event.usage_line] {
            let _ = self
                .output
                .write_message(&RawServerMessage::Event(RawEventEnvelope {
                    protocol_version: RAW_PROTOCOL_VERSION,
                    sequence: 0,
                    request_id: self.request_id.clone(),
                    event: RawEvent::Info {
                        label: None,
                        depth: 0,
                        message: message.clone(),
                    },
                }));
        }
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
    use std::collections::VecDeque;

    struct PollInput {
        lines: VecDeque<String>,
    }

    impl PollInput {
        fn new(lines: &[&str]) -> Self {
            Self {
                lines: lines.iter().map(|line| line.to_string()).collect(),
            }
        }
    }

    impl Read for PollInput {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            Ok(0)
        }
    }

    impl BufRead for PollInput {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            Ok(&[])
        }

        fn consume(&mut self, _amt: usize) {}
    }

    impl RawInput for PollInput {
        fn try_read_line(&mut self) -> io::Result<Option<String>> {
            Ok(self.lines.pop_front())
        }
    }

    // ── parse_request_line ───────────────────────────────────────────────

    #[test]
    fn parse_request_line_extracts_request_id_on_decode_error() {
        let err = parse_request_line(r#"{"request_id":"abc","protocol_version":1}"#).unwrap_err();
        assert_eq!(err.request_id, "abc");
        assert_eq!(err.code.as_deref(), Some("invalid_request"));
    }

    #[test]
    fn parse_request_line_rejects_invalid_json() {
        let err = parse_request_line("not json at all").unwrap_err();
        assert_eq!(err.request_id, "");
        assert_eq!(err.code.as_deref(), Some("invalid_request_json"));
    }

    #[test]
    fn parse_request_line_succeeds_for_valid_request() {
        let line = r#"{"protocol_version":4,"request_id":"r-1","op":"stats"}"#;
        let envelope = parse_request_line(line).unwrap();
        assert_eq!(envelope.request_id, "r-1");
        assert_eq!(envelope.request, RawRequest::Stats);
    }

    #[test]
    fn parse_request_line_succeeds_for_user_turn() {
        let line =
            r#"{"protocol_version":4,"request_id":"r-2","op":"user_turn","text":"hello world"}"#;
        let envelope = parse_request_line(line).unwrap();
        assert_eq!(
            envelope.request,
            RawRequest::UserTurn {
                text: "hello world".to_string(),
            }
        );
    }

    #[test]
    fn parse_request_line_succeeds_for_interrupt() {
        let line = r#"{"protocol_version":4,"request_id":"r-int","op":"interrupt"}"#;
        let envelope = parse_request_line(line).unwrap();
        assert_eq!(envelope.request_id, "r-int");
        assert_eq!(envelope.request, RawRequest::Interrupt);
    }

    #[test]
    fn parse_request_line_missing_request_id_gives_empty_string() {
        let err = parse_request_line(r#"{"protocol_version":4}"#).unwrap_err();
        assert_eq!(err.request_id, "");
    }

    // ── write_message / sequences ────────────────────────────────────────

    #[test]
    fn write_message_assigns_monotonic_sequences() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let server = RawServer::new(input, output);
        server.write_ok_result("one", None).unwrap();
        server.write_ok_result("two", None).unwrap();

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        let sequences = text
            .lines()
            .map(|line| {
                serde_json::from_str::<serde_json::Value>(line).unwrap()["sequence"]
                    .as_u64()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(sequences, vec![1, 2]);
    }

    #[test]
    fn write_accepted_request_emits_user_turn_marker() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let server = RawServer::new(input, output);
        server
            .write_accepted_request(&RawRequestEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                request_id: "turn-1".to_string(),
                request: RawRequest::UserTurn {
                    text: "hello replay".to_string(),
                },
            })
            .unwrap();

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["type"], "request");
        assert_eq!(value["request_id"], "turn-1");
        assert_eq!(value["op"], "user_turn");
        assert_eq!(value["text"], "hello replay");
    }

    #[test]
    fn write_accepted_request_emits_ralph_marker() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let server = RawServer::new(input, output);
        server
            .write_accepted_request(&RawRequestEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                request_id: "ralph-1".to_string(),
                request: RawRequest::RunRalphInline {
                    script: "echo done".to_string(),
                },
            })
            .unwrap();

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["type"], "request");
        assert_eq!(value["request_id"], "ralph-1");
        assert_eq!(value["op"], "run_ralph_inline");
        assert_eq!(value["script"], "echo done");
    }

    #[test]
    fn write_accepted_request_skips_stats_request() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let server = RawServer::new(input, output);
        server
            .write_accepted_request(&RawRequestEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                request_id: "stats".to_string(),
                request: RawRequest::Stats,
            })
            .unwrap();

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        assert!(text.is_empty());
    }

    // ── write_ok_result / write_error_result ─────────────────────────────

    #[test]
    fn write_ok_result_emits_result_type_with_ok_true() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let server = RawServer::new(input, output);
        server
            .write_ok_result("r-1", Some(serde_json::json!({"key": "value"})))
            .unwrap();

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["type"], "result");
        assert_eq!(value["ok"], true);
        assert_eq!(value["request_id"], "r-1");
        assert_eq!(value["data"]["key"], "value");
    }

    #[test]
    fn write_error_result_emits_result_type_with_ok_false() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let server = RawServer::new(input, output);
        server
            .write_error_result("r-2", Some("test_code"), "something broke")
            .unwrap();

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["type"], "result");
        assert_eq!(value["ok"], false);
        assert_eq!(value["request_id"], "r-2");
        assert_eq!(value["error"]["code"], "test_code");
        assert_eq!(value["error"]["message"], "something broke");
    }

    #[test]
    fn write_error_result_without_code() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let server = RawServer::new(input, output);
        server
            .write_error_result("r-3", None, "generic error")
            .unwrap();

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["error"]["code"], serde_json::Value::Null);
        assert_eq!(value["error"]["message"], "generic error");
    }

    // ── read_request ─────────────────────────────────────────────────────

    #[test]
    fn read_request_returns_none_on_eof() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        assert!(server.read_request().unwrap().is_none());
    }

    #[test]
    fn read_request_skips_blank_lines() {
        let input = BufReader::new(
            b"\n\n{\"protocol_version\":4,\"request_id\":\"r-1\",\"op\":\"stats\"}\n".as_slice(),
        );
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        let req = server.read_request().unwrap().unwrap();
        assert_eq!(req.request_id, "r-1");
    }

    #[test]
    fn read_request_reports_malformed_json_and_continues() {
        let input = BufReader::new(
            b"not json\n{\"protocol_version\":4,\"request_id\":\"r-1\",\"op\":\"stats\"}\n"
                .as_slice(),
        );
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        let req = server.read_request().unwrap().unwrap();
        assert_eq!(req.request_id, "r-1");

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        assert!(
            text.contains("invalid_request_json"),
            "expected error for malformed JSON, got: {text}"
        );
    }

    #[test]
    fn read_request_reports_valid_json_with_missing_op_and_continues() {
        let input = BufReader::new(
            b"{\"request_id\":\"bad\",\"protocol_version\":4}\n{\"protocol_version\":4,\"request_id\":\"good\",\"op\":\"shutdown\"}\n"
                .as_slice(),
        );
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        let req = server.read_request().unwrap().unwrap();
        assert_eq!(req.request_id, "good");
        assert_eq!(req.request, RawRequest::Shutdown);

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        assert!(text.contains("\"request_id\":\"bad\""));
        assert!(text.contains("invalid_request"));
    }

    // ── emit_event ───────────────────────────────────────────────────────

    #[test]
    fn emit_event_uses_active_request_id() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        server.set_request_id("req-42".to_string());
        server
            .emit_event(RawEvent::Info {
                label: None,
                depth: 0,
                message: "test".to_string(),
            })
            .unwrap();

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["request_id"], "req-42");
        assert_eq!(value["type"], "event");
        assert_eq!(value["event"], "info");
    }

    #[test]
    fn emit_event_uses_empty_string_when_no_request_id() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let server = RawServer::new(input, output);
        server
            .emit_event(RawEvent::Info {
                label: None,
                depth: 0,
                message: "test".to_string(),
            })
            .unwrap();

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["request_id"], "");
    }

    #[test]
    fn clear_request_id_reverts_to_empty() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        server.set_request_id("req-1".to_string());
        server.clear_request_id();
        server
            .emit_event(RawEvent::Info {
                label: None,
                depth: 0,
                message: "test".to_string(),
            })
            .unwrap();

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["request_id"], "");
    }

    #[test]
    fn should_interrupt_polls_interrupt_request() {
        let input = PollInput::new(&[
            r#"{"protocol_version":4,"request_id":"interrupt-1","op":"interrupt"}"#,
        ]);
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        server.set_request_id("turn-1".to_string());

        assert!(server.should_interrupt());

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["type"], "result");
        assert_eq!(value["request_id"], "interrupt-1");
        assert_eq!(value["ok"], true);
    }

    #[test]
    fn should_interrupt_rejects_non_interrupt_request_as_busy() {
        let input =
            PollInput::new(&[r#"{"protocol_version":4,"request_id":"stats-1","op":"stats"}"#]);
        let output = Vec::new();
        let server = RawServer::new(input, output);

        assert!(!server.should_interrupt());

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        assert!(text.contains("\"request_id\":\"stats-1\""));
        assert!(text.contains("\"code\":\"busy\""));
    }

    // ── prompt handling ──────────────────────────────────────────────────

    #[test]
    fn prompt_wait_rejects_non_prompt_requests() {
        let input = BufReader::new(
            br#"{"protocol_version":4,"request_id":"bad","op":"stats"}
{"protocol_version":4,"request_id":"good","op":"prompt_response","prompt_id":"prompt-1","response":"yes"}
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
        assert!(text.contains("\"type\":\"prompt_ack\""));
        assert!(text.contains("\"response_request_id\":\"good\""));
        assert!(text.contains("\"response\":\"yes\""));
        assert!(text.contains("\"request_id\":\"bad\""));
        assert!(text.contains("\"code\":\"busy\""));
    }

    #[test]
    fn prompt_wait_rejects_stale_prompt_id() {
        let input = BufReader::new(
            br#"{"protocol_version":4,"request_id":"stale","op":"prompt_response","prompt_id":"prompt-999","response":"yes"}
{"protocol_version":4,"request_id":"correct","op":"prompt_response","prompt_id":"prompt-1","response":"no"}
"#
            .as_slice(),
        );
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        server.set_request_id("req-1".to_string());
        let line = server.read_operator_line("Allow?").unwrap();
        assert_eq!(line, Some(OperatorLine::Line("no".to_string())));

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        assert!(
            text.contains("stale_prompt"),
            "expected stale_prompt error, got: {text}"
        );
    }

    #[test]
    fn prompt_wait_returns_eof_on_input_close() {
        let input = BufReader::new([].as_slice());
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        server.set_request_id("req-1".to_string());
        let line = server.read_operator_line("Continue?").unwrap();
        assert_eq!(line, Some(OperatorLine::Eof));
    }

    #[test]
    fn prompt_ids_increment_across_calls() {
        // First prompt uses prompt-1, second uses prompt-2.
        let input = BufReader::new(
            br#"{"protocol_version":4,"request_id":"a","op":"prompt_response","prompt_id":"prompt-1","response":"yes"}
{"protocol_version":4,"request_id":"b","op":"prompt_response","prompt_id":"prompt-2","response":"no"}
"#
            .as_slice(),
        );
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        server.set_request_id("req-1".to_string());

        let first = server.read_operator_line("First?").unwrap();
        assert_eq!(first, Some(OperatorLine::Line("yes".to_string())));

        let second = server.read_operator_line("Second?").unwrap();
        assert_eq!(second, Some(OperatorLine::Line("no".to_string())));
    }

    #[test]
    fn prompt_uses_fallback_request_id_when_none_set() {
        let input = BufReader::new(
            br#"{"protocol_version":4,"request_id":"resp","op":"prompt_response","prompt_id":"prompt-1","response":"yes"}
"#
            .as_slice(),
        );
        let output = Vec::new();
        let mut server = RawServer::new(input, output);
        // Deliberately do NOT set a request id.
        let line = server.read_operator_line("Continue?").unwrap();
        assert_eq!(line, Some(OperatorLine::Line("yes".to_string())));

        let text = server
            .output
            .with_writer(|writer| String::from_utf8_lossy(writer).into_owned());
        // Prompt should use fallback request_id "prompt".
        assert!(
            text.contains("\"request_id\":\"prompt\""),
            "expected fallback request_id, got: {text}"
        );
    }

    // ── SharedOutput clone ───────────────────────────────────────────────

    #[test]
    fn shared_output_clones_share_sequence_counter() {
        let output: SharedOutput<Vec<u8>> = SharedOutput::new(Vec::new());
        let clone = output.clone();

        output
            .write_message(&RawServerMessage::Result(RawResultEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                sequence: 0,
                request_id: "a".to_string(),
                ok: true,
                data: None,
                error: None,
            }))
            .unwrap();
        clone
            .write_message(&RawServerMessage::Result(RawResultEnvelope {
                protocol_version: RAW_PROTOCOL_VERSION,
                sequence: 0,
                request_id: "b".to_string(),
                ok: true,
                data: None,
                error: None,
            }))
            .unwrap();

        let text = output.with_writer(|w| String::from_utf8_lossy(w).into_owned());
        let sequences: Vec<u64> = text
            .lines()
            .map(|line| {
                serde_json::from_str::<serde_json::Value>(line).unwrap()["sequence"]
                    .as_u64()
                    .unwrap()
            })
            .collect();
        assert_eq!(sequences, vec![1, 2]);
    }

    // ── RawToolOutputObserver ────────────────────────────────────────────

    #[test]
    fn tool_output_observer_emits_tool_output_event() {
        let output: SharedOutput<Vec<u8>> = SharedOutput::new(Vec::new());
        let observer = RawToolOutputObserver::new(output.clone());

        observer.on_tool_output(&ToolOutputEvent {
            request_id: "r-1".to_string(),
            tool_name: "bash".to_string(),
            tool_use_id: "tu-1".to_string(),
            stream: "stdout".to_string(),
            text: Some("hello\n".to_string()),
            data_b64: None,
        });

        let text = output.with_writer(|w| String::from_utf8_lossy(w).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["type"], "event");
        assert_eq!(value["event"], "tool_output");
        assert_eq!(value["request_id"], "r-1");
        assert_eq!(value["tool_name"], "bash");
        assert_eq!(value["tool_use_id"], "tu-1");
        assert_eq!(value["stream"], "stdout");
        assert_eq!(value["text"], "hello\n");
    }

    #[test]
    fn raw_script_output_sink_emits_tool_output_event() {
        let output: SharedOutput<Vec<u8>> = SharedOutput::new(Vec::new());
        let sink = RawScriptOutputSink::new(output.clone(), "ralph-1");

        ScriptOutputSink::on_script_output(&sink, "stderr", b"ralph says hi\n");

        let text = output.with_writer(|w| String::from_utf8_lossy(w).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["type"], "event");
        assert_eq!(value["event"], "tool_output");
        assert_eq!(value["request_id"], "ralph-1");
        assert_eq!(value["tool_name"], "ralph");
        assert_eq!(value["tool_use_id"], "ralph-script");
        assert_eq!(value["stream"], "stderr");
        assert_eq!(value["text"], "ralph says hi\n");
    }

    #[test]
    fn raw_event_renderer_emits_labeled_text_event() {
        let output: SharedOutput<Vec<u8>> = SharedOutput::new(Vec::new());
        let interrupted = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut renderer = RawEventRenderer::new(output.clone(), "ralph-2", "fix", interrupted);

        renderer.print_text(&(), "working");

        let text = output.with_writer(|w| String::from_utf8_lossy(w).into_owned());
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["type"], "event");
        assert_eq!(value["event"], "assistant_text_delta");
        assert_eq!(value["request_id"], "ralph-2");
        assert_eq!(value["label"], "fix");
        assert_eq!(value["text"], "working");
    }

    #[test]
    fn usage_report_observer_emits_info_events() {
        let output: SharedOutput<Vec<u8>> = SharedOutput::new(Vec::new());
        let observer = RawUsageReportObserver::new(output.clone(), "r-usage");

        observer.on_usage_report(&UsageReportEvent {
            token_line: "[tokens: input=1 cache_creation=2 cached_input=3 output=4]".to_string(),
            usage_line: "[usage: test]".to_string(),
        });

        let text = output.with_writer(|w| String::from_utf8_lossy(w).into_owned());
        let values = text
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0]["type"], "event");
        assert_eq!(values[0]["event"], "info");
        assert_eq!(values[0]["request_id"], "r-usage");
        assert_eq!(
            values[0]["message"],
            "[tokens: input=1 cache_creation=2 cached_input=3 output=4]"
        );
        assert_eq!(values[1]["type"], "event");
        assert_eq!(values[1]["event"], "info");
        assert_eq!(values[1]["request_id"], "r-usage");
        assert_eq!(values[1]["message"], "[usage: test]");
    }
}
