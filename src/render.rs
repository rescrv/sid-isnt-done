//! Output rendering for chat and agent streaming.
//!
//! This module provides a [`PlainTextRenderer`] that formats tool calls in a
//! human-readable way instead of dumping raw JSON.

use std::io::{self, Stdout, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use claudius::{Renderer, StopReason, StreamContext};

/// ANSI escape code for dim text (used for thinking blocks).
const ANSI_DIM: &str = "\x1b[2m";

/// ANSI escape code for italic text (used for thinking blocks).
const ANSI_ITALIC: &str = "\x1b[3m";

/// ANSI escape code to reset all styling.
const ANSI_RESET: &str = "\x1b[0m";

// Selected from the Zenburn Vim theme's 256-color palette.
/// ANSI escape code for muted metadata.
const ANSI_METADATA: &str = "\x1b[38;5;245m";

/// ANSI escape code for thinking blocks.
const ANSI_THINKING: &str = "\x1b[38;5;245m";

/// ANSI escape code for tool names.
const ANSI_TOOL_LABEL: &str = "\x1b[38;5;109m";

/// ANSI escape code for tool input.
const ANSI_TOOL_INPUT: &str = "\x1b[38;5;229m";

/// ANSI escape code for successful tool result labels.
const ANSI_TOOL_RESULT_OK: &str = "\x1b[38;5;108m";

/// ANSI escape code for errors and failed tool result labels.
const ANSI_ERROR: &str = "\x1b[38;5;217m";

/// ANSI escape code for tool result bodies.
const ANSI_TOOL_RESULT_BODY: &str = "\x1b[38;5;187m";

/// ANSI escape code for field labels within tool input.
const ANSI_FIELD_LABEL: &str = "\x1b[38;5;109m";

/// Plain text renderer with optional ANSI styling.
///
/// This renderer outputs text directly to stdout with optional ANSI escape codes for styling
/// thinking blocks and tool use.  Tool inputs are accumulated and rendered in a human-readable
/// format instead of raw JSON.
pub struct PlainTextRenderer {
    stdout: Stdout,
    use_color: bool,
    in_thinking: bool,
    in_tool_result: bool,
    line_start: bool,
    interrupted: Option<Arc<AtomicBool>>,
    /// Name of the current tool being called.
    current_tool_name: Option<String>,
    /// Accumulated JSON chunks for the current tool input.
    tool_input_buf: String,
    /// Last command-like preview rendered from a partially parsed tool input.
    tool_input_preview: Option<ToolInputPreview>,
}

impl PlainTextRenderer {
    /// Creates a new PlainTextRenderer with ANSI colors enabled.
    pub fn new() -> Self {
        Self {
            stdout: io::stdout(),
            use_color: true,
            in_thinking: false,
            in_tool_result: false,
            line_start: true,
            interrupted: None,
            current_tool_name: None,
            tool_input_buf: String::new(),
            tool_input_preview: None,
        }
    }

    /// Creates a new PlainTextRenderer with specified color setting.
    pub fn with_color(use_color: bool) -> Self {
        Self {
            stdout: io::stdout(),
            use_color,
            in_thinking: false,
            in_tool_result: false,
            line_start: true,
            interrupted: None,
            current_tool_name: None,
            tool_input_buf: String::new(),
            tool_input_preview: None,
        }
    }

    /// Attaches an interrupt flag to the renderer.
    pub fn with_interrupt(mut self, interrupted: Arc<AtomicBool>) -> Self {
        self.interrupted = Some(interrupted);
        self
    }

    /// Creates a new PlainTextRenderer with specified color and interrupt flag.
    pub fn with_color_and_interrupt(use_color: bool, interrupted: Arc<AtomicBool>) -> Self {
        Self::with_color(use_color).with_interrupt(interrupted)
    }

    /// Flushes stdout to ensure immediate display of streamed content.
    fn flush(&mut self) {
        let _ = self.stdout.flush();
    }

    fn reset_thinking(&mut self) {
        if self.in_thinking {
            if self.use_color {
                print!("{ANSI_RESET}");
            }
            // Add newline separator when transitioning from thinking to other content.
            println!();
            self.line_start = true;
            self.in_thinking = false;
        }
    }

    fn reset_tool_result(&mut self) {
        if self.in_tool_result {
            if self.use_color {
                print!("{ANSI_RESET}");
            }
            self.in_tool_result = false;
        }
    }

    fn reset_styles(&mut self) {
        self.reset_thinking();
        self.reset_tool_result();
    }

    /// Writes text with proper indentation based on context depth.
    ///
    /// Each line is prefixed with indentation corresponding to the nesting depth.
    fn write_with_indent(&mut self, context: &dyn StreamContext, text: &str) {
        let prefix = "  ".repeat(context.depth());
        for line in text.split_inclusive('\n') {
            if self.line_start {
                print!("{prefix}");
            }
            print!("{line}");
            self.line_start = line.ends_with('\n');
        }
        self.flush();
    }

    /// Returns the prefix that introduces a thinking block.
    ///
    /// Top-level agents get a leading blank line to separate thinking from prior
    /// output; nested agents (which already carry indentation and a label) do not.
    fn thinking_prefix(context: &dyn StreamContext) -> &'static str {
        if context.depth() == 0 && context.label().is_none() {
            "\n[thinking] "
        } else {
            "[thinking] "
        }
    }

    /// Render the accumulated tool input in a human-readable format.
    fn render_tool_input(&mut self, context: &dyn StreamContext) {
        let json_str = std::mem::take(&mut self.tool_input_buf);
        let tool_name = self.current_tool_name.take();
        let preview = self.tool_input_preview.take();

        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json_str);
        let obj = match parsed {
            Ok(serde_json::Value::Object(map)) => map,
            _ => {
                // Fallback: just print the raw JSON if we can't parse it.
                if self.use_color {
                    self.write_with_indent(context, ANSI_TOOL_INPUT);
                }
                self.write_with_indent(context, &json_str);
                if self.use_color {
                    self.write_with_indent(context, ANSI_RESET);
                }
                self.write_with_indent(context, "\n");
                return;
            }
        };

        let tool = tool_name.as_deref().unwrap_or("");
        match tool {
            "bash" => self.render_bash_input(context, &obj, preview.as_ref()),
            "str_replace_based_edit_tool" => {
                self.render_editor_input(context, &obj, preview.as_ref())
            }
            _ => self.render_generic_input(context, &obj),
        }
    }

    fn render_partial_tool_input_preview(&mut self, context: &dyn StreamContext) {
        let Some(obj) = parse_partial_tool_input_object(&self.tool_input_buf) else {
            return;
        };
        let tool_name = self.current_tool_name.as_deref().unwrap_or("");
        let Some(preview) = tool_input_preview(tool_name, &obj) else {
            return;
        };
        if self.tool_input_preview.as_ref() == Some(&preview) {
            return;
        }

        self.write_tool_input_preview(context, &preview);
        self.tool_input_preview = Some(preview);
    }

    fn write_tool_input_preview(
        &mut self,
        context: &dyn StreamContext,
        preview: &ToolInputPreview,
    ) {
        match preview {
            ToolInputPreview::BashCommand(command) => {
                if self.use_color {
                    self.write_with_indent(
                        context,
                        &format!("{ANSI_TOOL_INPUT}{command}{ANSI_RESET}\n"),
                    );
                } else {
                    self.write_with_indent(context, &format!("{command}\n"));
                }
            }
            ToolInputPreview::EditorCommand { command, path } => {
                if self.use_color {
                    self.write_with_indent(context, &format!("{ANSI_FIELD_LABEL}{command}"));
                    if let Some(path) = path {
                        self.write_with_indent(
                            context,
                            &format!("{ANSI_RESET} {ANSI_TOOL_INPUT}{path}"),
                        );
                    }
                    self.write_with_indent(context, &format!("{ANSI_RESET}\n"));
                } else if let Some(path) = path {
                    self.write_with_indent(context, &format!("{command} {path}\n"));
                } else {
                    self.write_with_indent(context, &format!("{command}\n"));
                }
            }
            ToolInputPreview::GenericField { label, value } => {
                self.write_field(context, label, value);
            }
        }
    }

    /// Render bash tool input: just the command.
    fn render_bash_input(
        &mut self,
        context: &dyn StreamContext,
        obj: &serde_json::Map<String, serde_json::Value>,
        preview: Option<&ToolInputPreview>,
    ) {
        if let Some(serde_json::Value::Bool(true)) = obj.get("restart") {
            if self.use_color {
                self.write_with_indent(context, &format!("{ANSI_METADATA}(restart){ANSI_RESET}\n"));
            } else {
                self.write_with_indent(context, "(restart)\n");
            }
        }

        if let Some(serde_json::Value::String(cmd)) = obj.get("command") {
            if preview.is_some_and(|preview| preview.is_bash_command(cmd)) {
                return;
            }
            if self.use_color {
                self.write_with_indent(context, &format!("{ANSI_TOOL_INPUT}{cmd}{ANSI_RESET}\n"));
            } else {
                self.write_with_indent(context, &format!("{cmd}\n"));
            }
        }
    }

    /// Render str_replace_based_edit_tool input with structured formatting.
    fn render_editor_input(
        &mut self,
        context: &dyn StreamContext,
        obj: &serde_json::Map<String, serde_json::Value>,
        preview: Option<&ToolInputPreview>,
    ) {
        let command = obj
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let path = obj.get("path").and_then(|v| v.as_str()).unwrap_or("???");

        // Show the sub-command and path.
        if preview.is_some_and(|preview| preview.is_editor_command(command, path)) {
            // Already shown while the tool input was streaming.
        } else if self.use_color {
            self.write_with_indent(
                context,
                &format!(
                    "{ANSI_FIELD_LABEL}{command}{ANSI_RESET} {ANSI_TOOL_INPUT}{path}{ANSI_RESET}\n"
                ),
            );
        } else {
            self.write_with_indent(context, &format!("{command} {path}\n"));
        }

        match command {
            "view" => {
                if let Some(range) = obj.get("view_range") {
                    let range_str = format_json_value(range);
                    self.write_field(context, "range", &range_str);
                }
            }
            "create" => {
                if let Some(serde_json::Value::String(text)) = obj.get("file_text") {
                    self.write_content_block(context, "file_text", text);
                }
            }
            "str_replace" => {
                if let Some(serde_json::Value::String(text)) = obj.get("old_str") {
                    self.write_content_block(context, "old_str", text);
                }
                if let Some(serde_json::Value::String(text)) = obj.get("new_str") {
                    self.write_content_block(context, "new_str", text);
                } else if obj.get("new_str") == Some(&serde_json::Value::Null) {
                    self.write_field(context, "new_str", "(delete)");
                }
            }
            "insert" => {
                if let Some(line) = obj.get("insert_line") {
                    self.write_field(context, "insert_line", &format_json_value(line));
                }
                if let Some(serde_json::Value::String(text)) = obj.get("insert_text") {
                    self.write_content_block(context, "insert_text", text);
                }
            }
            _ => {
                // Unknown editor sub-command; show remaining fields generically.
                for (key, val) in obj {
                    if key == "command" || key == "path" {
                        continue;
                    }
                    self.write_field(context, key, &format_json_value(val));
                }
            }
        }
    }

    /// Render a generic tool call by showing each field on its own line.
    fn render_generic_input(
        &mut self,
        context: &dyn StreamContext,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) {
        for (key, val) in obj {
            match val {
                serde_json::Value::String(s) if s.contains('\n') || s.len() > 120 => {
                    self.write_content_block(context, key, s);
                }
                _ => {
                    self.write_field(context, key, &format_json_value(val));
                }
            }
        }
    }

    /// Write a single-line `key: value` field.
    fn write_field(&mut self, context: &dyn StreamContext, label: &str, value: &str) {
        if self.use_color {
            self.write_with_indent(
                context,
                &format!(
                    "{ANSI_FIELD_LABEL}{label}:{ANSI_RESET} {ANSI_TOOL_INPUT}{value}{ANSI_RESET}\n"
                ),
            );
        } else {
            self.write_with_indent(context, &format!("{label}: {value}\n"));
        }
    }

    /// Write a multi-line content block with a label header.
    fn write_content_block(&mut self, context: &dyn StreamContext, label: &str, content: &str) {
        if self.use_color {
            self.write_with_indent(
                context,
                &format!("{ANSI_FIELD_LABEL}{label}:{ANSI_RESET}\n"),
            );
            self.write_with_indent(context, ANSI_TOOL_INPUT);
            self.write_with_indent(context, content);
            if !content.ends_with('\n') {
                self.write_with_indent(context, "\n");
            }
            self.write_with_indent(context, ANSI_RESET);
        } else {
            self.write_with_indent(context, &format!("{label}:\n"));
            self.write_with_indent(context, content);
            if !content.ends_with('\n') {
                self.write_with_indent(context, "\n");
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ToolInputPreview {
    BashCommand(String),
    EditorCommand {
        command: String,
        path: Option<String>,
    },
    GenericField {
        label: String,
        value: String,
    },
}

impl ToolInputPreview {
    fn is_bash_command(&self, command: &str) -> bool {
        matches!(self, Self::BashCommand(previewed) if previewed == command)
    }

    fn is_editor_command(&self, command: &str, path: &str) -> bool {
        matches!(
            self,
            Self::EditorCommand {
                command: previewed_command,
                path: Some(previewed_path),
            } if previewed_command == command && previewed_path == path
        )
    }
}

fn parse_partial_tool_input_object(
    json_str: &str,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut candidate = String::with_capacity(json_str.len() + 1);
    candidate.push_str(json_str);
    candidate.push('}');

    let parsed: serde_json::Value = serde_json::from_str(&candidate).ok()?;
    match parsed {
        serde_json::Value::Object(map) => Some(map),
        _ => None,
    }
}

fn tool_input_preview(
    tool_name: &str,
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<ToolInputPreview> {
    match tool_name {
        "bash" => obj
            .get("command")
            .and_then(|value| value.as_str())
            .map(|command| ToolInputPreview::BashCommand(command.to_string())),
        "str_replace_based_edit_tool" => {
            let command = obj.get("command").and_then(|value| value.as_str())?;
            let path = obj
                .get("path")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            Some(ToolInputPreview::EditorCommand {
                command: command.to_string(),
                path,
            })
        }
        _ => {
            for label in ["command", "cmd"] {
                if let Some(value) = obj.get(label).and_then(|value| value.as_str()) {
                    return Some(ToolInputPreview::GenericField {
                        label: label.to_string(),
                        value: value.to_string(),
                    });
                }
            }
            obj.get("path")
                .and_then(|value| value.as_str())
                .map(|path| ToolInputPreview::GenericField {
                    label: "path".to_string(),
                    value: path.to_string(),
                })
        }
    }
}

/// Format a JSON value for display.  Strings are unquoted; everything else uses compact JSON.
fn format_json_value(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        _ => serde_json::to_string(val).unwrap_or_default(),
    }
}

impl Default for PlainTextRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for PlainTextRenderer {
    fn start_agent(&mut self, context: &dyn StreamContext) {
        let Some(label) = context.label() else {
            return;
        };
        self.reset_styles();
        self.write_with_indent(context, &format!("[agent: {label}]\n"));
    }

    fn finish_agent(&mut self, context: &dyn StreamContext, stop_reason: Option<&StopReason>) {
        let Some(label) = context.label() else {
            return;
        };
        self.reset_styles();
        if let Some(stop_reason) = stop_reason {
            self.write_with_indent(
                context,
                &format!("[agent: {label} done: {stop_reason:?}]\n"),
            );
        } else {
            self.write_with_indent(context, &format!("[agent: {label} done]\n"));
        }
    }

    fn print_text(&mut self, context: &dyn StreamContext, text: &str) {
        self.reset_styles();
        self.write_with_indent(context, text);
    }

    fn print_thinking(&mut self, context: &dyn StreamContext, text: &str) {
        if self.use_color {
            if !self.in_thinking {
                self.write_with_indent(context, ANSI_THINKING);
                self.write_with_indent(context, ANSI_DIM);
                self.write_with_indent(context, ANSI_ITALIC);
                self.write_with_indent(context, Self::thinking_prefix(context));
                self.in_thinking = true;
            }
            self.write_with_indent(context, text);
        } else {
            if !self.in_thinking {
                self.write_with_indent(context, Self::thinking_prefix(context));
                self.in_thinking = true;
            }
            self.write_with_indent(context, text);
        }
    }

    fn print_error(&mut self, context: &dyn StreamContext, error: &str) {
        self.reset_styles();
        if context.depth() == 0 && context.label().is_none() {
            if self.use_color {
                eprintln!("\n{ANSI_ERROR}{error}{ANSI_RESET}");
            } else {
                eprintln!("\n{error}");
            }
        } else {
            if self.use_color {
                self.write_with_indent(context, &format!("\n{ANSI_ERROR}{error}{ANSI_RESET}\n"));
            } else {
                self.write_with_indent(context, &format!("\n{error}\n"));
            }
        }
    }

    fn print_info(&mut self, context: &dyn StreamContext, info: &str) {
        self.reset_styles();
        if context.depth() == 0 && context.label().is_none() {
            println!("{info}");
            self.line_start = true;
            self.flush();
        } else {
            self.write_with_indent(context, &format!("{info}\n"));
        }
    }

    fn start_tool_use(&mut self, context: &dyn StreamContext, name: &str, id: &str) {
        self.reset_styles();

        if self.use_color {
            self.write_with_indent(
                context,
                &format!(
                    "\n{ANSI_TOOL_LABEL}[tool: {name}]{ANSI_RESET} {ANSI_METADATA}{ANSI_DIM}({id}){ANSI_RESET}\n"
                ),
            );
        } else {
            self.write_with_indent(context, &format!("\n[tool: {name}] ({id})\n"));
        }

        // Start accumulating tool input.
        self.current_tool_name = Some(name.to_string());
        self.tool_input_buf.clear();
        self.tool_input_preview = None;
    }

    fn print_tool_input(&mut self, context: &dyn StreamContext, partial_json: &str) {
        // Accumulate and opportunistically render a command-like preview.  The
        // final structured input is still rendered at tool-use end.
        self.tool_input_buf.push_str(partial_json);
        self.render_partial_tool_input_preview(context);
    }

    fn finish_tool_use(&mut self, context: &dyn StreamContext) {
        // Render the accumulated input in human-readable form.
        self.render_tool_input(context);
    }

    fn start_tool_result(
        &mut self,
        context: &dyn StreamContext,
        tool_use_id: &str,
        is_error: bool,
    ) {
        self.reset_styles();
        self.in_tool_result = true;
        if self.use_color {
            let label_color = if is_error {
                ANSI_ERROR
            } else {
                ANSI_TOOL_RESULT_OK
            };
            let status = if is_error { "error" } else { "ok" };
            self.write_with_indent(
                context,
                &format!(
                    "\n{label_color}[tool result: {tool_use_id} ({status})]{ANSI_RESET}\n{ANSI_TOOL_RESULT_BODY}"
                ),
            );
        } else if is_error {
            self.write_with_indent(context, &format!("\n[tool result: {tool_use_id} error]\n"));
        } else {
            self.write_with_indent(context, &format!("\n[tool result: {tool_use_id}]\n"));
        }
    }

    fn print_tool_result_text(&mut self, context: &dyn StreamContext, text: &str) {
        self.write_with_indent(context, text);
    }

    fn finish_tool_result(&mut self, context: &dyn StreamContext) {
        self.reset_tool_result();
        self.write_with_indent(context, "\n");
    }

    fn finish_response(&mut self, context: &dyn StreamContext) {
        self.reset_styles();
        self.write_with_indent(context, "\n");
    }

    fn print_interrupted(&mut self, context: &dyn StreamContext) {
        self.reset_styles();
        let message = if context.depth() == 0 && context.label().is_none() {
            "\n[interrupted]\n"
        } else {
            "[interrupted]\n"
        };
        self.write_with_indent(context, message);
    }

    fn should_interrupt(&self) -> bool {
        self.interrupted
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_default_has_color() {
        let renderer = PlainTextRenderer::new();
        assert!(renderer.use_color);
    }

    #[test]
    fn renderer_without_color() {
        let renderer = PlainTextRenderer::with_color(false);
        assert!(!renderer.use_color);
    }

    #[test]
    fn renderer_without_input_returns_none() {
        let mut renderer = PlainTextRenderer::new();
        assert_eq!(renderer.read_operator_line("> ").unwrap(), None);
    }

    #[test]
    fn render_bash_tool_input() {
        // Verify bash formatting doesn't panic and accumulates correctly.
        let mut renderer = PlainTextRenderer::with_color(false);
        renderer.start_tool_use(&(), "bash", "test_id");
        renderer.print_tool_input(&(), r#"{"command":"ls -la","restart":false}"#);
        renderer.finish_tool_use(&());
    }

    #[test]
    fn render_bash_restart() {
        let mut renderer = PlainTextRenderer::with_color(false);
        renderer.start_tool_use(&(), "bash", "test_id");
        renderer.print_tool_input(&(), r#"{"command":"echo hi","restart":true}"#);
        renderer.finish_tool_use(&());
    }

    #[test]
    fn render_editor_str_replace() {
        let mut renderer = PlainTextRenderer::with_color(false);
        renderer.start_tool_use(&(), "str_replace_based_edit_tool", "test_id");
        renderer.print_tool_input(
            &(),
            r#"{"command":"str_replace","path":"/src/main.rs","old_str":"hello","new_str":"world"}"#,
        );
        renderer.finish_tool_use(&());
    }

    #[test]
    fn render_editor_view() {
        let mut renderer = PlainTextRenderer::with_color(false);
        renderer.start_tool_use(&(), "str_replace_based_edit_tool", "test_id");
        renderer.print_tool_input(
            &(),
            r#"{"command":"view","path":"/src/main.rs","view_range":[1,20]}"#,
        );
        renderer.finish_tool_use(&());
    }

    #[test]
    fn render_editor_create() {
        let mut renderer = PlainTextRenderer::with_color(false);
        renderer.start_tool_use(&(), "str_replace_based_edit_tool", "test_id");
        renderer.print_tool_input(
            &(),
            r#"{"command":"create","path":"/src/new.rs","file_text":"fn main() {\n    println!(\"hello\");\n}\n"}"#,
        );
        renderer.finish_tool_use(&());
    }

    #[test]
    fn render_editor_insert() {
        let mut renderer = PlainTextRenderer::with_color(false);
        renderer.start_tool_use(&(), "str_replace_based_edit_tool", "test_id");
        renderer.print_tool_input(
            &(),
            r#"{"command":"insert","path":"/src/main.rs","insert_line":5,"insert_text":"use std::io;\n"}"#,
        );
        renderer.finish_tool_use(&());
    }

    #[test]
    fn render_generic_tool() {
        let mut renderer = PlainTextRenderer::with_color(false);
        renderer.start_tool_use(&(), "some_custom_tool", "test_id");
        renderer.print_tool_input(&(), r#"{"query":"find me something","limit":10}"#);
        renderer.finish_tool_use(&());
    }

    #[test]
    fn render_invalid_json_fallback() {
        let mut renderer = PlainTextRenderer::with_color(false);
        renderer.start_tool_use(&(), "bash", "test_id");
        renderer.print_tool_input(&(), "this is not json");
        renderer.finish_tool_use(&());
    }

    #[test]
    fn partial_tool_input_parse_appends_one_closing_brace() {
        let parsed = parse_partial_tool_input_object(r#"{"command":"echo hi""#).unwrap();
        assert_eq!(
            parsed.get("command").and_then(serde_json::Value::as_str),
            Some("echo hi")
        );

        assert!(parse_partial_tool_input_object(r#"{"command":"echo hi","#).is_none());
        assert!(parse_partial_tool_input_object(r#"{"command":"echo hi"}"#).is_none());
    }

    #[test]
    fn partial_tool_input_preview_extracts_bash_command() {
        let parsed = parse_partial_tool_input_object(r#"{"command":"echo hi""#).unwrap();
        assert_eq!(
            tool_input_preview("bash", &parsed),
            Some(ToolInputPreview::BashCommand("echo hi".to_string()))
        );
    }

    #[test]
    fn partial_tool_input_preview_extracts_editor_command_and_path() {
        let command_only = parse_partial_tool_input_object(r#"{"command":"create""#).unwrap();
        assert_eq!(
            tool_input_preview("str_replace_based_edit_tool", &command_only),
            Some(ToolInputPreview::EditorCommand {
                command: "create".to_string(),
                path: None,
            })
        );

        let with_path =
            parse_partial_tool_input_object(r#"{"command":"create","path":"/src/new.rs""#).unwrap();
        assert_eq!(
            tool_input_preview("str_replace_based_edit_tool", &with_path),
            Some(ToolInputPreview::EditorCommand {
                command: "create".to_string(),
                path: Some("/src/new.rs".to_string()),
            })
        );
    }

    #[test]
    fn partial_tool_input_preview_extracts_generic_command_field() {
        let parsed = parse_partial_tool_input_object(r#"{"cmd":"rg foo""#).unwrap();
        assert_eq!(
            tool_input_preview("custom_tool", &parsed),
            Some(ToolInputPreview::GenericField {
                label: "cmd".to_string(),
                value: "rg foo".to_string(),
            })
        );
    }

    #[test]
    fn render_chunked_input() {
        // Simulate streaming chunks.
        let mut renderer = PlainTextRenderer::with_color(false);
        renderer.start_tool_use(&(), "bash", "test_id");
        renderer.print_tool_input(&(), r#"{"comm"#);
        renderer.print_tool_input(&(), r#"and":"echo hello"#);
        renderer.print_tool_input(&(), r#"}"#);
        renderer.finish_tool_use(&());
    }
}
