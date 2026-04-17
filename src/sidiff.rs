use std::cmp::{max, min};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
use std::process::{Command, Stdio};
use tree_sitter::{Language, Node, Parser};

const DEFAULT_PAGER: &str = "less -R";
const MOVE_MIN_LINES: usize = 3;
const MAX_ROLE_GROUPS: usize = 8;
const ROLE_GLYPHS: [&str; MAX_ROLE_GROUPS] = [
    "\u{2460}", "\u{2461}", "\u{2462}", "\u{2463}", "\u{2464}", "\u{2465}", "\u{2466}", "\u{2467}",
];

const RESET: &str = "\x1b[0m";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhitespaceMode {
    Hide,
    Muted,
    Normal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TintIntensity {
    Low,
    Medium,
    High,
}

impl TintIntensity {
    fn alpha(self) -> f32 {
        match self {
            Self::Low => 0.18,
            Self::Medium => 0.30,
            Self::High => 0.44,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SidiffOptions {
    pub use_color: bool,
    pub whitespace_mode: WhitespaceMode,
    pub tint_intensity: TintIntensity,
}

impl Default for SidiffOptions {
    fn default() -> Self {
        Self {
            use_color: true,
            whitespace_mode: WhitespaceMode::Muted,
            tint_intensity: TintIntensity::Medium,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Diff {
    pub preamble: Vec<String>,
    pub files: Vec<DiffFile>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiffFile {
    pub old_path: Option<String>,
    pub new_path: Option<String>,
    pub header: Vec<String>,
    pub hunks: Vec<Hunk>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Hunk {
    pub header: String,
    pub old_start: usize,
    pub old_count: usize,
    pub new_start: usize,
    pub new_count: usize,
    pub section: String,
    pub lines: Vec<DiffLine>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffLine {
    pub op: DiffOp,
    pub content: String,
    pub old_lineno: Option<usize>,
    pub new_lineno: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffOp {
    Context,
    Add,
    Remove,
    Note,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct LineId {
    file: usize,
    hunk: usize,
    line: usize,
}

#[derive(Clone, Debug)]
struct Analysis {
    lines: HashMap<LineId, LineAnalysis>,
    moves: Vec<MoveBlock>,
    roles: Vec<RoleGroup>,
    syntax: HashMap<LineId, Vec<SyntaxRange>>,
}

#[derive(Clone, Debug, Default)]
struct LineAnalysis {
    novelty: LineNovelty,
    ranges: Vec<NoveltyRange>,
    paired_with: Option<LineId>,
    move_id: Option<usize>,
    role_ranges: Vec<RoleRange>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum LineNovelty {
    #[default]
    None,
    Unpaired,
    Content,
    Mixed,
    WhitespaceOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NoveltyRange {
    start: usize,
    end: usize,
    weight: NoveltyWeight,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NoveltyWeight {
    Unchanged,
    Muted,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RoleRange {
    start: usize,
    end: usize,
    role: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MoveBlock {
    remove_file: usize,
    remove_hunk: usize,
    remove_start: usize,
    add_file: usize,
    add_hunk: usize,
    add_start: usize,
    len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RoleGroup {
    removed: String,
    added: String,
    count: usize,
    color: Rgb,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SyntaxRange {
    start: usize,
    end: usize,
    class: SyntaxClass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SyntaxClass {
    Comment,
    String,
    Number,
    Keyword,
    Type,
    Identifier,
    Operator,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Side {
    Pre,
    Post,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Rgb {
    r: u8,
    g: u8,
    b: u8,
}

impl Rgb {
    const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    fn mix_with(self, base: Rgb, alpha: f32) -> Self {
        let mix = |fg: u8, bg: u8| -> u8 {
            ((fg as f32 * alpha) + (bg as f32 * (1.0 - alpha)))
                .round()
                .clamp(0.0, 255.0) as u8
        };
        Self::new(
            mix(self.r, base.r),
            mix(self.g, base.g),
            mix(self.b, base.b),
        )
    }
}

const ROLE_PALETTE: [Rgb; MAX_ROLE_GROUPS] = [
    Rgb::new(230, 159, 0),
    Rgb::new(86, 180, 233),
    Rgb::new(0, 158, 115),
    Rgb::new(240, 228, 66),
    Rgb::new(0, 114, 178),
    Rgb::new(213, 94, 0),
    Rgb::new(204, 121, 167),
    Rgb::new(128, 128, 128),
];

#[derive(Clone, Debug)]
struct RenderedLine {
    text: String,
}

#[derive(Clone, Debug)]
struct LineRef {
    id: LineId,
    content: String,
}

#[derive(Clone, Debug)]
struct ChangePair {
    remove: LineRef,
    add: LineRef,
}

#[derive(Clone, Debug)]
struct RunRef {
    file: usize,
    hunk: usize,
    start: usize,
    len: usize,
    normalized: Vec<String>,
    op: DiffOp,
}

#[derive(Clone, Debug)]
struct ReconLine {
    id: LineId,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LanguageKind {
    Rust,
    Python,
    Bash,
}

/// Parse a git-style or plain unified diff into sidiff's line-oriented IR.
pub fn parse_unified_diff(input: &str) -> Diff {
    let mut diff = Diff::default();
    let mut current_file: Option<DiffFile> = None;
    let mut current_hunk: Option<Hunk> = None;

    for raw in input.lines() {
        let line = raw.strip_suffix('\r').unwrap_or(raw).to_string();

        if line.starts_with("diff --git ") {
            flush_hunk(&mut current_file, &mut current_hunk);
            flush_file(&mut diff, &mut current_file);
            let (old_path, new_path) = parse_diff_git_paths(&line);
            current_file = Some(DiffFile {
                old_path,
                new_path,
                header: vec![line],
                hunks: vec![],
            });
            continue;
        }

        if line.starts_with("--- ") && should_start_plain_file(&current_file, &current_hunk) {
            flush_hunk(&mut current_file, &mut current_hunk);
            flush_file(&mut diff, &mut current_file);
            current_file = Some(DiffFile {
                old_path: parse_header_path(&line, "--- "),
                new_path: None,
                header: vec![line],
                hunks: vec![],
            });
            continue;
        }

        if line.starts_with("--- ") {
            ensure_file(&mut current_file);
            if let Some(file) = current_file.as_mut() {
                file.old_path = parse_header_path(&line, "--- ");
                file.header.push(line);
            }
            continue;
        }

        if line.starts_with("+++ ") {
            ensure_file(&mut current_file);
            if let Some(file) = current_file.as_mut() {
                file.new_path = parse_header_path(&line, "+++ ");
                file.header.push(line);
            }
            continue;
        }

        if line.starts_with("@@ ") {
            ensure_file(&mut current_file);
            flush_hunk(&mut current_file, &mut current_hunk);
            current_hunk = Some(parse_hunk_header(&line));
            continue;
        }

        if let Some(hunk) = current_hunk.as_mut() {
            append_hunk_line(hunk, &line);
            continue;
        }

        if let Some(file) = current_file.as_mut() {
            file.header.push(line);
        } else {
            diff.preamble.push(line);
        }
    }

    flush_hunk(&mut current_file, &mut current_hunk);
    flush_file(&mut diff, &mut current_file);
    diff
}

/// Render a diff as annotated text. This is used directly for piped stdout and
/// as the input fed to an external pager when stdout is a terminal.
pub fn render_diff(input: &str, options: SidiffOptions) -> String {
    let diff = parse_unified_diff(input);
    let analysis = analyze_diff(&diff);
    let lines = build_rendered_lines(&diff, &analysis, options);
    let mut out = String::new();
    for line in lines {
        out.push_str(&line.text);
        out.push('\n');
    }
    out
}

/// Render a unified diff and feed it to the external pager named by `PAGER`.
///
/// An empty `PAGER` disables paging and writes directly to stdout. When `PAGER`
/// is unset, sidiff defaults to `less -R` so ANSI truecolor survives the pager.
pub fn run_pager(input: &str, options: SidiffOptions) -> io::Result<()> {
    let rendered = render_diff(input, options);
    match pager_command_from_env() {
        PagerCommand::Disabled => write_rendered_stdout(&rendered),
        PagerCommand::Shell(command) => run_shell_pager(&command, &rendered),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PagerCommand {
    Disabled,
    Shell(OsString),
}

fn pager_command_from_env() -> PagerCommand {
    pager_command_from_value(env::var_os("PAGER"))
}

fn pager_command_from_value(value: Option<OsString>) -> PagerCommand {
    match value {
        Some(value) if os_str_is_blank(&value) => PagerCommand::Disabled,
        Some(value) => PagerCommand::Shell(value),
        None => PagerCommand::Shell(OsString::from(DEFAULT_PAGER)),
    }
}

fn os_str_is_blank(value: &OsStr) -> bool {
    value.to_string_lossy().trim().is_empty()
}

fn run_shell_pager(command: &OsStr, rendered: &str) -> io::Result<()> {
    let mut child = shell_command(command)
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|err| {
            io::Error::new(
                err.kind(),
                format!(
                    "failed to launch pager '{}': {err}",
                    command.to_string_lossy()
                ),
            )
        })?;

    if let Some(mut stdin) = child.stdin.take() {
        match stdin.write_all(rendered.as_bytes()) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::BrokenPipe => {}
            Err(err) => return Err(err),
        }
    }

    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "pager '{}' exited with status {status}",
            command.to_string_lossy()
        )))
    }
}

#[cfg(unix)]
fn shell_command(command: &OsStr) -> Command {
    let mut shell = Command::new("sh");
    shell.arg("-c").arg(command);
    shell
}

#[cfg(windows)]
fn shell_command(command: &OsStr) -> Command {
    let mut shell = Command::new("cmd");
    shell.arg("/C").arg(command);
    shell
}

fn write_rendered_stdout(rendered: &str) -> io::Result<()> {
    let mut stdout = io::stdout();
    stdout.write_all(rendered.as_bytes())?;
    stdout.flush()
}

fn flush_file(diff: &mut Diff, current_file: &mut Option<DiffFile>) {
    if let Some(file) = current_file.take() {
        diff.files.push(file);
    }
}

fn flush_hunk(current_file: &mut Option<DiffFile>, current_hunk: &mut Option<Hunk>) {
    if let Some(hunk) = current_hunk.take() {
        ensure_file(current_file);
        if let Some(file) = current_file.as_mut() {
            file.hunks.push(hunk);
        }
    }
}

fn ensure_file(current_file: &mut Option<DiffFile>) {
    if current_file.is_none() {
        *current_file = Some(DiffFile::default());
    }
}

fn should_start_plain_file(current_file: &Option<DiffFile>, current_hunk: &Option<Hunk>) -> bool {
    current_file.is_none()
        || current_hunk.is_some()
        || current_file
            .as_ref()
            .is_some_and(|file| !file.hunks.is_empty())
}

fn parse_diff_git_paths(line: &str) -> (Option<String>, Option<String>) {
    let rest = line.strip_prefix("diff --git ").unwrap_or(line);
    let mut parts = rest.split_whitespace();
    let old = parts.next().map(clean_diff_path);
    let new = parts.next().map(clean_diff_path);
    (old, new)
}

fn parse_header_path(line: &str, prefix: &str) -> Option<String> {
    let path = line.strip_prefix(prefix)?.split_whitespace().next()?;
    if path == "/dev/null" {
        None
    } else {
        Some(clean_diff_path(path))
    }
}

fn clean_diff_path(path: &str) -> String {
    path.strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path)
        .to_string()
}

fn parse_hunk_header(line: &str) -> Hunk {
    let mut parts = line.splitn(4, ' ');
    let _open = parts.next();
    let old_part = parts.next().unwrap_or("-0,0");
    let new_part = parts.next().unwrap_or("+0,0");
    let section = parts
        .next()
        .and_then(|part| part.strip_prefix("@@"))
        .unwrap_or("")
        .trim()
        .to_string();
    let (old_start, old_count) = parse_range_part(old_part, '-');
    let (new_start, new_count) = parse_range_part(new_part, '+');
    Hunk {
        header: line.to_string(),
        old_start,
        old_count,
        new_start,
        new_count,
        section,
        lines: vec![],
    }
}

fn parse_range_part(part: &str, prefix: char) -> (usize, usize) {
    let rest = part.strip_prefix(prefix).unwrap_or(part);
    let mut split = rest.splitn(2, ',');
    let start = split
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let count = split
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    (start, count)
}

fn append_hunk_line(hunk: &mut Hunk, line: &str) {
    let old_next = hunk
        .lines
        .iter()
        .filter(|line| matches!(line.op, DiffOp::Context | DiffOp::Remove))
        .count();
    let new_next = hunk
        .lines
        .iter()
        .filter(|line| matches!(line.op, DiffOp::Context | DiffOp::Add))
        .count();

    if line.starts_with("\\ ") {
        hunk.lines.push(DiffLine {
            op: DiffOp::Note,
            content: line.to_string(),
            old_lineno: None,
            new_lineno: None,
        });
        return;
    }

    let (op, content) = match line.as_bytes().first().copied() {
        Some(b' ') => (DiffOp::Context, &line[1..]),
        Some(b'+') => (DiffOp::Add, &line[1..]),
        Some(b'-') => (DiffOp::Remove, &line[1..]),
        _ => (DiffOp::Context, line),
    };
    hunk.lines.push(DiffLine {
        op,
        content: content.to_string(),
        old_lineno: match op {
            DiffOp::Context | DiffOp::Remove => Some(hunk.old_start + old_next),
            DiffOp::Add | DiffOp::Note => None,
        },
        new_lineno: match op {
            DiffOp::Context | DiffOp::Add => Some(hunk.new_start + new_next),
            DiffOp::Remove | DiffOp::Note => None,
        },
    });
}

fn analyze_diff(diff: &Diff) -> Analysis {
    let mut analysis = Analysis {
        lines: HashMap::new(),
        moves: vec![],
        roles: vec![],
        syntax: syntax_maps(diff),
    };
    let pairs = analyze_change_pairs(diff, &mut analysis.lines);
    detect_moves(diff, &mut analysis);
    analysis.roles = detect_roles(&pairs);
    apply_roles(diff, &mut analysis);
    analysis
}

fn analyze_change_pairs(diff: &Diff, lines: &mut HashMap<LineId, LineAnalysis>) -> Vec<ChangePair> {
    let mut pairs = vec![];
    for (file_idx, file) in diff.files.iter().enumerate() {
        for (hunk_idx, hunk) in file.hunks.iter().enumerate() {
            let mut index = 0usize;
            while index < hunk.lines.len() {
                if matches!(hunk.lines[index].op, DiffOp::Context | DiffOp::Note) {
                    index += 1;
                    continue;
                }
                let start = index;
                while index < hunk.lines.len()
                    && matches!(hunk.lines[index].op, DiffOp::Add | DiffOp::Remove)
                {
                    index += 1;
                }
                let mut removes = vec![];
                let mut adds = vec![];
                for line_idx in start..index {
                    let line = &hunk.lines[line_idx];
                    let id = LineId {
                        file: file_idx,
                        hunk: hunk_idx,
                        line: line_idx,
                    };
                    match line.op {
                        DiffOp::Remove => removes.push(LineRef {
                            id,
                            content: line.content.clone(),
                        }),
                        DiffOp::Add => adds.push(LineRef {
                            id,
                            content: line.content.clone(),
                        }),
                        DiffOp::Context | DiffOp::Note => {}
                    }
                }
                for (remove, add) in pair_change_lines(removes, adds) {
                    let old_meta = analyze_paired_line(&remove.content, &add.content, Side::Pre);
                    let new_meta = analyze_paired_line(&remove.content, &add.content, Side::Post);
                    lines.insert(
                        remove.id,
                        LineAnalysis {
                            paired_with: Some(add.id),
                            ..old_meta
                        },
                    );
                    lines.insert(
                        add.id,
                        LineAnalysis {
                            paired_with: Some(remove.id),
                            ..new_meta
                        },
                    );
                    pairs.push(ChangePair { remove, add });
                }
                for line_idx in start..index {
                    let id = LineId {
                        file: file_idx,
                        hunk: hunk_idx,
                        line: line_idx,
                    };
                    if lines.contains_key(&id) {
                        continue;
                    }
                    let line = &hunk.lines[line_idx];
                    if matches!(line.op, DiffOp::Add | DiffOp::Remove) {
                        lines.insert(id, analyze_unpaired_line(&line.content));
                    }
                }
            }
        }
    }
    pairs
}

fn pair_change_lines(removes: Vec<LineRef>, adds: Vec<LineRef>) -> Vec<(LineRef, LineRef)> {
    let mut pairs = vec![];
    let mut used_adds = HashSet::new();
    for (remove_idx, remove) in removes.iter().enumerate() {
        let mut best: Option<(usize, f32)> = None;
        for (add_idx, add) in adds.iter().enumerate() {
            if used_adds.contains(&add_idx) {
                continue;
            }
            let score = line_similarity(&remove.content, &add.content);
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((add_idx, score));
            }
        }
        let Some((add_idx, score)) = best else {
            continue;
        };
        let positional = remove_idx < adds.len() && !used_adds.contains(&remove_idx);
        if score >= 0.18 || positional {
            let chosen = if positional && score < 0.18 {
                remove_idx
            } else {
                add_idx
            };
            if !used_adds.insert(chosen) {
                continue;
            }
            pairs.push((remove.clone(), adds[chosen].clone()));
        }
    }
    pairs
}

fn line_similarity(old: &str, new: &str) -> f32 {
    if old == new {
        return 1.0;
    }
    let old_tokens = tokenize(old);
    let new_tokens = tokenize(new);
    if old_tokens.is_empty() && new_tokens.is_empty() {
        return 1.0;
    }
    let old_words: Vec<&str> = old_tokens
        .iter()
        .filter(|token| token.kind != TokenKind::Whitespace)
        .map(|token| token.text.as_str())
        .collect();
    let new_words: Vec<&str> = new_tokens
        .iter()
        .filter(|token| token.kind != TokenKind::Whitespace)
        .map(|token| token.text.as_str())
        .collect();
    if old_words.is_empty() && new_words.is_empty() {
        return if normalize_whitespace(old) == normalize_whitespace(new) {
            1.0
        } else {
            0.0
        };
    }
    lcs_len(&old_words, &new_words) as f32 / max(old_words.len(), new_words.len()) as f32
}

fn analyze_paired_line(old: &str, new: &str, side: Side) -> LineAnalysis {
    if strip_all_whitespace(old) == strip_all_whitespace(new) {
        let content = match side {
            Side::Pre => old,
            Side::Post => new,
        };
        return LineAnalysis {
            novelty: LineNovelty::WhitespaceOnly,
            ranges: whole_ranges(content, NoveltyWeight::Muted),
            ..LineAnalysis::default()
        };
    }

    let old_tokens = tokenize(old);
    let new_tokens = tokenize(new);
    let (old_marks, new_marks) = diff_token_marks(&old_tokens, &new_tokens);
    let marks = match side {
        Side::Pre => old_marks,
        Side::Post => new_marks,
    };
    let tokens = match side {
        Side::Pre => old_tokens,
        Side::Post => new_tokens,
    };
    let has_muted = marks.contains(&NoveltyWeight::Muted);
    let has_full = marks.contains(&NoveltyWeight::Full);
    let novelty = if has_muted && has_full {
        LineNovelty::Mixed
    } else if has_full {
        LineNovelty::Content
    } else {
        LineNovelty::None
    };
    LineAnalysis {
        novelty,
        ranges: tokens
            .iter()
            .zip(marks)
            .map(|(token, weight)| NoveltyRange {
                start: token.start,
                end: token.end,
                weight,
            })
            .collect(),
        ..LineAnalysis::default()
    }
}

fn analyze_unpaired_line(content: &str) -> LineAnalysis {
    LineAnalysis {
        novelty: LineNovelty::Unpaired,
        ranges: whole_ranges(content, NoveltyWeight::Full),
        ..LineAnalysis::default()
    }
}

fn whole_ranges(content: &str, weight: NoveltyWeight) -> Vec<NoveltyRange> {
    if content.is_empty() {
        vec![]
    } else {
        vec![NoveltyRange {
            start: 0,
            end: content.len(),
            weight,
        }]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Identifier,
    Whitespace,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Token {
    start: usize,
    end: usize,
    kind: TokenKind,
    text: String,
}

fn tokenize(input: &str) -> Vec<Token> {
    let mut tokens = vec![];
    let mut iter = input.char_indices().peekable();
    while let Some((start, ch)) = iter.next() {
        let kind = if is_ident_start(ch) {
            TokenKind::Identifier
        } else if ch.is_whitespace() {
            TokenKind::Whitespace
        } else {
            TokenKind::Other
        };
        let mut end = start + ch.len_utf8();
        while let Some(&(idx, next)) = iter.peek() {
            let same = match kind {
                TokenKind::Identifier => is_ident_continue(next),
                TokenKind::Whitespace => next.is_whitespace(),
                TokenKind::Other => {
                    !next.is_whitespace() && !is_ident_start(next) && next.is_ascii_punctuation()
                }
            };
            if !same {
                break;
            }
            iter.next();
            end = idx + next.len_utf8();
        }
        tokens.push(Token {
            start,
            end,
            kind,
            text: input[start..end].to_string(),
        });
    }
    tokens
}

fn diff_token_marks(old: &[Token], new: &[Token]) -> (Vec<NoveltyWeight>, Vec<NoveltyWeight>) {
    let pairs = lcs_pairs_by(old, new, |left, right| {
        left.kind == right.kind && left.text == right.text
    });
    let mut old_marks = vec![NoveltyWeight::Full; old.len()];
    let mut new_marks = vec![NoveltyWeight::Full; new.len()];
    let mut old_cursor = 0usize;
    let mut new_cursor = 0usize;
    for (old_match, new_match) in pairs {
        mark_changed_segment(
            old,
            new,
            &mut old_marks,
            &mut new_marks,
            old_cursor,
            old_match,
            new_cursor,
            new_match,
        );
        old_marks[old_match] = NoveltyWeight::Unchanged;
        new_marks[new_match] = NoveltyWeight::Unchanged;
        old_cursor = old_match + 1;
        new_cursor = new_match + 1;
    }
    mark_changed_segment(
        old,
        new,
        &mut old_marks,
        &mut new_marks,
        old_cursor,
        old.len(),
        new_cursor,
        new.len(),
    );
    (old_marks, new_marks)
}

#[allow(clippy::too_many_arguments)]
fn mark_changed_segment(
    old: &[Token],
    new: &[Token],
    old_marks: &mut [NoveltyWeight],
    new_marks: &mut [NoveltyWeight],
    old_start: usize,
    old_end: usize,
    new_start: usize,
    new_end: usize,
) {
    let all_whitespace = old[old_start..old_end]
        .iter()
        .chain(new[new_start..new_end].iter())
        .all(|token| token.kind == TokenKind::Whitespace);
    for idx in old_start..old_end {
        old_marks[idx] = if all_whitespace || old[idx].kind == TokenKind::Whitespace {
            NoveltyWeight::Muted
        } else {
            NoveltyWeight::Full
        };
    }
    for idx in new_start..new_end {
        new_marks[idx] = if all_whitespace || new[idx].kind == TokenKind::Whitespace {
            NoveltyWeight::Muted
        } else {
            NoveltyWeight::Full
        };
    }
}

fn lcs_len<T: Eq>(left: &[T], right: &[T]) -> usize {
    let mut row = vec![0usize; right.len() + 1];
    for left_item in left {
        let mut prev = 0usize;
        for (j, right_item) in right.iter().enumerate() {
            let saved = row[j + 1];
            if left_item == right_item {
                row[j + 1] = prev + 1;
            } else {
                row[j + 1] = max(row[j + 1], row[j]);
            }
            prev = saved;
        }
    }
    row[right.len()]
}

fn lcs_pairs_by<T>(left: &[T], right: &[T], eq: impl Fn(&T, &T) -> bool) -> Vec<(usize, usize)> {
    let mut table = vec![vec![0usize; right.len() + 1]; left.len() + 1];
    for i in (0..left.len()).rev() {
        for j in (0..right.len()).rev() {
            table[i][j] = if eq(&left[i], &right[j]) {
                table[i + 1][j + 1] + 1
            } else {
                max(table[i + 1][j], table[i][j + 1])
            };
        }
    }
    let mut i = 0usize;
    let mut j = 0usize;
    let mut pairs = vec![];
    while i < left.len() && j < right.len() {
        if eq(&left[i], &right[j]) {
            pairs.push((i, j));
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    pairs
}

fn strip_all_whitespace(input: &str) -> String {
    input.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn normalize_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

fn detect_moves(diff: &Diff, analysis: &mut Analysis) {
    let runs = collect_runs(diff);
    let removes: Vec<&RunRef> = runs
        .iter()
        .filter(|run| run.op == DiffOp::Remove && run.len >= MOVE_MIN_LINES)
        .collect();
    let adds: Vec<&RunRef> = runs
        .iter()
        .filter(|run| run.op == DiffOp::Add && run.len >= MOVE_MIN_LINES)
        .collect();
    let mut used_adds = HashSet::new();

    for remove in removes {
        let mut best: Option<(&RunRef, f32)> = None;
        for add in &adds {
            let add_key = (add.file, add.hunk, add.start);
            if used_adds.contains(&add_key) {
                continue;
            }
            let score = run_similarity(&remove.normalized, &add.normalized);
            if score >= 0.68 && best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((*add, score));
            }
        }
        let Some((add, _score)) = best else {
            continue;
        };
        used_adds.insert((add.file, add.hunk, add.start));
        let len = min(remove.len, add.len);
        let move_id = analysis.moves.len();
        analysis.moves.push(MoveBlock {
            remove_file: remove.file,
            remove_hunk: remove.hunk,
            remove_start: remove.start,
            add_file: add.file,
            add_hunk: add.hunk,
            add_start: add.start,
            len,
        });
        for offset in 0..len {
            for id in [
                LineId {
                    file: remove.file,
                    hunk: remove.hunk,
                    line: remove.start + offset,
                },
                LineId {
                    file: add.file,
                    hunk: add.hunk,
                    line: add.start + offset,
                },
            ] {
                analysis.lines.entry(id).or_default().move_id = Some(move_id);
            }
        }
    }
}

fn collect_runs(diff: &Diff) -> Vec<RunRef> {
    let mut runs = vec![];
    for (file_idx, file) in diff.files.iter().enumerate() {
        for (hunk_idx, hunk) in file.hunks.iter().enumerate() {
            let mut index = 0usize;
            while index < hunk.lines.len() {
                let op = hunk.lines[index].op;
                if !matches!(op, DiffOp::Add | DiffOp::Remove) {
                    index += 1;
                    continue;
                }
                let start = index;
                while index < hunk.lines.len() && hunk.lines[index].op == op {
                    index += 1;
                }
                let lines = &hunk.lines[start..index];
                runs.push(RunRef {
                    file: file_idx,
                    hunk: hunk_idx,
                    start,
                    len: lines.len(),
                    normalized: lines
                        .iter()
                        .map(|line| normalize_whitespace(&line.content))
                        .collect(),
                    op,
                });
            }
        }
    }
    runs
}

fn run_similarity(remove: &[String], add: &[String]) -> f32 {
    if remove.is_empty() || add.is_empty() {
        return 0.0;
    }
    let common = lcs_len(remove, add);
    if common < MOVE_MIN_LINES {
        return 0.0;
    }
    common as f32 / max(remove.len(), add.len()) as f32
}

fn detect_roles(pairs: &[ChangePair]) -> Vec<RoleGroup> {
    let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
    for pair in pairs {
        for (removed, added) in identifier_pairs(&pair.remove.content, &pair.add.content) {
            if removed != added {
                *counts.entry((removed, added)).or_insert(0) += 1;
            }
        }
    }
    let mut ordered: Vec<((String, String), usize)> = counts
        .into_iter()
        .filter(|((removed, added), count)| removed != added && *count >= 3)
        .collect();
    ordered.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| a.0.0.cmp(&b.0.0))
            .then_with(|| a.0.1.cmp(&b.0.1))
    });
    ordered
        .into_iter()
        .take(MAX_ROLE_GROUPS)
        .enumerate()
        .map(|(idx, ((removed, added), count))| RoleGroup {
            removed,
            added,
            count,
            color: ROLE_PALETTE[idx],
        })
        .collect()
}

fn identifier_pairs(old: &str, new: &str) -> Vec<(String, String)> {
    let old_tokens: Vec<Token> = tokenize(old)
        .into_iter()
        .filter(|token| token.kind == TokenKind::Identifier)
        .collect();
    let new_tokens: Vec<Token> = tokenize(new)
        .into_iter()
        .filter(|token| token.kind == TokenKind::Identifier)
        .collect();
    let pairs = lcs_pairs_by(&old_tokens, &new_tokens, |left, right| {
        left.text == right.text
    });
    let mut result = vec![];
    let mut old_cursor = 0usize;
    let mut new_cursor = 0usize;
    for (old_match, new_match) in pairs {
        pair_identifier_segment(
            &old_tokens,
            &new_tokens,
            old_cursor,
            old_match,
            new_cursor,
            new_match,
            &mut result,
        );
        old_cursor = old_match + 1;
        new_cursor = new_match + 1;
    }
    pair_identifier_segment(
        &old_tokens,
        &new_tokens,
        old_cursor,
        old_tokens.len(),
        new_cursor,
        new_tokens.len(),
        &mut result,
    );
    result
}

#[allow(clippy::too_many_arguments)]
fn pair_identifier_segment(
    old: &[Token],
    new: &[Token],
    old_start: usize,
    old_end: usize,
    new_start: usize,
    new_end: usize,
    out: &mut Vec<(String, String)>,
) {
    let len = min(
        old_end.saturating_sub(old_start),
        new_end.saturating_sub(new_start),
    );
    for offset in 0..len {
        out.push((
            old[old_start + offset].text.clone(),
            new[new_start + offset].text.clone(),
        ));
    }
}

fn apply_roles(diff: &Diff, analysis: &mut Analysis) {
    if analysis.roles.is_empty() {
        return;
    }
    for (file_idx, file) in diff.files.iter().enumerate() {
        for (hunk_idx, hunk) in file.hunks.iter().enumerate() {
            for (line_idx, line) in hunk.lines.iter().enumerate() {
                let side = match line.op {
                    DiffOp::Remove => Side::Pre,
                    DiffOp::Add => Side::Post,
                    DiffOp::Context | DiffOp::Note => continue,
                };
                let mut ranges = vec![];
                for token in tokenize(&line.content) {
                    if token.kind != TokenKind::Identifier {
                        continue;
                    }
                    for (role_idx, role) in analysis.roles.iter().enumerate() {
                        let target = match side {
                            Side::Pre => &role.removed,
                            Side::Post => &role.added,
                        };
                        if token.text == *target {
                            ranges.push(RoleRange {
                                start: token.start,
                                end: token.end,
                                role: role_idx,
                            });
                        }
                    }
                }
                if !ranges.is_empty() {
                    let id = LineId {
                        file: file_idx,
                        hunk: hunk_idx,
                        line: line_idx,
                    };
                    analysis.lines.entry(id).or_default().role_ranges = ranges;
                }
            }
        }
    }
}

fn syntax_maps(diff: &Diff) -> HashMap<LineId, Vec<SyntaxRange>> {
    let mut maps = HashMap::new();
    for (file_idx, file) in diff.files.iter().enumerate() {
        for side in [Side::Pre, Side::Post] {
            let path = match side {
                Side::Pre => display_old_path(file),
                Side::Post => display_new_path(file),
            };
            let Some(language) = language_for_path(&path) else {
                continue;
            };
            let (source, recon) = reconstruct_side(file_idx, file, side);
            if source.is_empty() {
                continue;
            }
            let spans = syntax_spans(language, &source);
            distribute_syntax_spans(&mut maps, &recon, &spans);
        }
    }
    maps
}

fn reconstruct_side(file_idx: usize, file: &DiffFile, side: Side) -> (String, Vec<ReconLine>) {
    let mut source = String::new();
    let mut recon = vec![];
    for (hunk_idx, hunk) in file.hunks.iter().enumerate() {
        for (line_idx, line) in hunk.lines.iter().enumerate() {
            let include = matches!(
                (side, line.op),
                (Side::Pre, DiffOp::Context | DiffOp::Remove)
                    | (Side::Post, DiffOp::Context | DiffOp::Add)
            );
            if !include {
                continue;
            }
            let start = source.len();
            source.push_str(&line.content);
            let end = source.len();
            source.push('\n');
            recon.push(ReconLine {
                id: LineId {
                    file: file_idx,
                    hunk: hunk_idx,
                    line: line_idx,
                },
                start,
                end,
            });
        }
    }
    (source, recon)
}

fn syntax_spans(language: LanguageKind, source: &str) -> Vec<SyntaxRange> {
    let mut spans = lexical_syntax_spans(language, source);
    let mut parser = Parser::new();
    let tree_language: Language = match language {
        LanguageKind::Rust => tree_sitter_rust::LANGUAGE.into(),
        LanguageKind::Python => tree_sitter_python::LANGUAGE.into(),
        LanguageKind::Bash => tree_sitter_bash::LANGUAGE.into(),
    };
    if parser.set_language(&tree_language).is_err() {
        return spans;
    }
    let Some(tree) = parser.parse(source, None) else {
        return spans;
    };
    collect_tree_syntax(tree.root_node(), source, &mut spans);
    spans.sort_by(|left, right| {
        left.start
            .cmp(&right.start)
            .then_with(|| right.end.cmp(&left.end))
    });
    spans
}

fn collect_tree_syntax(node: Node<'_>, source: &str, spans: &mut Vec<SyntaxRange>) {
    if let Some(class) = classify_tree_node(node, source) {
        spans.push(SyntaxRange {
            start: node.start_byte(),
            end: node.end_byte(),
            class,
        });
        if matches!(class, SyntaxClass::Comment | SyntaxClass::String) {
            return;
        }
    }
    for idx in 0..node.child_count() {
        if let Some(child) = node.child(idx) {
            collect_tree_syntax(child, source, spans);
        }
    }
}

fn classify_tree_node(node: Node<'_>, source: &str) -> Option<SyntaxClass> {
    let kind = node.kind();
    if node.end_byte() <= node.start_byte() {
        return None;
    }
    if kind.contains("comment") {
        return Some(SyntaxClass::Comment);
    }
    if kind.contains("string") || kind.contains("char_literal") {
        return Some(SyntaxClass::String);
    }
    if kind.contains("integer") || kind.contains("float") || kind.contains("number") {
        return Some(SyntaxClass::Number);
    }
    if matches!(
        kind,
        "type_identifier" | "primitive_type" | "scoped_type_identifier"
    ) {
        return Some(SyntaxClass::Type);
    }
    if matches!(
        kind,
        "identifier" | "field_identifier" | "shorthand_field_identifier"
    ) {
        return Some(SyntaxClass::Identifier);
    }
    let text = source
        .get(node.start_byte()..node.end_byte())
        .unwrap_or_default();
    if is_keyword(text) {
        return Some(SyntaxClass::Keyword);
    }
    if is_operator(text) {
        return Some(SyntaxClass::Operator);
    }
    None
}

fn lexical_syntax_spans(language: LanguageKind, source: &str) -> Vec<SyntaxRange> {
    let mut spans = vec![];
    let bytes = source.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        let ch = source[idx..].chars().next().unwrap_or('\0');
        if ch == '#' && language != LanguageKind::Rust {
            let end = source[idx..]
                .find('\n')
                .map(|offset| idx + offset)
                .unwrap_or(source.len());
            spans.push(SyntaxRange {
                start: idx,
                end,
                class: SyntaxClass::Comment,
            });
            idx = end;
            continue;
        }
        if ch == '/' && source[idx..].starts_with("//") && language == LanguageKind::Rust {
            let end = source[idx..]
                .find('\n')
                .map(|offset| idx + offset)
                .unwrap_or(source.len());
            spans.push(SyntaxRange {
                start: idx,
                end,
                class: SyntaxClass::Comment,
            });
            idx = end;
            continue;
        }
        if ch == '"' || ch == '\'' {
            let quote = ch;
            let start = idx;
            idx += ch.len_utf8();
            let mut escaped = false;
            while idx < source.len() {
                let next = source[idx..].chars().next().unwrap_or('\0');
                idx += next.len_utf8();
                if escaped {
                    escaped = false;
                } else if next == '\\' {
                    escaped = true;
                } else if next == quote {
                    break;
                }
            }
            spans.push(SyntaxRange {
                start,
                end: idx,
                class: SyntaxClass::String,
            });
            continue;
        }
        if ch.is_ascii_digit() {
            let start = idx;
            idx += ch.len_utf8();
            while idx < source.len() {
                let next = source[idx..].chars().next().unwrap_or('\0');
                if !(next.is_ascii_alphanumeric() || next == '_' || next == '.') {
                    break;
                }
                idx += next.len_utf8();
            }
            spans.push(SyntaxRange {
                start,
                end: idx,
                class: SyntaxClass::Number,
            });
            continue;
        }
        if is_ident_start(ch) {
            let start = idx;
            idx += ch.len_utf8();
            while idx < source.len() {
                let next = source[idx..].chars().next().unwrap_or('\0');
                if !is_ident_continue(next) {
                    break;
                }
                idx += next.len_utf8();
            }
            let word = &source[start..idx];
            spans.push(SyntaxRange {
                start,
                end: idx,
                class: if is_keyword(word) {
                    SyntaxClass::Keyword
                } else {
                    SyntaxClass::Identifier
                },
            });
            continue;
        }
        idx += ch.len_utf8();
    }
    spans
}

fn distribute_syntax_spans(
    maps: &mut HashMap<LineId, Vec<SyntaxRange>>,
    recon: &[ReconLine],
    spans: &[SyntaxRange],
) {
    for span in spans {
        if span.start >= span.end {
            continue;
        }
        for line in recon {
            let start = max(span.start, line.start);
            let end = min(span.end, line.end);
            if start >= end {
                continue;
            }
            maps.entry(line.id).or_default().push(SyntaxRange {
                start: start - line.start,
                end: end - line.start,
                class: span.class,
            });
        }
    }
}

fn language_for_path(path: &str) -> Option<LanguageKind> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".rs") {
        Some(LanguageKind::Rust)
    } else if lower.ends_with(".py") || lower.ends_with(".pyw") {
        Some(LanguageKind::Python)
    } else if lower.ends_with(".sh")
        || lower.ends_with(".bash")
        || lower.ends_with(".zsh")
        || lower.ends_with("/bash")
        || lower.ends_with("/sh")
    {
        Some(LanguageKind::Bash)
    } else {
        None
    }
}

fn is_keyword(text: &str) -> bool {
    matches!(
        text,
        "as" | "async"
            | "await"
            | "break"
            | "case"
            | "class"
            | "const"
            | "continue"
            | "def"
            | "do"
            | "done"
            | "dyn"
            | "elif"
            | "else"
            | "enum"
            | "export"
            | "false"
            | "False"
            | "fi"
            | "fn"
            | "for"
            | "from"
            | "if"
            | "impl"
            | "import"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "return"
            | "self"
            | "Self"
            | "struct"
            | "then"
            | "trait"
            | "true"
            | "True"
            | "type"
            | "use"
            | "where"
            | "while"
    )
}

fn is_operator(text: &str) -> bool {
    matches!(
        text,
        "=" | "=="
            | "!="
            | "=>"
            | "->"
            | "::"
            | "+"
            | "-"
            | "*"
            | "/"
            | "%"
            | "&&"
            | "||"
            | "!"
            | "<"
            | ">"
            | "<="
            | ">="
            | "|"
            | "&"
            | "^"
            | "$"
    )
}

fn build_rendered_lines(
    diff: &Diff,
    analysis: &Analysis,
    options: SidiffOptions,
) -> Vec<RenderedLine> {
    let mut out = vec![];
    if !diff.preamble.is_empty() {
        for line in &diff.preamble {
            out.push(RenderedLine {
                text: paint_fg(line, Rgb::new(150, 150, 150), options.use_color),
            });
        }
    }

    for (file_idx, file) in diff.files.iter().enumerate() {
        let file_label = format!(
            "file {} -> {}",
            file.old_path.as_deref().unwrap_or("/dev/null"),
            file.new_path.as_deref().unwrap_or("/dev/null")
        );
        out.push(RenderedLine {
            text: paint_fg(&file_label, Rgb::new(120, 200, 220), options.use_color),
        });
        for header in &file.header {
            out.push(RenderedLine {
                text: paint_fg(header, Rgb::new(120, 120, 120), options.use_color),
            });
        }
        for (hunk_idx, hunk) in file.hunks.iter().enumerate() {
            out.push(RenderedLine {
                text: paint_fg(&hunk.header, Rgb::new(105, 180, 255), options.use_color),
            });
            let mut last_move = None;
            for (line_idx, line) in hunk.lines.iter().enumerate() {
                let id = LineId {
                    file: file_idx,
                    hunk: hunk_idx,
                    line: line_idx,
                };
                let meta = analysis.lines.get(&id);
                if should_hide_line(meta, line, options.whitespace_mode) {
                    continue;
                }
                if meta.and_then(|meta| meta.move_id) != last_move {
                    last_move = meta.and_then(|meta| meta.move_id);
                    if let Some(move_id) = last_move {
                        out.push(RenderedLine {
                            text: render_move_label(diff, analysis, move_id, id, options),
                        });
                    }
                }
                out.push(RenderedLine {
                    text: render_diff_line(diff, analysis, id, line, options),
                });
            }
        }
    }

    if !analysis.roles.is_empty() {
        out.push(RenderedLine {
            text: render_role_summary(&analysis.roles, options),
        });
    }
    out
}

fn should_hide_line(
    meta: Option<&LineAnalysis>,
    line: &DiffLine,
    whitespace_mode: WhitespaceMode,
) -> bool {
    whitespace_mode == WhitespaceMode::Hide
        && matches!(line.op, DiffOp::Add | DiffOp::Remove)
        && meta.is_some_and(|meta| meta.novelty == LineNovelty::WhitespaceOnly)
}

fn render_move_label(
    diff: &Diff,
    analysis: &Analysis,
    move_id: usize,
    current: LineId,
    options: SidiffOptions,
) -> String {
    let Some(block) = analysis.moves.get(move_id) else {
        return String::new();
    };
    let (label, color) = if current.file == block.add_file
        && current.hunk == block.add_hunk
        && current.line >= block.add_start
    {
        let line =
            &diff.files[block.remove_file].hunks[block.remove_hunk].lines[block.remove_start];
        (
            format!(
                "  M moved from {}:L{}",
                display_old_path(&diff.files[block.remove_file]),
                line.old_lineno.unwrap_or(0)
            ),
            Rgb::new(175, 145, 255),
        )
    } else {
        let line = &diff.files[block.add_file].hunks[block.add_hunk].lines[block.add_start];
        (
            format!(
                "  M moved to {}:L{}",
                display_new_path(&diff.files[block.add_file]),
                line.new_lineno.unwrap_or(0)
            ),
            Rgb::new(175, 145, 255),
        )
    };
    paint_fg(&label, color, options.use_color)
}

fn render_diff_line(
    diff: &Diff,
    analysis: &Analysis,
    id: LineId,
    line: &DiffLine,
    options: SidiffOptions,
) -> String {
    let meta = analysis.lines.get(&id);
    let old = line
        .old_lineno
        .map(|num| format!("{num:>4}"))
        .unwrap_or_else(|| "    ".to_string());
    let new = line
        .new_lineno
        .map(|num| format!("{num:>4}"))
        .unwrap_or_else(|| "    ".to_string());
    let op = match line.op {
        DiffOp::Context => " ",
        DiffOp::Add => "+",
        DiffOp::Remove => "-",
        DiffOp::Note => "\\",
    };
    let move_marker = if meta.and_then(|meta| meta.move_id).is_some() {
        "M"
    } else {
        " "
    };
    let role_glyph = if options.use_color {
        " "
    } else {
        meta.and_then(|meta| meta.role_ranges.first())
            .map(|range| ROLE_GLYPHS[range.role])
            .unwrap_or(" ")
    };
    let mut line_out = format!("{role_glyph}{move_marker} {old} {new} {op} ");
    if line.op == DiffOp::Note {
        line_out.push_str(&paint_fg(
            &line.content,
            Rgb::new(150, 150, 150),
            options.use_color,
        ));
        return line_out;
    }
    let syntax = analysis.syntax.get(&id).map(Vec::as_slice).unwrap_or(&[]);
    line_out.push_str(&render_content(
        &line.content,
        line.op,
        meta,
        syntax,
        &analysis.roles,
        options,
    ));
    if meta.is_some_and(|meta| meta.paired_with.is_none())
        && matches!(line.op, DiffOp::Add | DiffOp::Remove)
    {
        let path = match line.op {
            DiffOp::Add => display_new_path(&diff.files[id.file]),
            DiffOp::Remove => display_old_path(&diff.files[id.file]),
            DiffOp::Context | DiffOp::Note => display_path(&diff.files[id.file]),
        };
        line_out.push_str(&paint_fg(
            &format!("  [{path}]"),
            Rgb::new(110, 110, 110),
            options.use_color,
        ));
    }
    line_out
}

fn render_content(
    content: &str,
    op: DiffOp,
    meta: Option<&LineAnalysis>,
    syntax: &[SyntaxRange],
    roles: &[RoleGroup],
    options: SidiffOptions,
) -> String {
    if content.is_empty() {
        return String::new();
    }
    let boundaries = span_boundaries(content, meta, syntax);
    let mut out = String::new();
    for pair in boundaries.windows(2) {
        let start = pair[0];
        let end = pair[1];
        if start == end {
            continue;
        }
        let text = &content[start..end];
        let novelty = meta
            .and_then(|meta| novelty_at(meta, start, end))
            .unwrap_or(match op {
                DiffOp::Context => NoveltyWeight::Unchanged,
                DiffOp::Add | DiffOp::Remove => NoveltyWeight::Full,
                DiffOp::Note => NoveltyWeight::Unchanged,
            });
        let syntax_class = syntax_at(syntax, start, end);
        let role = meta.and_then(|meta| role_at(meta, start, end));
        out.push_str(&style_span(
            text,
            op,
            novelty,
            syntax_class,
            role.and_then(|idx| roles.get(idx)),
            options,
        ));
    }
    out
}

fn span_boundaries(
    content: &str,
    meta: Option<&LineAnalysis>,
    syntax: &[SyntaxRange],
) -> Vec<usize> {
    let mut boundaries = vec![0, content.len()];
    if let Some(meta) = meta {
        for range in &meta.ranges {
            boundaries.push(range.start);
            boundaries.push(range.end);
        }
        for range in &meta.role_ranges {
            boundaries.push(range.start);
            boundaries.push(range.end);
        }
    }
    for range in syntax {
        boundaries.push(range.start);
        boundaries.push(range.end);
    }
    boundaries.retain(|idx| *idx <= content.len() && content.is_char_boundary(*idx));
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

fn novelty_at(meta: &LineAnalysis, start: usize, end: usize) -> Option<NoveltyWeight> {
    meta.ranges
        .iter()
        .find(|range| range.start <= start && range.end >= end)
        .map(|range| range.weight)
}

fn syntax_at(syntax: &[SyntaxRange], start: usize, end: usize) -> Option<SyntaxClass> {
    syntax
        .iter()
        .rev()
        .find(|range| range.start <= start && range.end >= end)
        .map(|range| range.class)
}

fn role_at(meta: &LineAnalysis, start: usize, end: usize) -> Option<usize> {
    meta.role_ranges
        .iter()
        .find(|range| range.start <= start && range.end >= end)
        .map(|range| range.role)
}

fn style_span(
    text: &str,
    op: DiffOp,
    novelty: NoveltyWeight,
    syntax: Option<SyntaxClass>,
    role: Option<&RoleGroup>,
    options: SidiffOptions,
) -> String {
    if !options.use_color {
        return text.to_string();
    }
    let mut fg = match (op, novelty) {
        (DiffOp::Add, NoveltyWeight::Full) => Rgb::new(80, 235, 150),
        (DiffOp::Remove, NoveltyWeight::Full) => Rgb::new(255, 105, 115),
        (DiffOp::Add, NoveltyWeight::Muted) | (DiffOp::Remove, NoveltyWeight::Muted) => {
            Rgb::new(125, 125, 125)
        }
        (DiffOp::Add, NoveltyWeight::Unchanged) => Rgb::new(100, 155, 120),
        (DiffOp::Remove, NoveltyWeight::Unchanged) => Rgb::new(165, 95, 100),
        (DiffOp::Context, _) => syntax_color(syntax).unwrap_or(Rgb::new(205, 205, 205)),
        (DiffOp::Note, _) => Rgb::new(150, 150, 150),
    };
    if matches!(op, DiffOp::Add | DiffOp::Remove)
        && novelty == NoveltyWeight::Unchanged
        && let Some(syntax_color) = syntax_color(syntax)
    {
        fg = syntax_color.mix_with(fg, 0.42);
    }
    let bg = role.map(|role| {
        role.color
            .mix_with(Rgb::new(18, 20, 24), options.tint_intensity.alpha())
    });
    paint(text, fg, bg, options.use_color)
}

fn syntax_color(class: Option<SyntaxClass>) -> Option<Rgb> {
    Some(match class? {
        SyntaxClass::Comment => Rgb::new(110, 145, 115),
        SyntaxClass::String => Rgb::new(225, 190, 120),
        SyntaxClass::Number => Rgb::new(185, 160, 255),
        SyntaxClass::Keyword => Rgb::new(115, 175, 255),
        SyntaxClass::Type => Rgb::new(105, 215, 210),
        SyntaxClass::Identifier => Rgb::new(220, 220, 220),
        SyntaxClass::Operator => Rgb::new(180, 180, 180),
    })
}

fn paint_fg(text: &str, fg: Rgb, use_color: bool) -> String {
    paint(text, fg, None, use_color)
}

fn paint(text: &str, fg: Rgb, bg: Option<Rgb>, use_color: bool) -> String {
    if !use_color || text.is_empty() {
        return text.to_string();
    }
    let mut out = format!("\x1b[38;2;{};{};{}m", fg.r, fg.g, fg.b);
    if let Some(bg) = bg {
        out.push_str(&format!("\x1b[48;2;{};{};{}m", bg.r, bg.g, bg.b));
    }
    out.push_str(text);
    out.push_str(RESET);
    out
}

fn render_role_summary(roles: &[RoleGroup], options: SidiffOptions) -> String {
    let mut parts = vec![];
    for (idx, role) in roles.iter().enumerate() {
        let label = format!(
            "{} {}->{} x{}",
            ROLE_GLYPHS[idx], role.removed, role.added, role.count
        );
        if options.use_color {
            parts.push(paint(
                &label,
                Rgb::new(235, 235, 235),
                Some(
                    role.color
                        .mix_with(Rgb::new(18, 20, 24), options.tint_intensity.alpha()),
                ),
                true,
            ));
        } else {
            parts.push(label);
        }
    }
    format!("roles: {}", parts.join("  "))
}

fn display_path(file: &DiffFile) -> String {
    file.new_path
        .as_ref()
        .or(file.old_path.as_ref())
        .cloned()
        .unwrap_or_else(|| "(unknown)".to_string())
}

fn display_old_path(file: &DiffFile) -> String {
    file.old_path
        .clone()
        .unwrap_or_else(|| "/dev/null".to_string())
}

fn display_new_path(file: &DiffFile) -> String {
    file.new_path
        .clone()
        .unwrap_or_else(|| "/dev/null".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
diff --git a/src/main.rs b/src/main.rs
index 1111111..2222222 100644
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,6 +1,6 @@
 fn main() {
-    let old_name = compute(1);
-    println!(\"{}\", old_name);
+    let new_name = compute(2);
+    println!(\"{}\", new_name);
 }
";

    #[test]
    fn parser_reads_git_unified_diff() {
        let diff = parse_unified_diff(SAMPLE);
        assert_eq!(diff.files.len(), 1);
        assert_eq!(diff.files[0].old_path.as_deref(), Some("src/main.rs"));
        assert_eq!(diff.files[0].new_path.as_deref(), Some("src/main.rs"));
        assert_eq!(diff.files[0].hunks.len(), 1);
        assert_eq!(diff.files[0].hunks[0].old_start, 1);
        assert_eq!(diff.files[0].hunks[0].new_start, 1);
        assert_eq!(diff.files[0].hunks[0].lines[1].old_lineno, Some(2));
        assert_eq!(diff.files[0].hunks[0].lines[3].new_lineno, Some(2));
    }

    #[test]
    fn whitespace_only_changes_are_detected() {
        let old = "let value = 1;";
        let new = "    let value = 1;";
        let meta = analyze_paired_line(old, new, Side::Post);
        assert_eq!(meta.novelty, LineNovelty::WhitespaceOnly);
        assert!(
            meta.ranges
                .iter()
                .all(|range| range.weight == NoveltyWeight::Muted)
        );
    }

    #[test]
    fn word_delta_keeps_unchanged_tokens_dim() {
        let meta = analyze_paired_line("let old_name = 1;", "let new_name = 1;", Side::Post);
        assert_eq!(meta.novelty, LineNovelty::Content);
        let tokens = tokenize("let new_name = 1;");
        let changed: Vec<&str> = tokens
            .iter()
            .filter(|token| {
                meta.ranges
                    .iter()
                    .any(|range| range.start == token.start && range.weight == NoveltyWeight::Full)
            })
            .map(|token| token.text.as_str())
            .collect();
        assert_eq!(changed, vec!["new_name"]);
    }

    #[test]
    fn recurring_identifier_pairs_become_roles() {
        let pairs = vec![
            pair("old_name", "new_name"),
            pair("old_name", "new_name"),
            pair("old_name", "new_name"),
        ];
        let roles = detect_roles(&pairs);
        assert_eq!(roles.len(), 1);
        assert_eq!(roles[0].removed, "old_name");
        assert_eq!(roles[0].added, "new_name");
    }

    #[test]
    fn moved_blocks_are_marked() {
        let input = "\
diff --git a/a.rs b/a.rs
--- a/a.rs
+++ b/a.rs
@@ -1,5 +1,2 @@
-fn alpha() {}
-fn beta() {}
-fn gamma() {}
 fn stay() {}
@@ -20,2 +17,5 @@
 fn other() {}
+fn alpha() {}
+fn beta() {}
+fn gamma() {}
";
        let diff = parse_unified_diff(input);
        let analysis = analyze_diff(&diff);
        assert_eq!(analysis.moves.len(), 1);
        assert!(
            analysis
                .lines
                .values()
                .filter(|line| line.move_id == Some(0))
                .count()
                >= 6
        );
    }

    #[test]
    fn no_color_output_has_role_glyphs_and_no_ansi() {
        let input = "\
diff --git a/lib.rs b/lib.rs
--- a/lib.rs
+++ b/lib.rs
@@ -1,4 +1,4 @@
-let old_name = old_name + old_name;
+let new_name = new_name + new_name;
-call(old_name);
+call(new_name);
";
        let rendered = render_diff(
            input,
            SidiffOptions {
                use_color: false,
                ..SidiffOptions::default()
            },
        );
        assert!(!rendered.contains("\x1b["));
        assert!(rendered.contains("\u{2460}"));
        assert!(rendered.contains("old_name->new_name"));
    }

    #[test]
    fn pager_command_honors_pager_env_value() {
        assert_eq!(
            pager_command_from_value(Some(std::ffi::OsString::from("cat"))),
            PagerCommand::Shell(std::ffi::OsString::from("cat"))
        );
        assert_eq!(
            pager_command_from_value(Some(std::ffi::OsString::from(""))),
            PagerCommand::Disabled
        );
        assert_eq!(
            pager_command_from_value(Some(std::ffi::OsString::from("   "))),
            PagerCommand::Disabled
        );
        assert_eq!(
            pager_command_from_value(None),
            PagerCommand::Shell(std::ffi::OsString::from(DEFAULT_PAGER))
        );
    }

    #[test]
    fn tree_sitter_rust_highlight_produces_keyword_span() {
        let spans = syntax_spans(LanguageKind::Rust, "fn main() {\n    let x = 1;\n}\n");
        assert!(
            spans
                .iter()
                .any(|span| span.class == SyntaxClass::Keyword && span.start == 0)
        );
    }

    #[test]
    fn cvd_palette_keeps_pairs_distinct() {
        for matrix in [PROTANOPIA, DEUTERANOPIA, TRITANOPIA] {
            for (left, left_color) in ROLE_PALETTE.iter().enumerate() {
                for (right, right_color) in ROLE_PALETTE.iter().enumerate().skip(left + 1) {
                    let a = simulate_cvd(*left_color, matrix);
                    let b = simulate_cvd(*right_color, matrix);
                    assert!(
                        rgb_distance(a, b) >= 20.0,
                        "palette entries {left} and {right} are too close after simulation"
                    );
                }
            }
        }
    }

    fn pair(removed: &str, added: &str) -> ChangePair {
        ChangePair {
            remove: LineRef {
                id: LineId::default(),
                content: removed.to_string(),
            },
            add: LineRef {
                id: LineId::default(),
                content: added.to_string(),
            },
        }
    }

    const PROTANOPIA: [[f32; 3]; 3] = [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ];
    const DEUTERANOPIA: [[f32; 3]; 3] = [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ];
    const TRITANOPIA: [[f32; 3]; 3] = [
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ];

    fn simulate_cvd(color: Rgb, matrix: [[f32; 3]; 3]) -> Rgb {
        let input = [color.r as f32, color.g as f32, color.b as f32];
        let channel = |row: [f32; 3]| -> u8 {
            row.iter()
                .zip(input)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                .round()
                .clamp(0.0, 255.0) as u8
        };
        Rgb::new(channel(matrix[0]), channel(matrix[1]), channel(matrix[2]))
    }

    fn rgb_distance(left: Rgb, right: Rgb) -> f32 {
        let dr = left.r as f32 - right.r as f32;
        let dg = left.g as f32 - right.g as f32;
        let db = left.b as f32 - right.b as f32;
        (dr * dr + dg * dg + db * db).sqrt()
    }
}
