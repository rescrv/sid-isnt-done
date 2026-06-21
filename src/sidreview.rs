//! Terminal review pager for unified diffs.

use std::cmp::min;
use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};
use serde::{Deserialize, Serialize};

use crate::sidiff::{
    Diff, DiffFile, DiffLine, DiffOp, build_review_line_metadata, file_is_pure_addition,
    parse_unified_diff, render_pure_addition_file_with_bat,
};

/// Run the sidreview terminal UI for a unified diff.
pub fn run_tui(input: &str) -> io::Result<()> {
    let mut app = ReviewApp::from_input_with_color(input, true);
    let fold_store = FoldStateStore::for_current_ref()?;
    if let Some(store) = &fold_store
        && let Some(state) = store.load()?
    {
        app.apply_fold_state(&state);
    }
    let mut terminal = ratatui::try_init()?;
    let result = run_app(&mut terminal, &mut app, fold_store.as_ref());
    let restore_result = ratatui::try_restore();
    match (result, restore_result) {
        (Err(err), _) => Err(err),
        (Ok(()), Err(err)) => Err(err),
        (Ok(()), Ok(())) => Ok(()),
    }
}

/// Render the review pager's current expanded text without terminal styling.
pub fn render_plain(input: &str) -> String {
    let app = ReviewApp::from_input_with_color(input, false);
    let mut out = app
        .rows()
        .into_iter()
        .map(|row| row.text)
        .collect::<Vec<_>>()
        .join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

fn run_app(
    terminal: &mut DefaultTerminal,
    app: &mut ReviewApp,
    fold_store: Option<&FoldStateStore>,
) -> io::Result<()> {
    loop {
        terminal.draw(|frame| render_review(frame, app))?;
        if app.should_quit {
            return Ok(());
        }
        if event::poll(Duration::from_millis(250))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && app.handle_key(key.code) == ReviewAction::FoldStateChanged
            && let Some(store) = fold_store
        {
            store.save(&app.fold_state())?;
        }
    }
}

fn render_review(frame: &mut Frame, app: &mut ReviewApp) {
    let [body_area, footer_area] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(frame.area());
    app.set_viewport(body_area.width as usize, body_area.height as usize);

    let rows = app
        .rows()
        .into_iter()
        .map(|row| render_row_line(row, app.selected))
        .collect::<Vec<_>>();
    let body = Paragraph::new(rows)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll_top.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(body, body_area);

    let footer = Paragraph::new(app.status_text()).style(
        Style::default()
            .fg(Color::Rgb(205, 205, 205))
            .bg(Color::Rgb(24, 26, 31)),
    );
    frame.render_widget(footer, footer_area);
}

fn row_style(row: &ReviewRow, selected: Option<usize>) -> Style {
    let mut style = match row.kind {
        ReviewRowKind::Title => Style::default()
            .fg(Color::Rgb(120, 200, 220))
            .add_modifier(Modifier::BOLD),
        ReviewRowKind::Header => Style::default().fg(Color::Rgb(145, 145, 145)),
        ReviewRowKind::Context => Style::default().fg(Color::Rgb(210, 210, 210)),
        ReviewRowKind::Add => Style::default().fg(Color::Rgb(95, 220, 145)),
        ReviewRowKind::MovedAdd => Style::default().fg(Color::Rgb(125, 125, 125)),
        ReviewRowKind::Remove => Style::default().fg(Color::Rgb(245, 115, 125)),
        ReviewRowKind::Note => Style::default().fg(Color::Rgb(160, 160, 160)),
        ReviewRowKind::Bat => Style::default().fg(Color::Rgb(210, 210, 210)),
        ReviewRowKind::Empty => Style::default().fg(Color::Rgb(160, 160, 160)),
    };
    if row.block == selected {
        style = style.bg(Color::Rgb(43, 48, 58));
        if row.kind == ReviewRowKind::Title {
            style = style.add_modifier(Modifier::BOLD);
        }
    }
    style
}

fn render_row_line(row: ReviewRow, selected: Option<usize>) -> Line<'static> {
    let style = row_style(&row, selected);
    if row.kind == ReviewRowKind::Bat || row.text.contains('\x1b') {
        ansi_styled_line(&row.text, style)
    } else {
        Line::styled(row.text, style)
    }
}

fn ansi_styled_line(text: &str, base: Style) -> Line<'static> {
    let mut spans = Vec::new();
    let mut style = base;
    let mut segment_start = 0usize;
    let bytes = text.as_bytes();
    let mut index = 0usize;

    while index < bytes.len() {
        if bytes[index] != b'\x1b' {
            index += 1;
            continue;
        }

        if index > segment_start {
            spans.push(Span::styled(text[segment_start..index].to_string(), style));
        }

        if let Some(end) = csi_sequence_end(text, index) {
            if bytes.get(index + 1) == Some(&b'[') && bytes[end - 1] == b'm' {
                apply_sgr(&mut style, base, &text[index + 2..end - 1]);
            }
            index = end;
            segment_start = index;
        } else {
            index += 1;
            segment_start = index;
        }
    }

    if segment_start < text.len() {
        spans.push(Span::styled(text[segment_start..].to_string(), style));
    }
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    Line::from(spans)
}

fn csi_sequence_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(start) != Some(&b'\x1b') || bytes.get(start + 1) != Some(&b'[') {
        return None;
    }
    bytes[start + 2..]
        .iter()
        .position(|byte| (0x40..=0x7e).contains(byte))
        .map(|offset| start + 2 + offset + 1)
}

fn apply_sgr(style: &mut Style, base: Style, params: &str) {
    let codes = sgr_codes(params);
    let mut index = 0usize;
    while index < codes.len() {
        match codes[index] {
            0 => *style = base,
            1 => *style = style.add_modifier(Modifier::BOLD),
            2 => *style = style.add_modifier(Modifier::DIM),
            3 => *style = style.add_modifier(Modifier::ITALIC),
            4 => *style = style.add_modifier(Modifier::UNDERLINED),
            9 => *style = style.add_modifier(Modifier::CROSSED_OUT),
            22 => *style = style.remove_modifier(Modifier::BOLD | Modifier::DIM),
            23 => *style = style.remove_modifier(Modifier::ITALIC),
            24 => *style = style.remove_modifier(Modifier::UNDERLINED),
            29 => *style = style.remove_modifier(Modifier::CROSSED_OUT),
            30..=37 | 90..=97 => style.fg = ansi_color(codes[index]),
            39 => style.fg = base.fg,
            40..=47 | 100..=107 => style.bg = ansi_color(codes[index] - 10),
            49 => style.bg = base.bg,
            38 | 48 => {
                let target_foreground = codes[index] == 38;
                let Some((color, consumed)) = extended_ansi_color(&codes[index + 1..]) else {
                    index += 1;
                    continue;
                };
                if target_foreground {
                    style.fg = Some(color);
                } else {
                    style.bg = Some(color);
                }
                index += consumed + 1;
                continue;
            }
            _ => {}
        }
        index += 1;
    }
}

fn sgr_codes(params: &str) -> Vec<u16> {
    if params.is_empty() {
        return vec![0];
    }
    params
        .split(';')
        .map(|param| {
            if param.is_empty() {
                0
            } else {
                param.parse::<u16>().unwrap_or(u16::MAX)
            }
        })
        .collect()
}

fn extended_ansi_color(codes: &[u16]) -> Option<(Color, usize)> {
    match codes {
        [2, r, g, b, ..] => Some((Color::Rgb(*r as u8, *g as u8, *b as u8), 4)),
        [5, index, ..] => Some((Color::Indexed(*index as u8), 2)),
        _ => None,
    }
}

fn ansi_color(code: u16) -> Option<Color> {
    Some(match code {
        30 => Color::Black,
        31 => Color::Red,
        32 => Color::Green,
        33 => Color::Yellow,
        34 => Color::Blue,
        35 => Color::Magenta,
        36 => Color::Cyan,
        37 => Color::Gray,
        90 => Color::DarkGray,
        91 => Color::LightRed,
        92 => Color::LightGreen,
        93 => Color::LightYellow,
        94 => Color::LightBlue,
        95 => Color::LightMagenta,
        96 => Color::LightCyan,
        97 => Color::White,
        _ => return None,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReviewApp {
    blocks: Vec<ReviewBlock>,
    selected: Option<usize>,
    scroll_top: usize,
    viewport_width: usize,
    viewport_height: usize,
    should_quit: bool,
}

const DEFAULT_VIEWPORT_WIDTH: usize = u16::MAX as usize;

impl ReviewApp {
    #[cfg(test)]
    fn from_input(input: &str) -> Self {
        Self::from_input_with_color(input, false)
    }

    fn from_input_with_color(input: &str, use_color: bool) -> Self {
        Self::new(build_blocks_with_color(
            &parse_unified_diff(input),
            use_color,
        ))
    }

    fn new(blocks: Vec<ReviewBlock>) -> Self {
        let selected = if blocks.is_empty() { None } else { Some(0) };
        Self {
            blocks,
            selected,
            scroll_top: 0,
            viewport_width: DEFAULT_VIEWPORT_WIDTH,
            viewport_height: 1,
            should_quit: false,
        }
    }

    fn handle_key(&mut self, key: KeyCode) -> ReviewAction {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.should_quit = true;
                ReviewAction::Quit
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.scroll_down_line();
                ReviewAction::None
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.scroll_up_line();
                ReviewAction::None
            }
            KeyCode::Char('J') | KeyCode::PageDown | KeyCode::Char(' ') => {
                self.select_next();
                ReviewAction::None
            }
            KeyCode::Char('K') | KeyCode::PageUp => {
                self.select_previous();
                ReviewAction::None
            }
            KeyCode::Char('g') | KeyCode::Home => {
                self.select_first();
                ReviewAction::None
            }
            KeyCode::Char('G') | KeyCode::End => {
                self.select_last_and_scroll_to_end();
                ReviewAction::None
            }
            KeyCode::Char('f') | KeyCode::Enter => self.toggle_selected_fold(),
            _ => ReviewAction::None,
        }
    }

    fn set_viewport(&mut self, width: usize, height: usize) {
        self.viewport_width = width.max(1);
        self.viewport_height = height.max(1);
        self.clamp_scroll();
    }

    #[cfg(test)]
    fn set_viewport_height(&mut self, height: usize) {
        self.viewport_height = height.max(1);
        self.clamp_scroll();
    }

    fn select_next(&mut self) {
        let Some(selected) = self.selected else {
            return;
        };
        if selected + 1 < self.blocks.len() {
            self.selected = Some(selected + 1);
            self.scroll_selected_to_start();
        } else {
            self.scroll_to_end();
        }
    }

    fn select_previous(&mut self) {
        let Some(selected) = self.selected else {
            return;
        };
        if selected > 0 {
            self.selected = Some(selected - 1);
            self.scroll_selected_to_start();
        } else {
            self.scroll_top = 0;
        }
    }

    fn scroll_down_line(&mut self) {
        self.scroll_top = self.scroll_top.saturating_add(1);
        self.clamp_scroll();
    }

    fn scroll_up_line(&mut self) {
        self.scroll_top = self.scroll_top.saturating_sub(1);
    }

    fn select_first(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        self.selected = Some(0);
        self.scroll_top = 0;
    }

    fn select_last_and_scroll_to_end(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        self.selected = Some(self.blocks.len() - 1);
        self.scroll_to_end();
    }

    fn toggle_selected_fold(&mut self) -> ReviewAction {
        let Some(selected) = self.selected else {
            return ReviewAction::None;
        };
        let was_folded = self.blocks[selected].folded;
        self.blocks[selected].folded = !was_folded;
        if was_folded {
            self.scroll_selected_to_start();
        } else {
            self.select_next();
        }
        ReviewAction::FoldStateChanged
    }

    fn apply_fold_state(&mut self, state: &FoldState) {
        let folded = state.folded.iter().collect::<BTreeSet<_>>();
        for block in &mut self.blocks {
            block.folded = folded.contains(&block.title);
        }
        self.clamp_scroll();
    }

    fn fold_state(&self) -> FoldState {
        FoldState {
            folded: self
                .blocks
                .iter()
                .filter(|block| block.folded)
                .map(|block| block.title.clone())
                .collect(),
        }
    }

    fn scroll_selected_to_start(&mut self) {
        if let Some(selected) = self.selected {
            self.scroll_top = self.block_start(selected);
            self.clamp_scroll();
        }
    }

    fn scroll_to_end(&mut self) {
        self.scroll_top = self.max_scroll();
    }

    fn clamp_scroll(&mut self) {
        self.scroll_top = min(self.scroll_top, self.max_scroll());
    }

    fn max_scroll(&self) -> usize {
        self.visible_height().saturating_sub(self.viewport_height)
    }

    fn block_start(&self, target: usize) -> usize {
        self.rows()
            .iter()
            .take_while(|row| row.block.map(|block| block < target).unwrap_or(false))
            .map(|row| self.row_height(row))
            .sum()
    }

    fn rows(&self) -> Vec<ReviewRow> {
        if self.blocks.is_empty() {
            return vec![ReviewRow {
                block: None,
                kind: ReviewRowKind::Empty,
                text: "no diff chunks".to_string(),
            }];
        }

        let mut rows = Vec::new();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let prefix = if block.folded { ">" } else { "v" };
            let folded = if block.folded {
                format!(" ({} lines folded)", block.lines.len())
            } else {
                String::new()
            };
            rows.push(ReviewRow {
                block: Some(block_idx),
                kind: ReviewRowKind::Title,
                text: format!("{prefix} {}{folded}", block.title),
            });
            if block.folded {
                continue;
            }
            rows.extend(block.lines.iter().map(|line| ReviewRow {
                block: Some(block_idx),
                kind: line.kind,
                text: line.render(),
            }));
        }
        rows
    }

    fn status_text(&self) -> String {
        let progress = self.view_progress();
        let progress_text = format!(
            "current-chunk {}  current-file {}  current-review {}",
            progress.chunk.render(),
            progress.file.render(),
            progress.review.render()
        );
        match progress.block_index {
            Some(block_index) => format!(
                "{progress_text}  {}  j/k scroll  J/K folds  f fold  G end  q quit",
                self.blocks[block_index].title
            ),
            None => format!("{progress_text}  no diff chunks  q quit"),
        }
    }

    fn view_progress(&self) -> ViewProgress {
        let total_rows = self.visible_height();
        if total_rows == 0 {
            return ViewProgress::empty();
        }

        let top_row = self.scroll_top.min(total_rows - 1);
        let viewport_end = self.viewport_end(total_rows);
        let review = Progress::new(viewport_end, total_rows);
        let Some((block_index, block_start, block)) = self.block_at_visible_row(top_row) else {
            return ViewProgress {
                review,
                ..ViewProgress::empty()
            };
        };
        let block_height = self.block_height(block_index);
        let block_end = block_start + block_height;
        let chunk = Progress::new(viewport_end.min(block_end) - block_start, block_height);
        let file = self.file_progress(block.file_index, viewport_end);
        ViewProgress {
            block_index: Some(block_index),
            chunk,
            file,
            review,
        }
    }

    fn visible_height(&self) -> usize {
        if self.blocks.is_empty() {
            return 0;
        }
        self.rows().iter().map(|row| self.row_height(row)).sum()
    }

    fn viewport_end(&self, total_rows: usize) -> usize {
        self.scroll_top
            .saturating_add(self.viewport_height)
            .min(total_rows)
    }

    fn block_at_visible_row(&self, row: usize) -> Option<(usize, usize, &ReviewBlock)> {
        let mut row_start = 0usize;
        let mut block_start = 0usize;
        let mut current_block = None;
        for review_row in self.rows() {
            let Some(block_index) = review_row.block else {
                continue;
            };
            if current_block != Some(block_index) {
                current_block = Some(block_index);
                block_start = row_start;
            }
            let row_height = self.row_height(&review_row);
            let row_end = row_start + row_height;
            if row < row_end {
                return Some((block_index, block_start, &self.blocks[block_index]));
            }
            row_start = row_end;
        }
        None
    }

    fn file_progress(&self, file_index: Option<usize>, viewport_end: usize) -> Progress {
        let Some(file_index) = file_index else {
            return Progress::empty();
        };

        let mut block_start = 0usize;
        let mut file_total = 0usize;
        let mut file_current = 0usize;
        for (block_index, block) in self.blocks.iter().enumerate() {
            let block_height = self.block_height(block_index);
            if block.file_index == Some(file_index) {
                file_current += viewport_end.saturating_sub(block_start).min(block_height);
                file_total += block_height;
            }
            block_start += block_height;
        }

        if file_current == 0 {
            Progress::empty()
        } else {
            Progress::new(file_current, file_total)
        }
    }

    fn block_height(&self, block_index: usize) -> usize {
        self.rows()
            .iter()
            .filter(|row| row.block == Some(block_index))
            .map(|row| self.row_height(row))
            .sum()
    }

    fn row_height(&self, row: &ReviewRow) -> usize {
        let line = render_row_line(row.clone(), None);
        Paragraph::new(line)
            .wrap(Wrap { trim: false })
            .line_count(self.viewport_width.min(u16::MAX as usize) as u16)
            .max(1)
    }

    #[cfg(test)]
    fn visible_text(&self) -> Vec<String> {
        self.rows()
            .into_iter()
            .skip(self.scroll_top)
            .take(self.viewport_height)
            .map(|row| row.text)
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ViewProgress {
    block_index: Option<usize>,
    chunk: Progress,
    file: Progress,
    review: Progress,
}

impl ViewProgress {
    fn empty() -> Self {
        Self {
            block_index: None,
            chunk: Progress::empty(),
            file: Progress::empty(),
            review: Progress::empty(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Progress {
    current: usize,
    total: usize,
}

impl Progress {
    fn empty() -> Self {
        Self {
            current: 0,
            total: 0,
        }
    }

    fn new(current: usize, total: usize) -> Self {
        if total == 0 {
            return Self::empty();
        }
        Self {
            current: current.clamp(1, total),
            total,
        }
    }

    fn percent(self) -> usize {
        if self.total == 0 {
            0
        } else {
            self.current.saturating_mul(100) / self.total
        }
    }

    fn render(self) -> String {
        format!("{}/{} {}%", self.current, self.total, self.percent())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReviewAction {
    None,
    Quit,
    FoldStateChanged,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
struct FoldState {
    folded: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FoldStateStore {
    path: PathBuf,
}

impl FoldStateStore {
    fn for_current_ref() -> io::Result<Option<Self>> {
        Self::for_current_ref_in(None)
    }

    fn for_current_ref_in(cwd: Option<&Path>) -> io::Result<Option<Self>> {
        let Some(ref_name) = current_git_ref(cwd)? else {
            return Ok(None);
        };
        let Some(git_dir) = git_output(cwd, &["rev-parse", "--absolute-git-dir"])? else {
            return Ok(None);
        };
        Ok(Some(Self {
            path: PathBuf::from(git_dir).join("sidreview").join(ref_name),
        }))
    }

    fn load(&self) -> io::Result<Option<FoldState>> {
        let payload = match fs::read(&self.path) {
            Ok(payload) => payload,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        };
        serde_json::from_slice(&payload)
            .map(Some)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
    }

    fn save(&self, state: &FoldState) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut payload = serde_json::to_vec_pretty(state).map_err(io::Error::other)?;
        payload.push(b'\n');
        let temp_path = self.path.with_extension("tmp");
        fs::write(&temp_path, payload)?;
        fs::rename(temp_path, &self.path)
    }
}

fn current_git_ref(cwd: Option<&Path>) -> io::Result<Option<String>> {
    if let Some(ref_name) = git_output(cwd, &["symbolic-ref", "-q", "HEAD"])? {
        return Ok(Some(ref_name));
    }
    let Some(tag_refs) = git_output(
        cwd,
        &[
            "for-each-ref",
            "--points-at",
            "HEAD",
            "--format=%(refname)",
            "refs/tags",
        ],
    )?
    else {
        return Ok(None);
    };
    Ok(tag_refs
        .lines()
        .find(|line| *line == "refs/tags/latest")
        .or_else(|| tag_refs.lines().next())
        .map(str::to_string))
}

fn git_output(cwd: Option<&Path>, args: &[&str]) -> io::Result<Option<String>> {
    let mut command = Command::new("git");
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = match command.args(args).output() {
        Ok(output) => output,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err),
    };
    if !output.status.success() {
        return Ok(None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_string()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReviewBlock {
    title: String,
    lines: Vec<ReviewLine>,
    folded: bool,
    file_index: Option<usize>,
}

impl ReviewBlock {
    fn new(title: String, lines: Vec<ReviewLine>) -> Self {
        Self {
            title,
            lines,
            folded: false,
            file_index: None,
        }
    }

    fn for_file(file_index: usize, title: String, lines: Vec<ReviewLine>) -> Self {
        Self {
            title,
            lines,
            folded: false,
            file_index: Some(file_index),
        }
    }

    #[cfg(test)]
    fn height(&self) -> usize {
        1 + if self.folded { 0 } else { self.lines.len() }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReviewLine {
    kind: ReviewRowKind,
    old_lineno: Option<usize>,
    new_lineno: Option<usize>,
    marker: &'static str,
    content: String,
}

impl ReviewLine {
    fn header(content: impl Into<String>) -> Self {
        Self {
            kind: ReviewRowKind::Header,
            old_lineno: None,
            new_lineno: None,
            marker: " ",
            content: content.into(),
        }
    }

    fn bat(content: impl Into<String>) -> Self {
        Self {
            kind: ReviewRowKind::Bat,
            old_lineno: None,
            new_lineno: None,
            marker: " ",
            content: content.into(),
        }
    }

    fn from_diff_line_with_content(
        line: &DiffLine,
        content: Option<&str>,
        muted_moved_addition: bool,
    ) -> Self {
        let (kind, marker, content) = match line.op {
            DiffOp::Context => (
                ReviewRowKind::Context,
                " ",
                content.unwrap_or(line.content.as_str()),
            ),
            DiffOp::Add => (
                if muted_moved_addition {
                    ReviewRowKind::MovedAdd
                } else {
                    ReviewRowKind::Add
                },
                "+",
                content.unwrap_or(line.content.as_str()),
            ),
            DiffOp::Remove => (
                ReviewRowKind::Remove,
                "-",
                content.unwrap_or(line.content.as_str()),
            ),
            DiffOp::Note => (
                ReviewRowKind::Note,
                "\\",
                content
                    .unwrap_or_else(|| line.content.strip_prefix("\\ ").unwrap_or(&line.content)),
            ),
        };
        Self {
            kind,
            old_lineno: line.old_lineno,
            new_lineno: line.new_lineno,
            marker,
            content: content.to_string(),
        }
    }

    fn render(&self) -> String {
        if self.kind == ReviewRowKind::Header {
            return format!("     {}", self.content);
        }
        if self.kind == ReviewRowKind::Bat {
            return self.content.clone();
        }
        format!(
            "{:>4} {:>4} {} {}",
            display_lineno(self.old_lineno),
            display_lineno(self.new_lineno),
            self.marker,
            self.content
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReviewRowKind {
    Title,
    Header,
    Context,
    Add,
    MovedAdd,
    Remove,
    Note,
    Bat,
    Empty,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReviewRow {
    block: Option<usize>,
    kind: ReviewRowKind,
    text: String,
}

#[cfg(test)]
fn build_blocks(diff: &Diff) -> Vec<ReviewBlock> {
    build_blocks_with_color(diff, false)
}

fn build_blocks_with_color(diff: &Diff, use_color: bool) -> Vec<ReviewBlock> {
    let metadata = build_review_line_metadata(diff, use_color);
    let mut blocks = Vec::new();
    if !diff.preamble.is_empty() {
        blocks.push(ReviewBlock::new(
            "preamble".to_string(),
            diff.preamble.iter().map(ReviewLine::header).collect(),
        ));
    }

    for (file_idx, file) in diff.files.iter().enumerate() {
        if file_is_pure_addition(file)
            && let Ok(rendered) = render_pure_addition_file_with_bat(file, use_color)
        {
            let mut lines = Vec::new();
            lines.extend(file.header.iter().map(ReviewLine::header));
            lines.extend(rendered.lines().map(ReviewLine::bat));
            blocks.push(ReviewBlock::for_file(
                file_idx,
                format!("file {}", display_path(file)),
                lines,
            ));
            continue;
        }

        if file.hunks.is_empty() {
            blocks.push(ReviewBlock::for_file(
                file_idx,
                format!("file {}", display_path(file)),
                file.header.iter().map(ReviewLine::header).collect(),
            ));
            continue;
        }

        for (hunk_idx, hunk) in file.hunks.iter().enumerate() {
            let mut lines = Vec::new();
            if hunk_idx == 0 {
                lines.extend(file.header.iter().map(ReviewLine::header));
            }
            lines.push(ReviewLine::header(&hunk.header));
            lines.extend(hunk.lines.iter().enumerate().map(|(line_idx, line)| {
                let muted_moved_addition =
                    metadata.is_muted_moved_addition(file_idx, hunk_idx, line_idx);
                let content = if muted_moved_addition {
                    None
                } else {
                    metadata.get(file_idx, hunk_idx, line_idx)
                };
                ReviewLine::from_diff_line_with_content(line, content, muted_moved_addition)
            }));
            blocks.push(ReviewBlock::for_file(
                file_idx,
                format!("{} {}", display_path(file), hunk.header),
                lines,
            ));
        }
    }
    blocks
}

fn display_path(file: &DiffFile) -> String {
    file.new_path
        .as_ref()
        .or(file.old_path.as_ref())
        .cloned()
        .unwrap_or_else(|| "(unknown)".to_string())
}

fn display_lineno(lineno: Option<usize>) -> String {
    lineno
        .map(|lineno| lineno.to_string())
        .unwrap_or_else(|| " ".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> io::Result<Self> {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("sidreview-{name}-{}-{unique}", process::id()));
            fs::create_dir(&path)?;
            let path = fs::canonicalize(path)?;
            Ok(Self { path })
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn run_git(cwd: &Path, args: &[&str]) -> bool {
        Command::new("git")
            .current_dir(cwd)
            .args(args)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    const TWO_HUNKS: &str = "\
diff --git a/a.rs b/a.rs
--- a/a.rs
+++ b/a.rs
@@ -1,2 +1,2 @@
-old_one
+new_one
 keep_one
@@ -10,2 +10,2 @@
-old_two
+new_two
 keep_two
";

    const THREE_HUNKS: &str = "\
diff --git a/a.rs b/a.rs
--- a/a.rs
+++ b/a.rs
@@ -1,2 +1,2 @@
-old_one
+new_one
 keep_one
@@ -10,2 +10,2 @@
-old_two
+new_two
 keep_two
@@ -20,8 +20,8 @@
-old_three
+new_three
 keep_three
 keep_four
 keep_five
 keep_six
 keep_seven
 keep_eight
 keep_nine
";

    const PURE_ADDITION: &str = "\
diff --git a/new.rs b/new.rs
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/new.rs
@@ -0,0 +1,3 @@
+fn main() {
+    println!(\"hello\");
+}
";

    const TWO_FILES: &str = "\
diff --git a/a.rs b/a.rs
--- a/a.rs
+++ b/a.rs
@@ -1,1 +1,1 @@
-old_a
+new_a
diff --git a/b.rs b/b.rs
--- a/b.rs
+++ b/b.rs
@@ -1,1 +1,1 @@
-old_b
+new_b
";

    const MOVED_BLOCKS: &str = "\
diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1,4 +1,1 @@
-moved alpha
-moved beta
-moved gamma
 stay
@@ -10,1 +7,4 @@
 elsewhere
+moved alpha
+moved beta
+moved gamma
";

    #[test]
    fn ansi_styled_line_parses_sgr_colors_and_reset() {
        let base = Style::default().bg(Color::Rgb(43, 48, 58));
        let line = ansi_styled_line("plain \x1b[38;2;1;2;3mbold\x1b[0m base", base);

        assert_eq!(line.spans.len(), 3);
        assert_eq!(line.spans[0].content, "plain ");
        assert_eq!(line.spans[0].style, base);
        assert_eq!(line.spans[1].content, "bold");
        assert_eq!(line.spans[1].style.fg, Some(Color::Rgb(1, 2, 3)));
        assert_eq!(line.spans[1].style.bg, base.bg);
        assert_eq!(line.spans[2].content, " base");
        assert_eq!(line.spans[2].style, base);
    }

    #[test]
    fn diff_rows_parse_embedded_ansi_colors() {
        let line = render_row_line(
            ReviewRow {
                block: Some(0),
                kind: ReviewRowKind::Add,
                text: "       1 + \x1b[38;2;1;2;3mfn main\x1b[0m".to_string(),
            },
            None,
        );

        assert_eq!(line.spans.len(), 2);
        assert_eq!(line.spans[0].content, "       1 + ");
        assert_eq!(line.spans[1].content, "fn main");
        assert_eq!(line.spans[1].style.fg, Some(Color::Rgb(1, 2, 3)));
    }

    #[test]
    fn paired_change_rows_use_semantic_spans() {
        let input = "\
diff --git a/query.rs b/query.rs
--- a/query.rs
+++ b/query.rs
@@ -64,1 +64,1 @@
-pub fn with_step(&self, step: Time) -> Result<QueryParams, Error> {
+pub fn with_step(&self, step: Time) -> Result<QueryParams, SError> {
";
        let blocks = build_blocks_with_color(&parse_unified_diff(input), true);
        let added = blocks[0]
            .lines
            .iter()
            .find(|line| line.kind == ReviewRowKind::Add)
            .unwrap();
        assert!(added.content.contains('\x1b'));

        let rendered = render_row_line(
            ReviewRow {
                block: Some(0),
                kind: added.kind,
                text: added.render(),
            },
            None,
        );
        let bright_add = Some(Color::Rgb(80, 235, 150));
        assert!(
            rendered
                .spans
                .iter()
                .any(|span| span.content == "SError" && span.style.fg == bright_add)
        );
        assert!(rendered.spans.iter().all(|span| {
            !(span.content.contains("pub")
                || span.content.contains("with_step")
                || span.content.contains("Result"))
                || span.style.fg != bright_add
        }));
    }

    #[test]
    fn review_changed_lines_highlight_only_changed_words() {
        let input = "\
diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1,1 +1,1 @@
-alpha old beta
+alpha new beta
";
        let blocks = build_blocks_with_color(&parse_unified_diff(input), true);
        let removed = blocks[0]
            .lines
            .iter()
            .find(|line| line.kind == ReviewRowKind::Remove)
            .unwrap();
        let added = blocks[0]
            .lines
            .iter()
            .find(|line| line.kind == ReviewRowKind::Add)
            .unwrap();

        assert!(removed.content.contains("\x1b[38;2;165;95;100malpha"));
        assert!(removed.content.contains("\x1b[38;2;255;105;115mold"));
        assert!(!removed.content.contains("\x1b[38;2;255;105;115malpha"));

        assert!(added.content.contains("\x1b[38;2;100;155;120malpha"));
        assert!(added.content.contains("\x1b[38;2;80;235;150mnew"));
        assert!(!added.content.contains("\x1b[38;2;80;235;150malpha"));
    }

    #[test]
    fn moved_addition_rows_render_muted_gray() {
        let blocks = build_blocks_with_color(&parse_unified_diff(MOVED_BLOCKS), true);
        let moved = blocks[1]
            .lines
            .iter()
            .find(|line| line.content == "moved alpha")
            .unwrap();

        assert_eq!(moved.kind, ReviewRowKind::MovedAdd);
        assert_eq!(moved.render(), "        8 + moved alpha");

        let rendered = render_row_line(
            ReviewRow {
                block: Some(1),
                kind: moved.kind,
                text: moved.render(),
            },
            None,
        );
        assert_eq!(rendered.style.fg, Some(Color::Rgb(125, 125, 125)));
    }

    #[test]
    fn pure_addition_file_uses_bat_block_when_available() {
        let diff = parse_unified_diff(PURE_ADDITION);
        if render_pure_addition_file_with_bat(&diff.files[0], false).is_err() {
            return;
        }

        let blocks = build_blocks_with_color(&diff, false);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].title, "file new.rs");
        assert!(
            blocks[0]
                .lines
                .iter()
                .any(|line| line.kind == ReviewRowKind::Bat && line.content == "fn main() {")
        );
        assert!(
            !blocks[0]
                .lines
                .iter()
                .any(|line| line.kind == ReviewRowKind::Add)
        );
        assert!(
            !blocks[0]
                .lines
                .iter()
                .any(|line| line.content == "@@ -0,0 +1,3 @@")
        );
    }

    #[test]
    fn builds_one_selectable_block_per_hunk() {
        assert_eq!(
            build_blocks(&parse_unified_diff(TWO_HUNKS)),
            vec![
                ReviewBlock {
                    title: "a.rs @@ -1,2 +1,2 @@".to_string(),
                    lines: vec![
                        ReviewLine {
                            kind: ReviewRowKind::Header,
                            old_lineno: None,
                            new_lineno: None,
                            marker: " ",
                            content: "diff --git a/a.rs b/a.rs".to_string(),
                        },
                        ReviewLine {
                            kind: ReviewRowKind::Header,
                            old_lineno: None,
                            new_lineno: None,
                            marker: " ",
                            content: "--- a/a.rs".to_string(),
                        },
                        ReviewLine {
                            kind: ReviewRowKind::Header,
                            old_lineno: None,
                            new_lineno: None,
                            marker: " ",
                            content: "+++ b/a.rs".to_string(),
                        },
                        ReviewLine {
                            kind: ReviewRowKind::Header,
                            old_lineno: None,
                            new_lineno: None,
                            marker: " ",
                            content: "@@ -1,2 +1,2 @@".to_string(),
                        },
                        ReviewLine {
                            kind: ReviewRowKind::Remove,
                            old_lineno: Some(1),
                            new_lineno: None,
                            marker: "-",
                            content: "old_one".to_string(),
                        },
                        ReviewLine {
                            kind: ReviewRowKind::Add,
                            old_lineno: None,
                            new_lineno: Some(1),
                            marker: "+",
                            content: "new_one".to_string(),
                        },
                        ReviewLine {
                            kind: ReviewRowKind::Context,
                            old_lineno: Some(2),
                            new_lineno: Some(2),
                            marker: " ",
                            content: "keep_one".to_string(),
                        },
                    ],
                    folded: false,
                    file_index: Some(0),
                },
                ReviewBlock {
                    title: "a.rs @@ -10,2 +10,2 @@".to_string(),
                    lines: vec![
                        ReviewLine {
                            kind: ReviewRowKind::Header,
                            old_lineno: None,
                            new_lineno: None,
                            marker: " ",
                            content: "@@ -10,2 +10,2 @@".to_string(),
                        },
                        ReviewLine {
                            kind: ReviewRowKind::Remove,
                            old_lineno: Some(10),
                            new_lineno: None,
                            marker: "-",
                            content: "old_two".to_string(),
                        },
                        ReviewLine {
                            kind: ReviewRowKind::Add,
                            old_lineno: None,
                            new_lineno: Some(10),
                            marker: "+",
                            content: "new_two".to_string(),
                        },
                        ReviewLine {
                            kind: ReviewRowKind::Context,
                            old_lineno: Some(11),
                            new_lineno: Some(11),
                            marker: " ",
                            content: "keep_two".to_string(),
                        },
                    ],
                    folded: false,
                    file_index: Some(0),
                },
            ]
        );
    }

    #[test]
    fn status_line_reports_chunk_file_and_review_progress() {
        let app = ReviewApp::from_input(TWO_HUNKS);

        assert_eq!(
            app.status_text(),
            "current-chunk 1/8 12%  current-file 1/13 7%  current-review 1/13 7%  a.rs @@ -1,2 +1,2 @@  j/k scroll  J/K folds  f fold  G end  q quit"
        );
    }

    #[test]
    fn status_progress_includes_visible_rows_when_line_scrolling() {
        let mut app = ReviewApp::from_input(TWO_HUNKS);
        app.set_viewport_height(4);
        for _ in 0..8 {
            app.handle_key(KeyCode::Char('j'));
        }

        assert_eq!(app.selected, Some(0));
        assert_eq!(
            app.status_text(),
            "current-chunk 4/5 80%  current-file 12/13 92%  current-review 12/13 92%  a.rs @@ -10,2 +10,2 @@  j/k scroll  J/K folds  f fold  G end  q quit"
        );
    }

    #[test]
    fn status_file_progress_resets_per_file() {
        let mut app = ReviewApp::from_input(TWO_FILES);
        app.select_next();

        assert_eq!(
            app.status_text(),
            "current-chunk 1/7 14%  current-file 1/7 14%  current-review 8/14 57%  b.rs @@ -1,1 +1,1 @@  j/k scroll  J/K folds  f fold  G end  q quit"
        );
    }

    #[test]
    fn next_moves_selection_to_next_chunk_and_scrolls_to_its_start() {
        let mut app = ReviewApp::from_input(THREE_HUNKS);
        app.set_viewport_height(4);
        app.select_next();
        assert_eq!(
            app,
            ReviewApp {
                blocks: build_blocks(&parse_unified_diff(THREE_HUNKS)),
                selected: Some(1),
                scroll_top: 8,
                viewport_width: DEFAULT_VIEWPORT_WIDTH,
                viewport_height: 4,
                should_quit: false,
            }
        );
        assert_eq!(
            app.visible_text(),
            vec![
                "v a.rs @@ -10,2 +10,2 @@".to_string(),
                "     @@ -10,2 +10,2 @@".to_string(),
                "  10      - old_two".to_string(),
                "       10 + new_two".to_string(),
            ]
        );
    }

    #[test]
    fn repeated_next_can_scroll_to_the_end_of_the_last_chunk() {
        let mut app = ReviewApp::from_input(THREE_HUNKS);
        app.set_viewport_height(4);
        app.select_next();
        app.select_next();
        assert_eq!(
            app.visible_text(),
            vec![
                "v a.rs @@ -20,8 +20,8 @@".to_string(),
                "     @@ -20,8 +20,8 @@".to_string(),
                "  20      - old_three".to_string(),
                "       20 + new_three".to_string(),
            ]
        );
        app.select_next();
        assert_eq!(
            app,
            ReviewApp {
                blocks: build_blocks(&parse_unified_diff(THREE_HUNKS)),
                selected: Some(2),
                scroll_top: 20,
                viewport_width: DEFAULT_VIEWPORT_WIDTH,
                viewport_height: 4,
                should_quit: false,
            }
        );
        assert_eq!(
            app.visible_text(),
            vec![
                "  24   24   keep_six".to_string(),
                "  25   25   keep_seven".to_string(),
                "  26   26   keep_eight".to_string(),
                "  27   27   keep_nine".to_string(),
            ]
        );
        assert_eq!(
            app.status_text(),
            "current-chunk 11/11 100%  current-file 24/24 100%  current-review 24/24 100%  a.rs @@ -20,8 +20,8 @@  j/k scroll  J/K folds  f fold  G end  q quit"
        );
    }

    #[test]
    fn lowercase_j_and_k_scroll_one_visible_line() {
        let mut app = ReviewApp::from_input(THREE_HUNKS);
        app.set_viewport_height(4);

        assert_eq!(app.handle_key(KeyCode::Char('j')), ReviewAction::None);
        assert_eq!(app.selected, Some(0));
        assert_eq!(app.scroll_top, 1);
        assert_eq!(
            app.visible_text(),
            vec![
                "     diff --git a/a.rs b/a.rs".to_string(),
                "     --- a/a.rs".to_string(),
                "     +++ b/a.rs".to_string(),
                "     @@ -1,2 +1,2 @@".to_string(),
            ]
        );

        assert_eq!(app.handle_key(KeyCode::Char('k')), ReviewAction::None);
        assert_eq!(app.selected, Some(0));
        assert_eq!(app.scroll_top, 0);
    }

    #[test]
    fn uppercase_j_and_k_move_one_fold_at_a_time() {
        let mut app = ReviewApp::from_input(THREE_HUNKS);
        app.set_viewport_height(4);

        assert_eq!(app.handle_key(KeyCode::Char('J')), ReviewAction::None);
        assert_eq!(app.selected, Some(1));
        assert_eq!(app.scroll_top, 8);

        assert_eq!(app.handle_key(KeyCode::Char('K')), ReviewAction::None);
        assert_eq!(app.selected, Some(0));
        assert_eq!(app.scroll_top, 0);
    }

    #[test]
    fn wrapped_rows_count_for_scroll_and_chunk_navigation() {
        let input = "\
diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1,1 +1,1 @@
-old_one
+alpha beta gamma delta epsilon zeta eta theta iota kappa
@@ -10,1 +10,1 @@
-old_two
+new_two
";
        let mut app = ReviewApp::from_input(input);
        app.set_viewport(24, 4);

        let wrapped_first_height = app.block_height(0);
        assert!(wrapped_first_height > app.blocks[0].height());

        app.select_next();
        assert_eq!(app.selected, Some(1));
        assert_eq!(app.scroll_top, wrapped_first_height);

        for _ in 0..100 {
            app.handle_key(KeyCode::Char('j'));
        }
        assert_eq!(app.scroll_top, app.max_scroll());
        assert!(app.max_scroll() > app.rows().len().saturating_sub(app.viewport_height));
    }

    #[test]
    fn folding_selected_block_advances_to_next_block() {
        let mut app = ReviewApp::from_input(THREE_HUNKS);
        app.set_viewport_height(4);

        assert_eq!(
            app.handle_key(KeyCode::Char('f')),
            ReviewAction::FoldStateChanged
        );
        assert!(app.blocks[0].folded);
        assert!(!app.blocks[1].folded);
        assert_eq!(app.selected, Some(1));
        assert_eq!(app.scroll_top, 1);

        assert_eq!(
            app.handle_key(KeyCode::Char('f')),
            ReviewAction::FoldStateChanged
        );
        assert!(app.blocks[1].folded);
        assert!(!app.blocks[2].folded);
        assert_eq!(app.selected, Some(2));
        assert_eq!(app.scroll_top, 2);
    }

    #[test]
    fn unfolding_selected_block_does_not_advance() {
        let mut app = ReviewApp::from_input(THREE_HUNKS);
        app.set_viewport_height(4);
        app.blocks[1].folded = true;
        app.selected = Some(1);
        app.scroll_selected_to_start();

        assert_eq!(
            app.handle_key(KeyCode::Char('f')),
            ReviewAction::FoldStateChanged
        );
        assert!(!app.blocks[1].folded);
        assert_eq!(app.selected, Some(1));
        assert_eq!(app.scroll_top, 8);
    }

    #[test]
    fn line_scroll_clamps_at_document_edges() {
        let mut app = ReviewApp::from_input(THREE_HUNKS);
        app.set_viewport_height(4);

        for _ in 0..100 {
            app.handle_key(KeyCode::Char('j'));
        }
        assert_eq!(app.scroll_top, app.max_scroll());
        assert_eq!(app.selected, Some(0));
        assert_eq!(
            app.visible_text(),
            vec![
                "  24   24   keep_six".to_string(),
                "  25   25   keep_seven".to_string(),
                "  26   26   keep_eight".to_string(),
                "  27   27   keep_nine".to_string(),
            ]
        );

        for _ in 0..100 {
            app.handle_key(KeyCode::Char('k'));
        }
        assert_eq!(app.scroll_top, 0);
        assert_eq!(app.selected, Some(0));
    }

    #[test]
    fn folding_selected_block_leaves_other_blocks_expanded() {
        let mut app = ReviewApp::from_input(TWO_HUNKS);
        app.set_viewport_height(20);
        app.toggle_selected_fold();
        assert_eq!(
            app.rows(),
            vec![
                ReviewRow {
                    block: Some(0),
                    kind: ReviewRowKind::Title,
                    text: "> a.rs @@ -1,2 +1,2 @@ (7 lines folded)".to_string(),
                },
                ReviewRow {
                    block: Some(1),
                    kind: ReviewRowKind::Title,
                    text: "v a.rs @@ -10,2 +10,2 @@".to_string(),
                },
                ReviewRow {
                    block: Some(1),
                    kind: ReviewRowKind::Header,
                    text: "     @@ -10,2 +10,2 @@".to_string(),
                },
                ReviewRow {
                    block: Some(1),
                    kind: ReviewRowKind::Remove,
                    text: "  10      - old_two".to_string(),
                },
                ReviewRow {
                    block: Some(1),
                    kind: ReviewRowKind::Add,
                    text: "       10 + new_two".to_string(),
                },
                ReviewRow {
                    block: Some(1),
                    kind: ReviewRowKind::Context,
                    text: "  11   11   keep_two".to_string(),
                },
            ]
        );
    }

    #[test]
    fn fold_state_round_trips_as_json() {
        let temp = TestDir::new("fold-json").unwrap();
        let store = FoldStateStore {
            path: temp.path.join(".git/sidreview/refs/heads/main"),
        };

        let mut app = ReviewApp::from_input(TWO_HUNKS);
        app.select_next();
        assert_eq!(app.toggle_selected_fold(), ReviewAction::FoldStateChanged);
        store.save(&app.fold_state()).unwrap();

        let saved: serde_json::Value =
            serde_json::from_slice(&fs::read(&store.path).unwrap()).unwrap();
        assert_eq!(saved, json!({"folded": ["a.rs @@ -10,2 +10,2 @@"]}));

        let mut restored = ReviewApp::from_input(TWO_HUNKS);
        restored.apply_fold_state(&store.load().unwrap().unwrap());
        assert!(!restored.blocks[0].folded);
        assert!(restored.blocks[1].folded);
    }

    #[test]
    fn applying_fold_state_before_first_draw_keeps_first_block_at_top() {
        let mut app = ReviewApp::from_input(TWO_HUNKS);
        app.apply_fold_state(&FoldState::default());
        app.set_viewport_height(4);

        assert_eq!(app.scroll_top, 0);
        assert_eq!(
            app.visible_text(),
            vec![
                "v a.rs @@ -1,2 +1,2 @@".to_string(),
                "     diff --git a/a.rs b/a.rs".to_string(),
                "     --- a/a.rs".to_string(),
                "     +++ b/a.rs".to_string(),
            ]
        );
    }

    #[test]
    fn fold_state_save_records_unfolds() {
        let temp = TestDir::new("fold-unfold").unwrap();
        let store = FoldStateStore {
            path: temp.path.join(".git/sidreview/refs/heads/main"),
        };

        let mut app = ReviewApp::from_input(TWO_HUNKS);
        app.toggle_selected_fold();
        app.select_first();
        app.toggle_selected_fold();
        store.save(&app.fold_state()).unwrap();

        let saved: serde_json::Value =
            serde_json::from_slice(&fs::read(&store.path).unwrap()).unwrap();
        assert_eq!(saved, json!({"folded": []}));
    }

    #[test]
    fn fold_state_store_uses_current_branch_ref() {
        let temp = TestDir::new("branch-ref").unwrap();
        if !run_git(&temp.path, &["init"]) {
            return;
        }
        assert!(run_git(&temp.path, &["checkout", "-b", "main"]));

        let store = FoldStateStore::for_current_ref_in(Some(&temp.path))
            .unwrap()
            .unwrap();
        assert_eq!(store.path, temp.path.join(".git/sidreview/refs/heads/main"));
    }

    #[test]
    fn fold_state_store_uses_exact_tag_ref_for_detached_head() {
        let temp = TestDir::new("tag-ref").unwrap();
        if !run_git(&temp.path, &["init"]) {
            return;
        }
        assert!(run_git(&temp.path, &["checkout", "-b", "main"]));
        assert!(run_git(
            &temp.path,
            &[
                "-c",
                "user.name=sidreview test",
                "-c",
                "user.email=sidreview@example.com",
                "commit",
                "--allow-empty",
                "-m",
                "initial",
            ],
        ));
        assert!(run_git(&temp.path, &["tag", "latest"]));
        assert!(run_git(&temp.path, &["checkout", "latest"]));

        let store = FoldStateStore::for_current_ref_in(Some(&temp.path))
            .unwrap()
            .unwrap();
        assert_eq!(
            store.path,
            temp.path.join(".git/sidreview/refs/tags/latest")
        );
    }

    #[test]
    fn empty_diff_has_no_selected_chunk() {
        assert_eq!(
            ReviewApp::from_input(""),
            ReviewApp {
                blocks: vec![],
                selected: None,
                scroll_top: 0,
                viewport_width: DEFAULT_VIEWPORT_WIDTH,
                viewport_height: 1,
                should_quit: false,
            }
        );
        assert_eq!(render_plain(""), "no diff chunks\n".to_string());
        assert_eq!(
            ReviewApp::from_input("").status_text(),
            "current-chunk 0/0 0%  current-file 0/0 0%  current-review 0/0 0%  no diff chunks  q quit"
        );
    }
}
