//! Socket listener and connector transport for `sid --listen` and `sid --connect`.
//!
//! The listener keeps the raw JSONL stream process-local and lets frontends
//! reconnect without restarting the agent session.  Complete server messages
//! are retained in memory and replayed to the newest connection.  Prompts are
//! replayed semantically: unanswered prompts stay as prompts, while answered
//! prompts are represented by their later `prompt_ack` messages.

use std::collections::HashSet;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::raw_mode::{RawInput, RawServer};
use crate::raw_protocol::{
    RAW_PROTOCOL_VERSION, RawReplayComplete, RawRequestEnvelope, RawServerMessage,
};

const VMADDR_CID_ANY_VALUE: u32 = u32::MAX;
const VMADDR_CID_HOST_VALUE: u32 = 2;

/// Listen on `spec` and return a raw JSONL server backed by that socket.
///
/// `unix:///absolute/path` binds a Unix-domain socket on Unix platforms.
/// `vsock://CID:PORT` binds an AF_VSOCK stream listener on Linux.  `CID` may
/// be omitted, `any`, or `-1` to bind `VMADDR_CID_ANY`.
pub fn listen(spec: &str) -> io::Result<RawServer<Box<dyn RawInput>, Box<dyn Write + Send>>> {
    let spec = ListenSpec::parse(spec)?;
    let (sender, receiver) = mpsc::channel();
    let state = Arc::new(Mutex::new(ListenState::default()));
    let guard = start_accept_loop(spec, sender, state.clone())?;
    let input: Box<dyn RawInput> = Box::new(ListeningInput::new(receiver, state.clone(), guard));
    let output: Box<dyn Write + Send> = Box::new(ReplayOutput::new(state));
    Ok(RawServer::new(input, output))
}

/// A client connection to a reconnectable raw JSONL server.
///
/// Use [`RawConnection::connect`] to connect to a listen-compatible spec and
/// exchange typed raw protocol messages.
pub struct RawConnection {
    reader: BufReader<Box<dyn ConnectedSocket>>,
    writer: RawConnectionWriter,
}

/// Shared writer half for a raw listener connection.
#[derive(Clone)]
pub struct RawConnectionWriter {
    writer: Arc<Mutex<Box<dyn ConnectedSocket>>>,
}

impl RawConnectionWriter {
    fn new(writer: Box<dyn ConnectedSocket>) -> Self {
        Self {
            writer: Arc::new(Mutex::new(writer)),
        }
    }

    /// Write one client request to the connection.
    pub fn write_request(&self, request: &RawRequestEnvelope) -> io::Result<()> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| io::Error::other("raw connection writer lock poisoned"))?;
        serde_json::to_writer(&mut *writer, request).map_err(|err| {
            io::Error::other(format!("failed to encode raw client request: {err}"))
        })?;
        writer.write_all(b"\n")?;
        writer.flush()
    }
}

impl RawConnection {
    /// Connect to a raw JSONL listener.
    ///
    /// `unix:///absolute/path` connects to a Unix-domain socket on Unix
    /// platforms.  `vsock://CID:PORT` connects to an AF_VSOCK stream peer on
    /// Linux.  To stay compatible with listener specs, omitted, `any`, or `-1`
    /// CIDs target the host CID by convention.
    pub fn connect(spec: &str) -> io::Result<Self> {
        let reader = connect_socket(spec)?;
        let writer = RawConnectionWriter::new(reader.try_clone_socket()?);
        Ok(Self {
            reader: BufReader::new(reader),
            writer,
        })
    }

    /// Read one server message from the connection.
    pub fn read_message(&mut self) -> io::Result<Option<RawServerMessage>> {
        let mut line = String::new();
        loop {
            line.clear();
            let read = self.reader.read_line(&mut line)?;
            if read == 0 {
                return Ok(None);
            }
            if line.trim().is_empty() {
                continue;
            }
            return serde_json::from_str(&line).map(Some).map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("failed to decode raw server message: {err}"),
                )
            });
        }
    }

    /// Write one client request to the connection.
    pub fn write_request(&mut self, request: &RawRequestEnvelope) -> io::Result<()> {
        self.writer.write_request(request)
    }

    /// Clone the connection writer for another thread.
    pub fn writer_handle(&self) -> RawConnectionWriter {
        self.writer.clone()
    }
}

#[derive(Debug, Eq, PartialEq)]
enum ListenSpec {
    Unix(PathBuf),
    Vsock { cid: u32, port: u32 },
}

#[derive(Debug, Eq, PartialEq)]
enum ConnectSpec {
    Unix(PathBuf),
    Vsock { cid: u32, port: u32 },
}

impl ListenSpec {
    fn parse(spec: &str) -> io::Result<Self> {
        if let Some(path) = spec.strip_prefix("unix://") {
            return parse_unix_spec(path).map(ListenSpec::Unix);
        }
        if let Some(address) = spec.strip_prefix("vsock://") {
            return parse_vsock_spec(address);
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "listen spec must start with unix:// or vsock://",
        ))
    }
}

impl ConnectSpec {
    fn parse(spec: &str) -> io::Result<Self> {
        if let Some(path) = spec.strip_prefix("unix://") {
            return parse_unix_spec(path).map(ConnectSpec::Unix);
        }
        if let Some(address) = spec.strip_prefix("vsock://") {
            return parse_vsock_connect_spec(address);
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "connect spec must start with unix:// or vsock://",
        ))
    }
}

fn parse_unix_spec(path: &str) -> io::Result<PathBuf> {
    if !path.starts_with('/') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unix listener URL must be unix:///absolute/path",
        ));
    }
    Ok(PathBuf::from(percent_decode(path)?))
}

fn parse_vsock_spec(address: &str) -> io::Result<ListenSpec> {
    if address.is_empty() || address.contains('/') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vsock listener URL must be vsock://CID:PORT or vsock://PORT",
        ));
    }
    let (cid, port) = match address.rsplit_once(':') {
        Some((cid, port)) => (parse_vsock_cid(cid)?, parse_vsock_port(port)?),
        None => (VMADDR_CID_ANY_VALUE, parse_vsock_port(address)?),
    };
    Ok(ListenSpec::Vsock { cid, port })
}

fn parse_vsock_connect_spec(address: &str) -> io::Result<ConnectSpec> {
    if address.is_empty() || address.contains('/') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vsock connect URL must be vsock://CID:PORT or vsock://PORT",
        ));
    }
    let (cid, port) = match address.rsplit_once(':') {
        Some((cid, port)) => (parse_vsock_connect_cid(cid)?, parse_vsock_port(port)?),
        None => (VMADDR_CID_HOST_VALUE, parse_vsock_port(address)?),
    };
    Ok(ConnectSpec::Vsock { cid, port })
}

fn parse_vsock_cid(cid: &str) -> io::Result<u32> {
    match cid {
        "" | "any" | "-1" => Ok(VMADDR_CID_ANY_VALUE),
        _ => cid.parse::<u32>().map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid vsock CID {cid:?}: {err}"),
            )
        }),
    }
}

fn parse_vsock_connect_cid(cid: &str) -> io::Result<u32> {
    match cid {
        "" | "any" | "-1" => Ok(VMADDR_CID_HOST_VALUE),
        _ => cid.parse::<u32>().map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid vsock CID {cid:?}: {err}"),
            )
        }),
    }
}

fn parse_vsock_port(port: &str) -> io::Result<u32> {
    if port.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vsock listener URL is missing a port",
        ));
    }
    port.parse::<u32>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid vsock port {port:?}: {err}"),
        )
    })
}

fn percent_decode(input: &str) -> io::Result<String> {
    let bytes = input.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "incomplete percent escape in listener URL",
                ));
            }
            let high = hex_value(bytes[index + 1]).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "invalid percent escape in listener URL",
                )
            })?;
            let low = hex_value(bytes[index + 2]).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "invalid percent escape in listener URL",
                )
            })?;
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("listener URL path is not valid UTF-8: {err}"),
        )
    })
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn start_accept_loop(
    spec: ListenSpec,
    sender: Sender<InputLine>,
    state: Arc<Mutex<ListenState>>,
) -> io::Result<ListenGuard> {
    match spec {
        ListenSpec::Unix(path) => start_unix_accept_loop(path, sender, state),
        ListenSpec::Vsock { cid, port } => start_vsock_accept_loop(cid, port, sender, state),
    }
}

fn connect_socket(spec: &str) -> io::Result<Box<dyn ConnectedSocket>> {
    match ConnectSpec::parse(spec)? {
        ConnectSpec::Unix(path) => connect_unix_socket(path),
        ConnectSpec::Vsock { cid, port } => connect_vsock_socket(cid, port),
    }
}

#[cfg(unix)]
fn connect_unix_socket(path: PathBuf) -> io::Result<Box<dyn ConnectedSocket>> {
    use std::os::unix::net::UnixStream;

    Ok(Box::new(UnixStream::connect(path)?))
}

#[cfg(not(unix))]
fn connect_unix_socket(_path: PathBuf) -> io::Result<Box<dyn ConnectedSocket>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "unix:// connections are not supported on this platform",
    ))
}

#[cfg(target_os = "linux")]
fn connect_vsock_socket(cid: u32, port: u32) -> io::Result<Box<dyn ConnectedSocket>> {
    Ok(Box::new(vsock::VsockStream::connect(cid, port)?))
}

#[cfg(not(target_os = "linux"))]
fn connect_vsock_socket(_cid: u32, _port: u32) -> io::Result<Box<dyn ConnectedSocket>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "vsock:// connections are only supported on Linux",
    ))
}

#[cfg(unix)]
fn start_unix_accept_loop(
    path: PathBuf,
    sender: Sender<InputLine>,
    state: Arc<Mutex<ListenState>>,
) -> io::Result<ListenGuard> {
    use std::fs;
    use std::os::unix::fs::FileTypeExt;
    use std::os::unix::net::UnixListener;

    if let Ok(metadata) = fs::symlink_metadata(&path) {
        if metadata.file_type().is_socket() {
            fs::remove_file(&path)?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("refusing to replace non-socket path {}", path.display()),
            ));
        }
    }

    let listener = UnixListener::bind(&path)?;
    thread::spawn(move || {
        for accepted in listener.incoming() {
            match accepted {
                Ok(stream) => install_connected_socket(Box::new(stream), &sender, &state),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
    });
    Ok(ListenGuard::unlink_on_drop(path))
}

#[cfg(not(unix))]
fn start_unix_accept_loop(
    _path: PathBuf,
    _sender: Sender<InputLine>,
    _state: Arc<Mutex<ListenState>>,
) -> io::Result<ListenGuard> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "unix:// listeners are not supported on this platform",
    ))
}

#[cfg(target_os = "linux")]
fn start_vsock_accept_loop(
    cid: u32,
    port: u32,
    sender: Sender<InputLine>,
    state: Arc<Mutex<ListenState>>,
) -> io::Result<ListenGuard> {
    let listener = vsock::VsockListener::bind(cid, port)?;
    thread::spawn(move || {
        loop {
            match listener.accept() {
                Ok(stream) => install_connected_socket(Box::new(stream), &sender, &state),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
    });
    Ok(ListenGuard::default())
}

#[cfg(not(target_os = "linux"))]
fn start_vsock_accept_loop(
    _cid: u32,
    _port: u32,
    _sender: Sender<InputLine>,
    _state: Arc<Mutex<ListenState>>,
) -> io::Result<ListenGuard> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "vsock:// listeners are only supported on Linux",
    ))
}

fn install_connected_socket(
    stream: Box<dyn ConnectedSocket>,
    sender: &Sender<InputLine>,
    state: &Arc<Mutex<ListenState>>,
) {
    let Ok(writer) = stream.try_clone_socket() else {
        return;
    };
    let generation = match state.lock() {
        Ok(mut state) => match state.connect(writer) {
            Ok(generation) => generation,
            Err(_) => return,
        },
        Err(_) => return,
    };
    spawn_reader(stream, generation, sender.clone(), state.clone());
}

fn spawn_reader(
    stream: Box<dyn ConnectedSocket>,
    generation: u64,
    sender: Sender<InputLine>,
    state: Arc<Mutex<ListenState>>,
) {
    thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if current_generation(&state) != Some(generation) {
                        break;
                    }
                    if sender.send(InputLine { generation, line }).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
}

fn current_generation(state: &Arc<Mutex<ListenState>>) -> Option<u64> {
    state.lock().ok().map(|state| state.generation)
}

trait ConnectedSocket: Read + Write + Send {
    fn try_clone_socket(&self) -> io::Result<Box<dyn ConnectedSocket>>;

    fn shutdown_socket(&self);
}

#[cfg(unix)]
impl ConnectedSocket for std::os::unix::net::UnixStream {
    fn try_clone_socket(&self) -> io::Result<Box<dyn ConnectedSocket>> {
        Ok(Box::new(self.try_clone()?))
    }

    fn shutdown_socket(&self) {
        let _ = self.shutdown(std::net::Shutdown::Both);
    }
}

#[derive(Default)]
struct ListenGuard {
    unlink_on_drop: Option<PathBuf>,
}

impl ListenGuard {
    fn unlink_on_drop(path: PathBuf) -> Self {
        Self {
            unlink_on_drop: Some(path),
        }
    }
}

impl Drop for ListenGuard {
    fn drop(&mut self) {
        if let Some(path) = self.unlink_on_drop.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

struct InputLine {
    generation: u64,
    line: String,
}

struct HistoryLine {
    line: Vec<u8>,
    kind: HistoryKind,
}

impl HistoryLine {
    fn parse(line: &[u8]) -> Self {
        let kind = match serde_json::from_slice::<RawServerMessage>(line) {
            Ok(RawServerMessage::Prompt(prompt)) => HistoryKind::Prompt {
                prompt_id: prompt.prompt_id,
            },
            Ok(RawServerMessage::PromptAck(prompt_ack)) => HistoryKind::PromptAck {
                prompt_id: prompt_ack.prompt_id,
            },
            _ => HistoryKind::Other,
        };
        Self {
            line: line.to_vec(),
            kind,
        }
    }

    fn should_replay(&self, acknowledged_prompts: &HashSet<String>) -> bool {
        match &self.kind {
            HistoryKind::Prompt { prompt_id } => !acknowledged_prompts.contains(prompt_id),
            HistoryKind::PromptAck { .. } | HistoryKind::Other => true,
        }
    }
}

enum HistoryKind {
    Prompt { prompt_id: String },
    PromptAck { prompt_id: String },
    Other,
}

#[derive(Default)]
struct ListenState {
    current: Option<Box<dyn ConnectedSocket>>,
    generation: u64,
    history: Vec<HistoryLine>,
    acknowledged_prompts: HashSet<String>,
    pending: Vec<u8>,
}

impl ListenState {
    fn connect(&mut self, mut stream: Box<dyn ConnectedSocket>) -> io::Result<u64> {
        self.generation = self.generation.saturating_add(1);
        if let Some(current) = self.current.take() {
            current.shutdown_socket();
        }
        for line in &self.history {
            if line.should_replay(&self.acknowledged_prompts) {
                stream.write_all(&line.line)?;
            }
        }
        write_replay_complete(&mut stream)?;
        stream.flush()?;
        self.current = Some(stream);
        Ok(self.generation)
    }

    fn push_output(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            if let Some(newline) = bytes.iter().position(|byte| *byte == b'\n') {
                self.pending.extend_from_slice(&bytes[..=newline]);
                let line = std::mem::take(&mut self.pending);
                self.write_line(&line);
                bytes = &bytes[newline + 1..];
            } else {
                self.pending.extend_from_slice(bytes);
                break;
            }
        }
    }

    fn write_line(&mut self, line: &[u8]) {
        let history_line = HistoryLine::parse(line);
        if let HistoryKind::PromptAck { prompt_id } = &history_line.kind {
            self.acknowledged_prompts.insert(prompt_id.clone());
        }
        self.history.push(history_line);
        let mut disconnect = false;
        if let Some(current) = self.current.as_mut() {
            disconnect = current
                .write_all(line)
                .and_then(|_| current.flush())
                .is_err();
        }
        if disconnect && let Some(current) = self.current.take() {
            current.shutdown_socket();
        }
    }

    fn flush_current(&mut self) {
        let disconnect = self
            .current
            .as_mut()
            .is_some_and(|current| current.flush().is_err());
        if disconnect && let Some(current) = self.current.take() {
            current.shutdown_socket();
        }
    }
}

fn write_replay_complete(stream: &mut Box<dyn ConnectedSocket>) -> io::Result<()> {
    let payload = serde_json::to_vec(&RawServerMessage::ReplayComplete(RawReplayComplete {
        protocol_version: RAW_PROTOCOL_VERSION,
        sequence: 0,
    }))
    .map_err(|err| io::Error::other(format!("failed to encode replay marker: {err}")))?;
    stream.write_all(&payload)?;
    stream.write_all(b"\n")
}

struct ReplayOutput {
    state: Arc<Mutex<ListenState>>,
}

impl ReplayOutput {
    fn new(state: Arc<Mutex<ListenState>>) -> Self {
        Self { state }
    }
}

impl Write for ReplayOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("raw listen state lock poisoned"))?;
        state.push_output(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("raw listen state lock poisoned"))?;
        state.flush_current();
        Ok(())
    }
}

struct ListeningInput {
    receiver: Receiver<InputLine>,
    state: Arc<Mutex<ListenState>>,
    buffer: Vec<u8>,
    position: usize,
    generation: u64,
    _guard: ListenGuard,
}

impl ListeningInput {
    fn new(
        receiver: Receiver<InputLine>,
        state: Arc<Mutex<ListenState>>,
        guard: ListenGuard,
    ) -> Self {
        Self {
            receiver,
            state,
            buffer: Vec::new(),
            position: 0,
            generation: 0,
            _guard: guard,
        }
    }

    fn current_generation(&self) -> io::Result<u64> {
        self.state
            .lock()
            .map(|state| state.generation)
            .map_err(|_| io::Error::other("raw listen state lock poisoned"))
    }

    fn clear_buffer(&mut self) {
        self.buffer.clear();
        self.position = 0;
        self.generation = 0;
    }
}

impl Read for ListeningInput {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let length = available.len().min(buf.len());
        buf[..length].copy_from_slice(&available[..length]);
        self.consume(length);
        Ok(length)
    }
}

impl BufRead for ListeningInput {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        loop {
            if self.position < self.buffer.len() {
                if self.generation == self.current_generation()? {
                    return Ok(&self.buffer[self.position..]);
                }
                self.clear_buffer();
            }

            let input = match self.receiver.recv() {
                Ok(input) => input,
                Err(_) => return Ok(&[]),
            };
            if input.generation == self.current_generation()? {
                self.buffer = input.line.into_bytes();
                self.position = 0;
                self.generation = input.generation;
            }
        }
    }

    fn consume(&mut self, amt: usize) {
        self.position = self.position.saturating_add(amt).min(self.buffer.len());
        if self.position >= self.buffer.len() {
            self.clear_buffer();
        }
    }
}

impl RawInput for ListeningInput {
    fn try_read_line(&mut self) -> io::Result<Option<String>> {
        loop {
            if self.position < self.buffer.len() {
                if self.generation == self.current_generation()? {
                    let line = String::from_utf8(self.buffer[self.position..].to_vec()).map_err(
                        |err| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("raw listener buffered input was not UTF-8: {err}"),
                            )
                        },
                    )?;
                    self.clear_buffer();
                    return Ok(Some(line));
                }
                self.clear_buffer();
            }

            match self.receiver.try_recv() {
                Ok(input) if input.generation == self.current_generation()? => {
                    return Ok(Some(input.line));
                }
                Ok(_) => {}
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return Ok(None),
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod vsock {
    use std::io::{self, Read, Write};
    use std::mem;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    use super::ConnectedSocket;

    pub(super) struct VsockListener {
        fd: OwnedFd,
    }

    pub(super) struct VsockStream {
        fd: OwnedFd,
    }

    impl VsockListener {
        pub(super) fn bind(cid: u32, port: u32) -> io::Result<Self> {
            let fd = syscall_fd(|| unsafe {
                libc::socket(libc::AF_VSOCK, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0)
            })?;
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            let mut address = unsafe { mem::zeroed::<libc::sockaddr_vm>() };
            address.svm_family = libc::AF_VSOCK as libc::sa_family_t;
            address.svm_cid = cid;
            address.svm_port = port;
            syscall_unit(|| unsafe {
                libc::bind(
                    fd.as_raw_fd(),
                    (&address as *const libc::sockaddr_vm).cast::<libc::sockaddr>(),
                    mem::size_of::<libc::sockaddr_vm>() as libc::socklen_t,
                )
            })?;
            syscall_unit(|| unsafe { libc::listen(fd.as_raw_fd(), 128) })?;
            Ok(Self { fd })
        }

        pub(super) fn accept(&self) -> io::Result<VsockStream> {
            let fd = syscall_fd(|| unsafe {
                libc::accept4(
                    self.fd.as_raw_fd(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    libc::SOCK_CLOEXEC,
                )
            })?;
            Ok(VsockStream {
                fd: unsafe { OwnedFd::from_raw_fd(fd) },
            })
        }
    }

    impl VsockStream {
        pub(super) fn connect(cid: u32, port: u32) -> io::Result<Self> {
            let fd = syscall_fd(|| unsafe {
                libc::socket(libc::AF_VSOCK, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0)
            })?;
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            let mut address = unsafe { mem::zeroed::<libc::sockaddr_vm>() };
            address.svm_family = libc::AF_VSOCK as libc::sa_family_t;
            address.svm_cid = cid;
            address.svm_port = port;
            syscall_unit(|| unsafe {
                libc::connect(
                    fd.as_raw_fd(),
                    (&address as *const libc::sockaddr_vm).cast::<libc::sockaddr>(),
                    mem::size_of::<libc::sockaddr_vm>() as libc::socklen_t,
                )
            })?;
            Ok(Self { fd })
        }
    }

    impl ConnectedSocket for VsockStream {
        fn try_clone_socket(&self) -> io::Result<Box<dyn ConnectedSocket>> {
            let fd = syscall_fd(|| unsafe { libc::dup(self.fd.as_raw_fd()) })?;
            Ok(Box::new(VsockStream {
                fd: unsafe { OwnedFd::from_raw_fd(fd) },
            }))
        }

        fn shutdown_socket(&self) {
            let _ = unsafe { libc::shutdown(self.fd.as_raw_fd(), libc::SHUT_RDWR) };
        }
    }

    impl Read for VsockStream {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            syscall_size(|| unsafe {
                libc::recv(
                    self.fd.as_raw_fd(),
                    buf.as_mut_ptr().cast::<libc::c_void>(),
                    buf.len(),
                    0,
                )
            })
        }
    }

    impl Write for VsockStream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            syscall_size(|| unsafe {
                libc::send(
                    self.fd.as_raw_fd(),
                    buf.as_ptr().cast::<libc::c_void>(),
                    buf.len(),
                    libc::MSG_NOSIGNAL,
                )
            })
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn syscall_fd(call: impl FnOnce() -> libc::c_int) -> io::Result<libc::c_int> {
        let result = call();
        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(result)
        }
    }

    fn syscall_unit(call: impl FnOnce() -> libc::c_int) -> io::Result<()> {
        syscall_fd(call).map(|_| ())
    }

    fn syscall_size(call: impl FnOnce() -> libc::ssize_t) -> io::Result<usize> {
        let result = call();
        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(result as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_vsock_spec_defaults_to_any_cid() {
        assert_eq!(
            ListenSpec::parse("vsock://1024").unwrap(),
            ListenSpec::Vsock {
                cid: VMADDR_CID_ANY_VALUE,
                port: 1024,
            }
        );
        assert_eq!(
            ListenSpec::parse("vsock://any:2048").unwrap(),
            ListenSpec::Vsock {
                cid: VMADDR_CID_ANY_VALUE,
                port: 2048,
            }
        );
    }

    #[test]
    fn parse_vsock_connect_spec_defaults_to_host_cid() {
        assert_eq!(
            ConnectSpec::parse("vsock://1024").unwrap(),
            ConnectSpec::Vsock {
                cid: VMADDR_CID_HOST_VALUE,
                port: 1024,
            }
        );
        assert_eq!(
            ConnectSpec::parse("vsock://any:2048").unwrap(),
            ConnectSpec::Vsock {
                cid: VMADDR_CID_HOST_VALUE,
                port: 2048,
            }
        );
        assert_eq!(
            ConnectSpec::parse("vsock://3:2048").unwrap(),
            ConnectSpec::Vsock { cid: 3, port: 2048 }
        );
    }

    #[test]
    fn parse_unix_spec_requires_absolute_path() {
        let err = ListenSpec::parse("unix://relative.sock").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[cfg(unix)]
    #[test]
    fn unix_listener_replays_history_to_reconnecting_client() {
        use std::os::unix::net::UnixStream;
        use std::time::{Duration, SystemTime, UNIX_EPOCH};

        let id = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let path = PathBuf::from(format!("/tmp/sid-lt-{id}-{nanos}.sock"));
        let spec = format!("unix://{}", path.display());
        let server = match listen(&spec) {
            Ok(server) => server,
            Err(err) if err.kind() == io::ErrorKind::PermissionDenied => return,
            Err(err) => panic!("failed to start unix listener: {err}"),
        };
        server.write_ok_result("first", None).unwrap();

        let first = UnixStream::connect(&path).unwrap();
        first
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut first_reader = BufReader::new(first.try_clone().unwrap());
        assert_eq!(read_request_id(&mut first_reader), "first");
        assert_eq!(read_type(&mut first_reader), "replay_complete");

        server.write_ok_result("second", None).unwrap();
        assert_eq!(read_request_id(&mut first_reader), "second");

        let second = UnixStream::connect(&path).unwrap();
        second
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut second_reader = BufReader::new(second);
        assert_eq!(read_request_id(&mut second_reader), "first");
        assert_eq!(read_request_id(&mut second_reader), "second");
        assert_eq!(read_type(&mut second_reader), "replay_complete");

        server.write_ok_result("third", None).unwrap();
        assert_eq!(read_request_id(&mut second_reader), "third");

        let mut stale = String::new();
        let result = first_reader.read_line(&mut stale);
        assert!(
            matches!(result, Ok(0) | Err(_)),
            "previous stream stayed readable: {result:?}, line={stale:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn replay_output_replays_history_and_disconnects_previous_stream() {
        use std::os::unix::net::UnixStream;
        use std::time::Duration;

        let state = Arc::new(Mutex::new(ListenState::default()));
        let mut output = ReplayOutput::new(state.clone());
        let (first_server, first_client) = UnixStream::pair().unwrap();
        first_client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        assert_eq!(
            state
                .lock()
                .unwrap()
                .connect(Box::new(first_server))
                .unwrap(),
            1
        );
        let mut first_reader = BufReader::new(first_client.try_clone().unwrap());
        assert_eq!(read_type(&mut first_reader), "replay_complete");

        output.write_all(br#"{"sequence":1}"#).unwrap();
        output.write_all(b"\n").unwrap();
        let mut first_line = String::new();
        first_reader.read_line(&mut first_line).unwrap();
        assert_eq!(first_line, "{\"sequence\":1}\n");

        let (second_server, second_client) = UnixStream::pair().unwrap();
        second_client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        assert_eq!(
            state
                .lock()
                .unwrap()
                .connect(Box::new(second_server))
                .unwrap(),
            2
        );

        let mut second_reader = BufReader::new(second_client);
        let mut replayed = String::new();
        second_reader.read_line(&mut replayed).unwrap();
        assert_eq!(replayed, "{\"sequence\":1}\n");
        assert_eq!(read_type(&mut second_reader), "replay_complete");

        output.write_all(b"{\"sequence\":2}\n").unwrap();
        let mut live = String::new();
        second_reader.read_line(&mut live).unwrap();
        assert_eq!(live, "{\"sequence\":2}\n");

        let mut stale = String::new();
        let result = first_reader.read_line(&mut stale);
        assert!(
            matches!(result, Ok(0) | Err(_)),
            "previous stream stayed readable: {result:?}, line={stale:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn replay_output_replays_only_unacknowledged_prompts_as_prompts() {
        use std::os::unix::net::UnixStream;
        use std::time::Duration;

        let state = Arc::new(Mutex::new(ListenState::default()));
        let mut output = ReplayOutput::new(state.clone());
        output
            .write_all(prompt_line("prompt-1").as_bytes())
            .unwrap();

        let (first_server, first_client) = UnixStream::pair().unwrap();
        first_client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        state
            .lock()
            .unwrap()
            .connect(Box::new(first_server))
            .unwrap();
        let mut first_reader = BufReader::new(first_client);
        assert_eq!(read_type(&mut first_reader), "prompt");

        output
            .write_all(prompt_ack_line("prompt-1", "yes").as_bytes())
            .unwrap();

        let (second_server, second_client) = UnixStream::pair().unwrap();
        second_client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        state
            .lock()
            .unwrap()
            .connect(Box::new(second_server))
            .unwrap();
        let mut second_reader = BufReader::new(second_client);
        assert_eq!(read_type(&mut second_reader), "prompt_ack");
    }

    fn read_request_id(reader: &mut impl BufRead) -> String {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        serde_json::from_str::<serde_json::Value>(&line).unwrap()["request_id"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn read_type(reader: &mut impl BufRead) -> String {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        serde_json::from_str::<serde_json::Value>(&line).unwrap()["type"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn prompt_line(prompt_id: &str) -> String {
        format!(
            "{}\n",
            serde_json::json!({
                "type": "prompt",
                "protocol_version": 2,
                "sequence": 1,
                "request_id": "request",
                "prompt_id": prompt_id,
                "kind": "confirmation",
                "message": "Continue?",
                "choices": ["yes", "no"],
            })
        )
    }

    fn prompt_ack_line(prompt_id: &str, response: &str) -> String {
        format!(
            "{}\n",
            serde_json::json!({
                "type": "prompt_ack",
                "protocol_version": 2,
                "sequence": 2,
                "request_id": "request",
                "response_request_id": "response-request",
                "prompt_id": prompt_id,
                "response": response,
            })
        )
    }

    // ── percent_decode ───────────────────────────────────────────────────

    #[test]
    fn percent_decode_passthrough() {
        assert_eq!(percent_decode("/tmp/foo.sock").unwrap(), "/tmp/foo.sock");
    }

    #[test]
    fn percent_decode_space() {
        assert_eq!(percent_decode("/tmp/my%20sock").unwrap(), "/tmp/my sock");
    }

    #[test]
    fn percent_decode_multiple() {
        assert_eq!(percent_decode("%2F%2F").unwrap(), "//");
    }

    #[test]
    fn percent_decode_mixed_case_hex() {
        assert_eq!(percent_decode("%4a%4A").unwrap(), "JJ");
    }

    #[test]
    fn percent_decode_incomplete_escape() {
        assert!(percent_decode("%2").is_err());
        assert!(percent_decode("abc%").is_err());
    }

    #[test]
    fn percent_decode_invalid_hex_digit() {
        assert!(percent_decode("%GG").is_err());
    }

    // ── hex_value ────────────────────────────────────────────────────────

    #[test]
    fn hex_value_digits() {
        assert_eq!(hex_value(b'0'), Some(0));
        assert_eq!(hex_value(b'9'), Some(9));
    }

    #[test]
    fn hex_value_lowercase() {
        assert_eq!(hex_value(b'a'), Some(10));
        assert_eq!(hex_value(b'f'), Some(15));
    }

    #[test]
    fn hex_value_uppercase() {
        assert_eq!(hex_value(b'A'), Some(10));
        assert_eq!(hex_value(b'F'), Some(15));
    }

    #[test]
    fn hex_value_invalid() {
        assert_eq!(hex_value(b'g'), None);
        assert_eq!(hex_value(b'G'), None);
        assert_eq!(hex_value(b' '), None);
        assert_eq!(hex_value(b'z'), None);
    }

    // ── parse_vsock_port ─────────────────────────────────────────────────

    #[test]
    fn parse_vsock_port_valid() {
        assert_eq!(parse_vsock_port("1024").unwrap(), 1024);
        assert_eq!(parse_vsock_port("0").unwrap(), 0);
    }

    #[test]
    fn parse_vsock_port_empty() {
        assert!(parse_vsock_port("").is_err());
    }

    #[test]
    fn parse_vsock_port_non_numeric() {
        assert!(parse_vsock_port("abc").is_err());
    }

    // ── parse_vsock_cid ──────────────────────────────────────────────────

    #[test]
    fn parse_vsock_cid_any_variants() {
        assert_eq!(parse_vsock_cid("").unwrap(), VMADDR_CID_ANY_VALUE);
        assert_eq!(parse_vsock_cid("any").unwrap(), VMADDR_CID_ANY_VALUE);
        assert_eq!(parse_vsock_cid("-1").unwrap(), VMADDR_CID_ANY_VALUE);
    }

    #[test]
    fn parse_vsock_cid_numeric() {
        assert_eq!(parse_vsock_cid("3").unwrap(), 3);
    }

    #[test]
    fn parse_vsock_cid_invalid() {
        assert!(parse_vsock_cid("xyz").is_err());
    }

    // ── parse_vsock_connect_cid ──────────────────────────────────────────

    #[test]
    fn parse_vsock_connect_cid_any_maps_to_host() {
        assert_eq!(parse_vsock_connect_cid("").unwrap(), VMADDR_CID_HOST_VALUE);
        assert_eq!(
            parse_vsock_connect_cid("any").unwrap(),
            VMADDR_CID_HOST_VALUE
        );
        assert_eq!(
            parse_vsock_connect_cid("-1").unwrap(),
            VMADDR_CID_HOST_VALUE
        );
    }

    #[test]
    fn parse_vsock_connect_cid_numeric() {
        assert_eq!(parse_vsock_connect_cid("5").unwrap(), 5);
    }

    #[test]
    fn parse_vsock_connect_cid_invalid() {
        assert!(parse_vsock_connect_cid("xyz").is_err());
    }

    // ── ListenSpec / ConnectSpec parse edge cases ─────────────────────────

    #[test]
    fn listen_spec_rejects_unknown_scheme() {
        assert!(ListenSpec::parse("tcp://localhost:1234").is_err());
    }

    #[test]
    fn connect_spec_rejects_unknown_scheme() {
        assert!(ConnectSpec::parse("tcp://localhost:1234").is_err());
    }

    #[test]
    fn vsock_listen_spec_rejects_path_separator() {
        assert!(ListenSpec::parse("vsock://1024/extra").is_err());
    }

    #[test]
    fn vsock_connect_spec_rejects_path_separator() {
        assert!(ConnectSpec::parse("vsock://1024/extra").is_err());
    }

    #[test]
    fn vsock_listen_spec_rejects_empty_address() {
        assert!(ListenSpec::parse("vsock://").is_err());
    }

    #[test]
    fn vsock_connect_spec_rejects_empty_address() {
        assert!(ConnectSpec::parse("vsock://").is_err());
    }

    #[test]
    fn listen_spec_cid_with_port() {
        assert_eq!(
            ListenSpec::parse("vsock://3:5000").unwrap(),
            ListenSpec::Vsock { cid: 3, port: 5000 }
        );
    }

    #[test]
    fn connect_spec_minus_one_cid() {
        assert_eq!(
            ConnectSpec::parse("vsock://-1:9000").unwrap(),
            ConnectSpec::Vsock {
                cid: VMADDR_CID_HOST_VALUE,
                port: 9000,
            }
        );
    }

    #[cfg(unix)]
    #[test]
    fn listen_spec_unix_with_percent_encoded_space() {
        assert_eq!(
            ListenSpec::parse("unix:///tmp/my%20socket.sock").unwrap(),
            ListenSpec::Unix(PathBuf::from("/tmp/my socket.sock")),
        );
    }

    #[cfg(unix)]
    #[test]
    fn connect_spec_unix_roundtrip() {
        assert_eq!(
            ConnectSpec::parse("unix:///tmp/test.sock").unwrap(),
            ConnectSpec::Unix(PathBuf::from("/tmp/test.sock")),
        );
    }

    // ── HistoryLine::should_replay ───────────────────────────────────────

    #[test]
    fn history_line_unanswered_prompt_should_replay() {
        let line = prompt_line("p-1");
        let history = HistoryLine::parse(line.as_bytes());
        let acked = HashSet::new();
        assert!(history.should_replay(&acked));
    }

    #[test]
    fn history_line_answered_prompt_should_not_replay() {
        let line = prompt_line("p-1");
        let history = HistoryLine::parse(line.as_bytes());
        let acked: HashSet<String> = ["p-1".to_string()].into_iter().collect();
        assert!(!history.should_replay(&acked));
    }

    #[test]
    fn history_line_prompt_ack_always_replays() {
        let line = prompt_ack_line("p-1", "yes");
        let history = HistoryLine::parse(line.as_bytes());
        let acked = HashSet::new();
        assert!(history.should_replay(&acked));
    }

    #[test]
    fn history_line_other_always_replays() {
        let line = b"{\"type\":\"result\",\"protocol_version\":2,\"sequence\":1,\"request_id\":\"r\",\"ok\":true}\n";
        let history = HistoryLine::parse(line);
        let acked = HashSet::new();
        assert!(history.should_replay(&acked));
    }

    // ── ListenState::push_output ─────────────────────────────────────────

    #[test]
    fn push_output_buffers_partial_lines() {
        let mut state = ListenState::default();
        state.push_output(b"partial");
        assert_eq!(state.history.len(), 0);
        assert_eq!(state.pending, b"partial");

        state.push_output(b" line\n");
        assert_eq!(state.history.len(), 1);
        assert!(state.pending.is_empty());
    }

    #[test]
    fn push_output_handles_multiple_lines_in_one_call() {
        let mut state = ListenState::default();
        state.push_output(b"line1\nline2\n");
        assert_eq!(state.history.len(), 2);
        assert!(state.pending.is_empty());
    }

    #[test]
    fn push_output_handles_trailing_partial() {
        let mut state = ListenState::default();
        state.push_output(b"line1\npartial");
        assert_eq!(state.history.len(), 1);
        assert_eq!(state.pending, b"partial");
    }

    #[test]
    fn listening_input_try_read_line_is_nonblocking_and_filters_stale_generations() {
        let (sender, receiver) = mpsc::channel();
        let state = Arc::new(Mutex::new(ListenState::default()));
        state.lock().unwrap().generation = 2;
        let mut input = ListeningInput::new(receiver, state, ListenGuard::default());

        assert_eq!(RawInput::try_read_line(&mut input).unwrap(), None);

        sender
            .send(InputLine {
                generation: 1,
                line: "stale\n".to_string(),
            })
            .unwrap();
        sender
            .send(InputLine {
                generation: 2,
                line: "live\n".to_string(),
            })
            .unwrap();

        assert_eq!(
            RawInput::try_read_line(&mut input).unwrap(),
            Some("live\n".to_string())
        );
        assert_eq!(RawInput::try_read_line(&mut input).unwrap(), None);
    }

    // ── ListenState::connect (acknowledged prompt suppression) ───────────

    #[cfg(unix)]
    #[test]
    fn connect_replays_all_when_no_prompts_acknowledged() {
        use std::os::unix::net::UnixStream;
        use std::time::Duration;

        let mut state = ListenState::default();
        // Push a prompt and a result.
        state.push_output(prompt_line("p-1").as_bytes());
        state.push_output(
            b"{\"type\":\"result\",\"protocol_version\":2,\"sequence\":1,\"request_id\":\"r\",\"ok\":true}\n",
        );
        assert_eq!(state.history.len(), 2);

        let (server, client) = UnixStream::pair().unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        state.connect(Box::new(server)).unwrap();

        let mut reader = BufReader::new(client);
        assert_eq!(read_type(&mut reader), "prompt");
        assert_eq!(read_type(&mut reader), "result");
    }

    #[cfg(unix)]
    #[test]
    fn connect_skips_acknowledged_prompts() {
        use std::os::unix::net::UnixStream;
        use std::time::Duration;

        let mut state = ListenState::default();
        state.push_output(prompt_line("p-1").as_bytes());
        state.push_output(prompt_ack_line("p-1", "yes").as_bytes());
        assert_eq!(state.history.len(), 2);
        // p-1 should be in acknowledged_prompts now.
        assert!(state.acknowledged_prompts.contains("p-1"));

        let (server, client) = UnixStream::pair().unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        state.connect(Box::new(server)).unwrap();

        // Only the prompt_ack should be replayed, not the prompt.
        let mut reader = BufReader::new(client);
        assert_eq!(read_type(&mut reader), "prompt_ack");
    }

    // ── ListenGuard ──────────────────────────────────────────────────────

    #[test]
    fn listen_guard_default_does_not_unlink() {
        let guard = ListenGuard::default();
        assert!(guard.unlink_on_drop.is_none());
        drop(guard); // should not panic
    }

    #[cfg(unix)]
    #[test]
    fn listen_guard_unlinks_on_drop() {
        let path = PathBuf::from(format!("/tmp/sid-guard-test-{}.sock", std::process::id()));
        std::fs::write(&path, b"").unwrap();
        assert!(path.exists());

        let guard = ListenGuard::unlink_on_drop(path.clone());
        drop(guard);
        assert!(!path.exists());
    }
}
