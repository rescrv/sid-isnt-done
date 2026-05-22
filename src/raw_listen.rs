//! Socket listener transport for `sid --listen`.
//!
//! The listener keeps the raw JSONL stream process-local and lets frontends
//! reconnect without restarting the agent session.  Every complete server
//! message is retained in memory and replayed to the newest connection.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::raw_mode::RawServer;

const VMADDR_CID_ANY_VALUE: u32 = u32::MAX;

/// Listen on `spec` and return a raw JSONL server backed by that socket.
///
/// `unix:///absolute/path` binds a Unix-domain socket on Unix platforms.
/// `vsock://CID:PORT` binds an AF_VSOCK stream listener on Linux.  `CID` may
/// be omitted, `any`, or `-1` to bind `VMADDR_CID_ANY`.
pub fn listen(spec: &str) -> io::Result<RawServer<Box<dyn BufRead + Send>, Box<dyn Write + Send>>> {
    let spec = ListenSpec::parse(spec)?;
    let (sender, receiver) = mpsc::channel();
    let state = Arc::new(Mutex::new(ListenState::default()));
    let guard = start_accept_loop(spec, sender, state.clone())?;
    let input: Box<dyn BufRead + Send> =
        Box::new(ListeningInput::new(receiver, state.clone(), guard));
    let output: Box<dyn Write + Send> = Box::new(ReplayOutput::new(state));
    Ok(RawServer::new(input, output))
}

#[derive(Debug, Eq, PartialEq)]
enum ListenSpec {
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

#[derive(Default)]
struct ListenState {
    current: Option<Box<dyn ConnectedSocket>>,
    generation: u64,
    history: Vec<u8>,
    pending: Vec<u8>,
}

impl ListenState {
    fn connect(&mut self, mut stream: Box<dyn ConnectedSocket>) -> io::Result<u64> {
        self.generation = self.generation.saturating_add(1);
        if let Some(current) = self.current.take() {
            current.shutdown_socket();
        }
        stream.write_all(&self.history)?;
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
        self.history.extend_from_slice(line);
        let mut disconnect = false;
        if let Some(current) = self.current.as_mut() {
            disconnect = current
                .write_all(line)
                .and_then(|_| current.flush())
                .is_err();
        }
        if disconnect {
            if let Some(current) = self.current.take() {
                current.shutdown_socket();
            }
        }
    }

    fn flush_current(&mut self) {
        let disconnect = self
            .current
            .as_mut()
            .is_some_and(|current| current.flush().is_err());
        if disconnect {
            if let Some(current) = self.current.take() {
                current.shutdown_socket();
            }
        }
    }
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
    fn parse_unix_spec_requires_absolute_path() {
        let err = ListenSpec::parse("unix://relative.sock").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[cfg(unix)]
    #[test]
    fn unix_listener_replays_history_to_reconnecting_client() {
        use std::os::unix::net::UnixStream;
        use std::time::{Duration, SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("sid-listen-test-{nanos}.sock"));
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

        server.write_ok_result("second", None).unwrap();
        assert_eq!(read_request_id(&mut first_reader), "second");

        let second = UnixStream::connect(&path).unwrap();
        second
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut second_reader = BufReader::new(second);
        assert_eq!(read_request_id(&mut second_reader), "first");
        assert_eq!(read_request_id(&mut second_reader), "second");

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

        output.write_all(br#"{"sequence":1}"#).unwrap();
        output.write_all(b"\n").unwrap();
        let mut first_reader = BufReader::new(first_client.try_clone().unwrap());
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

    fn read_request_id(reader: &mut impl BufRead) -> String {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        serde_json::from_str::<serde_json::Value>(&line).unwrap()["request_id"]
            .as_str()
            .unwrap()
            .to_string()
    }
}
