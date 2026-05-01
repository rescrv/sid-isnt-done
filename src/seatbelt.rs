/// macOS Seatbelt sandbox integration.
///
/// Generates SBPL policies with configurable writable roots and provides
/// helpers for wrapping child processes in `sandbox-exec`.
use std::fmt;
use std::path::Path;
use std::process::Command;
use std::process::Stdio;
use std::str::FromStr;
use std::sync::OnceLock;

/// Colon-separated list of directories that a sandboxed process may write to.
///
/// Parses from a PATH-style string (e.g. `/workspace:/tmp/scratch`).  Empty
/// components are silently dropped.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WritableRoots(Vec<String>);

impl WritableRoots {
    /// Returns the roots as a slice of strings.
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }

    /// Returns the roots as a vec of `&str` suitable for [`build_policy`].
    pub fn as_str_slice(&self) -> Vec<&str> {
        self.0.iter().map(String::as_str).collect()
    }

    /// Returns `true` when no writable roots are configured.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Appends a root directory.
    pub fn push(&mut self, root: String) {
        if !root.is_empty() {
            self.0.push(root);
        }
    }
}

impl FromStr for WritableRoots {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let roots = s
            .split(':')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        Ok(WritableRoots(roots))
    }
}

impl fmt::Display for WritableRoots {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for root in &self.0 {
            if !first {
                write!(f, ":")?;
            }
            write!(f, "{root}")?;
            first = false;
        }
        Ok(())
    }
}

/// Absolute path to the macOS `sandbox-exec` binary.
const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// SBPL policy text before the writable-roots section.
///
/// The policy is deny-default, then allows:
///   - Process control (fork, exec, signals within the same sandbox)
///   - Full disk read
///   - Write only to caller-specified roots and /tmp
///   - Network denied by default; only localhost permitted
///   - Platform defaults (system libs, frameworks, /dev, /tmp, binaries)
const POLICY_BEFORE_WRITE_RULES: &str = r#"(version 1)

;; =======================================================================
;; 1. BASE POLICY — deny-default, process control, PTY, sysctls, IPC
;; =======================================================================

(deny default)

; Child processes inherit the sandbox.
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))

; /dev/null write
(allow file-write-data
  (require-all
    (path "/dev/null")
    (vnode-type CHARACTER-DEVICE)))

; Curated sysctl reads (CPU topology, kernel info, memory, load average).
(allow sysctl-read
  (sysctl-name "hw.activecpu")
  (sysctl-name "hw.busfrequency_compat")
  (sysctl-name "hw.byteorder")
  (sysctl-name "hw.cacheconfig")
  (sysctl-name "hw.cachelinesize_compat")
  (sysctl-name "hw.cpufamily")
  (sysctl-name "hw.cpufrequency_compat")
  (sysctl-name "hw.cputype")
  (sysctl-name "hw.l1dcachesize_compat")
  (sysctl-name "hw.l1icachesize_compat")
  (sysctl-name "hw.l2cachesize_compat")
  (sysctl-name "hw.l3cachesize_compat")
  (sysctl-name "hw.logicalcpu_max")
  (sysctl-name "hw.machine")
  (sysctl-name "hw.model")
  (sysctl-name "hw.memsize")
  (sysctl-name "hw.ncpu")
  (sysctl-name "hw.nperflevels")
  (sysctl-name-prefix "hw.optional.arm.")
  (sysctl-name-prefix "hw.optional.armv8_")
  (sysctl-name "hw.packages")
  (sysctl-name "hw.pagesize_compat")
  (sysctl-name "hw.pagesize")
  (sysctl-name "hw.physicalcpu")
  (sysctl-name "hw.physicalcpu_max")
  (sysctl-name "hw.logicalcpu")
  (sysctl-name "hw.cpufrequency")
  (sysctl-name "hw.tbfrequency_compat")
  (sysctl-name "hw.vectorunit")
  (sysctl-name "machdep.cpu.brand_string")
  (sysctl-name "kern.argmax")
  (sysctl-name "kern.hostname")
  (sysctl-name "kern.maxfilesperproc")
  (sysctl-name "kern.maxproc")
  (sysctl-name "kern.osproductversion")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.osvariant_status")
  (sysctl-name "kern.osversion")
  (sysctl-name "kern.secure_kernel")
  (sysctl-name "kern.usrstack64")
  (sysctl-name "kern.version")
  (sysctl-name "sysctl.proc_cputype")
  (sysctl-name "vm.loadavg")
  (sysctl-name-prefix "hw.perflevel")
  (sysctl-name-prefix "kern.proc.pgrp.")
  (sysctl-name-prefix "kern.proc.pid.")
  (sysctl-name-prefix "net.routetable."))

; Java CPU grading (misclassified as write, conceptually a read).
(allow sysctl-write
  (sysctl-name "kern.grade_cputype"))

; IOKit root domain.
(allow iokit-open
  (iokit-registry-entry-class "RootDomainUserClient"))

; User directory lookups.
(allow mach-lookup
  (global-name "com.apple.system.opendirectoryd.libinfo"))

; Python multiprocessing SemLock.
(allow ipc-posix-sem)

; PyTorch/libomp shared memory.
(allow ipc-posix-shm-read-data
  ipc-posix-shm-write-create
  ipc-posix-shm-write-unlink
  (ipc-posix-name-regex #"^/__KMP_REGISTERED_LIB_[0-9]+$"))

; Power management.
(allow mach-lookup
  (global-name "com.apple.PowerManagement.control"))

; PTY support.
(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-read* file-write*
  (require-all
    (regex #"^/dev/ttys[0-9]+")
    (extension "com.apple.sandbox.pty")))
(allow file-ioctl (regex #"^/dev/ttys[0-9]+"))

; Read-only user preferences.
(allow ipc-posix-shm-read* (ipc-posix-name-prefix "apple.cfprefs."))
(allow mach-lookup
  (global-name "com.apple.cfprefsd.daemon")
  (global-name "com.apple.cfprefsd.agent")
  (local-name "com.apple.cfprefsd.agent"))
(allow user-preference-read)

;; =======================================================================
;; 2. READ POLICY — full disk read
;; =======================================================================

(allow file-read*)

;; =======================================================================
;; 3. WRITE POLICY — writable roots + /tmp
;; =======================================================================

"#;

/// SBPL policy text after the writable-roots section.
const POLICY_AFTER_WRITE_RULES: &str = r#"
;; =======================================================================
;; 4. NETWORK POLICY — deny by default, localhost only + platform services
;; =======================================================================

; Explicit deny-all for network access (defense-in-depth over `deny default`).
(deny network*)

; Allow binding on localhost only (needed for dev servers).
(allow network-bind (local ip "localhost:*"))

; Restrict traffic to loopback.
(allow network-inbound (local ip "localhost:*"))
(allow network-outbound (remote ip "localhost:*"))

; AF_SYSTEM sockets for local platform services.
(allow system-socket
  (require-all
    (socket-domain AF_SYSTEM)
    (socket-protocol 2)))

; Mach lookups for DNS resolution and TLS certificate validation.
(allow mach-lookup
    (global-name "com.apple.bsd.dirhelper")
    (global-name "com.apple.system.opendirectoryd.membership")
    (global-name "com.apple.SecurityServer")
    (global-name "com.apple.networkd")
    (global-name "com.apple.ocspd")
    (global-name "com.apple.trustd.agent")
    (global-name "com.apple.SystemConfiguration.DNSConfiguration")
    (global-name "com.apple.SystemConfiguration.configd"))

; Route-table reads.
(allow sysctl-read
  (sysctl-name-regex #"^net.routetable"))

; TLS session cache writes.
(allow file-write*
  (subpath (param "DARWIN_USER_CACHE_DIR")))

;; =======================================================================
;; 5. PLATFORM DEFAULTS — system libs, frameworks, /dev, /tmp, binaries
;; =======================================================================

; Read access to standard system paths.
(allow file-read* file-test-existence
  (subpath "/Library/Apple")
  (subpath "/Library/Filesystems/NetFSPlugins")
  (subpath "/Library/Preferences/Logging")
  (subpath "/private/var/db/DarwinDirectory/local/recordStore.data")
  (subpath "/private/var/db/timezone")
  (subpath "/usr/lib")
  (subpath "/usr/share")
  (subpath "/Library/Preferences")
  (subpath "/var/db")
  (subpath "/private/var/db"))

; Map system frameworks and dylibs for the loader.
(allow file-map-executable
  (subpath "/Library/Apple/System/Library/Frameworks")
  (subpath "/Library/Apple/System/Library/PrivateFrameworks")
  (subpath "/Library/Apple/usr/lib")
  (subpath "/System/Library/Extensions")
  (subpath "/System/Library/Frameworks")
  (subpath "/System/Library/PrivateFrameworks")
  (subpath "/System/Library/SubFrameworks")
  (subpath "/System/iOSSupport/System/Library/Frameworks")
  (subpath "/System/iOSSupport/System/Library/PrivateFrameworks")
  (subpath "/System/iOSSupport/System/Library/SubFrameworks")
  (subpath "/usr/lib"))

; System framework resources.
(allow file-read* file-test-existence
  (subpath "/Library/Apple/System/Library/Frameworks")
  (subpath "/Library/Apple/System/Library/PrivateFrameworks")
  (subpath "/Library/Apple/usr/lib")
  (subpath "/System/Library/Frameworks")
  (subpath "/System/Library/PrivateFrameworks")
  (subpath "/System/Library/SubFrameworks")
  (subpath "/System/iOSSupport/System/Library/Frameworks")
  (subpath "/System/iOSSupport/System/Library/PrivateFrameworks")
  (subpath "/System/iOSSupport/System/Library/SubFrameworks")
  (subpath "/usr/lib"))

; Guarded vnodes.
(allow system-mac-syscall (mac-policy-name "vnguard"))

; Container detection.
(allow system-mac-syscall
  (require-all
    (mac-policy-name "Sandbox")
    (mac-syscall-number 67)))

; Symlink resolution for standard paths.
(allow file-read-metadata file-test-existence
  (literal "/etc")
  (literal "/tmp")
  (literal "/var")
  (literal "/private/etc/localtime"))

; Firmlink parent traversal.
(allow file-read-metadata file-test-existence
  (path-ancestors "/System/Volumes/Data/private"))

; Root directory.
(allow file-read* file-test-existence
  (literal "/"))

; Alternate chflags.
(allow system-fsctl (fsctl-command FSIOC_CAS_BSDFLAGS))

; Standard special files.
(allow file-read* file-test-existence
  (literal "/dev/autofs_nowait")
  (literal "/dev/random")
  (literal "/dev/urandom")
  (literal "/private/etc/master.passwd")
  (literal "/private/etc/passwd")
  (literal "/private/etc/protocols")
  (literal "/private/etc/services"))

; /dev/null and /dev/zero read/write.
(allow file-read* file-test-existence file-write-data
  (literal "/dev/null")
  (literal "/dev/zero"))

; File descriptors.
(allow file-read-data file-test-existence file-write-data
  (subpath "/dev/fd"))

; DTrace helper.
(allow file-read* file-test-existence file-write-data file-ioctl
  (literal "/dev/dtracehelper"))

; Scratch space (tmp directories).
(allow file-read* file-test-existence file-write* (subpath "/tmp"))
(allow file-read* file-write* (subpath "/private/tmp"))
(allow file-read* file-write* (subpath "/var/tmp"))
(allow file-read* file-write* (subpath "/private/var/tmp"))

; Standard config directories.
(allow file-read* (subpath "/etc"))
(allow file-read* (subpath "/private/etc"))

; CoreServices version plists.
(allow file-read* file-test-existence
  (literal "/System/Library/CoreServices")
  (literal "/System/Library/CoreServices/.SystemVersionPlatform.plist")
  (literal "/System/Library/CoreServices/SystemVersion.plist"))

; /var metadata.
(allow file-read-metadata (subpath "/var"))
(allow file-read-metadata (subpath "/private/var"))

; System agents and services.
(allow mach-lookup
  (global-name "com.apple.analyticsd")
  (global-name "com.apple.analyticsd.messagetracer")
  (global-name "com.apple.appsleep")
  (global-name "com.apple.bsd.dirhelper")
  (global-name "com.apple.cfprefsd.agent")
  (global-name "com.apple.cfprefsd.daemon")
  (global-name "com.apple.diagnosticd")
  (global-name "com.apple.dt.automationmode.reader")
  (global-name "com.apple.espd")
  (global-name "com.apple.logd")
  (global-name "com.apple.logd.events")
  (global-name "com.apple.runningboard")
  (global-name "com.apple.secinitd")
  (global-name "com.apple.system.DirectoryService.libinfo_v1")
  (global-name "com.apple.system.logger")
  (global-name "com.apple.system.notification_center")
  (global-name "com.apple.system.opendirectoryd.membership")
  (global-name "com.apple.trustd")
  (global-name "com.apple.trustd.agent")
  (global-name "com.apple.xpc.activity.unmanaged")
  (local-name "com.apple.cfprefsd.agent"))

; Syslog socket.
(allow network-outbound (literal "/private/var/run/syslog"))

; macOS notifications shared memory.
(allow ipc-posix-shm-read*
  (ipc-posix-name "apple.shm.notification_center"))

; Eligibility plist.
(allow file-read*
  (literal "/private/var/db/eligibilityd/eligibility.plist"))

; Audio and power management.
(allow mach-lookup (global-name "com.apple.audio.audiohald"))
(allow mach-lookup (global-name "com.apple.audio.AudioComponentRegistrar"))

; System binaries.
(allow file-read-data (subpath "/bin"))
(allow file-read-metadata (subpath "/bin"))
(allow file-read-data (subpath "/sbin"))
(allow file-read-metadata (subpath "/sbin"))
(allow file-read-data (subpath "/usr/bin"))
(allow file-read-metadata (subpath "/usr/bin"))
(allow file-read-data (subpath "/usr/sbin"))
(allow file-read-metadata (subpath "/usr/sbin"))
(allow file-read-data (subpath "/usr/libexec"))
(allow file-read-metadata (subpath "/usr/libexec"))

; Library paths (Homebrew, system, Applications).
(allow file-read* (subpath "/Library/Preferences"))
(allow file-read* (subpath "/opt/homebrew/lib"))
(allow file-read* (subpath "/usr/local/lib"))
(allow file-read* (subpath "/Applications"))

; Terminal device handles.
(allow file-read* (regex "^/dev/fd/(0|1|2)$"))
(allow file-write* (regex "^/dev/fd/(1|2)$"))
(allow file-read* file-write* (literal "/dev/null"))
(allow file-read* file-write* (literal "/dev/tty"))
(allow file-read-metadata (literal "/dev"))
(allow file-read-metadata (regex "^/dev/.*$"))
(allow file-read-metadata (literal "/dev/stdin"))
(allow file-read-metadata (literal "/dev/stdout"))
(allow file-read-metadata (literal "/dev/stderr"))
(allow file-read-metadata (regex "^/dev/tty[^/]*$"))
(allow file-read-metadata (regex "^/dev/pty[^/]*$"))
(allow file-read* file-write* (regex "^/dev/ttys[0-9]+$"))
(allow file-read* file-write* (literal "/dev/ptmx"))
(allow file-ioctl (regex "^/dev/ttys[0-9]+$"))

; Firmlink volume metadata.
(allow file-read-metadata (literal "/System/Volumes") (vnode-type DIRECTORY))
(allow file-read-metadata (literal "/System/Volumes/Data") (vnode-type DIRECTORY))
(allow file-read-metadata (literal "/System/Volumes/Data/Users") (vnode-type DIRECTORY))

; App sandbox extensions.
(allow file-read* (extension "com.apple.app-sandbox.read"))
(allow file-read* file-write* (extension "com.apple.app-sandbox.read-write"))
"#;

/// Returns `true` when macOS `sandbox-exec` is available.
pub fn sandbox_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(sandbox_exec_accepts_policy)
}

fn sandbox_exec_accepts_policy() -> bool {
    if !Path::new(SANDBOX_EXEC).is_file() {
        return false;
    }
    Command::new(SANDBOX_EXEC)
        .arg("-p")
        .arg("(version 1)\n(allow default)\n")
        .arg("/usr/bin/true")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Resolve `DARWIN_USER_CACHE_DIR` for the current user.
///
/// Falls back to `/private/var/folders` when `getconf` is unavailable.
pub fn darwin_user_cache_dir() -> String {
    let output = Command::new("/usr/bin/getconf")
        .arg("DARWIN_USER_CACHE_DIR")
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
            dir.trim_end_matches('/').to_string()
        }
        _ => "/private/var/folders".to_string(),
    }
}

/// Escape a string for inclusion in an SBPL double-quoted literal.
///
/// SBPL uses Scheme-style string literals where backslash and double-quote
/// must be escaped.  Control characters (newlines, tabs, etc.) are also
/// escaped to prevent policy injection through crafted paths.
fn escape_sbpl_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                // Drop other control characters entirely.
            }
            c => out.push(c),
        }
    }
    out
}

/// Build an SBPL policy string with the given writable roots inlined.
///
/// Each entry becomes an `(allow file-write* (subpath "..."))` rule.
/// Path strings are escaped to prevent SBPL injection.
/// `DARWIN_USER_CACHE_DIR` remains a `sandbox-exec` parameter.
pub fn build_policy(writable_roots: &WritableRoots) -> String {
    let roots = writable_roots.as_slice();
    let mut policy = String::with_capacity(
        POLICY_BEFORE_WRITE_RULES.len() + POLICY_AFTER_WRITE_RULES.len() + roots.len() * 64,
    );
    policy.push_str(POLICY_BEFORE_WRITE_RULES);
    for root in roots {
        let escaped = escape_sbpl_string(root);
        policy.push_str(&format!("(allow file-write* (subpath \"{escaped}\"))\n"));
    }
    policy.push_str(POLICY_AFTER_WRITE_RULES);
    policy
}

/// Build the `shell_wrapper` argument list for [`claudius::BashPtyConfig`].
///
/// Returns `None` when sandboxing is unavailable (non-macOS).
/// The returned vec is `[sandbox-exec, -p, <policy>, -D, DARWIN_USER_CACHE_DIR=..., --]`.
pub fn shell_wrapper(writable_roots: &WritableRoots) -> Option<Vec<String>> {
    if !sandbox_available() {
        return None;
    }
    let policy = build_policy(writable_roots);
    let cache_dir = darwin_user_cache_dir();
    Some(vec![
        SANDBOX_EXEC.to_string(),
        "-p".to_string(),
        policy,
        "-D".to_string(),
        format!("DARWIN_USER_CACHE_DIR={cache_dir}"),
        "--".to_string(),
    ])
}

/// Build a [`tokio::process::Command`] that runs `program` with `args` inside
/// the seatbelt sandbox.
///
/// When sandboxing is unavailable, returns a plain command without wrapping.
pub fn sandboxed_command(
    program: &str,
    args: &[&str],
    writable_roots: &WritableRoots,
) -> tokio::process::Command {
    if sandbox_available() {
        let policy = build_policy(writable_roots);
        let cache_dir = darwin_user_cache_dir();
        let mut cmd = tokio::process::Command::new(SANDBOX_EXEC);
        cmd.arg("-p").arg(policy);
        cmd.arg("-D")
            .arg(format!("DARWIN_USER_CACHE_DIR={cache_dir}"));
        cmd.arg("--");
        cmd.arg(program);
        cmd.args(args);
        cmd
    } else {
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args);
        cmd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots(paths: &[&str]) -> WritableRoots {
        let mut wr = WritableRoots::default();
        for p in paths {
            wr.push(p.to_string());
        }
        wr
    }

    #[test]
    fn writable_roots_from_str_splits_on_colon() {
        let wr: WritableRoots = "/a:/b:/c".parse().unwrap();
        assert_eq!(wr.as_slice(), &["/a", "/b", "/c"]);
    }

    #[test]
    fn writable_roots_from_str_drops_empty_components() {
        let wr: WritableRoots = ":/a::/b:".parse().unwrap();
        assert_eq!(wr.as_slice(), &["/a", "/b"]);
    }

    #[test]
    fn writable_roots_display_round_trips() {
        let wr: WritableRoots = "/a:/b".parse().unwrap();
        assert_eq!(wr.to_string(), "/a:/b");
    }

    #[test]
    fn writable_roots_push_ignores_empty() {
        let mut wr = WritableRoots::default();
        wr.push(String::new());
        assert!(wr.is_empty());
    }

    #[test]
    fn build_policy_contains_writable_root_rules() {
        let policy = build_policy(&roots(&["/home/user/src", "/tmp/scratch"]));
        assert!(
            policy.contains("(allow file-write* (subpath \"/home/user/src\"))"),
            "policy should contain first writable root"
        );
        assert!(
            policy.contains("(allow file-write* (subpath \"/tmp/scratch\"))"),
            "policy should contain second writable root"
        );
    }

    #[test]
    fn build_policy_with_no_roots_has_no_writable_root_rules() {
        let policy = build_policy(&WritableRoots::default());
        assert!(
            !policy.contains("(allow file-write* (subpath \"/home"),
            "policy should have no writable-root rules"
        );
    }

    #[test]
    fn build_policy_is_valid_sbpl_structure() {
        let policy = build_policy(&roots(&["/workspace"]));
        assert!(policy.starts_with("(version 1)"));
        assert!(policy.contains("(deny default)"));
        assert!(policy.contains("(allow file-read*)"));
        assert!(policy.contains("(param \"DARWIN_USER_CACHE_DIR\")"));
    }

    #[test]
    fn shell_wrapper_returns_sandbox_exec_args() {
        if !sandbox_available() {
            return;
        }
        let wrapper = shell_wrapper(&roots(&["/workspace"])).expect("sandbox should be available");
        assert_eq!(wrapper[0], SANDBOX_EXEC);
        assert_eq!(wrapper[1], "-p");
        assert!(wrapper[2].contains("(version 1)"));
        assert_eq!(wrapper[3], "-D");
        assert!(wrapper[4].starts_with("DARWIN_USER_CACHE_DIR="));
        assert_eq!(wrapper[5], "--");
    }

    #[test]
    fn escape_sbpl_string_handles_plain_path() {
        assert_eq!(escape_sbpl_string("/home/user/src"), "/home/user/src");
    }

    #[test]
    fn escape_sbpl_string_escapes_backslash() {
        assert_eq!(escape_sbpl_string(r"/path\to"), r"/path\\to");
    }

    #[test]
    fn escape_sbpl_string_escapes_double_quote() {
        assert_eq!(escape_sbpl_string(r#"/path "evil""#), r#"/path \"evil\""#);
    }

    #[test]
    fn escape_sbpl_string_escapes_newline_and_tab() {
        assert_eq!(escape_sbpl_string("/path\n/inject"), "/path\\n/inject");
        assert_eq!(escape_sbpl_string("/path\t/tab"), "/path\\t/tab");
    }

    #[test]
    fn escape_sbpl_string_drops_control_chars() {
        assert_eq!(escape_sbpl_string("/path\x01\x7f/ctl"), "/path/ctl");
    }

    #[test]
    fn build_policy_escapes_quote_in_writable_root() {
        let policy = build_policy(&roots(&["/home/user/O'Brien\"s"]));
        assert!(
            policy.contains(r#"(subpath "/home/user/O'Brien\"s")"#),
            "double-quote in path must be escaped: {policy}"
        );
    }

    #[test]
    fn build_policy_escapes_newline_in_writable_root() {
        let policy = build_policy(&roots(&["/home/user/evil\n(deny default)"]));
        assert!(
            !policy.contains("\n(deny default)\")"),
            "newline in path must not produce raw SBPL: {policy}"
        );
        assert!(
            policy.contains(r#"(subpath "/home/user/evil\n(deny default)")"#),
            "newline should be escaped to literal \\n: {policy}"
        );
    }
}
