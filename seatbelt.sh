#!/bin/bash
# seatbelt.sh - Run a command in a macOS Seatbelt sandbox.
#
# Policy: deny-default, then allow:
#   Filesystem: full read, write only to WRITABLE_ROOT and /tmp
#   Network:    localhost only (inbound + outbound)
#
# Usage: WRITABLE_ROOT=/path ./seatbelt.sh <command> [args...]
#
# Derived from the Codex sandbox profiles:
#   codex-rs/sandboxing/src/seatbelt_base_policy.sbpl
#   codex-rs/sandboxing/src/seatbelt_network_policy.sbpl
#   codex-rs/sandboxing/src/restricted_read_only_platform_defaults.sbpl

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 1
fi

# Use the absolute path to sandbox-exec to defend against PATH injection.
SANDBOX_EXEC=/usr/bin/sandbox-exec

if [ ! -x "$SANDBOX_EXEC" ]; then
    echo "error: $SANDBOX_EXEC not found; this script requires macOS" >&2
    exit 1
fi

# Resolve the writable root.  Honor a caller-provided WRITABLE_ROOT, otherwise
# default to ~/src.  Canonicalize if the directory already exists; fall back to
# the literal expansion so the sandbox still works before the first mkdir.
WRITABLE_ROOT="${SID_WORKSPACE_ROOT:-$HOME/src}"
if [ -d "$WRITABLE_ROOT" ]; then
    WRITABLE_ROOT="$(cd "$WRITABLE_ROOT" && pwd -P)"
fi

# DARWIN_USER_CACHE_DIR is needed by the network-policy section so that
# libnetwork/nsurlsession can write its TLS session cache.
DARWIN_USER_CACHE_DIR="$(getconf DARWIN_USER_CACHE_DIR 2>/dev/null || echo "/private/var/folders")"
# Strip trailing slash that getconf sometimes appends.
DARWIN_USER_CACHE_DIR="${DARWIN_USER_CACHE_DIR%/}"

# --- Assemble the policy ------------------------------------------------
#
# The policy is built from four sections in the same order the Codex Rust
# code uses:
#   1. Base policy         (deny default, process control, PTY, sysctls, IPC)
#   2. Read policy         (full disk read)
#   3. Write policy        (WRITABLE_ROOT + /tmp)
#   4. Network policy      (localhost only + platform services)
#   5. Platform defaults   (system libs, frameworks, /dev, /tmp, binaries)

read -r -d '' POLICY <<'SBPL' || true
(version 1)

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
;; 3. WRITE POLICY — WRITABLE_ROOT (parameterized) + /tmp
;; =======================================================================

; Write access to the workspace root.
(allow file-write* (subpath (param "WRITABLE_ROOT")))

;; =======================================================================
;; 4. NETWORK POLICY — localhost only + platform services
;; =======================================================================

; Allow binding on any local port (needed for dev servers).
(allow network-bind (local ip "*:*"))

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
SBPL

exec "$SANDBOX_EXEC" \
    -p "$POLICY" \
    -D "WRITABLE_ROOT=$WRITABLE_ROOT" \
    -D "DARWIN_USER_CACHE_DIR=$DARWIN_USER_CACHE_DIR" \
    -- "$@"
