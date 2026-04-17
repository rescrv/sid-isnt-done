# SID(1)

## NAME

sid - run a small, rc-configured coding agent in the current workspace

## SYNOPSIS

```text
sid [OPTIONS]
SID_HOME=DIR sid [OPTIONS]
sid --bash-debug COMMAND
sid-seatbelt [--writable-roots DIR[:DIR...]] -- COMMAND [ARG...]
```

Generated help spells long options with one leading dash.  The parser also
accepts the double-dash forms used below.

## DESCRIPTION

`sid` starts an interactive coding-agent session rooted at the current working
directory.  The workspace is mounted for the model's virtual filesystem as `/`;
that virtual mount is not an operating-system chroot.  Agent definitions, tool
definitions, and optional skills are read from rc-style configuration files
rather than from hardcoded tool lists.

When no `sid` configuration exists, `sid` starts the built-in `sid` agent with
no configured external tools.  When `agents.conf` or `tools.conf` exists,
configuration is loaded from `SID_HOME` if it is set and non-empty; otherwise
configuration is loaded from the current directory.

The interactive prompt accepts ordinary user messages and slash commands.  Use
`/help` inside a running session for chat commands such as changing the model,
saving or loading transcripts, clearing context, and printing session stats.

## OPTIONS

`--param-model MODEL`
: Use `MODEL` for the session.  The default is supplied by `claudius` and is
  currently printed by `sid --help`.

`--param-system PROMPT`
: Set the initial system prompt.  Agent prompt files and agent configuration can
  override this value when a configured workspace is loaded.

`--param-max-tokens TOKENS`
: Set the maximum response tokens per model request.

`--param-temperature TEMP`
: Set sampling temperature.  `TEMP` must be between `0.0` and `1.0`.

`--param-top-p TOP_P`
: Set nucleus sampling.  `TOP_P` must be between `0.0` and `1.0`.

`--param-top-k TOP_K`
: Set top-k sampling.

`--param-thinking TOKENS`
: Enable extended thinking with the given token budget.

`--param-no-color`
: Disable ANSI color and style output.

`--bash-debug COMMAND`
: Run `COMMAND` through the configured built-in bash tool and exit.  This is
  useful for checking tool configuration without starting an interactive chat.

`--help`
: Print the command-line help.

## CONFIGURATION

Configuration uses two required files when configuration is present:

```text
agents.conf
tools.conf
```

Agent prompts live in `agents/`.  External tool executables and manifests live
in `tools/`.  Skills live in `skills/<skill>/SKILL.md` unless `SID_SKILLS_PATH`
is set.

The bundled starter configuration is in `init/`:

```sh
SID_HOME=init sid
```

## AGENTS

An agent is an rc-conf service in `agents.conf`.  Service names are discovered
by `rc_conf`; each service may define fields under the service prefix:

```sh
DEFAULT_AGENT="build"

build_ENABLED="YES"
build_NAME="Let's Go"
build_DESC="buildit"
build_TOOLS="bash edit format"
build_SKILLS="*"
build_MODEL="claude-sonnet-4-5"
build_MAX_TOKENS="8192"
build_THINKING="on"
```

`<agent>_ENABLED`
: Controls whether the agent can start.  `YES` starts immediately, `MANUAL`
  asks the operator before starting, and `NO` disables the agent.

`<agent>_NAME`
: Optional display name.

`<agent>_DESC`
: Optional description.

`<agent>_TOOLS`
: Space-split list of configured tool names exposed to the agent.

`<agent>_SKILLS`
: Space-split list of skills to mount.  Use `*` to mount every loaded skill.

`<agent>_MODEL`, `<agent>_SYSTEM`, `<agent>_MAX_TOKENS`
: Optional model, system prompt, and response-token overrides.

`<agent>_TEMPERATURE`, `<agent>_TOP_P`, `<agent>_TOP_K`
: Optional sampling controls.

`<agent>_STOP_SEQUENCES`
: Space-split stop sequence list.  Shell-style quoting is supported by
  `shvar`.

`<agent>_THINKING`
: `on`, `yes`, or `true` enables the default thinking budget.  A number sets an
  explicit budget.  `off`, `no`, or `false` disables thinking.

`<agent>_USE_COLOR`, `<agent>_NO_COLOR`
: Optional terminal color controls.

`<agent>_SESSION_BUDGET`
: Optional token budget for the session.

`<agent>_TRANSCRIPT_PATH`
: Optional path for transcript auto-save.

`<agent>_CACHING_ENABLED`
: Optional prompt-cache toggle.

The prompt file for an agent is `agents/<agent>.md`.  If the agent is an alias,
`sid` follows the rc-conf alias lookup order and uses the first matching prompt
file.  Prompt-file content becomes the agent's system prompt unless overridden
by `<agent>_SYSTEM`.

If `DEFAULT_AGENT` is unset, `sid` starts the first enabled agent.  If no agent
is enabled, it starts the first manual agent after confirmation.

## TOOLS

Tools are rc-conf services in `tools.conf`.  There are no implicit external
tools: a tool named by an agent must also be defined in `tools.conf`.
Canonical tool ids and model-visible external tool names must be 1-64 ASCII
letters, digits, underscores, or hyphens.

```sh
bash_ENABLED="MANUAL"
edit_ENABLED="MANUAL"
fmt_ENABLED="YES"

format_INHERIT="YES"
format_ALIASES="fmt"
```

`<tool>_ENABLED`
: Controls whether the tool can be used.  `YES` allows calls, `MANUAL` prompts
  the operator for every call, and `NO` disables the tool.

`<tool>_ALIASES`
: Defines aliases resolved before filesystem lookup.  In the example above,
  `format` resolves to canonical tool `fmt`.

`bash`
: Built-in bash capability.  It is exposed to the model as Anthropic's bash
  tool.  It runs in the host filesystem namespace, not a chroot.  The initial
  working directory is the workspace root; host `/` remains visible subject to
  normal OS permissions and the optional macOS Seatbelt policy.  It does not
  need a `tools/bash` executable or `tools/bash.json` manifest.

`edit`
: Built-in text-editor capability.  It is exposed to the model as Anthropic's
  text editor tool, but calls are routed through `tools/edit`; the starter
  script execs `sid-editor-tool`.  The helper process is not chrooted, but the
  editor protocol resolves file paths under `WORKSPACE_ROOT`; `/etc/passwd` in
  an editor request means `$WORKSPACE_ROOT/etc/passwd`, not host `/etc/passwd`.

External tools must provide both files below for the canonical tool id:

```text
tools/<id>
tools/<id>.json
```

The executable must be marked executable.  The manifest supplies the model
description and input schema:

```json
{
  "protocol_version": 1,
  "description": "Format source files in the workspace.",
  "input_schema": {
    "type": "object",
    "properties": {
      "paths": {
        "type": "array",
        "items": { "type": "string" }
      }
    },
    "required": ["paths"]
  }
}
```

Manifest rules:

- `protocol_version` is required and must be `1`.
- `description` is required and must not be empty.
- `input_schema` is required and must be a JSON object.
- The manifest does not contain the tool name.

The model-visible name is the name listed in `<agent>_TOOLS`, not necessarily
the canonical id.  Thus `format` can resolve to canonical executable `tools/fmt`
while still appearing to the model as `format`.

## TOOL PROTOCOL

Tool executables are rc-style programs.  They must respond to `rcvar` and
`run`.

```sh
#!/bin/sh
set -eu

PREFIX=${RCVAR_ARGV0:?missing RCVAR_ARGV0}

case "${1:-}" in
rcvar)
    printf '%s\n' \
        "${PREFIX}_REQUEST_FILE" \
        "${PREFIX}_RESULT_FILE" \
        "${PREFIX}_SCRATCH_DIR" \
        "${PREFIX}_WORKSPACE_ROOT" \
        "${PREFIX}_AGENT_ID" \
        "${PREFIX}_TOOL_ID" \
        "${PREFIX}_TOOL_NAME" \
        "${PREFIX}_TOOL_PROTOCOL" \
        "${PREFIX}_RC_CONF_PATH" \
        "${PREFIX}_RC_D_PATH"
    ;;
run)
    export REQUEST_FILE=$(printenv "${PREFIX}_REQUEST_FILE")
    export RESULT_FILE=$(printenv "${PREFIX}_RESULT_FILE")
    export WORKSPACE_ROOT=$(printenv "${PREFIX}_WORKSPACE_ROOT")
    exec ./tools/fmt.impl
    ;;
*)
    echo "usage: $0 [rcvar|run]" >&2
    exit 129
    ;;
esac
```

For each tool call, `sid` creates a fresh scratch directory under the system
temporary directory, writes `request.json`, writes an rc-conf overlay, invokes
`tools/<id> run`, then reads `result.json`.  The tool process runs with the
workspace root as its current directory and inherits standard input, standard
output, and standard error.  Tool processes are not chrooted; host `/` is still
the process root unless the operating system or sandbox policy denies a
particular operation.

The request file has this shape:

```json
{
  "protocol_version": 1,
  "request_id": "sidreq_123",
  "tool": {
    "id": "fmt"
  },
  "invocation": {
    "tool_use_id": "toolu_abc",
    "input": {
      "paths": ["src/lib.rs"]
    }
  },
  "agent": {
    "id": "build"
  },
  "workspace": {
    "root": "/abs/workspace",
    "cwd": "/abs/workspace"
  },
  "files": {
    "scratch_dir": "/tmp/sid-tool/sidreq_123",
    "result_file": "/tmp/sid-tool/sidreq_123/result.json"
  }
}
```

A successful result is:

```json
{
  "protocol_version": 1,
  "request_id": "sidreq_123",
  "ok": true,
  "output": {
    "kind": "text",
    "text": "Formatted 3 files."
  }
}
```

A handled failure is:

```json
{
  "protocol_version": 1,
  "request_id": "sidreq_123",
  "ok": false,
  "error": {
    "code": "invalid_input",
    "message": "paths must not be empty"
  }
}
```

Protocol rules:

- Standard output and standard error are for the human terminal.
- `sid` only parses `result.json`.
- Exit status `0` means process transport succeeded, so `result.json` must
  exist and be valid.
- Nonzero exit status is treated as process failure; any partial result file is
  ignored.
- `request_id` in the result must match the request.
- Protocol v1 output is text-only.

## ENVIRONMENT

`SID_HOME`
: Configuration root.  If unset or empty, the current working directory is
  used.

`SID_SKILLS_PATH`
: Colon-separated list of directories to scan for `*/SKILL.md`.  If unset,
  `sid` scans `skills/` under the configuration root.

`SID_WORKSPACE_ROOT`
: Set by `sid` for child processes to the absolute workspace root.

`RCVAR_ARGV0`
: Set during tool invocation to the invoked tool name rendered as an rc
  variable prefix.

`RC_CONF_PATH`
: Set during tool invocation to `<config-root>/tools.conf:<scratch>/tool-invoke.conf`.

`RC_D_PATH`
: Set during tool invocation to `<config-root>/tools`.

For each configured tool service, the overlay binds these variables under that
service's prefix:

```text
<tool>_REQUEST_FILE
<tool>_RESULT_FILE
<tool>_SCRATCH_DIR
<tool>_WORKSPACE_ROOT
<tool>_AGENT_ID
<tool>_TOOL_ID
<tool>_TOOL_NAME
<tool>_TOOL_PROTOCOL
<tool>_RC_CONF_PATH
<tool>_RC_D_PATH
```

Aliases get their own prefix.  If the model invokes `format`, the tool reads
`format_REQUEST_FILE`; if it invokes `fmt`, it reads `fmt_REQUEST_FILE`.

## SANDBOXING

On macOS, `sid` wraps bash and external tool processes with
`/usr/bin/sandbox-exec` when it is available.  This is a sandbox wrapper, not a
chroot: processes still see host `/`.  The generated policy allows full
filesystem reads, writes to the workspace and the system temporary directory,
and loopback networking.  On systems without `sandbox-exec`, commands run
without this wrapper.

`sid-seatbelt` is a helper for running an arbitrary command under the same
macOS Seatbelt policy:

```sh
sid-seatbelt --writable-roots "$PWD:/tmp" -- make test
```

## FILES

`agents.conf`
: Agent services and default-agent selection.

`agents/<agent>.md`
: Agent prompt markdown.

`tools.conf`
: Tool services, enable states, and aliases.

`tools/<id>`
: Rc-style executable for an external tool or the `edit` bridge.

`tools/<id>.json`
: Tool manifest for model-visible external tools.

`skills/<skill>/SKILL.md`
: Optional skill document mounted read-only under `/skills/<skill>/`.

`/tmp/sid-tool/sidreq_*`
: Per-call scratch directories on typical Unix systems.  The exact parent is
  the platform's system temporary directory.

## EXIT STATUS

`0`
: Successful interactive session, accepted manual abort, or successful
  `--bash-debug` command.

`1`
: Help display, startup failure, configuration failure, client initialization
  failure, I/O failure, or `--bash-debug` failure.

`64`
: Command-line parse failure reported by `arrrg`.

## EXAMPLES

Start with the bundled manual-confirmation tools:

```sh
SID_HOME=init sid
```

Run a one-shot bash configuration check:

```sh
SID_HOME=init sid --bash-debug 'pwd && ls'
```

Define a formatter tool:

```sh
cat >tools.conf <<'EOF'
fmt_ENABLED="YES"
format_INHERIT="YES"
format_ALIASES="fmt"
EOF

mkdir -p tools agents
```

Expose it to an agent:

```sh
cat >agents.conf <<'EOF'
DEFAULT_AGENT="build"
build_ENABLED="YES"
build_TOOLS="format"
EOF

cat >agents/build.md <<'EOF'
# Build

You are an expert builder.
EOF
```

Implement `tools/fmt`, mark it executable, and place the schema in
`tools/fmt.json`.  Calls to `format` will execute the canonical `fmt` tool while
preserving `format` as the model-visible tool name.

## SEE ALSO

[SID-EDITOR-TOOL(1)](#sid-editor-tool1),
[SID-SEATBELT(1)](#sid-seatbelt1),
[RCINVOKE(1)](#rcinvoke1),
[SANDBOX-EXEC(1)](#sandbox-exec1)

# SID-EDITOR-TOOL(1)

## NAME

sid-editor-tool - execute the sid text-editor tool protocol

## SYNOPSIS

```text
sid-editor-tool
tools/edit run
```

## DESCRIPTION

`sid-editor-tool` is the helper used by the configured `edit` tool.  It is not
an interactive editor.  It reads a sid tool request from `REQUEST_FILE`,
executes one filesystem edit operation relative to `WORKSPACE_ROOT`, and writes
a sid tool result to `RESULT_FILE`.  The helper process is not chrooted, but
editor paths are workspace-rooted by the protocol implementation.

The starter `init/tools/edit` script is the normal entrypoint.  It receives
prefixed rc-conf variables from `sid`, exports the unprefixed environment used
by `sid-editor-tool`, then execs `sid-editor-tool`.

## COMMANDS

`view`
: Read a file.  Input fields are `path` and optional `view_range`.

`str_replace`
: Replace one exact string in a file.  Input fields are `path`, `old_str`, and
  optional `new_str`.

`insert`
: Insert text at a line.  Input fields are `path`, `insert_line`, and either
  `insert_text` or `new_str`.

`create`
: Create a new file.  Input fields are `path` and `file_text`.

## ENVIRONMENT

`REQUEST_FILE`
: Path to the JSON request envelope.

`RESULT_FILE`
: Path where the JSON result envelope must be written.

`WORKSPACE_ROOT`
: Workspace root used for filesystem operations.  Leading slashes in editor
  paths are stripped before joining with this directory, so editor path `/`
  names `WORKSPACE_ROOT`, not host `/`.

## EXIT STATUS

`0`
: The helper read the request and wrote a protocol result.  The result may
  still contain `"ok": false` for handled editor failures.

`nonzero`
: The helper failed before it could complete protocol transport, usually
  because a required environment variable was missing or a request/result file
  could not be read or written.

## SEE ALSO

[SID(1)](#sid1), [SID TOOL PROTOCOL](#tool-protocol)

# SID-SEATBELT(1)

## NAME

sid-seatbelt - run a command inside sid's macOS Seatbelt sandbox policy

## SYNOPSIS

```text
sid-seatbelt [--writable-roots DIR[:DIR...]] -- COMMAND [ARG...]
```

Generated help spells long options with one leading dash.  The parser also
accepts the double-dash form used above.

## DESCRIPTION

`sid-seatbelt` execs `COMMAND` under `/usr/bin/sandbox-exec` using the same
policy builder that `sid` uses for sandboxed bash and external tool processes.
It is a macOS helper; it exits with an error when `/usr/bin/sandbox-exec` is not
available.  It does not chroot `COMMAND`; host `/` remains the process root.

The policy is deny-by-default, permits full filesystem reads, permits writes to
configured writable roots and temporary directories, and limits network access
to loopback.

## OPTIONS

`--writable-roots DIR[:DIR...]`
: Colon-separated list of directories that should be writable inside the
  sandbox.

## EXAMPLES

Run tests with the current workspace and `/tmp` writable:

```sh
sid-seatbelt --writable-roots "$PWD:/tmp" -- cargo test
```

Start a local development server that may bind a loopback port:

```sh
sid-seatbelt --writable-roots "$PWD:/tmp" -- npm run dev
```

## EXIT STATUS

`1`
: No command was supplied, `sandbox-exec` is unavailable, or exec failed.

Otherwise, `sid-seatbelt` replaces itself with `sandbox-exec`; the final status
is the status reported by the sandboxed command.

## SEE ALSO

[SID(1)](#sid1), [SANDBOX-EXEC(1)](#sandbox-exec1)

# RCINVOKE(1)

## NAME

rcinvoke - invoke rc-style services by reading their advertised variables

## DESCRIPTION

`rcinvoke` is not implemented by this repository, but sid tools are shaped to be
compatible with it.  A sid tool executable answers `rcvar` with the variables it
needs, and answers `run` by performing the tool operation.

During a sid tool call, `RC_CONF_PATH` points at the workspace `tools.conf` plus
sid's per-call overlay, and `RC_D_PATH` points at the configured `tools/`
directory.  A tool can use those values to invoke another configured tool
without reconstructing sid's environment by hand.

## EXAMPLES

Invoke another configured tool from inside a sid tool:

```sh
rcinvoke --rc-conf-path "$RC_CONF_PATH" --rc-d-path "$RC_D_PATH" format
```

## SEE ALSO

[SID(1)](#sid1), [SID TOOL PROTOCOL](#tool-protocol)

# SANDBOX-EXEC(1)

## NAME

sandbox-exec - run a process under a macOS sandbox profile

## DESCRIPTION

`sandbox-exec` is the macOS program sid uses when it is available.  `sid` builds
an SBPL policy at runtime and passes it to `/usr/bin/sandbox-exec` for bash,
external tools, and `sid-seatbelt`.  The policy restricts operations; it does
not replace `/` with the workspace.

When `sandbox-exec` is unavailable, `sid` runs child processes without the
Seatbelt wrapper.  `sid-seatbelt` is stricter: it is specifically a
`sandbox-exec` frontend and exits with an error if the program is missing.

## POLICY

The generated sid policy:

- denies by default;
- allows child process execution and same-sandbox signaling;
- allows full filesystem reads;
- allows writes to the workspace, configured writable roots, and temporary
  directories;
- allows loopback networking for local servers and tools;
- includes platform allowances needed for common shells, build tools, language
  runtimes, and system libraries.

## SEE ALSO

[SID(1)](#sid1), [SID-SEATBELT(1)](#sid-seatbelt1)
