# sid isn't done

`sid` is a minimal framework for a coding agent. The workspace defines:

- agents in `agents.conf` plus `agents/<agent>.md`
- tools in `tools.conf`
- rc-style tool entrypoints in `tools/<tool>`
- tool manifests in `tools/<tool>.json`

There are no hardcoded built-in tools. Every tool goes through the same external contract.

## Agents

An agent is a prompt plus a list of tools:

```ignore
$ cat agents.conf
build_ENABLED="YES"
plan_ENABLED="MANUAL"

build_NAME="Let's Go!"
build_DESC="buildit"
build_TOOLS="format shell"
build_SKILLS="*"
```

```ignore
$ cat agents/build.md
# Build

You are an expert builder.
```

## Tool Config

`tools.conf` controls enable/disable state and aliases:

```ignore
$ cat tools.conf
fmt_ENABLED="YES"
shell_ENABLED="YES"

format_ALIASES="fmt"
```

Resolution rules:

- `tools.conf` is the source of enable/disable and alias behavior
- alias resolution happens before filesystem lookup
- only the canonical resolved tool id needs an executable and manifest
- the model-visible name is the invoked tool name from agent configuration

In the example above:

- `fmt` resolves to `tools/fmt` and `tools/fmt.json`
- `format` also resolves to `tools/fmt` and `tools/fmt.json`
- the model sees `format` because that is the invoked tool name

## Tool Layout

Each canonical tool provides:

```text
tools/<id>
tools/<id>.json
```

`tools/<id>` is an rc-style executable: it must answer `rcvar` and `run`, so it can be launched
by both `sid` and `rcinvoke`.

Minimal shell shape:

```sh
#!/bin/sh
set -eu

case "${1:-}" in
  rcvar)
    printf '%s\n' \
      "${RCVAR_ARGV0}_REQUEST_FILE" \
      "${RCVAR_ARGV0}_RESULT_FILE" \
      "${RCVAR_ARGV0}_WORKSPACE_ROOT"
    ;;
  run)
    REQUEST_FILE=$(printenv "${RCVAR_ARGV0}_REQUEST_FILE")
    RESULT_FILE=$(printenv "${RCVAR_ARGV0}_RESULT_FILE")
    WORKSPACE_ROOT=$(printenv "${RCVAR_ARGV0}_WORKSPACE_ROOT")
    exec ./tools/fmt.impl
    ;;
  *)
    echo "usage: $0 [rcvar|run]" >&2
    exit 129
    ;;
esac
```

Example manifest:

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

- `protocol_version` is required and must be `1`
- `description` is required and becomes the tool description sent to the model
- `input_schema` is required and is sent verbatim as the tool JSON schema
- the manifest does not contain the tool name

## Invocation Contract

For each tool call, `sid` creates a fresh scratch directory, writes `request.json`, writes an
overlay `rc.conf`, launches `<tool> run`, and expects `result.json` back.  The tool's `rcvar`
output determines which invocation variables are bound.

`request.json`:

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

Success `result.json`:

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

Handled tool failure:

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

- `stdout` and `stderr` are for the human terminal only
- `sid` only parses `result.json`
- exit code `0` means process transport succeeded, so `result.json` must exist and be valid
- nonzero exit status is treated as process failure and any partial `result.json` is ignored
- `request_id` in `result.json` must match `request.json`
- v1 output is text-only

## Execution Environment

The tool is launched with:

- `cwd = workspace root`
- inherited `stdin`, `stdout`, and `stderr`
- `RCVAR_ARGV0=<invoked tool name rendered as an rc variable>`
- `RC_CONF_PATH=<abs tools.conf>:<abs scratch overlay>`
- `RC_D_PATH=<abs workspace>/tools`

The overlay binds per-tool invocation values under the invoked tool name.  If the model calls
`format`, the tool can advertise and consume:

- `format_REQUEST_FILE=<abs path>`
- `format_RESULT_FILE=<abs path>`
- `format_SCRATCH_DIR=<abs path>`
- `format_WORKSPACE_ROOT=<abs path>`
- `format_AGENT_ID=<agent id>`
- `format_TOOL_ID=<canonical tool id>`
- `format_TOOL_NAME=format`
- `format_TOOL_PROTOCOL=1`
- `format_RC_CONF_PATH=<same as RC_CONF_PATH>`
- `format_RC_D_PATH=<same as RC_D_PATH>`

Aliases get their own prefix, so invoking `fmt` yields `fmt_REQUEST_FILE`, while invoking
`format` yields `format_REQUEST_FILE`.  This means a tool can recurse through
`rcinvoke --rc-conf-path "$RC_CONF_PATH" --rc-d-path "$RC_D_PATH" <tool>` without manually
rebuilding the workspace contract.
