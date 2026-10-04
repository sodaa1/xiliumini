# xiliumini

`xiliumini` is a minimal Python Agent CLI built around an OpenAI-compatible model,
LangGraph, and a small set of bounded tools. Its main execution path is a
Planner Supervisor → Verifier workflow with bounded retries and a deterministic Final
node. The Supervisor delegates research to searchAgent and implementation to codeAgent,
then continues planning from their results. Failed verification returns to the Supervisor.

## Requirements

- Python 3.12 or newer (managed automatically by `uv` when needed)
- [uv](https://docs.astral.sh/uv/)
- An API key and model for an OpenAI-compatible endpoint
- Git available on PATH for checkpoints (the default); checkpoint `off` needs no Git

## Install

```powershell
git clone <repository-url>
cd xiliumini
uv sync
uv tool install --editable .
Copy-Item .env.example .env
```

If `uv` reports that its tool directory is not on `PATH`, run `uv tool update-shell`
once and open a new terminal.

Edit `.env` locally. It is ignored by Git and must never be committed.

```dotenv
XILIUMINI_API_KEY=your-provider-key
XILIUMINI_MODEL=your-model-name
XILIUMINI_BASE_URL=https://your-provider.example/v1
XILIUMINI_TEMPERATURE=0
XILIUMINI_TIMEOUT_SECONDS=60
XILIUMINI_DATA_DIR=.xiliumini
XILIUMINI_CONTEXT_WINDOW_TOKENS=64000
XILIUMINI_COMPRESSION_TRIGGER_RATIO=0.8
XILIUMINI_COMPRESSION_KEEP_TOKENS=8000
XILIUMINI_CHECKPOINT_MODE=light
XILIUMINI_TRACE_MODE=full
# Optional: a unique ID for one run; reuse is rejected
# XILIUMINI_TRACE_ID=my-run
# Optional: required only for web research
TAVILY_API_KEY=your-tavily-key
```

`XILIUMINI_BASE_URL` is optional for the default OpenAI endpoint. The API key is
represented as a Pydantic secret and is never written to CLI errors or tool output.

## Commands

Enter the project directory and start the full-screen Textual conversation interface. No
`uv run` prefix or subcommand is needed; `xiliumini chat` is an explicit alias:

```powershell
cd E:\Learning\xiliumini
xiliumini
xiliumini chat
```

The centered, large ASCII logo settles into a Claude Code-style welcome header with the
tagline “你的专属智能 Agent” and a session status line. After the first message is submitted, the Logo hides and the header
collapses to the session status so the conversation gets the available space. Submit
natural-language messages in the bottom input; one turn runs at a time and the same session
is reused. Each turn places plans, tool calls/results, handoffs, searches, checkpoints, and
verifier details in a collapsed `思考过程` section. Select the section or press Enter to inspect
it; when the turn finishes, its title shows the elapsed time. Final answers and terminal errors
remain visible directly below the collapsed details. Terminals without color support receive a
static plain-text logo.

Check Python, configuration, the data directory, and model construction:

```powershell
xiliumini doctor
```

Run the Supervisor workflow. `--max-attempts` limits Supervisor rounds and defaults to 3:

```powershell
xiliumini ask "帮我实现一个 Conway's Game of Life，要求 TDD：先写测试，再写实现，最后跑 demo" --max-attempts 3
xiliumini ask "Calculate (17 + 5) * 3" --no-stream
```

Harness controls are root options, so place them before `ask` or `chat` (the existing
`ask --max-attempts` spelling remains supported). With the TUI, `--workspace` selects the
session data root; with `ask`, it continues to select an explicit execution workspace:

```powershell
xiliumini --workspace ".xiliumini/workspaces/my-task" --max-attempts 5 `
  --approval-mode inline --checkpoint-mode strict --trace-mode on ask "Implement it"
xiliumini --workspace ".xiliumini" --max-attempts 5 `
  --approval-mode inline --checkpoint-mode strict --trace-mode on chat
```

`--approval-mode` accepts `inline`, `auto`, or `deny`; `--checkpoint-mode` accepts
`light`, `strict`, or `off`; the CLI `--trace-mode` accepts `on` or `off`, where `on`
maps to the Runtime `full` mode. Runtime/Settings still support `summary`. An explicit
workspace must resolve below `<XILIUMINI_DATA_DIR>/workspaces`, cannot be that root,
and cannot traverse or use link components. `--workspace` and `--resume` are mutually
exclusive. If checkpoint/trace flags are omitted, existing Settings/environment values
remain in effect; an explicitly supplied root flag overrides them.

Normal output identifies every completed graph stage:

```text
  ↳ planner: call_code_agent
  ↳ code_agent: bash ok
📋 Planner (attempt 1/3): ...
✅ Verifier: ...
📝 Final: ...
```

Failed verification uses `❌ Verifier` and retries Planner while attempts remain.
`--no-stream` hides Planner, Verifier, and specialist-progress events, leaving only
the Final line.

Each session receives an isolated workspace at:

```text
<XILIUMINI_DATA_DIR>/workspaces/<session-id>/
```

The Supervisor has `todo_write`, `call_search_agent`, and `call_code_agent` tools.
searchAgent has only `web_search` (Tavily). Without a key, it returns
`{"ok": false, "error": "missing TAVILY_API_KEY", "query": "..."}` without a network request.
Configure the key in the environment or local `.env`; it is not stored in graph state.

codeAgent uses `file_read`, `file_write`, `file_edit`, `grep`, `bash`, `todo_update`,
`notepad_read`, and `notepad_append`. Todo progress is persisted in
`TODO.md` at the session workspace root; durable notes are in `NOTEPAD.md`. A new user
task resets todo progress but preserves workspace files and notes. Retries preserve the
current task's progress.

## Memory system

Runtime assembles Memory for every graph node; Agents consume the assembled object and
must not write Memory directly. The object has three bounded layers:

- Rules: fixed workspace-safety rules plus project-wide user preferences.
- Working Memory: current node/task/session, plan, todos, acceptance criteria, research,
  sources, the latest six Agent handoffs, Agent/Verifier summaries, last error, and attempts.
- History Summary: bounded `HISTORY_SUMMARY.md`, `NOTEPAD.md`, the previous context summary,
  and the latest three compression events.

Rule precedence is fixed safety rules, then the current explicit task instruction, then
saved user preferences. Planner receives `preference_write` and records a preference only
when the user explicitly asks to remember or update a long-term default. An explicit
“forget” request removes it. Ordinary instructions and one-task overrides are never saved.
Preferences are stored once per project data directory at
`<XILIUMINI_DATA_DIR>/USER_PREFERENCES.md`, so later tasks and different sessions share them.
Secret-like values are rejected.

Planner estimates message tokens before every model call. At 80% of the configured context
window by default, it keeps the fixed rules, complete current task, and newest 8,000 tokens,
summarizes older messages with the unbound model, then atomically writes
`HISTORY_SUMMARY.md`. Model, empty-summary, or write failures leave the original messages
and state unchanged. If the non-compressible rules and current task already exceed the
budget, Runtime returns a redacted `memory_error` before the normal Planner request.

Canonical per-session files are:

```text
<XILIUMINI_DATA_DIR>/workspaces/<session-id>/
├── TODO.md
├── NOTEPAD.md
└── HISTORY_SUMMARY.md
```

On first access, valid legacy `.xiliumini/todos.json` and `.xiliumini/notepad.md` files are
copied to their canonical forms without deleting the originals. Canonical files win when
both exist. Invalid UTF-8 or malformed data is preserved unchanged and reported safely.

Tool paths
must be relative to that session workspace; absolute paths, parent traversal, UNC
paths, drive-qualified paths, and symlink escapes are rejected.

The Bash tool accepts `python <workspace-relative.py>`, `python -m pytest ...`,
`python -m ruff check ...`, `python -m ruff format --check ...`, `python -m pyright ...`,
`python -m compileall ...`, and `python -m pip check`. Its optional `cwd` is a
workspace-relative directory, which allows checks to run from a nested project without
using `cd` or a shell wrapper. Recognized installation commands (`pip install`,
`python -m pip install`, `uv add/sync/pip install`, `npm/pnpm install`, `yarn install/add`),
downloads (`curl`, `wget`), and development servers (`uvicorn`, `python -m http.server`)
require approval. BashTool's `inline` default invokes an application-supplied
`approval_handler(ApprovalRequest) -> ApprovalDecision`; without a handler it rejects
the command. `auto` permits recognized risky commands and `deny` rejects them.
The TUI supplies an approval modal in `inline` mode with the tool, risk reason, workspace,
and full command. Press `Y` or `Enter` to approve and `N` or `Escape` to deny; closing the app
denies pending requests and releases blocked workers. The non-interactive `ask` command keeps
its `typer.confirm` prompt. `auto` and `deny` never prompt.
Handlers are run-local ContextVar data and are not written to GraphState, checkpoints, or
traces. These modes are CLI/programmatic options rather than Settings environment variables.
Risky results carry
`requires_approval=True`, including refusals. Approval preserves argv execution,
workspace-relative cwd, timeout and output bounds; shell operators remain forbidden.
Rejections, timeouts, truncated output, and non-zero exits have distinct CLI
progress messages.

The tool uses argv without a shell and bounds runtime and output. This is not an OS
sandbox: generated Python code still runs with the same user permissions as the
xiliumini process. Only run tasks and generated code you trust.

Inspect the command contracts:

```powershell
xiliumini --help
xiliumini --version
xiliumini chat --help
xiliumini sessions
```

The TUI and programmatic session API persist a bounded transcript in
`.xiliumini/session/session.json` and a readable mirror in `SESSION_SUMMARY.md`. Each input
first runs through an intent graph: ordinary conversation uses the no-tool chat responder,
while workspace work continues through the full Planner/Verifier workflow. CLI transcript
turns use this same `stream_session_events` path.
The legacy analysis Agent is no longer registered in the Supervisor path. Each specialist
has an independent bounded ReAct conversation (search: 4 loops, code: 10 loops).
Planner has an 8-loop budget per round. Only summaries, research and tool evidence enter
graph state; full specialist messages stay in their function results.

Shared state, nodes, and graph routing live in `src/xiliumini/graph/state.py`,
`nodes.py`, and `workflow.py`. `src/xiliumini/core/agent.py` translates LangGraph
`updates` and `custom` streams into stable CLI events. Modules under
`src/xiliumini/agents/` are specialist sub-Agents invoked only for focused work.

The Typer entry point is `src/xiliumini/cli/__init__.py` (there is no separate
`cli/app.py`). Programmatic integrations can import `stream_agent_events` or the multi-turn
`stream_session_events` from `xiliumini.core` or `xiliumini.core.agent`:

```python
from pathlib import Path

from xiliumini.core import stream_agent_events, stream_session_events

for item in stream_agent_events(
    "Implement the requested change",
    workspace=Path(".xiliumini/workspaces/my-task"),
    checkpoint_mode="light",
    trace_mode="on",
):
    print(item)  # {"type": "custom_event" | "graph_event", "event": {...}}

for item in stream_session_events(
    "继续完成刚才的任务",
    session_workspace=Path(".xiliumini"),
    checkpoint_mode="light",
    trace_mode="on",
):
    print(item)
```

This compatibility iterator delegates to Runtime and adapts each stable RuntimeEvent once;
it does not create a second Checkpoint or Trace recorder. Pass the same path as
`workspace` and `resume_workspace` to resume; conflicting paths produce a redacted
workspace error.

`stream_session_events` defaults `session_workspace` to `.xiliumini`. It stores at most ten
recent messages, limits each message to 4,000 characters, and supplies at most 7,000
characters of routing context. That context includes up to 30 recently modified relative
paths from `.xiliumini/workspaces/<session-id>/`; file contents and session/checkpoint/trace
metadata are not included.

## Checkpoints and execution traces

Each run owns independent recorders, including interleaved sessions on one Runtime.
Checkpoint defaults to `light`; Trace defaults to `full`. Omitted Settings values use
these defaults; invalid or blank environment modes raise a configuration error.
Only internal mode normalizers and direct recorder/runtime contexts fall back to defaults
for invalid modes. Blank optional `XILIUMINI_TRACE_ID` is treated as unset and generates an ID.

| Setting | Mode | Behavior |
| --- | --- | --- |
| `XILIUMINI_CHECKPOINT_MODE` | `light` | Save state summary, recovery guide and Git snapshot at start, after each node and at termination |
| | `strict` | Also save `state.json` and append each custom/node event to checkpoint `events.jsonl` |
| | `off` | No checkpoint I/O or Git; resume is refused |
| `XILIUMINI_TRACE_MODE` | `full` | Record all received custom events and node updates plus lifecycle events |
| | `summary` | Persist lifecycle, nodes, failures, approvals, handoffs and checkpoints; count all received canonical tool events |
| | `off` | No trace I/O |

```text
<XILIUMINI_DATA_DIR>/workspaces/<session-id>/
├── TODO.md / NOTEPAD.md / HISTORY_SUMMARY.md
└── .xiliumini/
    ├── checkpoints/
    │   ├── checkpoint.json       # latest metadata, state_summary and SHA-256 manifest
    │   ├── RECOVERY.md           # task, status, manifest, commit and resume command
    │   ├── repo.git/             # independent local Git history
    │   ├── state.json           # strict only
    │   └── events.jsonl         # strict only
    └── traces/<trace-id>/
        ├── events.jsonl         # sequenced UTC events with redacted payloads
        ├── trace.json           # run statistics and bounded timeline
        └── timeline.md          # human-readable summary
```

Resume using the same data-directory configuration as the saved run:

```powershell
xiliumini --resume ".xiliumini/workspaces/<session-id>"
```

`--resume` is a root option and cannot accompany a subcommand or `--workspace`. The saved
task is reused; root `--max-attempts` defaults to three and can override the recovery limit.
Programmatic
`Runtime.resume(workspace, task=..., max_attempts=...)` allows overrides. Recovery selects
Verifier after Planner, Planner after retryable failed verification, and Final after
successful/exhausted verification or a Final checkpoint. It does not resume inside a tool.

Recovery restores checkpoint file bytes and **deletes ordinary files added since that
checkpoint**, including ignored files. Back up changes you want to keep before resuming.
Root `.xiliumini/` and `.git/` (including case variants) are preserved, as are unrelated empty directories; a directory
occupying a target file path is removed to restore that file. Empty directories are not
stored by Git, so a checkpoint-era empty directory removed before recovery is not recreated.
Schema, all required GraphState fields/types, paths and commit are validated before changes;
a pre-restore commit enables
best-effort rollback on failure. This is a single-recorder protocol without concurrent-writer
locking or atomic recovery from hard process termination. Retained recovery backups/internal
Git history support manual inspection when rollback also fails.

Trace statistics include UTC `started_at`/`ended_at`, monotonic `duration_ms`, `status`,
`node_visits`, `tool_calls`, `failed_tool_calls`, `approval_count`, `checkpoint_count` and
`handoff_count`. Failure and approval counters require `ok is False` and
`requires_approval is True`; approval_count counts risky tool results, not approval prompts.
`timeline_head` keeps the first 20 events, `timeline_tail` the following/latest 80 without
duplicates, and `timeline_omitted` counts the gap above 100 persisted events.

Statuses are `completed`, `failed` and `interrupted`. Closing a stream, GeneratorExit or
Ctrl+C records interrupted; SystemExit records failed. Finalization attempts Trace.end
once and preserves the original exception. The child graph stream is explicitly closed
and its background nodes stopped before the terminal snapshot and trace end.
Checkpoint manifests hash immutable committed blobs, so background file changes cannot
produce a manifest that disagrees with its saved commit. Enabled persistence errors stop execution with
stable `checkpoint_error` or `trace_error`, rather than silently disabling recording.
The CLI returns 1 for these errors/interruption and 2 for configuration/option errors,
without a traceback. Damaged or out-of-bounds recovery fails before file mutation.

State/event files redact sensitive keys and SecretStr, replace the workspace absolute
path with `<workspace>`, and bound text. RECOVERY.md intentionally includes a quoted
absolute resume command. Git snapshots preserve ordinary file bytes, including potentially
sensitive files; metadata redaction does not redact snapshot contents. Keep the workspace
and its internal Git history private.

## Development

Tests use fake models and never contact a real provider:

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

After the automated suite passes, `doctor` and one `ask` command can be used as a
manual API smoke test with the local `.env` configuration.

See [PRD.md](PRD.md) and [SPEC.md](SPEC.md) for the product contract and staged roadmap.
