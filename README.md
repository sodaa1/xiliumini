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
# Optional: required only for web research
TAVILY_API_KEY=your-tavily-key
```

`XILIUMINI_BASE_URL` is optional for the default OpenAI endpoint. The API key is
represented as a Pydantic secret and is never written to CLI errors or tool output.

## Commands

Enter the project directory and start an interactive Agent conversation. No `uv run`
prefix or subcommand is needed:

```powershell
cd E:\Learning\xiliumini
xiliumini
```

Inside the conversation, type natural-language messages directly. Use `/help`,
`/status`, `/new`, or `/exit` for local controls.

Check Python, configuration, the data directory, and model construction:

```powershell
xiliumini doctor
```

Run the Supervisor workflow. `--max-attempts` limits Supervisor rounds and defaults to 3:

```powershell
xiliumini ask "帮我实现一个 Conway's Game of Life，要求 TDD：先写测试，再写实现，最后跑 demo" --max-attempts 3
xiliumini ask "Calculate (17 + 5) * 3" --no-stream
```

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

The Bash tool accepts only `python <workspace-relative.py>`, `python -m pytest ...`,
`python -m ruff check ...`, `python -m ruff format --check ...`, `python -m pyright ...`,
`python -m compileall ...`, and `python -m pip check`. Its optional `cwd` is a
workspace-relative directory, which allows checks to run from a nested project without
using `cd` or a shell wrapper. It does not install packages or accept arbitrary shell
commands. Rejections, timeouts, truncated output, and non-zero exits have distinct CLI
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

Persistent chat sessions and traces remain separate roadmap work; Task3 here implements
Runtime-managed working/history memory rather than full chat transcript persistence.
The legacy analysis Agent is no longer registered in the Supervisor path. Each specialist
has an independent bounded ReAct conversation (search: 4 loops, code: 10 loops).
Planner has an 8-loop budget per round. Only summaries, research and tool evidence enter
graph state; full specialist messages stay in their function results.

Shared state, nodes, and graph routing live in `src/xiliumini/graph/state.py`,
`nodes.py`, and `workflow.py`. `src/xiliumini/core/agent.py` translates LangGraph
`updates` and `custom` streams into stable CLI events. Modules under
`src/xiliumini/agents/` are specialist sub-Agents invoked only for focused work.

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
