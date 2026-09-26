# xiliumini

`xiliumini` is a minimal Python Agent CLI built around an OpenAI-compatible model,
LangGraph, and a small set of bounded tools. Its main execution path is a
Planner → Actor → Verifier workflow with bounded retries and a deterministic Final
node. Each session has a persistent workspace with confined file tools and a restricted
Python/pytest command tool.

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
XILIUMINI_ANALYSIS_TIMEOUT_SECONDS=30
XILIUMINI_ANALYSIS_MAX_CHARS=8000
XILIUMINI_DATA_DIR=.xiliumini
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

Run the Plan-Act-Verify workflow. `--max-attempts` defaults to 3:

```powershell
xiliumini ask "帮我实现一个 Conway's Game of Life，要求 TDD：先写测试，再写实现，最后跑 demo" --max-attempts 3
xiliumini ask "Calculate (17 + 5) * 3" --no-stream
```

Normal output identifies every completed graph stage:

```text
📋 Planner: ...
🔧 Actor (attempt 1/3): ...
✅ Verifier: ...
📝 Final: ...
```

Failed verification uses `❌ Verifier` and retries Actor while attempts remain.
`--no-stream` hides Planner, Actor, Verifier, and action-progress events, leaving only
the Final line.

Each session receives an isolated workspace at:

```text
<XILIUMINI_DATA_DIR>/workspaces/<session-id>/
```

The main Actor can use `file_read`, `file_write`, `file_edit`, `grep`, and `command`.
Tool paths
must be relative to that session workspace; absolute paths, parent traversal, UNC
paths, drive-qualified paths, and symlink escapes are rejected.

The command tool accepts only `python <workspace-relative.py>` and
`python -m pytest ...`, uses argv without a shell, fixes the working directory to the
session workspace, and bounds runtime and output. This is not an OS sandbox: generated
Python code still runs with the same user permissions as the xiliumini process. Only
run tasks and generated code you trust.

Inspect the command contracts:

```powershell
xiliumini --help
xiliumini --version
xiliumini chat --help
xiliumini sessions
```

Persistent chat sessions and traces are intentionally reserved for Tasks 3 and 4.
The analysis Agent has its own context, binds no tools, and cannot recursively delegate.
The required `bash_tool.py` module is present but shell execution is disabled and the
tool is not registered in the MVP.

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
