# xiliumini

`xiliumini` is a minimal Python Agent CLI built around an OpenAI-compatible model,
LangGraph, and a small set of bounded tools. Task 0 provides the installable CLI,
validated configuration, calculator and timezone tools, isolated analysis delegation,
and a checkpointed ReAct loop. Task 0.1 adds a persistent workspace per session and
four workspace-confined file tools.

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
XILIUMINI_MAX_STEPS=8
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

Run the Task 0 ReAct loop:

```powershell
xiliumini ask "What time is it in Asia/Shanghai?"
xiliumini ask "Calculate (17 + 5) * 3" --no-stream
```

Each session receives an isolated workspace at:

```text
<XILIUMINI_DATA_DIR>/workspaces/<session-id>/
```

The main Actor can use `file_read`, `file_write`, `file_edit`, and `grep`. Tool paths
must be relative to that session workspace; absolute paths, parent traversal, UNC
paths, drive-qualified paths, and symlink escapes are rejected.

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

The primary ReAct loop lives in `src/xiliumini/core/agent.py`. Modules under
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
