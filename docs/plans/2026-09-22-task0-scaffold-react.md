# Task 0 Scaffold and Minimal ReAct Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the installable, testable Task 0 project defined by `PRD.md` and `SPEC.md`, including the Typer CLI, validated configuration, two Agent roles, prompts, the specified tools, an OpenAI-compatible provider boundary, and a minimal LangGraph ReAct loop.

**Architecture:** Use `E:/Learning/xiliumini` as a new, independent Git repository. The Typer package calls a runtime, the runtime calls a LangGraph workflow, the workflow binds provider models to tools, and analysis delegation uses a separate no-tools Agent so it cannot recurse.

**Tech Stack:** Python 3.12+, uv, Typer, Rich, LangGraph, LangChain, langchain-openai, Pydantic Settings, python-dotenv, pytest, pytest-asyncio, Ruff.

**Spec:** `SPEC.md`

## Global Constraints

- Implement only Task 0; do not add session persistence or trace persistence beyond the directory/package placeholders required by the specified tree.
- API keys must never appear in `repr`, CLI errors, model errors, test output, or tool output.
- Tests must not access the real network; inject fake models and fake analysis functions.
- The registered main-Agent tools are `calculator`, `current_time`, and `delegate_analysis`; `bash_tool.py` exists to satisfy the specified directory but shell execution remains disabled because the PRD lists Shell as a future target and the tool contract is unspecified.
- The analysis Agent binds no tools and cannot call `delegate_analysis` recursively.
- The ReAct loop is `START -> agent -> tools -> agent -> END` and stops with an explicit message when `XILIUMINI_MAX_STEPS` is reached.
- Keep the Task 0 delivery as one commit named exactly `chore: scaffold xiliumini typer cli`.
- Keep `.env` ignored and stage only Task 0 project files after reviewing `git diff`.

## Review Focus

- Missing or blank `XILIUMINI_API_KEY`/`XILIUMINI_MODEL` must produce exit code 2 without a traceback or secret leakage.
- Calculator names, attributes, calls, excessive expressions, unsafe powers, division by zero, and oversized results must return readable tool errors rather than execute code.
- Invalid IANA timezones must return a readable tool error; deterministic tests inject the clock.
- Delegated analysis must time out, truncate oversized output, and never receive the main Agent's tools.
- Repeated tool calls must terminate at `max_steps` with an explicit final response instead of looping forever.

---

### Task 0: Scaffold the complete Task 0 vertical slice

**Files:**

- Create/replace: `pyproject.toml`, `.env.example`, `.gitignore`, `README.md`
- Create: `src/xiliumini/__init__.py`, `src/xiliumini/config.py`, `src/xiliumini/errors.py`, `src/xiliumini/events.py`, `src/xiliumini/runtime.py`
- Create: `src/xiliumini/cli/__init__.py`, `src/xiliumini/cli/tui.py`
- Create: `src/xiliumini/prompts/__init__.py`, `src/xiliumini/prompts/main.py`, `src/xiliumini/prompts/analysis.py`
- Create: `src/xiliumini/core/__init__.py`
- Create: `src/xiliumini/agents/__init__.py`, `src/xiliumini/agents/main.py`, `src/xiliumini/agents/analysis.py`
- Create: `src/xiliumini/graph/__init__.py`, `src/xiliumini/graph/state.py`, `src/xiliumini/graph/workflow.py`
- Create: `src/xiliumini/providers/__init__.py`, `src/xiliumini/providers/openai_compatible.py`
- Create: `src/xiliumini/tools/__init__.py`, `src/xiliumini/tools/calculator.py`, `src/xiliumini/tools/current_time.py`, `src/xiliumini/tools/delegate_analysis.py`, `src/xiliumini/tools/bash_tool.py`
- Create: `src/xiliumini/storage/__init__.py`, `src/xiliumini/storage/sessions.py`, `src/xiliumini/storage/traces.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/test_config.py`, `tests/unit/test_calculator.py`, `tests/unit/test_current_time.py`, `tests/unit/test_delegate_analysis.py`, `tests/unit/test_provider.py`, `tests/unit/test_workflow.py`
- Create: `tests/integration/test_cli.py`, `tests/integration/test_runtime.py`

**Interfaces:**

- Produces: `Settings.from_env(env)`, `load_settings()`, and a sanitized exception hierarchy.
- Produces: Typer `app` with `doctor`, `ask`, `chat`, `sessions`, and `--version` contracts.
- Produces: `create_chat_model(settings) -> BaseChatModel` and provider error classification.
- Produces: `calculator(expression) -> str`, `current_time(timezone, clock=None) -> str`, and `make_delegate_analysis(analyze, timeout_seconds, max_chars)`.
- Produces: `AnalysisAgent.analyze(question) -> str`, `build_main_agent(model, tools, max_steps)`, `build_workflow(model, tools, max_steps)`, and `Runtime.astream(question, session_id) -> AsyncIterator[RuntimeEvent]`.

- [ ] **Step 1: Normalize the project metadata and exact package tree**

Set the console entry point to the CLI package and keep dependency bounds compatible with Python 3.12:

```toml
[project.scripts]
xiliumini = "xiliumini.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py312"
line-length = 100
```

Create every package in the Spec tree with `__init__.py`. Keep `storage/sessions.py`, `storage/traces.py`, and `cli/tui.py` importable but free of Task 1-4 behavior. Export `__version__ = "0.1.0"` from `xiliumini.__init__`.

- [ ] **Step 2: Write failing configuration and CLI contract tests**

Add tests that pin required settings, numeric bounds, secret redaction, help/version output, required commands, and doctor exit code 2:

```python
def test_settings_requires_api_key_and_model() -> None:
    with pytest.raises(ConfigError, match="XILIUMINI_API_KEY"):
        Settings.from_env({})


def test_secret_is_not_rendered() -> None:
    settings = Settings.from_env({"XILIUMINI_API_KEY": "never-print-me", "XILIUMINI_MODEL": "fake"})
    assert "never-print-me" not in repr(settings)


def test_doctor_missing_config_is_safe(monkeypatch) -> None:
    monkeypatch.delenv("XILIUMINI_API_KEY", raising=False)
    monkeypatch.delenv("XILIUMINI_MODEL", raising=False)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 2
    assert "Traceback" not in result.stdout
```

- [ ] **Step 3: Run the configuration and CLI tests to prove the red state**

Run:

```powershell
uv run pytest tests/unit/test_config.py tests/integration/test_cli.py -q
```

Expected: FAIL because the package and CLI package do not yet implement the tested contracts.

- [ ] **Step 4: Implement settings, safe exceptions, and Typer command boundaries**

Implement `Settings` with `SecretStr`, `env_prefix="XILIUMINI_"`, `.env` loading, `max_steps >= 1`, positive timeouts, `data_dir`, and readable field-name-only validation errors. Implement exception types for configuration, authentication, rate limits, timeouts, tools, and providers. The command skeletons must be real boundaries: `doctor` validates configuration/data-directory access; `ask` calls the runtime; `chat` accepts `--session` but reports Task 3 persistence as unavailable; `sessions` reports that no sessions exist; `--version` exits 0.

- [ ] **Step 5: Re-run the focused configuration and CLI tests**

Run:

```powershell
uv run pytest tests/unit/test_config.py tests/integration/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Write failing safe-tool tests**

Use table-driven tests for supported arithmetic and rejected syntax, inject a fixed UTC clock for timezone conversion, and assert that the Bash placeholder is never registered:

```python
@pytest.mark.parametrize(("expression", "expected"), [("1 + 2 * 3", "7"), ("2 ** 8", "256")])
def test_calculator(expression: str, expected: str) -> None:
    assert calculator.invoke({"expression": expression}) == expected


@pytest.mark.parametrize("expression", ["open('x')", "a.b", "name", "2 ** 10000"])
def test_calculator_rejects_unsafe_input(expression: str) -> None:
    assert "error" in calculator.invoke({"expression": expression}).lower()


def test_current_time_uses_injected_clock() -> None:
    now = lambda: datetime(2026, 1, 1, tzinfo=UTC)
    assert "2026-01-01" in current_time_value("UTC", clock=now)
```

- [ ] **Step 7: Run the safe-tool tests to prove the red state**

Run:

```powershell
uv run pytest tests/unit/test_calculator.py tests/unit/test_current_time.py -q
```

Expected: FAIL because tool modules do not yet exist.

- [ ] **Step 8: Implement calculator, current-time, and disabled Bash modules**

Implement calculator with `ast.parse(mode="eval")`, explicit numeric/operator node dispatch, maximum expression length 256, maximum absolute exponent 100, and maximum absolute result `1e100`; never use `eval`. Implement current time with `ZoneInfo`, an injected aware clock, and readable invalid-timezone errors. Define `bash_tool` as an unregistered function that always returns `"Shell tool is not enabled in the MVP."` so the required file is explicit without broadening the PRD.

- [ ] **Step 9: Re-run the safe-tool tests**

Run:

```powershell
uv run pytest tests/unit/test_calculator.py tests/unit/test_current_time.py -q
```

Expected: PASS.

- [ ] **Step 10: Write failing provider and isolated-analysis tests**

Patch `ChatOpenAI` so provider tests inspect construction without network access. Test that the analysis model binds no tools, delegation enforces timeout and output length, and provider exceptions expose codes without secrets:

```python
def test_provider_maps_settings(monkeypatch, settings) -> None:
    captured = {}
    monkeypatch.setattr(
        provider, "ChatOpenAI", lambda **kwargs: captured.update(kwargs) or FakeModel()
    )
    create_chat_model(settings)
    assert captured["model"] == settings.model
    assert captured["temperature"] == settings.temperature


@pytest.mark.asyncio
async def test_delegate_truncates_output() -> None:
    tool = make_delegate_analysis(lambda _: async_value("abcdef"), 1.0, 4)
    assert await tool.ainvoke({"question": "q"}) == "abcd"
```

- [ ] **Step 11: Run provider and delegation tests to prove the red state**

Run:

```powershell
uv run pytest tests/unit/test_provider.py tests/unit/test_delegate_analysis.py -q
```

Expected: FAIL because provider, prompts, Agents, and delegate tool are not implemented.

- [ ] **Step 12: Implement prompts, provider, and the two Agent roles**

Define concise system prompts: the main Agent may use registered tools and must not invent tool results; the analysis Agent performs text-only analysis and cannot invoke tools. `create_chat_model` constructs `ChatOpenAI(api_key, model, base_url, temperature, timeout)` from validated settings and maps authentication/rate-limit/timeout errors to sanitized application exceptions. `AnalysisAgent` owns a model with no bound tools; `build_main_agent` receives its tool list explicitly.

- [ ] **Step 13: Implement bounded analysis delegation and rerun its tests**

Use `asyncio.timeout(timeout_seconds)` around `AnalysisAgent.analyze`, convert timeout to a readable tool result, reject blank questions, and truncate by Unicode characters to `max_chars`. Do not pass messages, tools, or runtime state from the main Agent into the analysis Agent.

Run:

```powershell
uv run pytest tests/unit/test_provider.py tests/unit/test_delegate_analysis.py -q
```

Expected: PASS.

- [ ] **Step 14: Write failing LangGraph ReAct and runtime tests**

Use a deterministic fake chat model to cover direct answers, one tool call followed by a final answer, tool failure, streaming token order, and a model that repeatedly requests tools until `max_steps`:

```python
@pytest.mark.asyncio
async def test_workflow_stops_at_max_steps(repeating_tool_model) -> None:
    graph = build_workflow(repeating_tool_model, [calculator], max_steps=2)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="loop")], "session_id": "s", "step_count": 0}
    )
    assert result["step_count"] == 2
    assert "maximum" in result["messages"][-1].content.lower()


@pytest.mark.asyncio
async def test_runtime_emits_tokens_then_final(fake_direct_model) -> None:
    events = [event async for event in Runtime(fake_direct_model).astream("hello", "s")]
    assert isinstance(events[-1], FinalEvent)
    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "answer"
```

- [ ] **Step 15: Run graph and runtime tests to prove the red state**

Run:

```powershell
uv run pytest tests/unit/test_workflow.py tests/integration/test_runtime.py -q
```

Expected: FAIL because state, workflow, events, and runtime are not implemented.

- [ ] **Step 16: Implement typed events, graph state, and the minimal ReAct workflow**

Define `AgentState` with `messages`, `session_id`, and `step_count`. Build nodes for model invocation and `ToolNode`, conditional routing based on `AIMessage.tool_calls`, and a maximum-step terminal message. Use LangGraph's in-memory checkpointer interface only for Task 0; persistent resume remains Task 3. Emit `TokenEvent`, `ToolStartedEvent`, `ToolFinishedEvent`, `FinalEvent`, and `ErrorEvent` from the runtime without storing content.

- [ ] **Step 17: Connect the real Task 0 runtime to `ask` and rerun graph/runtime tests**

`ask` loads settings, constructs the provider and both Agents, registers only calculator/current-time/delegate tools, streams Rich output unless `--no-stream`, maps configuration errors to exit 2, and maps runtime/provider errors to exit 1. `chat` may run an in-memory loop but must not claim persistence support in Task 0.

Run:

```powershell
uv run pytest tests/unit/test_workflow.py tests/integration/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 18: Complete the minimal README and environment template**

Document Python/uv prerequisites, `uv sync`, copying `.env.example` to `.env`, required variables, `uv run xiliumini doctor`, `ask`, `chat`, `sessions`, version/help checks, and that tests use fakes. Include every optional variable from the PRD and the analysis bounds used by Task 0; do not include real credentials.

- [ ] **Step 19: Run Task 0 acceptance verification**

Run all checks from `E:/Learning/xiliumini`:

```powershell
uv sync
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run xiliumini --help
uv run xiliumini --version
uv run xiliumini doctor
```

Expected: dependency sync succeeds; tests, Ruff checks, help, and version pass; `doctor` returns exit 2 only when the required local secrets are intentionally absent and does so without traceback.

- [ ] **Step 20: Review the diff and create the single Task 0 commit**

Inspect `git status --short`, `git diff`, and `git diff --cached`. Confirm `.env` is ignored, then stage only Task 0 project files. Commit exactly:

```powershell
git commit -m "chore: scaffold xiliumini typer cli"
```

Then rerun `git status --short --branch` and report the commit hash plus verification results.
