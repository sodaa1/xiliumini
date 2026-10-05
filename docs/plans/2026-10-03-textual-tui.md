# xiliumini Textual TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-screen Textual client for persistent xiliumini sessions, including structured runtime events, synchronous Bash approval, and the approved animated Rich logo.

**Architecture:** Preserve the existing synchronous Runtime and run one session turn at a time in a Textual thread worker. Extend `ProgressEvent` with backward-compatible structured metadata, post custom Textual messages to the UI thread, and bridge BashTool's synchronous approval callback through a thread-safe `ApprovalGate` and modal screen.

**Tech Stack:** Python 3.12, Textual 8.x, Rich, Typer, pytest/pytest-asyncio, Ruff, Pyright

**Spec:** `docs/specs/2026-10-03-textual-tui-design.md`

## Global Constraints

- Work directly on `main`; do not create a Git worktree.
- Do not create per-task commits or a separate documentation commit; the user requested direct implementation without separate commits.
- Keep the public application class name `MokioClawTuiApp`; show `xiliumini` as the user-visible brand.
- Keep `xiliumini ask`, `doctor`, resume, RuntimeEvent consumers, and the `graph_event` / `custom_event` wrapper compatible.
- Only one session turn may execute at a time.
- UI widgets may only be mutated on the Textual UI thread; the worker communicates with `post_message`.
- Pending approvals must resolve as denied when the app closes.
- Use `textual>=8.2,<9`, matching the current stable 8.x API and the project's Python 3.12 floor.
- Follow TDD: add a focused failing test, observe the failure, implement the minimum behavior, then rerun the focused test.

## Review Focus

1. A custom runtime payload with a missing/non-string `type` must remain a normal `progress` event rather than crashing; pin this in Task 1.
2. A terminal narrower than the stage subtitle must render a bounded static logo without negative lengths or markup leakage; pin this in Task 3.
3. Approval resolution racing with Escape, a button press, or app shutdown must use the first decision and release the worker exactly once; pin this in Task 2 and Task 4.
4. A stream that raises before emitting `FinalEvent` or `ErrorEvent` must restore the input and leave the app usable; pin this in Task 4.
5. A second submit while a turn is active must not start another generator or mutate the same session concurrently; pin this in Task 4.

---

## File Map

- `src/xiliumini/events.py`: backward-compatible structured progress event fields.
- `src/xiliumini/core/agent.py`: custom payload translation and emitted checkpoint events.
- `src/xiliumini/cli/tui/__init__.py`: lazy public `run_tui` entry point.
- `src/xiliumini/cli/tui/approval.py`: approval synchronization and modal UI.
- `src/xiliumini/cli/tui/logo.py`: deterministic Rich logo frames.
- `src/xiliumini/cli/tui/app.py`: Textual application, messages, worker, event rendering, and lifecycle.
- `src/xiliumini/cli/tui/styles.tcss`: all TUI layout and presentation rules.
- `src/xiliumini/cli/__init__.py`: bare command and `chat` integration.
- `pyproject.toml`, `uv.lock`: Textual runtime dependency.
- `tests/unit/test_tui_approval.py`: approval gate and modal behavior.
- `tests/unit/test_tui_logo.py`: logo render and fallback behavior.
- `tests/unit/test_tui_app.py`: event mapping, worker, session, and shutdown behavior.
- `tests/unit/test_agent.py`: runtime event compatibility and checkpoint publishing.
- `tests/integration/test_cli.py`: CLI routing regression coverage.
- `README.md`, `SPEC.md`, `项目进程.md`: user-facing command and task status documentation.

### Task 1: Preserve structured custom events and publish checkpoints

**Files:**
- Modify: `src/xiliumini/events.py`
- Modify: `src/xiliumini/core/agent.py`
- Modify: `tests/unit/test_agent.py`

**Interfaces:**
- Produces: `ProgressEvent(stage: str, message: str, event_type: str = "progress", details: dict[str, Any] = {})`.
- Produces: `stream_agent` events whose `event_type` is the original custom payload type and whose `details` is an independent payload copy.
- Produces: `checkpoint_saved` ProgressEvents after successful visible checkpoint saves.

- [ ] **Step 1: Add failing structured-event tests**

Add tests proving that `_chunk_events` preserves `tool_call` fields, copies the payload, and falls back to `event_type == "progress"` for missing/non-string types while keeping old two-argument `ProgressEvent` construction valid.

- [ ] **Step 2: Run the focused tests and observe failure**

Run: `uv run pytest tests/unit/test_agent.py -k "structured_progress or malformed_custom_type or progress_event_compatibility" -v`  
Expected: FAIL because `ProgressEvent` has no `event_type` / `details` fields.

- [ ] **Step 3: Extend ProgressEvent and custom event translation**

Add the defaulted fields in `events.py`. Update `_chunk_events` in `core/agent.py` to validate `stage` and `message`, normalize the payload type, and attach `deepcopy(payload)` as details.

- [ ] **Step 4: Run the structured-event tests**

Run: `uv run pytest tests/unit/test_agent.py -k "structured_progress or malformed_custom_type or progress_event_compatibility" -v`  
Expected: PASS.

- [ ] **Step 5: Add failing checkpoint-stream tests**

Add tests for checkpoint mode `light` and `strict` proving a successful save emits a `ProgressEvent` with `event_type == "checkpoint_saved"`, correct status/latest node details, no duplicate trace recording, and unchanged generator-close cleanup.

- [ ] **Step 6: Run checkpoint tests and observe failure**

Run: `uv run pytest tests/unit/test_agent.py -k "checkpoint_event or close" -v`  
Expected: the new checkpoint-event assertions fail because saved checkpoints are not yielded.

- [ ] **Step 7: Emit checkpoint ProgressEvents without weakening cleanup**

Change the nested save helper to return a ProgressEvent when `CheckpointManager.save` returns a payload. Yield started/running saves immediately; on normal exhaustion emit the terminal save after closing the LangGraph iterator. On exception or `GeneratorExit`, persist cleanup without yielding from the closing generator.

- [ ] **Step 8: Verify the full agent test module**

Run: `uv run pytest tests/unit/test_agent.py -v`  
Expected: PASS.

### Task 2: Add Textual dependency and approval bridge

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Delete: `src/xiliumini/cli/tui.py`
- Create: `src/xiliumini/cli/tui/__init__.py`
- Create: `src/xiliumini/cli/tui/approval.py`
- Create: `src/xiliumini/cli/tui/styles.tcss`
- Create: `tests/unit/test_tui_approval.py`

**Interfaces:**
- Produces: `ApprovalGate.wait(timeout: float | None = None) -> bool`.
- Produces: `ApprovalGate.resolve(*, approved: bool) -> bool`, returning whether this call installed the first decision.
- Produces: `ApprovalGate.resolved: bool`.
- Produces: `ApprovalModal(request: ApprovalRequest, workspace: Path)` as `ModalScreen[bool]`.

- [ ] **Step 1: Add and lock Textual**

Add `textual>=8.2,<9` to project dependencies and run `uv lock`. Verify `uv.lock` resolves Textual 8.x and its dependencies without changing the Python floor.

- [ ] **Step 2: Replace the TUI placeholder with a package**

Remove the one-line `src/xiliumini/cli/tui.py`, create the package directory and an initially minimal `__init__.py`, and add the approved base palette/layout selectors to `styles.tcss`.

- [ ] **Step 3: Write failing ApprovalGate tests**

Test blocking until resolve, approve/deny values, `resolved`, first-decision-wins under repeated/concurrent resolve, and a short timeout raising `TimeoutError` rather than silently approving or denying.

- [ ] **Step 4: Run gate tests and observe failure**

Run: `uv run pytest tests/unit/test_tui_approval.py -k "gate" -v`  
Expected: FAIL because `ApprovalGate` does not exist.

- [ ] **Step 5: Implement ApprovalGate**

Use `threading.Event` plus a lock. `wait(None)` blocks indefinitely; a finite expired timeout raises `TimeoutError`; `resolve` stores only the first boolean decision and always wakes existing waiters.

- [ ] **Step 6: Run gate tests**

Run: `uv run pytest tests/unit/test_tui_approval.py -k "gate" -v`  
Expected: PASS.

- [ ] **Step 7: Write failing ApprovalModal tests**

Use a minimal Textual host app and pilot to assert the full tool/risk/workspace/command text is present, Y/Enter return `True`, N/Escape return `False`, and buttons return the same values.

- [ ] **Step 8: Run modal tests and observe failure**

Run: `uv run pytest tests/unit/test_tui_approval.py -k "modal" -v`  
Expected: FAIL because `ApprovalModal` does not exist.

- [ ] **Step 9: Implement ApprovalModal and modal styles**

Compose labels, a scrollable command area, and Approve/Deny buttons. Use priority bindings so Y/Enter/N/Escape work regardless of focused button, and dismiss with a boolean result.

- [ ] **Step 10: Verify approval tests**

Run: `uv run pytest tests/unit/test_tui_approval.py -v`  
Expected: PASS.

### Task 3: Implement the approved Rich logo

**Files:**
- Create: `src/xiliumini/cli/tui/logo.py`
- Create: `tests/unit/test_tui_logo.py`

**Interfaces:**
- Produces: `LOGO_FRAME_COUNT: int`.
- Produces: `render_logo(*, width: int = 80, frame: int | None = None, color: bool = True) -> Text`.
- The returned Rich `Text` uses a centered five-row block wordmark on wide terminals and
  always includes the tagline `你的专属智能 Agent`; narrow terminals retain the readable
  compact `x i l i u m i n i` form.

- [ ] **Step 1: Write failing logo tests**

Test exact copy, orange/cyan/blue styling in color mode, no spans in color-off mode, deterministic bounded frames, `NO_COLOR` selection, and widths smaller than the subtitle without negative repetition or exceptions.

- [ ] **Step 2: Run logo tests and observe failure**

Run: `uv run pytest tests/unit/test_tui_logo.py -v`  
Expected: FAIL because `render_logo` does not exist.

- [ ] **Step 3: Implement deterministic Rich logo frames**

Build the multiline result with `rich.text.Text`; vary only paw intensity and visible scan-line length across `LOGO_FRAME_COUNT` frames. Clamp the line width and use a static final frame when `frame is None`, color is disabled, or `NO_COLOR` is set.

- [ ] **Step 4: Verify logo tests**

Run: `uv run pytest tests/unit/test_tui_logo.py -v`  
Expected: PASS.

### Task 4: Build the Textual application and worker lifecycle

**Files:**
- Create: `src/xiliumini/cli/tui/app.py`
- Modify: `src/xiliumini/cli/tui/styles.tcss`
- Modify: `src/xiliumini/cli/tui/__init__.py`
- Create: `tests/unit/test_tui_app.py`

**Interfaces:**
- Consumes: `render_logo`, `ApprovalGate`, `ApprovalModal`, and `stream_session_events`.
- Produces: `AgentEventMessage(payload: dict[str, Any])`.
- Produces: `ApprovalRequestedMessage(request: ApprovalRequest, workspace: Path, gate: ApprovalGate)`.
- Produces: `TurnFinishedMessage(error: str | None = None)`.
- Produces: `MokioClawTuiApp(App[None])`, initialized with a preloaded `session_id: str` and
  session data-root `session_workspace: Path`.
- Produces: `run_tui(*, session_workspace: Path | None = None, max_attempts: int = 3, approval_mode: str = "inline", checkpoint_mode: str = "light", trace_mode: str = "on") -> None`.
- `run_tui` resolves the data root and loads/creates the session before calling `App.run()`, then
  passes the resulting ID into `MokioClawTuiApp`.

- [ ] **Step 1: Write failing composition and initial-state tests**

Assert Header/status, Logo, scrollable Conversation, Input, the injected persistent session ID,
startup status, and static-logo test mode using `App.run_test()`.

- [ ] **Step 2: Run initial app tests and observe failure**

Run: `uv run pytest tests/unit/test_tui_app.py -k "compose or initial or logo" -v`  
Expected: FAIL because the app does not exist.

- [ ] **Step 3: Implement composition, state, and Logo timer**

Compose widgets with stable IDs, display the preloaded session ID, focus Input, and update Rich
logo frames for approximately 0.8 seconds before collapsing to the compact header. Test mode
skips timers; no filesystem work runs on the Textual UI thread.

- [ ] **Step 4: Run initial app tests**

Run: `uv run pytest tests/unit/test_tui_app.py -k "compose or initial or logo" -v`  
Expected: PASS.

- [ ] **Step 5: Write failing event-rendering tests**

Post representative planner, tool call/result, search, handoff, checkpoint, verifier, final, error,
and unknown progress dictionaries. Assert default-collapsed reasoning, user expansion, elapsed-time
titles, multi-turn isolation, icons/status classes, safe truncation, and visible final/error results.

- [ ] **Step 6: Run event-rendering tests and observe failure**

Run: `uv run pytest tests/unit/test_tui_app.py -k "event or plan or truncat" -v`  
Expected: FAIL because message handlers and renderers are missing.

- [ ] **Step 7: Implement Textual messages and event dispatch**

Define the three message classes. Decode the existing `custom_event` / `graph_event` envelope, update only the owning panel, limit displayed long fields with an explicit truncation marker, and keep unknown events readable.

- [ ] **Step 8: Run event-rendering tests**

Run: `uv run pytest tests/unit/test_tui_app.py -k "event or plan or truncat" -v`  
Expected: PASS.

- [ ] **Step 9: Write failing worker and multi-turn tests**

Inject a deterministic stream factory. Submit two sequential inputs and assert one shared workspace/session, one generator per turn, disabled input while running, no second generator on rapid double-submit, final re-focus, generator close, and recovery after the factory raises before a terminal event.

- [ ] **Step 10: Run worker tests and observe failure**

Run: `uv run pytest tests/unit/test_tui_app.py -k "worker or turn or session or submit or exception" -v`  
Expected: FAIL because the thread worker is missing.

- [ ] **Step 11: Implement the exclusive thread worker**

On non-empty `Input.Submitted`, disable Input, mark status running, echo the user text, and launch one thread worker. Iterate `stream_session_events`, post `AgentEventMessage`, close the generator in `finally`, and always post `TurnFinishedMessage`. Reject/notify duplicate submits while busy.

- [ ] **Step 12: Run worker tests**

Run: `uv run pytest tests/unit/test_tui_app.py -k "worker or turn or session or submit or exception" -v`  
Expected: PASS.

- [ ] **Step 13: Write failing approval lifecycle tests**

From the stream worker's synchronous approval handler, assert `ApprovalRequestedMessage` opens the modal, Header becomes approval, approve/deny returns an `ApprovalDecision`, duplicate UI resolution does not change the decision, and app shutdown resolves every pending gate as denied.

- [ ] **Step 14: Run lifecycle tests and observe failure**

Run: `uv run pytest tests/unit/test_tui_app.py -k "approval or shutdown" -v`  
Expected: FAIL because the app has no approval bridge.

- [ ] **Step 15: Implement the approval bridge and shutdown cleanup**

Create gates only inside the worker approval callback, track pending gates under a lock, post the request, wait, and convert the result to `ApprovalDecision`. Modal callbacks resolve and remove gates. App unmount denies all remaining gates and prevents late worker messages from mutating widgets.

- [ ] **Step 16: Verify the full TUI app tests**

Run: `uv run pytest tests/unit/test_tui_app.py -v`  
Expected: PASS.

### Task 5: Route CLI entry points into the TUI

**Files:**
- Modify: `src/xiliumini/cli/__init__.py`
- Modify: `tests/integration/test_cli.py`

**Interfaces:**
- Consumes: `xiliumini.cli.tui.run_tui`.
- Preserves: `_create_cli_runtime`, `_render_events`, `ask`, `doctor`, and resume behavior.
- Produces: bare `xiliumini` and `xiliumini chat` calls to `run_tui` with root options.

- [ ] **Step 1: Write failing CLI routing tests**

Patch `run_tui` and assert the bare command and `chat` call it once with workspace, attempt, approval, checkpoint, and trace options. Assert explicit `ask`, `doctor`, and resume do not call it; retain the current rejection for unsupported `chat --session`.

- [ ] **Step 2: Run CLI routing tests and observe failure**

Run: `uv run pytest tests/integration/test_cli.py -k "tui or chat or help_and_version or resume" -v`  
Expected: new routing assertions fail because the default still starts the prompt loop.

- [ ] **Step 3: Replace the default prompt loop with lazy TUI launch**

Load configuration before alternate-screen startup, lazy-import `run_tui` (which preloads the
session before `App.run()`), pass `_CliOptions`, and keep existing Typer/Xiliumini error codes. Remove obsolete
prompt-loop-only code and update command help without importing Textual during unrelated
subcommands.

- [ ] **Step 4: Verify CLI integration**

Run: `uv run pytest tests/integration/test_cli.py -v`  
Expected: PASS.

### Task 6: Document, audit, and verify the completed feature

**Files:**
- Modify: `README.md`
- Modify: `SPEC.md`
- Modify: `项目进程.md`
- Modify as needed: files from Tasks 1-5 for review findings

**Interfaces:**
- Consumes: all completed TUI interfaces.
- Produces: accurate launch/interaction documentation and recorded real verification results.

- [ ] **Step 1: Update user and architecture documentation**

Document bare-command TUI startup, `chat` alias, input lifecycle, collapsible reasoning, approval
keys, Logo fallback, and retained non-interactive commands. Append a new task section to
`项目进程.md` without overwriting Task 5 history.

- [ ] **Step 2: Run focused TUI and event suites**

Run: `uv run pytest tests/unit/test_tui_approval.py tests/unit/test_tui_logo.py tests/unit/test_tui_app.py tests/unit/test_agent.py tests/integration/test_cli.py -q`  
Expected: all selected tests pass.

- [ ] **Step 3: Run the complete test suite**

Run: `uv run pytest -q`  
Expected: all tests pass with only documented skips.

- [ ] **Step 4: Run lint, format, type, dependency, and whitespace checks**

Run: `uv run ruff check .`  
Expected: PASS.

Run: `uv run ruff format --check .`  
Expected: PASS.

Run: `uv run pyright`  
Expected: 0 errors.

Run: `uv run python -m pip check`  
Expected: no broken requirements.

Run: `git diff --check`  
Expected: no whitespace errors.

- [ ] **Step 5: Perform a final requirements audit**

Compare the implementation against `docs/specs/2026-10-03-textual-tui-design.md`, confirm every acceptance criterion has test or manual evidence, and write the actual commands/results into `项目进程.md`.
