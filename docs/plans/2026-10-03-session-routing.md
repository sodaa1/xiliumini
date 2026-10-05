# Session Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist bounded multi-turn sessions and expose an event stream that routes chat directly while sending workspace work through the existing full workflow.

**Architecture:** A pure session module owns validated atomic JSON/Markdown persistence and bounded context construction. Runtime owns entry-graph and full-workflow execution, while `core.agent` wraps both in the existing public dictionary event format and records completed assistant turns.

**Tech Stack:** Python 3.12, pathlib, stdlib JSON/UUID/datetime, LangGraph, pytest, Pydantic-backed project configuration

**Spec:** `docs/specs/2026-10-03-session-routing-design.md`

## Global Constraints

- Work directly on `main`; do not create a worktree.
- Do not create intermediate Git commits; the user did not request a commit.
- Default persistent files are exactly `.xiliumini/session/session.json` and `.xiliumini/session/SESSION_SUMMARY.md`.
- Full-workflow tools remain inside `.xiliumini/workspaces/<session_id>/`.
- `MAX_SESSION_CONTEXT = 7000` and `MAX_TURN_CONTENT = 4000` are character limits.
- Persist at most ten recent messages while keeping `turn_index` monotonic.
- Existing `stream_agent_events`, CLI, checkpoint, trace, and resume behavior must remain compatible.
- Production changes follow red-green-refactor; completion requires the full repository verification suite.

## Review Focus

- A session file containing valid JSON but a duplicate, missing, or out-of-order turn must raise `SessionError` without being overwritten; Task 1 tests each ordering fault.
- An assistant turn with an unsupported route must be rejected before persistence; Task 1 tests an invalid route literal.
- A workspace entry that disappears during inventory collection must be skipped rather than abort context construction; Task 1 simulates the stat race.
- Closing the public generator after an intermediate workflow event must close the nested generator and must not add an assistant turn; Task 3 tests both effects.
- A workflow that yields an error after the saved user turn must preserve that user turn and omit an assistant turn; Task 3 tests persisted state after the error.

---

## File Structure

- Create `src/xiliumini/core/session.py`: session schema validation, turn mutation, atomic persistence, Markdown rendering, file inventory, and bounded context.
- Modify `src/xiliumini/errors.py`: add stable `SessionError(code="session_error")`.
- Modify `src/xiliumini/graph/workflow.py`: expose `build_complex_workflow` as the canonical full-workflow compatibility name.
- Modify `src/xiliumini/runtime.py`: add session-aware route execution while retaining Runtime ownership of model and harness services.
- Modify `src/xiliumini/core/agent.py`: expose and implement the dictionary-based multi-turn event transaction.
- Modify `src/xiliumini/core/__init__.py`: lazy-export `stream_session_events`.
- Create `tests/unit/test_session.py`: persistence, validation, limits, inventory, and context behavior.
- Modify `tests/integration/test_runtime.py`: chat short circuit and workflow handoff tests.
- Modify `tests/unit/test_agent.py`: public session event and transaction lifecycle tests.
- Modify `tests/unit/test_workflow.py`: compatibility-name behavior.
- Modify `README.md`, `SPEC.md`, and `项目进程.md`: final public API, file layout, and verified task record.

### Task 1: Validated Session Persistence and Context

**Files:**
- Create: `src/xiliumini/core/session.py`
- Modify: `src/xiliumini/errors.py`
- Create: `tests/unit/test_session.py`

**Interfaces:**
- Consumes: `atomic_write_utf8(path: Path, content: str) -> None` from `tools.workspace`.
- Produces: `load_or_create_session(workspace: Path) -> dict[str, Any]`, `append_user_turn(session: dict[str, Any], content: str) -> int`, `append_assistant_turn(session: dict[str, Any], *, turn: int, route: Literal["chat", "workflow"], content: str, summary: str = "") -> None`, `save_session(workspace: Path, session: dict[str, Any]) -> dict[str, Any]`, and `build_session_context(workspace: Path, session: dict[str, Any] | None = None) -> str`.
- Produces: `SessionError` with `code = "session_error"`.

- [ ] **Step 1: Write failing creation and reload tests**

Add tests asserting a missing session creates a UUID-backed object with `turn_index == 0`, empty `recent_turns`, parseable UTC timestamps, and that save/reload preserves the ID.

- [ ] **Step 2: Run the creation test and verify RED**

Run: `uv run pytest tests/unit/test_session.py -q`

Expected: collection failure because `xiliumini.core.session` does not exist.

- [ ] **Step 3: Implement constants, `SessionError`, schema validation, loading, and atomic JSON/Markdown saving**

Resolve storage as `<workspace>/session`, where the default workspace is the parent of `SESSION_ROOT` (`.xiliumini`); JSON is authoritative. Validate UUID, timestamps, types, turn order, role-specific fields, and `turn_index >= last persisted turn`. Never replace a corrupt existing source.

- [ ] **Step 4: Run creation/reload tests and verify GREEN**

Run: `uv run pytest tests/unit/test_session.py -q`

Expected: creation/reload tests pass.

- [ ] **Step 5: Write failing turn mutation and validation tests**

Assert per-message numbering (`1, 2, 3`), 4000-character truncation, ten-message retention, invalid assistant route rejection, next-turn enforcement, and corrupt/duplicate/gapped session rejection.

- [ ] **Step 6: Run turn tests and verify RED**

Run: `uv run pytest tests/unit/test_session.py -q`

Expected: failures identify missing append and retention behavior.

- [ ] **Step 7: Implement turn mutation and validation behavior**

Use UTC ISO timestamps. `append_user_turn` allocates the next index; `append_assistant_turn` requires the supplied index to equal `turn_index + 1`. Retain only the newest ten entries after either append.

- [ ] **Step 8: Write failing context tests**

Assert session metadata, relative files ordered by descending mtime, a 30-file maximum, newest ten messages, assistant-summary preference, strict 7000-character maximum, and graceful handling of a file removed between enumeration and stat.

- [ ] **Step 9: Run context tests and verify RED**

Run: `uv run pytest tests/unit/test_session.py -q`

Expected: failures identify missing inventory and context budgeting.

- [ ] **Step 10: Implement bounded context construction**

Inventory only regular files below `<workspace>/workspaces/<session_id>`, return relative POSIX paths, and budget header/newest dialogue before older dialogue and file lines.

- [ ] **Step 11: Run Task 1 tests and verify GREEN**

Run: `uv run pytest tests/unit/test_session.py -q`

Expected: all Task 1 tests pass.

### Task 2: Runtime Entry Routing

**Files:**
- Modify: `src/xiliumini/graph/workflow.py`
- Modify: `src/xiliumini/runtime.py`
- Modify: `tests/unit/test_workflow.py`
- Modify: `tests/integration/test_runtime.py`

**Interfaces:**
- Consumes: `build_entry_workflow(model)`, `GraphState.context_summary`, and Task 1 session ID/context.
- Produces: `build_complex_workflow(model: Any, memory_manager: Any, checkpointer: Any | None = None)` and `Runtime.stream_session(task: str, session_id: str, context_summary: str, max_attempts: int = 3) -> Generator[RuntimeEvent, None, None]`.

- [ ] **Step 1: Write failing compatibility and chat-route tests**

Assert `build_complex_workflow` has the same observable graph behavior as the existing builder. Assert `Runtime.stream_session` passes the bounded context to the entry graph, emits one `FinalEvent` for chat, and never invokes `stream_agent` or starts Todo state.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/unit/test_workflow.py tests/integration/test_runtime.py -q`

Expected: failures identify the missing builder name and Runtime method.

- [ ] **Step 3: Implement the compatibility builder and chat route**

Keep one full-graph implementation. Prepare fresh state with the session context, execute the entry graph under the Runtime context, and translate its chat answer to `FinalEvent`.

- [ ] **Step 4: Run focused tests and verify chat GREEN**

Run: `uv run pytest tests/unit/test_workflow.py tests/integration/test_runtime.py -q`

Expected: compatibility and chat tests pass.

- [ ] **Step 5: Write the failing workflow-handoff test**

Assert a workflow route preserves `intent_route`, `intent_reason`, `intent_confidence`, and `context_summary` in the state handed to `stream_agent`; verify the existing approval/checkpoint/trace path and isolated workspace are used once.

- [ ] **Step 6: Run the handoff test and verify RED**

Run: `uv run pytest tests/integration/test_runtime.py -q`

Expected: workflow handoff assertions fail before implementation.

- [ ] **Step 7: Implement workflow handoff through the existing `_run` lifecycle**

Start Todo only for workflow, then pass the routed state and existing `MemoryManager` to `_run`; do not duplicate checkpoint or trace setup.

- [ ] **Step 8: Run Task 2 tests and verify GREEN**

Run: `uv run pytest tests/unit/test_workflow.py tests/integration/test_runtime.py -q`

Expected: all Task 2 tests pass.

### Task 3: Public Multi-Turn Event Transaction

**Files:**
- Modify: `src/xiliumini/core/agent.py`
- Modify: `src/xiliumini/core/__init__.py`
- Modify: `src/xiliumini/runtime.py`
- Modify: `tests/unit/test_agent.py`

**Interfaces:**
- Consumes: Task 1 session functions and `Runtime.stream_session` from Task 2.
- Produces: `stream_session_events(task: str, *, session_workspace: Path | None = None, max_attempts: int = 3, approval_mode: str = "inline", approval_handler: Callable[[ApprovalRequest], ApprovalDecision] | None = None, checkpoint_mode: str = "light", trace_mode: str = "on") -> Generator[dict[str, Any], None, None]`.
- Produces: optional `data_dir` override on `create_runtime` so explicit session roots remain Runtime-owned without mutating Settings.

- [ ] **Step 1: Write failing chat transaction and public-export tests**

Assert the default session workspace is `.xiliumini`, producing `.xiliumini/session`; save user turn 1 before routing, save chat assistant turn 2 after `FinalEvent`, yield the existing dictionary event shape, and expose the function from `xiliumini.core`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/unit/test_agent.py -q`

Expected: failures identify the missing public API/export.

- [ ] **Step 3: Implement default-path resolution, Runtime construction, event adaptation, and completed chat persistence**

Treat `session_workspace` as the xiliumini data root when supplied; when omitted use `Path(SESSION_ROOT).parent`, which is `.xiliumini`. Pass that root as Runtime's `data_dir` override.

- [ ] **Step 4: Run chat transaction tests and verify GREEN**

Run: `uv run pytest tests/unit/test_agent.py -q`

Expected: chat transaction tests pass.

- [ ] **Step 5: Write failing workflow, error, prior-context, and close tests**

Assert workflow custom/graph events are forwarded, final answer and collected Planner/Verifier summary are saved, a second invocation receives the prior turns, errors preserve only the user turn, and early close closes the Runtime generator without an assistant append.

- [ ] **Step 6: Run lifecycle tests and verify RED**

Run: `uv run pytest tests/unit/test_agent.py -q`

Expected: new lifecycle cases fail before implementation.

- [ ] **Step 7: Implement workflow summary collection and generator-finalization behavior**

Record an assistant only after a `FinalEvent`; derive workflow summary from the latest Planner/Verifier events already observed. Always close the nested generator in `finally`.

- [ ] **Step 8: Run Task 3 tests and verify GREEN**

Run: `uv run pytest tests/unit/test_agent.py -q`

Expected: all Task 3 tests pass.

### Task 4: Documentation, Progress Record, and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `SPEC.md`
- Modify: `项目进程.md`

**Interfaces:**
- Consumes: completed Task 1-3 APIs and actual verification results.
- Produces: documented layout/API and a new Task 5 progress section without changing historical entries.

- [ ] **Step 1: Update user and architecture documentation**

Document the entry route, session files, isolated execution directory, turn/context limits, and `stream_session_events` example. Replace statements that persistent transcripts are only roadmap work.

- [ ] **Step 2: Run the complete verification suite**

Run: `uv run pytest -q`

Expected: all tests pass, with only documented Windows link skips.

Run: `uv run ruff check .`

Expected: `All checks passed!`

Run: `uv run ruff format --check .`

Expected: all files already formatted.

Run: `uv run pyright`

Expected: `0 errors, 0 warnings, 0 informations`.

Run: `git diff --check`

Expected: exit code 0.

- [ ] **Step 3: Append the verified Task 5 record**

Add status, implementation summary, changed architecture, and the exact fresh verification counts to `项目进程.md`; do not alter Tasks 0-4.

- [ ] **Step 4: Re-run documentation-sensitive checks after the progress edit**

Run: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`, and `git diff --check`.

Expected: every command exits 0.
