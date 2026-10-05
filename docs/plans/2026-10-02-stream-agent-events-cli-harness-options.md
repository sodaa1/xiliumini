# Stream Agent Events and CLI Harness Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the requested `stream_agent_events` compatibility API and root CLI harness options without duplicating the existing Checkpoint/Trace lifecycle.

**Architecture:** Keep `stream_agent` as the only raw LangGraph lifecycle implementation. Runtime owns per-run frozen context, CheckpointManager, and TraceRecorder; the new compatibility function builds an overridden Runtime and adapts its stable events to `custom_event`/`graph_event` dictionaries. Approval configuration travels through a run-local ContextVar into BashTool and is never persisted in GraphState or checkpoint files.

**Tech Stack:** Python 3.12, Typer, LangGraph, ContextVar, pytest, Ruff, Pyright.

**Spec:** `docs/specs/2026-09-30-task4-checkpoint-trace-design.md` plus the user-requested `stream_agent_events`/CLI signatures from 2026-10-02.

## Global Constraints

- Work directly on `main`; do not create a worktree.
- Preserve Trace modes `full/summary/off`; accept requested `on` only as an alias for `full` at the new API/CLI boundary.
- Keep Checkpoint/Trace recording single-owned by Runtime and `stream_agent`; never record the same raw event twice.
- Never persist `approval_handler`, model objects, ContextVars, callbacks, or absolute workspace paths in GraphState/checkpoints/traces.
- An explicit workspace must resolve beneath `<data_dir>/workspaces`; reject traversal, links, and foreign roots with redacted errors.
- Use TDD for every behavior change and run the complete repository gate before completion.

## Review Focus

- `trace_mode="on"` maps to `full`, while invalid values and existing `summary` behavior remain deterministic.
- Interleaved runs with different approval modes/handlers do not leak ContextVar values across yields, close, or failure.
- `--workspace` and `--resume` cannot be combined, and foreign/missing workspaces never leak absolute paths.
- Inline approval receives the exact frozen command/risk request once; `auto`/`deny` never prompt.
- The compatibility API produces one adapted event per existing RuntimeEvent while the underlying trace/checkpoint contains one copy of each raw event.

---

### Task 1: Run-local Approval Configuration

**Files:**
- Create: `src/xiliumini/tools/approval_context.py`
- Modify: `src/xiliumini/tools/__init__.py`
- Modify: `src/xiliumini/runtime.py`
- Modify: `tests/unit/test_bash_tool.py`
- Modify: `tests/integration/test_runtime.py`

**Interfaces:**
- Produces: immutable `ApprovalConfig(mode, handler)` and `approval_config: ContextVar[ApprovalConfig]`.
- Extends: `Runtime(..., approval_mode="inline", approval_handler=None)` and frozen `_RunContext`.
- Consumes: existing `BashTool(approval_mode=..., approval_handler=...)`.

- [x] **Step 1: Write failing approval propagation/isolation tests**

Assert `build_tools` binds mode/handler from the active context, defaults to inline/no handler, and two interleaved Runtime iterators see only their own approval configuration across `next()`, exception, and `close()`.

- [x] **Step 2: Run RED**

Run: `python -m pytest tests/unit/test_bash_tool.py tests/integration/test_runtime.py -k approval -q`

Expected: failures because Runtime does not bind approval configuration and build_tools always constructs default BashTool.

- [x] **Step 3: Implement minimal run-local propagation**

Bind/reset one approval ContextVar alongside the existing model/search ContextVars for generator creation, every `next()`, and `close()`. `build_tools` copies mode/handler into each BashTool. Do not add these fields to GraphState.

- [x] **Step 4: Run GREEN and static checks**

Run the Task 1 tests plus Ruff/Pyright for modified files.

### Task 2: Explicit Workspace Runtime and `stream_agent_events`

**Files:**
- Modify: `src/xiliumini/tools/workspace.py`
- Modify: `src/xiliumini/runtime.py`
- Modify: `src/xiliumini/core/agent.py`
- Modify: `src/xiliumini/core/__init__.py`
- Modify: `tests/unit/test_agent.py`
- Modify: `tests/integration/test_runtime.py`

**Interfaces:**
- Produces: `Runtime.stream_workspace(task, workspace, max_attempts=3)`.
- Produces: `stream_agent_events(task, *, workspace, max_attempts=3, approval_mode="inline", approval_handler=None, checkpoint_mode="light", resume_workspace=None, trace_mode="on") -> Iterator[dict[str, Any]]`.
- Keeps: `stream_agent(...) -> Generator[RuntimeEvent, None, None]` unchanged for existing callers.

- [x] **Step 1: Write failing explicit workspace tests**

Cover safe create/reuse under `data_dir/workspaces`, Todo reset for a new task, foreign root/link/traversal rejection, redacted errors, and resume taking precedence only when no conflicting workspace is supplied.

- [x] **Step 2: Run RED**

Run: `python -m pytest tests/integration/test_runtime.py -k workspace -q` and the new agent compatibility tests.

- [x] **Step 3: Implement explicit workspace validation and shared input construction**

Refactor Runtime's fresh-state creation into one private path used by UUID sessions and explicit workspaces. The compatibility function locally imports configuration/runtime modules to avoid package cycles, maps `on -> full`, and delegates all persistence to Runtime.

- [x] **Step 4: Implement event adaptation**

Map ProgressEvent to `{"type": "custom_event", "event": ...}` and Planner/Verifier/Final/Error to `{"type": "graph_event", "event": ...}` using JSON-safe primitive dictionaries. Do not expose model/callback objects or mutate RuntimeEvent instances.

- [x] **Step 5: Run GREEN and regression tests**

Assert raw trace/checkpoint counts are unchanged versus direct Runtime execution and resume preserves checkpoint state.

### Task 3: Root Typer Harness Options

**Files:**
- Modify: `src/xiliumini/cli/__init__.py`
- Modify: `tests/integration/test_cli.py`

**Interfaces:**
- Adds root options: `--workspace/-w`, `--max-attempts`, `--approval-mode inline|auto|deny`, `--checkpoint-mode light|strict|off`, `--trace-mode on|off`, and existing `--resume`.
- Extends: `create_runtime(settings, *, approval_mode=..., approval_handler=..., checkpoint_mode=..., trace_mode=...)` with backward-compatible optional overrides.

- [x] **Step 1: Write failing CLI parsing/dispatch tests**

Cover defaults and overrides for chat/ask/resume, `on -> full`, inline prompt handler decisions, `auto`/`deny` no prompt, workspace/resume conflict, subcommand behavior, exit codes, and path redaction.

- [x] **Step 2: Run RED**

Run: `python -m pytest tests/integration/test_cli.py -k "workspace or approval or checkpoint or trace or resume" -q`.

- [x] **Step 3: Implement CLI option context and approval prompt**

Store parsed root options in a small immutable CLI context. Construct Runtime with overrides for ask/chat/resume; pass explicit workspace to `stream_workspace`. Inline approval uses `typer.confirm` and returns `ApprovalDecision`; only inline mode installs the handler.

- [x] **Step 4: Run GREEN and full CLI regression**

Run all CLI integration tests, help snapshots/contains assertions, Ruff, and scoped Pyright.

### Task 4: Documentation, Review, and Acceptance

**Files:**
- Modify: `README.md`
- Modify: `SPEC.md`
- Modify: `.env.example` only if a documented environment alias is added
- Modify: `项目进程.md`

**Interfaces:**
- Documents the compatibility API, current file location (`cli/__init__.py`), CLI option scope, `on -> full`, workspace containment, and inline approval behavior.

- [x] **Step 1: Update documentation from actual behavior**

State that `summary` remains available through Settings/Runtime even though the requested CLI alias exposes on/off. Explain that approval callbacks are run-local and absent from checkpoints/traces.

- [x] **Step 2: Run focused suites**

Run approval, agent, runtime, CLI, checkpoint, and trace suites.

- [x] **Step 3: Run complete acceptance**

Run `pytest -q`, `ruff format --check .`, `ruff check .`, full Pyright with the Task4 venv, and `git diff --check`.

- [x] **Step 4: Perform independent code review and fix all Critical/Important findings**

Focus on duplicate persistence, ContextVar leakage, explicit workspace containment, approval prompt safety, CLI compatibility, and event adaptation.

- [x] **Step 5: Update project progress and commit**

Record exact RED/GREEN/full results and review outcome in `项目进程.md`; commit only verified work.
