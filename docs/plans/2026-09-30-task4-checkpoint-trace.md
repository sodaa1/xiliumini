# Task4 Checkpoint and Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add recoverable workspace checkpoints and redacted execution traces, then expose exact resume through the xiliumini Runtime and CLI.

**Architecture:** Runtime creates per-run CheckpointManager and TraceRecorder instances and passes them to `stream_agent`, which observes raw LangGraph `updates/custom` chunks while maintaining the latest merged state. Checkpoints use an internal detached Git directory for exact workspace snapshots; traces write bounded, redacted JSONL plus machine- and human-readable summaries.

**Tech Stack:** Python 3.12, LangGraph, LangChain messages, Pydantic Settings, Typer, Git CLI, pytest, Ruff, Pyright.

**Spec:** `docs/specs/2026-09-30-task4-checkpoint-trace-design.md`

## Global Constraints

- Work directly on `main`; do not create a worktree unless the user changes the repository rule.
- Preserve the existing `InMemorySaver`; disk Checkpoint augments it rather than replacing it.
- Use `.xiliumini/checkpoints` and `.xiliumini/traces`; never use the old `mokioclaw` name.
- Checkpoint modes are exactly `light | strict | off`, default/fallback `light`.
- Trace modes are exactly `full | summary | off`, default/fallback `full`.
- Every subprocess call uses argv with `shell=False`, a timeout, and bounded output.
- Never persist models, tools, callbacks, SecretStr values, environment variables, or absolute workspace paths.
- Exact restore may delete ordinary post-checkpoint files but must preserve `.xiliumini/**` and `.git/**`.
- No new third-party dependency; use stdlib plus existing LangChain/LangGraph packages.
- Follow TDD: each production change starts with a failing test that fails for the intended missing behavior.
- Before every commit, reread `AGENTS.md` and `项目进程.md`, append the task's actual status/summary/verification, and stage only that task's files.

## Review Focus

1. A malicious checkpoint containing `../`, absolute paths, symlink escapes, or an unknown type tag must fail before any workspace mutation; Task 3 adds each adversarial restore test.
2. A workspace that already contains `.git` and changing `.xiliumini` control data must retain both exactly across restore; Task 3 pins this preservation contract.
3. Interleaved generators from one Runtime must never share workspace, trace ID, counters, latest state, or recorder lifecycle; Task 7 adds the interleaving test.
4. Generator close/KeyboardInterrupt during a failed persistence operation must not emit duplicate end events or mask the original interruption; Task 6 adds close and failure-order tests.
5. Trace counters must use strict booleans so values such as `"false"`, `1`, or missing fields cannot count as approval/failure; Task 4 adds malformed-event tests.

---

### Task 0: Commit the Completed Approval Baseline

**Files:**
- Modify: `项目进程.md`
- Existing implementation: `src/xiliumini/core/approval.py`
- Existing implementation: `src/xiliumini/tools/bash_tool.py`
- Existing implementation: `src/xiliumini/tools/command.py`
- Existing tests: `tests/unit/test_approval.py`
- Existing tests: `tests/unit/test_bash_tool.py`

**Interfaces:**
- Consumes: the already completed and reviewed first Task4 step in the working tree.
- Produces: clean committed baseline with `ApprovalRequest`, `ApprovalDecision`, `classify_command_risk`, `normalize_approval_mode`, and `CommandResult.requires_approval`.

- [ ] **Step 1: Reinspect the current diff and verify no second-step files are staged**

Run: `git status --short && git diff -- src/xiliumini/core/approval.py src/xiliumini/tools/bash_tool.py src/xiliumini/tools/command.py tests/unit/test_approval.py tests/unit/test_bash_tool.py`

Expected: only the five Approval source/test files are uncommitted; the design commit remains separate.

- [ ] **Step 2: Run fresh Approval and full verification**

Run: `python -m pytest tests/unit/test_approval.py tests/unit/test_bash_tool.py -q && python -m pytest -q && python -m ruff check . && python -m ruff format --check . && python -m pyright src tests && git diff --check`

Expected: focused `56 passed`; full `266 passed, 4 skipped`; Ruff/Pyright/diff check pass.

- [ ] **Step 3: Append the actual first-step implementation and verification record**

Add a Task4 first-step subsection to `项目进程.md`; describe only the Approval work and exact fresh results.

- [ ] **Step 4: Commit only the Approval baseline**

```bash
git add src/xiliumini/core/approval.py src/xiliumini/tools/bash_tool.py src/xiliumini/tools/command.py tests/unit/test_approval.py tests/unit/test_bash_tool.py 项目进程.md
git commit -m "feat: add approval gate for risky commands"
```

### Task 1: Shared Harness Modes, Serialization, and Errors

**Files:**
- Create: `src/xiliumini/core/harness_io.py`
- Modify: `src/xiliumini/config.py`
- Modify: `src/xiliumini/errors.py`
- Modify: `.env.example`
- Create: `tests/unit/test_harness_io.py`
- Modify: `tests/unit/test_config.py`
- Modify: `项目进程.md`

**Interfaces:**
- Consumes: `atomic_write_utf8(path: Path, content: str)` from `tools.workspace`.
- Produces: `normalize_checkpoint_mode(mode: str | None) -> str`, `normalize_trace_mode(mode: str | None) -> str`, `sanitize_for_persistence(value: Any, workspace: Path, *, max_text: int = 20_000) -> Any`, `restore_persisted_value(value: Any, workspace: Path) -> Any`, `write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None`, and `append_jsonl(path: Path, payload: Mapping[str, Any]) -> None`.

- [ ] **Step 1: Write failing mode/config/error tests**

Add tests named:

```python
def test_checkpoint_mode_defaults_and_invalid_values_fall_back_to_light(): ...
def test_trace_mode_defaults_and_invalid_values_fall_back_to_full(): ...
def test_settings_accept_harness_modes_and_optional_trace_id(): ...
def test_checkpoint_and_trace_errors_have_stable_codes(): ...
```

Assert exact valid sets and defaults; invalid Settings values must be rejected while standalone normalizers fall back.

- [ ] **Step 2: Run the mode/config tests and verify RED**

Run: `python -m pytest tests/unit/test_harness_io.py tests/unit/test_config.py -q`

Expected: collection/import or assertion failures for missing harness interfaces and Settings fields.

- [ ] **Step 3: Implement modes, Settings fields, and controlled errors**

Add `CheckpointError.code = "checkpoint_error"` and `TraceError.code = "trace_error"`. Add typed Settings fields with defaults `light`, `full`, and `None`; document their env names in `.env.example`.

- [ ] **Step 4: Write failing JSON-safe serialization tests**

Cover nested Path, dataclass, tuple/list/dict, HumanMessage/AIMessage/ToolMessage, SecretStr, sensitive key names, absolute workspace replacement, long-text truncation, and rejection of an unknown tagged type during restore.

- [ ] **Step 5: Run serialization tests and verify RED**

Run: `python -m pytest tests/unit/test_harness_io.py -q`

Expected: failures for missing serialization and persistence helpers.

- [ ] **Step 6: Implement shared serialization and atomic I/O**

Use explicit type tags only for Path/tuple/dataclass/message values. Never fall back to arbitrary `repr`; unsupported objects raise the caller's controlled harness error. JSONL append must flush and fsync after one complete line.

- [ ] **Step 7: Run Task 1 tests and static checks**

Run: `python -m pytest tests/unit/test_harness_io.py tests/unit/test_config.py -q && python -m ruff check src/xiliumini/core/harness_io.py src/xiliumini/config.py src/xiliumini/errors.py tests/unit/test_harness_io.py tests/unit/test_config.py && python -m pyright src/xiliumini/core/harness_io.py src/xiliumini/config.py src/xiliumini/errors.py`

Expected: all pass with zero Ruff/Pyright errors.

- [ ] **Step 8: Record results and commit**

Commit message: `feat: add harness modes and safe persistence primitives`

### Task 2: Checkpoint Save and Git Snapshot

**Files:**
- Create: `src/xiliumini/core/checkpoint.py`
- Create: `tests/unit/test_checkpoint.py`
- Modify: `项目进程.md`

**Interfaces:**
- Consumes: Task 1 mode/serialization/I/O helpers and `CheckpointError`.
- Produces: `workspace_manifest(workspace: Path) -> list[dict[str, Any]]`, `snapshot_workspace_git(workspace: Path, root: Path, *, message: str) -> str`, `resume_command(workspace: Path) -> str`, `build_recovery_markdown(payload: Mapping[str, Any]) -> str`, and `CheckpointManager.save(...) -> dict[str, Any] | None`.

- [ ] **Step 1: Write failing light/off and manifest tests**

Test names and core assertions:

```python
def test_checkpoint_off_creates_no_files_and_returns_none(): ...
def test_manifest_is_sorted_hashed_and_excludes_control_directories(): ...
def test_light_save_writes_metadata_recovery_and_detached_git_snapshot(): ...
```

Assert workspace root has no new `.git`; checkpoint event has `type == "checkpoint_saved"`; manifest uses POSIX relative paths with size/SHA-256.

- [ ] **Step 2: Run the light/off tests and verify RED**

Run: `python -m pytest tests/unit/test_checkpoint.py -k "off or manifest or light" -q`

Expected: import or missing-interface failures.

- [ ] **Step 3: Implement CheckpointManager construction, manifest, Git snapshot, and light save**

`CheckpointManager.__init__(runtime, task="")` reads `runtime.workspace` and `runtime.checkpoint_mode`; root is `workspace / ".xiliumini" / "checkpoints"`. Git commands use explicit git-dir/work-tree, local identity, no shell, timeout, and bounded diagnostics.

- [ ] **Step 4: Write failing strict save tests**

Assert strict writes full `state.json`, appends one valid JSON line per supplied event, creates Git commits, and preserves prior JSONL lines; light never creates state.json/events.jsonl.

- [ ] **Step 5: Run strict tests and verify RED**

Run: `python -m pytest tests/unit/test_checkpoint.py -k strict -q`

Expected: failures for missing strict artifacts/event append behavior.

- [ ] **Step 6: Implement strict save and recovery Markdown**

`build_recovery_markdown` must include task, status, manifest paths, commit, and `resume_command(workspace)`. Write metadata last so a visible checkpoint always references completed preceding artifacts.

- [ ] **Step 7: Run Task 2 tests and static checks**

Run: `python -m pytest tests/unit/test_checkpoint.py -q && python -m ruff check src/xiliumini/core/checkpoint.py tests/unit/test_checkpoint.py && python -m pyright src/xiliumini/core/checkpoint.py`

Expected: all pass; Git tests stay inside pytest temporary directories.

- [ ] **Step 8: Record results and commit**

Commit message: `feat: save checkpoint state and workspace git snapshots`

### Task 3: Exact Checkpoint Restore

**Files:**
- Modify: `src/xiliumini/core/checkpoint.py`
- Modify: `tests/unit/test_checkpoint.py`
- Modify: `项目进程.md`

**Interfaces:**
- Consumes: Task 2 checkpoint schema and internal repo.git.
- Produces: `restore_workspace_git(workspace: Path, root: Path, commit: str) -> None` and `CheckpointManager.load_resume_inputs(runtime, task=None, max_attempts=3) -> tuple[dict[str, Any], dict[str, Any]]`.

- [ ] **Step 1: Write failing successful-restore tests**

Cover light and strict state sources, caller task/max-attempt overrides, runtime/workspace reinjection, Path/message round-trip, commit restoration, deletion of post-checkpoint files, and preservation of `.xiliumini/**` plus a pre-existing `.git/keep` file.

- [ ] **Step 2: Run successful-restore tests and verify RED**

Run: `python -m pytest tests/unit/test_checkpoint.py -k "restore and not rejects and not rollback" -q`

Expected: failures for missing load/restore behavior.

- [ ] **Step 3: Implement validated exact restore and input rebuilding**

Validate workspace containment under `runtime.data_dir / "workspaces"`, schema version, known fields, internal commit, and next-node combination before creating the pre-restore commit. Delete only validated ordinary paths absent from the target tree; never traverse or delete `.xiliumini`/`.git`.

- [ ] **Step 4: Write failing adversarial and rollback tests**

Add explicit tests for `../escape`, absolute manifest path, symlink escape when supported, unknown serialized type, corrupt JSON, missing commit, foreign commit, workspace outside data_dir, and injected failure after pre-restore that restores original files.

- [ ] **Step 5: Run adversarial tests and verify RED**

Run: `python -m pytest tests/unit/test_checkpoint.py -k "rejects or rollback" -q`

Expected: each new case fails because validation/rollback is absent.

- [ ] **Step 6: Implement preflight validation and rollback**

Resolve every candidate beneath the real workspace before mutation. Create pre-restore commit only after all preflight checks pass; on later failure restore that commit and raise stable `CheckpointError` without paths or Git stderr.

- [ ] **Step 7: Run Task 3 tests and static checks**

Run: `python -m pytest tests/unit/test_checkpoint.py -q && python -m ruff check src/xiliumini/core/checkpoint.py tests/unit/test_checkpoint.py && python -m pyright src/xiliumini/core/checkpoint.py`

Expected: all pass, including platform-conditional symlink coverage.

- [ ] **Step 8: Record results and commit**

Commit message: `feat: restore exact workspace checkpoints safely`

### Task 4: Trace Recorder and Bounded Timeline

**Files:**
- Create: `src/xiliumini/core/trace.py`
- Modify: `src/xiliumini/storage/traces.py`
- Create: `tests/unit/test_trace.py`
- Modify: `项目进程.md`

**Interfaces:**
- Consumes: Task 1 Trace mode, serializer, JSON I/O, and `TraceError`.
- Produces: `TraceRecorder.start`, `record_custom_event`, `record_graph_update`, and `end` with the signatures fixed by the spec.

- [ ] **Step 1: Write failing lifecycle/mode tests**

Assert off performs zero I/O; full writes run_start/custom/graph/run_end; summary filters non-key custom events but still creates all three files; repeated start/end cannot duplicate lifecycle records.

- [ ] **Step 2: Run lifecycle tests and verify RED**

Run: `python -m pytest tests/unit/test_trace.py -k "mode or start or end" -q`

Expected: import or missing-interface failures.

- [ ] **Step 3: Implement TraceRecorder lifecycle and mode filtering**

Generate/validate trace IDs, assign monotonically increasing sequence numbers, UTC timestamps, and monotonic duration. `storage/traces.py` should re-export the canonical implementation rather than duplicate it.

- [ ] **Step 4: Write failing statistics/timeline/redaction tests**

Cover node visits and every counter; require `ok is False` and `requires_approval is True`; malformed values (`"false"`, `1`, missing) do not count. Generate 99, 100, and 101 events to assert head 20/tail 80/no duplicate/omitted math. Assert secrets, absolute workspace, and long output do not appear in any artifact.

- [ ] **Step 5: Run statistics tests and verify RED**

Run: `python -m pytest tests/unit/test_trace.py -k "count or timeline or redact" -q`

Expected: failures for missing counters, bounds, or sanitization.

- [ ] **Step 6: Implement strict counters, trace.json, and timeline.md**

`end(...) -> dict | None` returns the same sanitized summary written to trace.json. Timeline lines contain sequence/time/type/node or bounded message only, never full output.

- [ ] **Step 7: Run Task 4 tests and static checks**

Run: `python -m pytest tests/unit/test_trace.py -q && python -m ruff check src/xiliumini/core/trace.py src/xiliumini/storage/traces.py tests/unit/test_trace.py && python -m pyright src/xiliumini/core/trace.py`

Expected: all pass with zero static errors.

- [ ] **Step 8: Record results and commit**

Commit message: `feat: record redacted execution traces`

### Task 5: Resume Routing and Canonical Harness Events

**Files:**
- Modify: `src/xiliumini/graph/state.py`
- Modify: `src/xiliumini/graph/workflow.py`
- Modify: `src/xiliumini/runtime.py`
- Modify: `src/xiliumini/agents/react.py`
- Modify: `src/xiliumini/tools/subagent_tools.py`
- Modify: `tests/unit/test_workflow.py`
- Modify: `tests/unit/test_react_memory.py`
- Modify: `tests/unit/test_nodes.py`
- Modify: `tests/agent_fakes.py`
- Modify: `项目进程.md`

**Interfaces:**
- Consumes: checkpoint `latest_node/next_node` contract and existing custom writer.
- Produces: `GraphState.resume_node`, `route_entry(state) -> Literal["planner", "verifier", "final"]`, top-level strict boolean `requires_approval` on tool_result, and `type="handoff"` custom events.

- [ ] **Step 1: Write failing resume-route tests**

Assert new state starts planner; planner checkpoint routes verifier; failed verifier under limit routes planner; passed/exhausted verifier routes final; unknown resume node is rejected.

- [ ] **Step 2: Run route tests and verify RED**

Run: `python -m pytest tests/unit/test_workflow.py -q`

Expected: failures because START is unconditional and resume_node is absent.

- [ ] **Step 3: Implement `resume_node` and conditional START routing**

Add the field to every production/test state factory. Replace only the START edge; preserve planner→verifier and verifier conditional edges.

- [ ] **Step 4: Write failing canonical-event tests**

Assert Bash `requires_approval` is copied from parsed tool output only when exactly True, and delegation emits one bounded handoff event after the result with no full instruction/tool output.

- [ ] **Step 5: Run event tests and verify RED**

Run: `python -m pytest tests/unit/test_react_memory.py tests/unit/test_nodes.py -q`

Expected: failures for missing top-level approval and handoff event.

- [ ] **Step 6: Implement canonical approval and handoff events**

Do not change existing ProgressEvent mapping; the extra fields remain available to Trace at the raw custom boundary.

- [ ] **Step 7: Run Task 5 tests and static checks**

Run: `python -m pytest tests/unit/test_workflow.py tests/unit/test_react_memory.py tests/unit/test_delegate_analysis.py tests/unit/test_state_and_prompts.py -q && python -m ruff check src/xiliumini/graph src/xiliumini/agents/react.py src/xiliumini/tools/subagent_tools.py && python -m pyright src/xiliumini/graph src/xiliumini/agents/react.py src/xiliumini/tools/subagent_tools.py`

Expected: all pass.

- [ ] **Step 8: Record results and commit**

Commit message: `feat: route resumed graphs and emit harness events`

### Task 6: Integrate Harness Lifecycle into stream_agent

**Files:**
- Modify: `src/xiliumini/core/agent.py`
- Modify: `tests/unit/test_agent.py`
- Modify: `项目进程.md`

**Interfaces:**
- Consumes: Tasks 2–5 managers/events/routes.
- Produces: optional `checkpoint_manager`, `trace_recorder`, `resumed`, and `resume_event` keyword parameters on `stream_agent`; internal latest-state merge and deterministic lifecycle ordering.

- [ ] **Step 1: Write failing normal-flow integration tests**

Use real lightweight fake recorder/manager objects and raw workflow chunks. Assert start occurs first; custom is traced before UI mapping; update is merged then traced/checkpointed; checkpoint_saved returns to Trace; completed save precedes Trace.end; UI events remain unchanged.

- [ ] **Step 2: Run normal-flow tests and verify RED**

Run: `python -m pytest tests/unit/test_agent.py -k harness -q`

Expected: failures for unsupported parameters and missing calls.

- [ ] **Step 3: Implement latest-state merge and normal lifecycle**

Copy nested state at persistence boundaries so a recorder cannot mutate graph inputs. Light saves only after node updates; strict also saves custom events. Off manager methods remain no-ops.

- [ ] **Step 4: Write failing failure/interruption tests**

Cover workflow exception, checkpoint failure, trace failure, `generator.close()`, and KeyboardInterrupt. Assert exactly one final status, no duplicate run_end, persistence errors do not mask KeyboardInterrupt/GeneratorExit, and original runtime error still maps outside this layer.

- [ ] **Step 5: Run failure tests and verify RED**

Run: `python -m pytest tests/unit/test_agent.py -k "failure or interrupt or close" -q`

Expected: failures for missing finalization/error ordering.

- [ ] **Step 6: Implement failed/interrupted finalization**

Use one guarded finalizer with explicit status. Never recursively checkpoint a checkpoint_saved event. Preserve exception identity for BaseException interruptions.

- [ ] **Step 7: Run Task 6 and existing agent regressions**

Run: `python -m pytest tests/unit/test_agent.py tests/unit/test_nodes.py tests/unit/test_code_agent.py tests/unit/test_search_agent.py -q && python -m ruff check src/xiliumini/core/agent.py tests/unit/test_agent.py && python -m pyright src/xiliumini/core/agent.py`

Expected: all pass and existing RuntimeEvent sequences are unchanged.

- [ ] **Step 8: Record results and commit**

Commit message: `feat: checkpoint and trace raw graph events`

### Task 7: Runtime Isolation and CLI Resume

**Files:**
- Modify: `src/xiliumini/runtime.py`
- Modify: `src/xiliumini/cli/__init__.py`
- Modify: `tests/integration/test_runtime.py`
- Modify: `tests/integration/test_cli.py`
- Modify: `项目进程.md`

**Interfaces:**
- Consumes: `CheckpointManager.load_resume_inputs`, `TraceRecorder`, and expanded `stream_agent`.
- Produces: per-run immutable context, Runtime constructor mode fields, `Runtime.resume(workspace: Path, *, task: str | None = None, max_attempts: int = 3) -> Iterator[RuntimeEvent]`, and CLI `--resume`.

- [ ] **Step 1: Write failing Runtime new-run and interleaving tests**

Assert create_runtime passes Settings modes/ID; each stream gets distinct manager instances; two interleaved sessions keep separate workspace/trace ID/counters/latest state; existing model/search ContextVars still reset between yields.

- [ ] **Step 2: Run Runtime tests and verify RED**

Run: `python -m pytest tests/integration/test_runtime.py -k "checkpoint or trace or interleave" -q`

Expected: failures for missing run contexts and manager injection.

- [ ] **Step 3: Implement per-run context and new-run manager wiring**

The shared Runtime object must not expose mutable current workspace. A frozen private context supplies `workspace`, `data_dir`, modes, trace ID, and session to both managers.

- [ ] **Step 4: Write failing Runtime resume and CLI tests**

Test successful resume event sequence and task/max override; `xiliumini --resume <workspace with spaces>`; missing/outside/corrupt workspaces; `--resume` with a subcommand; stable exit codes; no traceback or absolute-path leak.

- [ ] **Step 5: Run resume/CLI tests and verify RED**

Run: `python -m pytest tests/integration/test_runtime.py tests/integration/test_cli.py -k resume -q`

Expected: failures because Runtime.resume and root option do not exist.

- [ ] **Step 6: Implement Runtime.resume and CLI dispatch**

Factor the current event rendering loop so ask/chat/resume share it. Resume validates through CheckpointManager before Todo reset; never call `TodoStore.start_task()` for resumed work. Validate that the `runtime` marker returned by `load_resume_inputs` is the current Runtime, remove that marker before casting/passing GraphState to `stream_agent`, and preserve the restored workspace Path.

- [ ] **Step 7: Run Task 7 and complete Runtime/CLI regressions**

Run: `python -m pytest tests/integration/test_runtime.py tests/integration/test_cli.py -q && python -m ruff check src/xiliumini/runtime.py src/xiliumini/cli tests/integration/test_runtime.py tests/integration/test_cli.py && python -m pyright src/xiliumini/runtime.py src/xiliumini/cli`

Expected: all pass, including existing ask/chat behavior.

- [ ] **Step 8: Record results and commit**

Commit message: `feat: resume checkpointed runs from the CLI`

### Task 8: Documentation and Final Acceptance

**Files:**
- Modify: `README.md`
- Modify: `SPEC.md`
- Modify: `项目进程.md`

**Interfaces:**
- Consumes: all implemented behavior and actual verification output.
- Produces: user-facing setup, mode/file layout, exact restore warning, resume examples, Trace field definitions, and final Task4 implementation record.

- [ ] **Step 1: Update README and SPEC from actual behavior**

Document defaults, all six mode values, `.xiliumini` trees, `xiliumini --resume`, exact deletion semantics with preserved control dirs, trace statistics/head-tail policy, and controlled failure codes.

- [ ] **Step 2: Run focused harness suites**

Run: `python -m pytest tests/unit/test_harness_io.py tests/unit/test_checkpoint.py tests/unit/test_trace.py tests/unit/test_agent.py tests/unit/test_workflow.py tests/integration/test_runtime.py tests/integration/test_cli.py -q`

Expected: all focused tests pass.

- [ ] **Step 3: Run complete acceptance verification**

Run: `python -m pytest -q && python -m ruff format --check . && python -m ruff check . && python -m pyright src tests && git diff --check`

Expected: zero failures/errors; only existing platform skips are allowed and must be named in `项目进程.md`.

- [ ] **Step 4: Run local Git smoke without network**

Create a pytest/temp workspace through the public Checkpoint API, save, modify/add/delete ordinary files, load resume inputs, and verify exact restoration plus trace.json/timeline.md generation. Do not run against the repository root.

Expected: restored manifest equals the saved manifest; `.xiliumini` artifacts remain; summary counters match emitted events.

- [ ] **Step 5: Update final project progress record**

Record actual implementation summary, focused/full counts, Ruff/Pyright/diff results, platform skips, and smoke evidence. Do not claim a real interrupted provider run unless one was actually executed.

- [ ] **Step 6: Request independent whole-change review and fix all Critical/Important findings**

Review from the Approval baseline commit through the current working tree, with special focus on destructive restore containment, recorder lifecycle, redaction, strict booleans, and concurrent Runtime isolation. Add failing regression tests before every fix.

- [ ] **Step 7: Re-run complete acceptance after review fixes**

Run the exact Step 3 command again and record the fresh output.

- [ ] **Step 8: Commit final documentation and review fixes**

Commit message: `docs: complete Task4 checkpoint and trace guidance`
