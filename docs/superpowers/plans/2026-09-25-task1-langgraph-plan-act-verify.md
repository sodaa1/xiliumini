# Task1 LangGraph Plan-Act-Verify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current ReAct loop with a LangGraph Planner → Actor → Verifier → Final workflow that creates files and runs real TDD tests and demos inside a session workspace.

**Architecture:** `graph/` owns typed state, model-driven nodes, deterministic action execution, and routing. `core/agent.py` consumes `build_workflow().stream(inputs, stream_mode=["updates", "custom"])` and translates chunks into stable events; Runtime owns workspaces and CLI renders node events.

**Tech Stack:** Python 3.12+, LangGraph, LangChain Core, Pydantic 2, Typer, pytest, Ruff, Pyright, uv.

**Spec:** `docs/superpowers/specs/2026-09-25-task1-langgraph-plan-act-verify-design.md`

## Global Constraints

- Workflow: `START → planner → actor → verifier`, then verifier routes to actor/final, and `final → END`.
- `--max-attempts` defaults to 3 and rejects values below 1.
- Remove old ReAct `ACTOR_PROMPT`; new prompts live in `prompts/task1.py`.
- Actor returns one structured action plan per attempt; actions execute once in order.
- Explicit TDD evidence order is `test_red → implementation → test_green → demo`.
- Commands use argv, `shell=False`, fixed workspace, current Python, timeout, and bounded output.
- Generated Python is not OS-sandboxed; document that it runs with current-user permissions.
- Tests use fake models only; the real Game of Life check is a separate smoke test.
- Before every commit, reread `AGENTS.md` and `项目进程.md`, update actual verification, and align the commit body.

## Review Focus

- Missing red-before-implementation evidence must fail explicit TDD verification.
- Unknown tools and malformed model actions must become failed evidence, not escape the graph.
- `python -c`, non-pytest modules, absolute/traversal scripts, shells, and command connectors are rejected.
- `custom` data cannot be misclassified as `updates` or duplicate CLI events.
- Reused sessions reset graph fields but reuse the same workspace.

---

### Task 1: State and prompts

**Files:**
- Create: `src/xiliumini/graph/__init__.py`
- Create: `src/xiliumini/graph/state.py`
- Create: `src/xiliumini/prompts/task1.py`
- Modify: `src/xiliumini/prompts/__init__.py`, `src/xiliumini/agents/analysis.py`
- Delete: `src/xiliumini/prompts/main.py`, `src/xiliumini/prompts/analysis.py`
- Test: `tests/unit/test_state_and_prompts.py`

**Interfaces:**
- Produces `ActionKind`, `ActionPhase`, `ActionResult`, `GraphStatus`, `GraphState`.
- Produces `PLANNER_NODE_PROMPT`, `ACTOR_NODE_PROMPT`, `VERIFIER_NODE_PROMPT`, `FINAL_PROMPT`, `ANALYSIS_SYSTEM_PROMPT`.

- [ ] Write a failing test that constructs every GraphState field and asserts each prompt contains its JSON contract.
- [ ] Run `uv run pytest tests/unit/test_state_and_prompts.py -q`; expect import failure for `xiliumini.graph`.
- [ ] Implement `GraphState` with `task`, `todo`, `result`, `execution`, `graph_state`, `verification`, `attempt`, `max_attempts`, `final_answer`, `session_id`, `workspace`. `ActionResult` contains `kind`, `phase`, `label`, `ok`, `output`, `exit_code`, `timed_out`, `truncated`.
- [ ] Consolidate prompts in `task1.py`. Actor action JSON requires `kind`, `phase`, `label` and tool `name/args` or command `argv`. Final is a `.format()` template and never calls a model.
- [ ] Move the existing analysis prompt text unchanged, update exports/imports, delete old prompt modules.
- [ ] Run `uv run pytest tests/unit/test_state_and_prompts.py tests/unit/test_delegate_analysis.py -q`; expect PASS.
- [ ] Update project progress, reread repository rules, commit `feat: 定义 Task1 图状态与提示词`.

---

### Task 2: Workspace command tool

**Files:**
- Create: `src/xiliumini/tools/command.py`
- Modify: `src/xiliumini/tools/__init__.py`, `src/xiliumini/errors.py`
- Test: `tests/unit/test_command_tool.py`, `tests/unit/test_current_time.py`

**Interfaces:**
- Produces `CommandResult` and `CommandTool(workspace, timeout_seconds=30, max_output_bytes=20000)`.
- Produces `execute(argv) -> CommandResult`; BaseTool `_run(argv)` returns JSON.

- [ ] Write failing tests for workspace script success, `python -m pytest`, nonzero exit, timeout, truncation, and rejection of PowerShell/cmd/bash, `-c`, non-pytest modules, absolute/traversal scripts, and `&&`/pipes.
- [ ] Run `uv run pytest tests/unit/test_command_tool.py -q`; expect missing-module failure.
- [ ] Implement exact argv validation: first token `python`; allow only `-m pytest` or an existing workspace-relative `.py`; replace executable with `sys.executable`; reject shell tokens.
- [ ] Execute with `subprocess.run(..., cwd=workspace, shell=False, capture_output=True, timeout=..., check=False)`, redact the workspace, bound UTF-8 output, and convert errors/timeouts to CommandResult.
- [ ] Register `command` as the last workspace tool and add `CommandExecutionError(code="command_error")`.
- [ ] Run `uv run pytest tests/unit/test_command_tool.py tests/unit/test_current_time.py tests/unit/test_workspace.py tests/unit/test_file_tools.py -q`; expect PASS.
- [ ] Update progress, reread rules, commit `feat: 添加工作区受限命令工具`.

---

### Task 3: Core nodes

**Files:**
- Create: `src/xiliumini/graph/nodes.py`
- Modify: `src/xiliumini/errors.py`
- Test: `tests/unit/test_nodes.py`

**Interfaces:**
- Produces `planner_node(state, *, model)`, `actor_node(state, *, model, tools)`, `verifier_node(state, *, model)`, `final_node(state)`.
- Produces validated Planner/Actor/Verifier Pydantic schemas and `NodeOutputError(code="node_output_error")`.

- [ ] Write Planner tests for ordered todo, complete prompt input, one invalid-JSON repair call, and sanitized failure after a second invalid response.
- [ ] Run `uv run pytest tests/unit/test_nodes.py -q -k planner`; expect missing-module failure.
- [ ] Implement strict JSON-object parsing, Planner schema, and one repair call.
- [ ] Write Actor tests for sequential file/command execution, custom start/finish events, unknown tool, malformed JSON counting as a failed attempt, and evidence accumulation across retries.
- [ ] Run `uv run pytest tests/unit/test_nodes.py -q -k actor`; expect missing behavior.
- [ ] Implement discriminated `ToolAction`/`CommandAction`; pass task/todo/prior result/verification/tools to model; execute once in order; sanitize exceptions; append evidence; increment attempt once.
- [ ] Write Verifier/Final tests for valid red→implementation→green→demo, missing red, failed/timed-out/truncated final test, nonzero/missing demo, invalid JSON, non-TDD tool success, and Final formatting.
- [ ] Run `uv run pytest tests/unit/test_nodes.py -q -k "verifier or final"`; expect missing behavior.
- [ ] Implement deterministic guards before accepting model `passed`; every Verifier update includes unchanged attempt. Implement Final only with `FINAL_PROMPT.format(...)`.
- [ ] Run `uv run pytest tests/unit/test_nodes.py -q`; expect PASS.
- [ ] Update progress, reread rules, commit `feat: 实现 Task1 三核心节点`.

---

### Task 4: Workflow, event adapter, and Runtime

**Files:**
- Create: `src/xiliumini/graph/workflow.py`
- Modify: `src/xiliumini/graph/__init__.py`, `src/xiliumini/core/agent.py`, `src/xiliumini/core/__init__.py`, `src/xiliumini/events.py`, `src/xiliumini/runtime.py`
- Delete: `src/xiliumini/core/state.py`
- Test: `tests/unit/test_workflow.py`, `tests/unit/test_agent.py`, `tests/integration/test_runtime.py`

**Interfaces:**
- Produces `build_workflow(model, tools, checkpointer=None)` and `route_after_verifier(state)`.
- Produces `stream_agent(model, tools, inputs, checkpointer=None) -> Iterator[RuntimeEvent]`.
- Produces PlannerEvent, ActorEvent, VerifierEvent, ProgressEvent, FinalEvent, ErrorEvent.
- Produces synchronous `Runtime.stream(task, session_id, max_attempts=3)`.

- [ ] Write failing workflow tests using patched deterministic nodes: one-attempt pass, Actor-only retry, failed final at attempt limit, Planner called once.
- [ ] Run `uv run pytest tests/unit/test_workflow.py -q`; expect missing workflow.
- [ ] Build exact graph edges and route to final on passed or exhausted attempts.
- [ ] Replace old Agent tests with a FakeWorkflow yielding both modes; assert exact `.stream(inputs, config={thread_id}, stream_mode=["updates","custom"])`, exact stable events, ignored malformed/unknown chunks, and zero-attempt rejection.
- [ ] Run `uv run pytest tests/unit/test_agent.py -q`; expect old ReAct API failures.
- [ ] Replace ReAct code with stream adapter; remove Token/Tool events, `build_actor`, `MAX_STEPS_MESSAGE`, and RuntimeState.
- [ ] Rewrite Runtime tests for fresh fields per turn, workspace reuse/isolation, invalid session, workspace failure, node error, and provider error.
- [ ] Run `uv run pytest tests/integration/test_runtime.py -q`; expect old async-runtime failures.
- [ ] Implement synchronous Runtime stream, complete fresh GraphState input, workspace tools, stable error mapping, and graph delegation.
- [ ] Run `uv run pytest tests/unit/test_workflow.py tests/unit/test_agent.py tests/integration/test_runtime.py -q`; expect PASS.
- [ ] Update progress, reread rules, commit `feat: 构建 Task1 LangGraph 工作流`.

---

### Task 5: CLI, docs, and acceptance

**Files:**
- Modify: `src/xiliumini/cli/__init__.py`, `src/xiliumini/config.py`, `.env.example`
- Test: `tests/integration/test_cli.py`, `tests/unit/test_config.py`
- Modify: `README.md`, `SPEC.md`, `项目进程.md`

**Interfaces:**
- Produces `ask QUESTION --max-attempts INTEGER` default 3.
- Renders 📋 Planner, 🔧 Actor, ✅/❌ Verifier, 📝 Final; no-stream hides intermediates.

- [ ] Write failing CLI tests for default/explicit/zero attempts, four stage outputs, failed icon, no-stream, and chat default on every turn.
- [ ] Run `uv run pytest tests/integration/test_cli.py -q`; expect old TokenEvent/async failures.
- [ ] Convert CLI to synchronous Runtime iteration; add bounded option; render stable events; preserve Error and Ctrl+C behavior.
- [ ] Remove obsolete `max_steps` config and `.env.example` entry; attempts remain a per-command CLI value.
- [ ] Run `uv run pytest tests/integration/test_cli.py tests/unit/test_config.py -q`; expect PASS.
- [ ] Update README/SPEC with Plan-Act-Verify, graph files, command example, stage output, real Python execution, and the explicit non-sandbox warning.
- [ ] Run `uv run pytest -q`, `uv run ruff format --check .`, `uv run ruff check .`, `uv run pyright`, and `uv run xiliumini ask --help`; record exact results.
- [ ] Run `uv run xiliumini doctor`; if configured, run `uv run xiliumini ask "帮我实现一个 Conway's Game of Life，要求 TDD：先写测试，再写实现，最后跑 demo" --max-attempts 3`.
- [ ] Acceptance requires ✅ Verifier plus workspace files and ordered red/green/demo evidence. Any defect gets a failing regression test before its fix. External configuration failure is recorded and reported, never claimed as success.
- [ ] After successful automated and real acceptance, update progress, reread rules, and commit `feat: 完成 Task1 LangGraph 执行闭环`. Never add `.xiliumini/workspaces/` artifacts.
