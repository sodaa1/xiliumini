# Task3 Layered Memory and Context Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Runtime-owned three-layer memory, proactive Planner context compression, durable session notes/history, and project-wide user preference rules that survive sessions and tasks.

**Architecture:** `Runtime` creates a `MemoryManager` from project-level preferences and the current session workspace, then passes it explicitly through `stream_agent`, workflow, and graph nodes. Planner consumes assembled memory and calls a `before_model` hook before every model invocation; the hook estimates tokens, persists a summary before replacing old messages, and keeps fixed rules, the current task, and recent context. Dedicated stores own `TODO.md`, `NOTEPAD.md`, `HISTORY_SUMMARY.md`, and project-level `USER_PREFERENCES.md`; Agents only access them through Runtime-constructed snapshots and tools.

**Tech Stack:** Python 3.12+, LangGraph, LangChain Core messages/tools, Pydantic Settings, tiktoken, pytest, Ruff, Pyright

**Spec:** `docs/specs/2026-09-29-task3-memory-system-design.md`

## Global Constraints

- Work directly on `main`; do not create a Git worktree.
- All Agent file and command operations stay inside the session workspace and use relative paths.
- Runtime owns Memory assembly and persistence; Agents may only use dedicated Todo, Notepad, and Preference tools.
- Fixed safety rules override current-task instructions; current explicit instructions override saved preferences.
- Session files are `TODO.md`, `NOTEPAD.md`, and `HISTORY_SUMMARY.md`; project preferences are `.xiliumini/USER_PREFERENCES.md` by default.
- Defaults are 64,000 context tokens, compression ratio 0.8, and 8,000 retained tokens.
- Unit and integration tests do not access the network; real-provider validation is a separate smoke check.
- Before every commit, re-read `AGENTS.md` and `项目进程.md`, update the Task3 record with actual verification, and use the same core summary in the commit body.

## Review Focus

- A current task or fixed rules block that alone cannot fit the trigger budget must produce a bounded `memory_error` before invoking the provider, never silently drop either block.
- A corrupt new-format file must remain byte-for-byte unchanged and must not fall back to stale legacy data.
- Unicode-heavy content and an unknown OpenAI-compatible model name must still produce a deterministic token estimate via `cl100k_base`.
- An explicit one-task override must win for that task without mutating the stored long-term preference unless the user explicitly requests persistence.
- A failed or empty compression summary, including a History write failure, must leave the original message list unchanged.

---

### Task 1: Memory Types, Configuration, and Token Dependency

**Files:**
- Create: `src/xiliumini/graph/memory.py`
- Modify: `src/xiliumini/config.py`
- Modify: `src/xiliumini/graph/state.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.env.example`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_state_and_prompts.py`
- Create: `tests/unit/test_memory.py`

**Interfaces:**
- Consumes: existing `GraphState`, Pydantic settings, and JSON-compatible state conventions.
- Produces: `UserPreference`, `RulesLayer`, `WorkingMemory`, `CompressionEvent`, `HistorySummaryLayer`, `LayeredMemory`, `MemoryLimits`; Settings fields `context_window_tokens`, `compression_trigger_ratio`, `compression_keep_tokens`.

- [ ] **Step 1: Write failing configuration and type-contract tests**

Add tests asserting defaults `64000`, `0.8`, and `8000`; environment overrides; rejection of zero/negative context, ratio outside `(0, 1)`, and keep tokens greater than or equal to the trigger budget. Add a JSON serialization test for a literal `LayeredMemory`, and update the canonical fake `GraphState` fixture with every new field from the spec.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_state_and_prompts.py tests/unit/test_memory.py -q`

Expected: FAIL because the settings and memory types do not exist.

- [ ] **Step 3: Add the typed contracts and validated settings**

In `graph/memory.py`, define the TypedDicts plus an immutable `MemoryLimits(context_window_tokens: int, trigger_ratio: float, keep_tokens: int)`. In `config.py`, add the three environment-backed fields and cross-field validation. Add `tiktoken>=0.12,<1` as a direct dependency and refresh the lockfile. Extend `GraphState` with `memory`, `current_node`, `plan_summary`, `acceptance_criteria`, `agent_handoffs`, `code_agent_summary`, `verifier_summary`, `last_error`, `context_summary`, and `compression_events`.

- [ ] **Step 4: Run tests and static checks for GREEN**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_state_and_prompts.py tests/unit/test_memory.py -q`

Run: `uv run pyright src/xiliumini/config.py src/xiliumini/graph/state.py src/xiliumini/graph/memory.py`

Expected: focused tests pass and Pyright reports zero errors.

- [ ] **Step 5: Update progress and commit**

Record actual Task 1 results in `项目进程.md`, re-read both required files, then commit the types, configuration, dependency, tests, and progress entry with `feat: add Task3 memory contracts`.

### Task 2: Project-wide User Preference Store and Tool

**Files:**
- Create: `src/xiliumini/tools/preferences.py`
- Modify: `src/xiliumini/tools/__init__.py`
- Create: `tests/unit/test_preferences.py`

**Interfaces:**
- Consumes: `UserPreference`, existing atomic UTF-8 helpers, Runtime `data_dir`.
- Produces: `UserPreferenceStore(data_dir: Path, *, clock: Callable[[], datetime] | None = None)`, `read()`, `upsert(key, content)`, `remove(key)`, and `PreferenceWriteTool(store=...)` with action `upsert|remove`.

- [ ] **Step 1: Write failing store behavior tests**

Test an empty store, exact Markdown fenced JSON schema, stable-key upsert without duplication, remove, cross-instance recovery, 100-entry and 2,000-character bounds, and atomic preservation after malformed new-format input. Assert the corrupt file remains byte-for-byte unchanged.

- [ ] **Step 2: Write failing safety and tool tests**

Test rejection of blank/invalid keys, blank upsert content, missing remove key, and content containing `sk-`, `api_key`, `token=`, or `password=`. Assert the tool returns structured errors without an absolute path or submitted secret, and that ordinary file tools do not expose this project-level file.

- [ ] **Step 3: Run tests and confirm RED**

Run: `uv run pytest tests/unit/test_preferences.py -q`

Expected: FAIL because `xiliumini.tools.preferences` does not exist.

- [ ] **Step 4: Implement the store and dedicated Planner tool**

Use `USER_PREFERENCES.md`, schema version `1`, stable slug keys of at most 100 characters, at most 100 records, content at most 2,000 characters, UTC ISO timestamps, and existing atomic writes. `PreferenceWriteTool` delegates all validation and persistence to the store and never accepts a path.

- [ ] **Step 5: Run tests and static checks for GREEN**

Run: `uv run pytest tests/unit/test_preferences.py -q`

Run: `uv run ruff check src/xiliumini/tools/preferences.py tests/unit/test_preferences.py`

Expected: all preference tests and lint pass.

- [ ] **Step 6: Update progress and commit**

Record actual Task 2 results, re-read the required files, and commit with `feat: add project user preferences`.

### Task 3: Session Memory Files and Legacy Migration

**Files:**
- Modify: `src/xiliumini/tools/todo.py`
- Modify: `src/xiliumini/tools/notepad.py`
- Modify: `src/xiliumini/graph/memory.py`
- Modify: `tests/unit/test_todo_tools.py`
- Modify: `tests/unit/test_notepad_and_memory.py`
- Modify: `tests/unit/test_memory.py`

**Interfaces:**
- Consumes: existing Todo/Notepad tools and atomic workspace helpers.
- Produces: `TodoStore` backed by `TODO.md`; Notepad backed by `NOTEPAD.md`; `HistorySummaryStore(workspace)` with `read()` and `write(summary)` for `HISTORY_SUMMARY.md`.

- [ ] **Step 1: Write failing canonical-file tests**

Assert Todo writes schema-versioned fenced JSON to `TODO.md`, Notepad appends to `NOTEPAD.md`, and History read/write uses `HISTORY_SUMMARY.md`. Existing Todo transition and file-size tests must continue to exercise real stores.

- [ ] **Step 2: Write failing migration and corruption tests**

Create valid legacy `.xiliumini/todos.json` and `.xiliumini/notepad.md`, then assert first access atomically creates the canonical file and leaves the legacy file untouched. Assert canonical files win when both exist. For malformed canonical or legacy input, assert an error and byte-for-byte preservation rather than fallback or overwrite.

- [ ] **Step 3: Run tests and confirm RED**

Run: `uv run pytest tests/unit/test_todo_tools.py tests/unit/test_notepad_and_memory.py tests/unit/test_memory.py -q -k "todo or notepad or history or migration"`

Expected: FAIL because stores still use legacy paths and History store is missing.

- [ ] **Step 4: Implement canonical formats and one-way migration**

Centralize fenced JSON encode/decode helpers in `graph/memory.py` without permitting arbitrary paths. Migration validates old content before writing new content; new-format corruption never falls back. Preserve all current Todo transition rules and Notepad append limits.

- [ ] **Step 5: Run store regression tests for GREEN**

Run: `uv run pytest tests/unit/test_todo_tools.py tests/unit/test_notepad_and_memory.py tests/unit/test_memory.py -q`

Expected: all session persistence and migration tests pass.

- [ ] **Step 6: Update progress and commit**

Record actual Task 3 results, re-read the required files, and commit with `feat: migrate Task3 memory files`.

### Task 4: Layered Memory Assembly

**Files:**
- Modify: `src/xiliumini/graph/memory.py`
- Modify: `src/xiliumini/memory.py`
- Modify: `tests/unit/test_memory.py`
- Modify: `tests/unit/test_notepad_and_memory.py`

**Interfaces:**
- Consumes: canonical session stores, `UserPreferenceStore`, and complete `GraphState`.
- Produces: `MemoryManager(workspace, preference_store, limits, *, model_name)` and `assemble(state, *, current_node) -> LayeredMemory`.

- [ ] **Step 1: Write failing three-layer assembly tests**

Assert exact fixed rules, loaded project preferences, current node/task/session, plan/todo/acceptance state, deduplicated source order, last 6 handoffs, code/verifier summaries, last error, `{current, max}` attempts, bounded Notepad/History summaries, prior context summary, and last 3 compression events.

- [ ] **Step 2: Write failing isolation and precedence tests**

Assert mutation of the returned snapshot cannot mutate state; secrets, model objects, absolute workspace paths, and full tool arguments are absent after JSON serialization. Create a saved preference conflicting with the current task and assert both are represented separately with precedence encoded in fixed rules, while the preference file remains unchanged.

- [ ] **Step 3: Run tests and confirm RED**

Run: `uv run pytest tests/unit/test_memory.py tests/unit/test_notepad_and_memory.py -q -k "layer or snapshot or precedence"`

Expected: FAIL because `MemoryManager.assemble` is not implemented.

- [ ] **Step 4: Implement bounded assembly and retire the old snapshot implementation**

Implement fixed rules in code, copy JSON-safe state values, preserve first-seen source order, slice handoffs/events from the tail, and cap Notepad/History summaries to their latest 8,000 characters on UTF-8 boundaries. Move consumers toward `graph.memory`; keep `src/xiliumini/memory.py` as a temporary deprecation re-export only until Task 6 removes the final import.

- [ ] **Step 5: Run tests and static checks for GREEN**

Run: `uv run pytest tests/unit/test_memory.py tests/unit/test_notepad_and_memory.py -q`

Run: `uv run pyright src/xiliumini/graph/memory.py src/xiliumini/memory.py`

Expected: all assembly tests pass and Pyright reports zero errors.

- [ ] **Step 6: Update progress and commit**

Record actual Task 4 results, re-read the required files, and commit with `feat: assemble layered runtime memory`.

### Task 5: Token Estimation and Lossless Planner Compression

**Files:**
- Modify: `src/xiliumini/graph/memory.py`
- Modify: `src/xiliumini/agents/react.py`
- Modify: `tests/unit/test_memory.py`
- Create: `tests/unit/test_react_memory.py`

**Interfaces:**
- Consumes: LangChain `BaseMessage`, `MemoryLimits`, `HistorySummaryStore`, unbound chat model.
- Produces: `estimate_message_tokens(messages, model_name) -> int`, `MemoryManager.prepare_planner_messages(messages, state, *, model) -> list[BaseMessage]`, and optional `before_model: Callable[[list[BaseMessage]], list[BaseMessage]]` on `run_react`.

- [ ] **Step 1: Write failing token boundary tests**

Use literal English and Chinese message fixtures. Assert deterministic positive counts, known/unknown model behavior, no compression one token below the trigger, and compression exactly at and above the trigger. Inject a deterministic token counter into `MemoryManager` for threshold tests; keep one real-tiktoken fallback test.

- [ ] **Step 2: Write failing message retention and persistence tests**

Assert compression preserves the first system message, full current-task HumanMessage, and newest messages within 8,000 tokens; replaces older messages with one history-summary SystemMessage; writes History before returning replacements; updates `context_summary`; and retains only three events with correct before/after counts and attempt.

- [ ] **Step 3: Write failing no-loss and impossible-budget tests**

For model exception, empty summary, and History write failure, assert the returned list equals the original list and no state event is appended. For fixed rules/current task alone exceeding the trigger budget, assert `MemoryBudgetError` is raised before the regular Planner model is called.

- [ ] **Step 4: Write failing per-invocation hook test**

Run a scripted two-tool-call ReAct loop and assert `before_model` is called before all three model invocations with the progressively growing message list; no callback preserves Task2 behavior.

- [ ] **Step 5: Run tests and confirm RED**

Run: `uv run pytest tests/unit/test_memory.py tests/unit/test_react_memory.py -q -k "token or compress or before_model or budget"`

Expected: FAIL because token estimation, compression, and the hook are missing.

- [ ] **Step 6: Implement token estimation, transactional compression, and hook invocation**

Use `tiktoken.encoding_for_model` with `cl100k_base` fallback; include role/content/tool call JSON plus fixed overhead. Compression invokes the unbound model with a bounded summarization prompt, validates nonblank text, persists History, then returns a new list without mutating the original. `run_react` calls the callback immediately before every `bound.invoke`.

- [ ] **Step 7: Run tests and static checks for GREEN**

Run: `uv run pytest tests/unit/test_memory.py tests/unit/test_react_memory.py -q`

Run: `uv run ruff check src/xiliumini/graph/memory.py src/xiliumini/agents/react.py tests/unit/test_memory.py tests/unit/test_react_memory.py`

Expected: all compression and ReAct tests pass with clean lint.

- [ ] **Step 8: Update progress and commit**

Record actual Task 5 results, re-read the required files, and commit with `feat: compress planner context near token limit`.

### Task 6: Planner, Specialist, and Verifier Memory Integration

**Files:**
- Modify: `src/xiliumini/graph/nodes.py`
- Modify: `src/xiliumini/tools/subagent_tools.py`
- Modify: `src/xiliumini/agents/code_agent.py`
- Modify: `src/xiliumini/prompts/task1.py`
- Modify: `src/xiliumini/tools/__init__.py`
- Delete: `src/xiliumini/memory.py`
- Modify: `tests/unit/test_nodes.py`
- Modify: `tests/unit/test_code_agent.py`
- Modify: `tests/unit/test_state_and_prompts.py`

**Interfaces:**
- Consumes: `MemoryManager`, `PreferenceWriteTool`, expanded GraphState, and `run_react(before_model=...)`.
- Produces: `planner_node(state, *, model, memory_manager, max_loops=8)`, `verifier_node(state, *, model, memory_manager)`, bounded handoff/state summaries, and expanded `PlannerOutput`.

- [ ] **Step 1: Write failing Planner memory and output tests**

Assert the first Planner HumanMessage contains one `memory` object rather than duplicate loose state; bound tools include `preference_write`; output parses `summary`, `plan_summary`, `acceptance_criteria`, and readiness; the compression hook is supplied to `run_react`; malformed output still follows the existing single repair attempt.

- [ ] **Step 2: Write failing preference-intent tests**

Script explicit “remember this default” and “forget this preference” tool calls and assert persistence. For an ordinary one-task instruction and a current-task override, assert no preference tool call and no mutation of `USER_PREFERENCES.md`.

- [ ] **Step 3: Write failing handoff and node-state tests**

Exercise seven delegations and assert only the latest six handoffs remain in order. Assert code results update `code_agent_summary`, failures update `last_error`, Verifier updates `verifier_summary/current_node`, successful later evidence clears only the repaired error, and every returned memory snapshot reflects the node update.

- [ ] **Step 4: Write failing codeAgent memory test**

Assert codeAgent receives the Runtime-assembled memory from state, can use canonical Notepad tools, and no longer imports or calls the old root `build_memory_snapshot`.

- [ ] **Step 5: Run tests and confirm RED**

Run: `uv run pytest tests/unit/test_nodes.py tests/unit/test_code_agent.py tests/unit/test_state_and_prompts.py -q`

Expected: FAIL because nodes do not accept `MemoryManager` or maintain the new state.

- [ ] **Step 6: Integrate nodes, tools, prompts, and specialists**

Extend Planner's exact JSON contract, add preference guidance and precedence rules to the prompt, pass the compression callback, update SupervisorContext state with bounded handoffs/summaries, assemble memory after node state changes, and remove the obsolete root memory module once all imports use `xiliumini.graph.memory`.

- [ ] **Step 7: Run node regression tests for GREEN**

Run: `uv run pytest tests/unit/test_nodes.py tests/unit/test_code_agent.py tests/unit/test_search_agent.py tests/unit/test_state_and_prompts.py -q`

Expected: all node and specialist tests pass.

- [ ] **Step 8: Update progress and commit**

Record actual Task 6 results, re-read the required files, and commit with `feat: integrate layered memory into graph nodes`.

### Task 7: Runtime, Workflow, and Cross-task Recovery

**Files:**
- Modify: `src/xiliumini/runtime.py`
- Modify: `src/xiliumini/core/agent.py`
- Modify: `src/xiliumini/graph/workflow.py`
- Modify: `src/xiliumini/errors.py`
- Modify: `tests/unit/test_agent.py`
- Modify: `tests/unit/test_workflow.py`
- Modify: `tests/integration/test_runtime.py`

**Interfaces:**
- Consumes: Settings memory limits, `UserPreferenceStore`, `MemoryManager`, integrated node signatures.
- Produces: explicit `memory_manager` propagation from Runtime to graph nodes and stable `memory_error` mapping.

- [ ] **Step 1: Write failing complete-state and propagation tests**

Update Runtime state assertions for every new field. Assert one manager instance is passed Runtime → `stream_agent` → workflow → Planner/Verifier/Final, and no ContextVar or global manager leaks between interleaved stream events.

- [ ] **Step 2: Write failing cross-task and cross-session preference tests**

Persist one preference through the dedicated store in session A; start later tasks in sessions A and B with the same Runtime data directory; assert both initial Rules Layers include it. Remove the preference and assert the next task in both sessions omits it.

- [ ] **Step 3: Write failing Runtime error tests**

Assert corrupt `USER_PREFERENCES.md`, impossible fixed/current-task token budget, and session History corruption become secret-free `ErrorEvent(code="memory_error", ...)` before a provider call, with source files unchanged.

- [ ] **Step 4: Run tests and confirm RED**

Run: `uv run pytest tests/unit/test_agent.py tests/unit/test_workflow.py tests/integration/test_runtime.py -q`

Expected: FAIL because Runtime does not construct or propagate MemoryManager.

- [ ] **Step 5: Implement explicit Runtime ownership and error mapping**

Build limits from Settings in `create_runtime`; construct the preference store from `data_dir` and manager from the session workspace; initialize the complete GraphState; pass the manager through all graph boundaries; add a typed `MemorySystemError`/`MemoryBudgetError` family whose messages never expose content or paths.

- [ ] **Step 6: Run integration and CLI regression tests for GREEN**

Run: `uv run pytest tests/unit/test_agent.py tests/unit/test_workflow.py tests/integration/test_runtime.py tests/integration/test_cli.py -q`

Expected: Runtime, workflow, and CLI tests pass.

- [ ] **Step 7: Update progress and commit**

Record actual Task 7 results, re-read the required files, and commit with `feat: make runtime own layered memory`.

### Task 8: Documentation, Full Verification, and Real Smoke

**Files:**
- Modify: `README.md`
- Modify: `SPEC.md`
- Modify: `.env.example`
- Modify: `项目进程.md`

**Interfaces:**
- Consumes: completed Task3 behavior and actual verification output.
- Produces: user-facing configuration, architecture, persistence, precedence, migration, and recovery documentation.

- [ ] **Step 1: Update user and technical documentation**

Document the three layers; canonical session files; project-level preferences; explicit preference recording/removal; rule precedence; token defaults and environment variables; proactive compression; legacy migration; and the distinction between Runtime-managed Memory and Agent tools.

- [ ] **Step 2: Run focused Task3 tests**

Run: `uv run pytest tests/unit/test_memory.py tests/unit/test_preferences.py tests/unit/test_react_memory.py tests/unit/test_notepad_and_memory.py tests/unit/test_todo_tools.py tests/unit/test_nodes.py tests/integration/test_runtime.py -q`

Expected: all Task3 tests pass with no warnings introduced by the change.

- [ ] **Step 3: Run the full automated suite**

Run: `uv run pytest -q`

Expected: all tests pass; existing Windows symlink skips may remain and must be reported by exact count.

- [ ] **Step 4: Run formatting, lint, type, and diff checks**

Run: `uv run ruff format --check .`

Run: `uv run ruff check .`

Run: `uv run pyright`

Run: `git diff --check`

Expected: every command exits 0; Pyright reports zero errors.

- [ ] **Step 5: Run bounded CLI checks and optional real-provider smoke**

Run: `uv run xiliumini ask --help` and `uv run xiliumini doctor`. If local provider configuration is available, use test-only low limits to force one Planner compression, then confirm Planner continues, `HISTORY_SUMMARY.md` is created, and the final task completes. Do not claim the real smoke passed if credentials or provider availability prevent it.

- [ ] **Step 6: Complete Task3 progress record and final commit**

Write only actual commands and results into `项目进程.md`, including any skipped real smoke. Re-read `AGENTS.md` and the progress file, then commit documentation and final verification with `docs: complete Task3 memory system`.

## Self-review Result

- Spec coverage: all fixed Rules, Working Memory fields, History Summary fields, four persistence files, preference precedence, compression behavior, migration, Runtime ownership, node changes, errors, docs, and verification map to Tasks 1–8.
- Step scan: each task has a failing test, observed RED command, one bounded implementation step, GREEN verification, and a progress-aware commit.
- Type consistency: `MemoryManager`, `UserPreferenceStore`, `LayeredMemory`, node signatures, and `before_model` names are identical across producer and consumer tasks.
- Review Focus: all five listed failure modes are assigned concrete assertions in Tasks 2, 3, 4, 5, 6, or 7.
- Proportion: the plan fixes interfaces and observable assertions without prescribing full function bodies.
