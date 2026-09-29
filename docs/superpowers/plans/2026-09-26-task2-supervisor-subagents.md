# Task2 Supervisor 与专业子 Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 Planner → Actor 工作流改造成能维护 todo、调用 searchAgent 与 codeAgent、根据 Verifier 反馈重试的 Planner Supervisor。

**Architecture:** Planner 通过三个 LangChain 工具在 bounded ReAct 循环中写 todo、委派搜索和委派代码；两个子 Agent 各自建立模型上下文并返回可序列化摘要与工具证据。LangGraph 只保留 Planner、Verifier、Final 三个节点，Verifier 失败时回到 Planner，todo/notepad 使用 session workspace 内的持久文件。

**Tech Stack:** Python 3.12+、LangGraph、LangChain Core、Pydantic 2、`tavily-python>=0.8,<0.9`、pytest、Ruff、Pyright、uv。

**Spec:** `docs/superpowers/specs/2026-09-26-task2-supervisor-subagents-design.md`

## Global Constraints

- Python 保持 `>=3.12`；现有 LangChain/LangGraph/Pydantic 主版本范围不变。
- Tavily 使用官方 `tavily-python` 的 `TavilyClient.search()`；即使 SDK 支持 keyless mode，本项目仍必须在缺少 `TAVILY_API_KEY` 时返回 `{ok: False, error: "missing TAVILY_API_KEY"}`。
- 所有 workspace 文件路径必须继续拒绝绝对路径、父目录穿越、Windows 盘符、UNC 和符号链接逃逸。
- `BashTool` 必须使用 argv、`shell=False`、固定 workspace、超时和共享输出上限；不得增加任意字符串 Shell。
- 模型、工具实例、API key 和 SecretStr 不得写入 GraphState、事件、持久化 todo 或 notepad。
- searchAgent 与 codeAgent 的完整消息仅存在于各自函数返回值中，GraphState 只保存可序列化摘要、来源和证据。
- Prompt 常量继续集中在 `src/xiliumini/prompts/task1.py`；用户提供的 SEARCH_AGENT_PROMPT 和 CODE_AGENT_PROMPT 规则不得弱化。
- 每个生产代码行为必须先有失败测试；开始写测试前阅读 `superpowers:test-driven-development/references/writing-good-tests.md`。
- 每次提交前重新阅读 `AGENTS.md` 和 `项目进程.md`，更新 Task2 的实际摘要与验证结果，且提交正文只描述该提交包含的内容。

## Review Focus

- Tavily 返回 `None`、非列表 results 或缺少 title/url/content/score 时必须得到安全的标准化结果；Task 4 覆盖。
- 模型给出未知工具、缺失 tool-call ID 或无效 args 时必须形成受控 ToolMessage/失败事件而非 traceback；Tasks 5、6、7 覆盖。
- `.xiliumini/todos.json` 已损坏时不得静默覆盖用户数据，必须返回脱敏持久化错误；Task 2 覆盖。
- Verifier 失败后的新 Supervisor attempt 必须复用已完成 todo，且确定性校验只把未修复的最新必要失败视为阻塞；Tasks 7、8 覆盖。
- codeAgent 在完成 todo 后又出现失败、超时或截断检查时，Verifier 仍必须拒绝通过；Tasks 6、8 覆盖。

---

### Task 1: 共享状态与 Prompt 契约

**Files:**
- Modify: `src/xiliumini/graph/state.py`
- Modify: `src/xiliumini/prompts/task1.py`
- Modify: `src/xiliumini/prompts/__init__.py`
- Modify: `tests/unit/test_state_and_prompts.py`

**Interfaces:**
- Produces `TodoStatus`, `TodoDraft`, `TodoItem`, `ResearchNote`, `ToolEvent`, `AgentResult`, `GraphState`。
- `TodoDraft={id, content}`; `TodoItem={id, content, status, note}`; `ResearchNote={summary, queries, sources, attempt}`。
- `ToolEvent={agent, tool, args, output, ok, attempt, phase}` where `phase` may be `None`; `AgentResult={agent, instruction, ok, summary, attempt}`。
- Produces `PLANNER_NODE_PROMPT`, `SEARCH_AGENT_PROMPT`, `CODE_AGENT_PROMPT`, `VERIFIER_NODE_PROMPT`, `FINAL_PROMPT`。
- `GraphState` fields: `task`, `todos`, `research_notes`, `agent_results`, `tool_events`, `result`, `graph_state`, `verification`, `attempt`, `max_attempts`, `final_answer`, `session_id`, `workspace`。

- [ ] **Step 1: 写失败的状态与 Prompt 测试**

把 `tests/unit/test_state_and_prompts.py` 改为构造完整新状态，并断言：todo 状态包含四个精确值；Planner prompt 提到 `TodoWriteTool`、`CallSearchAgentTool`、`CallCodeAgentTool`；两个子 Agent prompt 包含用户要求的工具和规则；旧 `ACTOR_NODE_PROMPT` 不再导出。

- [ ] **Step 2: 运行测试并确认因新类型/常量不存在而失败**

Run: `uv run pytest tests/unit/test_state_and_prompts.py -q`
Expected: FAIL on missing `TodoItem` or `SEARCH_AGENT_PROMPT` import.

- [ ] **Step 3: 实现最小状态类型与 Prompt 常量**

在 `state.py` 定义 TypedDict 和 Literal；在 `task1.py` 放入用户给定的 search/code prompt，并写严格 Supervisor/Verifier JSON 协议；更新 `prompts/__init__.py` 导出，删除 Actor prompt。

- [ ] **Step 4: 运行聚焦测试**

Run: `uv run pytest tests/unit/test_state_and_prompts.py -q`
Expected: PASS.

- [ ] **Step 5: 更新进程记录并提交**

重新读取仓库规则，向 `项目进程.md` 的 Task2 追加本任务真实结果。

```powershell
git add src/xiliumini/graph/state.py src/xiliumini/prompts/task1.py src/xiliumini/prompts/__init__.py tests/unit/test_state_and_prompts.py 项目进程.md
git commit -m "feat: 定义 Task2 Supervisor 状态与提示词"
```

### Task 2: Todo、Notepad 与 layered memory 边界

**Files:**
- Create: `src/xiliumini/tools/todo.py`
- Create: `src/xiliumini/tools/notepad.py`
- Create: `src/xiliumini/memory.py`
- Create: `tests/unit/test_todo_tools.py`
- Create: `tests/unit/test_notepad_and_memory.py`
- Modify: `src/xiliumini/tools/__init__.py`

**Interfaces:**
- Consumes `TodoItem`, `TodoStatus`, `GraphState` from Task 1 and workspace helpers from `tools/workspace.py`。
- Produces `TodoStore(workspace: Path).read() -> list[TodoItem]`, `.write(items: list[TodoDraft]) -> list[TodoItem]`, `.update(todo_id: str, status: TodoStatus, note: str | None = None) -> list[TodoItem]`。
- Produces `TodoWriteTool(workspace: Path)` named `todo_write`, input `todos: list[{id, content}]`; same id/content preserves status, new ids start `pending`, omitted ids are removed only when not `in_progress`。
- Produces `TodoUpdateTool(workspace: Path)` named `todo_update`, input `todo_id`, `status`, optional `note`。
- Produces `NotepadReadTool(workspace: Path)` with no input, `NotepadAppendTool(workspace: Path)` with `entry: str`, and `build_memory_snapshot(state: GraphState) -> dict[str, object]`。

- [ ] **Step 1: 写失败的 Todo store/tool 测试**

覆盖稳定 ID、原子写入、`pending → in_progress → completed`、`pending/in_progress → blocked`、禁止 `pending → completed`、未知 ID、blocked 必须有说明、损坏 JSON 不被覆盖。

- [ ] **Step 2: 运行 Todo 测试并确认缺少模块失败**

Run: `uv run pytest tests/unit/test_todo_tools.py -q`
Expected: FAIL with `ModuleNotFoundError: xiliumini.tools.todo`.

- [ ] **Step 3: 实现 TodoStore、TodoWriteTool 和 TodoUpdateTool**

持久文件固定为 `.xiliumini/todos.json`；使用 workspace 路径校验与原子 UTF-8 写入。工具输出 JSON，至少包含 `ok`、`todos`，失败时包含脱敏 `error`。

- [ ] **Step 4: 运行 Todo 测试并确认通过**

Run: `uv run pytest tests/unit/test_todo_tools.py -q`
Expected: PASS.

- [ ] **Step 5: 写失败的 Notepad 与 memory 测试**

覆盖首次读取为空、逐行追加不覆盖、大小上限、memory snapshot 包含 session/todos/research/notepad，且不包含模型或 API key。

- [ ] **Step 6: 运行测试并确认缺少行为失败**

Run: `uv run pytest tests/unit/test_notepad_and_memory.py -q`
Expected: FAIL on missing `NotepadReadTool` or `build_memory_snapshot`.

- [ ] **Step 7: 实现 Notepad 工具与 memory snapshot 接口**

Notepad 固定为 `.xiliumini/notepad.md`；Append 使用原子重写实现追加，复用 1 MB 上限。snapshot 只返回 JSON-compatible 值。

- [ ] **Step 8: 运行本任务测试**

Run: `uv run pytest tests/unit/test_todo_tools.py tests/unit/test_notepad_and_memory.py -q`
Expected: PASS.

- [ ] **Step 9: 更新进程记录并提交**

```powershell
git add src/xiliumini/tools/todo.py src/xiliumini/tools/notepad.py src/xiliumini/memory.py src/xiliumini/tools/__init__.py tests/unit/test_todo_tools.py tests/unit/test_notepad_and_memory.py 项目进程.md
git commit -m "feat: 添加 Todo 与 layered memory 持久化边界"
```

### Task 3: codeAgent 工作区工具集与 BashTool

**Files:**
- Modify: `src/xiliumini/tools/bash_tool.py`
- Modify: `src/xiliumini/tools/command.py`
- Modify: `src/xiliumini/tools/__init__.py`
- Create: `tests/unit/test_bash_tool.py`
- Modify: `tests/unit/test_command_tool.py`

**Interfaces:**
- Consumes existing `CommandResult`, file tools, Task 2 Notepad tools。
- Produces `BashTool(workspace: Path, timeout_seconds=30, max_output_bytes=20_000)` named `bash`, input `argv: list[str]` and `phase: Literal["test_red", "test_green", "demo", "check", "other"]`。
- Produces `build_tools(state: GraphState) -> list[BaseTool]` with file read/write/edit, grep, Bash, notepad read/append；不包含 TodoUpdateTool。

- [ ] **Step 1: 写失败的 BashTool 和 build_tools 测试**

断言 argv + phase、`shell=False`、cwd 为 workspace、Python/pytest/Ruff/Pyright allowlist、Shell token 拒绝、workspace 外测试目标拒绝、超时与 stdout/stderr 共享上限，以及工具名称集合精确匹配。

- [ ] **Step 2: 运行并确认旧占位实现导致失败**

Run: `uv run pytest tests/unit/test_bash_tool.py tests/unit/test_command_tool.py -q`
Expected: FAIL because `BashTool` and expanded validation do not exist.

- [ ] **Step 3: 实现 BashTool 并抽取/扩展安全命令解析**

保留 `CommandTool` 兼容；共享命令解析只允许：workspace Python 文件、`python -m pytest`、`python -m ruff check`、`python -m ruff format --check`、`python -m pyright`。所有目标和配置路径须经 workspace 校验。

- [ ] **Step 4: 实现 build_tools**

在 `tools/__init__.py` 导出 `build_tools(state)`，只从 `state["workspace"]` 构造工具，不读取全局 cwd。

- [ ] **Step 5: 运行本任务测试**

Run: `uv run pytest tests/unit/test_bash_tool.py tests/unit/test_command_tool.py -q`
Expected: PASS.

- [ ] **Step 6: 更新进程记录并提交**

```powershell
git add src/xiliumini/tools/bash_tool.py src/xiliumini/tools/command.py src/xiliumini/tools/__init__.py tests/unit/test_bash_tool.py tests/unit/test_command_tool.py 项目进程.md
git commit -m "feat: 为 codeAgent 添加受限 Bash 工具集"
```

### Task 4: Tavily WebSearchTool 与配置文档

**Files:**
- Create: `src/xiliumini/tools/web_search_tool.py`
- Create: `tests/unit/test_web_search_tool.py`
- Modify: `src/xiliumini/tools/__init__.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.env.example`

**Interfaces:**
- Produces `WebSearchTool()` named `web_search` with input `query: str`, `max_results: int = 5` bounded to 1..10。
- `_run()` returns JSON text representing `{ok, query, answer?, results?, error?}`；每个 result 精确包含 `title`, `url`, `content`, `score`。

- [ ] **Step 1: 写失败的 WebSearchTool 测试**

覆盖缺 key 的精确错误、调用 `TavilyClient.search(query=..., max_results=...)`、正常结果、缺字段、`results=None`、非列表 results、异常脱敏且不泄漏 key。

- [ ] **Step 2: 运行测试并确认模块不存在**

Run: `uv run pytest tests/unit/test_web_search_tool.py -q`
Expected: FAIL with `ModuleNotFoundError: xiliumini.tools.web_search_tool`.

- [ ] **Step 3: 添加 Tavily 依赖并刷新锁文件**

Run: `uv add "tavily-python>=0.8,<0.9"`
Expected: `pyproject.toml` and `uv.lock` contain resolved Tavily SDK dependency.

- [ ] **Step 4: 实现 WebSearchTool**

按调用时读取 `os.getenv("TAVILY_API_KEY")`；存在 key 才构造 `TavilyClient(api_key=key)`。对响应做类型检查与标准化，异常只返回 `search failed` 类脱敏错误。

- [ ] **Step 5: 更新配置模板并运行测试**

在 `.env.example` 添加空的 `TAVILY_API_KEY=`。

Run: `uv run pytest tests/unit/test_web_search_tool.py -q`
Expected: PASS without real network requests.

- [ ] **Step 6: 更新进程记录并提交**

```powershell
git add src/xiliumini/tools/web_search_tool.py src/xiliumini/tools/__init__.py tests/unit/test_web_search_tool.py pyproject.toml uv.lock .env.example 项目进程.md
git commit -m "feat: 添加 Tavily WebSearchTool"
```

### Task 5: searchAgent ReAct 循环

**Files:**
- Create: `src/xiliumini/agents/react.py`
- Create: `src/xiliumini/agents/search_agent.py`
- Modify: `src/xiliumini/agents/__init__.py`
- Create: `tests/unit/test_search_agent.py`

**Interfaces:**
- Consumes `SEARCH_AGENT_PROMPT`, `WebSearchTool`, `load_settings()`, `create_chat_model()`。
- Produces `run_search_agent(state, instruction, *, writer=None, max_loops=4) -> dict` with keys `ok`, `summary`, `queries`, `sources`, `messages`, `tool_events`。
- `agents/react.py` produces shared content normalization and safe tool-call execution helpers; it must never serialize secrets or traceback text。

- [ ] **Step 1: 写失败的 searchAgent 测试**

用可 bind 的 scripted model 覆盖：仅绑定 WebSearchTool；HumanMessage 含 task/instruction/research；同一 AIMessage 的多个 tool calls 逐个执行；ToolMessage 使用对应 ID；query/source 去重保序；writer 发出 `tool_call` 和 `search_results`；无效 args、未知工具、缺失 ID、缺 key、max loops 都返回受控结果。

- [ ] **Step 2: 运行并确认模块不存在**

Run: `uv run pytest tests/unit/test_search_agent.py -q`
Expected: FAIL with `ModuleNotFoundError: xiliumini.agents.search_agent`.

- [ ] **Step 3: 实现共享 ReAct helper 与 run_search_agent**

严格保留公开签名。测试通过 monkeypatch agent 模块中的 `load_settings`、`create_chat_model` 和 `WebSearchTool` 注入 fake，不增加 model 参数到公开接口。

- [ ] **Step 4: 运行 searchAgent 测试**

Run: `uv run pytest tests/unit/test_search_agent.py -q`
Expected: PASS.

- [ ] **Step 5: 更新进程记录并提交**

```powershell
git add src/xiliumini/agents/react.py src/xiliumini/agents/search_agent.py src/xiliumini/agents/__init__.py tests/unit/test_search_agent.py 项目进程.md
git commit -m "feat: 实现 searchAgent ReAct 搜索循环"
```

### Task 6: codeAgent ReAct 循环

**Files:**
- Create: `src/xiliumini/agents/code_agent.py`
- Modify: `src/xiliumini/agents/__init__.py`
- Create: `tests/unit/test_code_agent.py`

**Interfaces:**
- Consumes `CODE_AGENT_PROMPT`, `build_tools(state)`, `TodoUpdateTool`, `TodoStore`, `build_memory_snapshot(state)` and Task 5 ReAct helpers。
- Produces `run_code_agent(state, instruction, *, writer=None, max_loops=10) -> dict` with keys `ok`, `summary`, `todos`, `messages`, `tool_events`。

- [ ] **Step 1: 写失败的 codeAgent 测试**

覆盖工具绑定集合、HumanMessage 的 task/instruction/session/research/memory、多个 tool calls 顺序、FileRead→FileEdit、todo 的 in_progress→completed 持久化、blocked、writer 三类事件、未知工具/无效 args/缺失 ID、loop limit、`test_red` 的预期非零不让 Agent 本身失败，以及 completed 后失败/超时检查仍保留失败证据。

- [ ] **Step 2: 运行并确认模块不存在**

Run: `uv run pytest tests/unit/test_code_agent.py -q`
Expected: FAIL with `ModuleNotFoundError: xiliumini.agents.code_agent`.

- [ ] **Step 3: 实现 run_code_agent**

严格保留公开签名；`TodoUpdateTool` 追加到 `build_tools(state)` 返回列表之后；最终 todo 必须从 TodoStore 重读。存在 blocked、遗留 in_progress、达到循环上限或终止性工具错误时 `ok=False`；带 `phase="test_red"` 的非零测试结果作为证据保留，但不单独令 Agent 失败。

- [ ] **Step 4: 运行 codeAgent 与依赖测试**

Run: `uv run pytest tests/unit/test_code_agent.py tests/unit/test_todo_tools.py tests/unit/test_bash_tool.py -q`
Expected: PASS.

- [ ] **Step 5: 更新进程记录并提交**

```powershell
git add src/xiliumini/agents/code_agent.py src/xiliumini/agents/__init__.py tests/unit/test_code_agent.py 项目进程.md
git commit -m "feat: 实现 codeAgent ReAct 编码循环"
```

### Task 7: Planner Supervisor 与子 Agent 调用工具

**Files:**
- Create: `src/xiliumini/tools/subagent_tools.py`
- Modify: `src/xiliumini/graph/nodes.py`
- Modify: `src/xiliumini/tools/__init__.py`
- Rewrite: `tests/unit/test_nodes.py`
- Create: `tests/unit/test_subagent_tools.py`

**Interfaces:**
- Consumes Tasks 2、5、6 的 Todo/search/code 入口。
- Produces `SupervisorContext(todos, research_notes, agent_results, tool_events, state)` holding mutable JSON-compatible snapshots for one planner invocation。
- Produces `CallSearchAgentTool(context, writer)` and `CallCodeAgentTool(context, writer)` accepting only `instruction: str` and returning JSON with `ok`, `summary` and the relevant `sources` or `todos` snapshot。
- Produces `planner_node(state: GraphState, *, model: Any, max_loops: int = 8) -> dict[str, Any]`。

- [ ] **Step 1: 写失败的 subagent tool 测试**

断言 search 结果追加到 research/agent_results/tool_events；code 结果同步 todos；子 Agent 异常被脱敏；返回 ToolMessage 可序列化；第二次调用保留第一次结果。

- [ ] **Step 2: 运行并确认模块不存在**

Run: `uv run pytest tests/unit/test_subagent_tools.py -q`
Expected: FAIL with `ModuleNotFoundError: xiliumini.tools.subagent_tools`.

- [ ] **Step 3: 实现 SupervisorContext 与两个调用工具**

工具闭包持有当前 state 的副本，不直接把模型或工具写入 GraphState；子 Agent writer 复用 Planner 当前 stream writer。

- [ ] **Step 4: 写失败的 planner Supervisor 测试**

覆盖 bind 三工具、先 TodoWrite 后 code、search→code 串联、同轮继续推理、严格结束 JSON、一次修复、未知工具/无效 args/缺 ID、max loops、attempt 加一，以及 Verifier 反馈在下一轮 HumanMessage 中出现且 completed todos 被复用。

- [ ] **Step 5: 运行 planner 测试并确认旧静态 Planner 失败**

Run: `uv run pytest tests/unit/test_nodes.py -q -k planner`
Expected: FAIL because current planner does not bind/call tools.

- [ ] **Step 6: 实现 planner_node bounded ReAct Supervisor**

返回更新字段至少包含 `todos`、`research_notes`、`agent_results`、`tool_events`、`result`、`attempt`、`graph_state="verifying"`。删除 Actor schema、actor_node 及专属执行 helper。

- [ ] **Step 7: 运行本任务测试**

Run: `uv run pytest tests/unit/test_subagent_tools.py tests/unit/test_nodes.py -q -k "subagent or planner"`
Expected: PASS.

- [ ] **Step 8: 更新进程记录并提交**

```powershell
git add src/xiliumini/tools/subagent_tools.py src/xiliumini/graph/nodes.py src/xiliumini/tools/__init__.py tests/unit/test_nodes.py tests/unit/test_subagent_tools.py 项目进程.md
git commit -m "feat: 将 Planner 改造为子 Agent Supervisor"
```

### Task 8: Verifier、Final 与 LangGraph 回环

**Files:**
- Modify: `src/xiliumini/graph/nodes.py`
- Modify: `src/xiliumini/graph/workflow.py`
- Modify: `tests/unit/test_nodes.py`
- Modify: `tests/unit/test_workflow.py`

**Interfaces:**
- Produces `verifier_node(state, *, model)` with model JSON judgment plus deterministic guard。
- Produces `final_node(state)` using the new Final template。
- Produces `route_after_verifier(state) -> Literal["planner", "final"]` and `build_workflow(model, checkpointer=None)`。

- [ ] **Step 1: 写失败的 Verifier/Final 测试**

覆盖 todos 全完成通过；pending/in_progress/blocked 拒绝；实现任务缺 code result 拒绝；研究任务缺来源拒绝；要求 tests/demo/Ruff/Pyright 时缺成功最新证据拒绝；TDD 要求 `test_red → code/todo activity → test_green` 的有序证据；completed 后的失败/超时/截断检查拒绝；同 phase 的后续成功能修复历史失败；非法模型 JSON 失败；Final 包含状态、次数、总结和未完成项。

- [ ] **Step 2: 运行并确认旧 Actor evidence 规则不匹配**

Run: `uv run pytest tests/unit/test_nodes.py -q -k "verifier or final"`
Expected: FAIL on new todo/agent evidence expectations.

- [ ] **Step 3: 实现确定性 guard、Verifier 和 Final**

先计算 guard；guard 失败时即使模型声称 passed 也返回 failed。latest evidence 按 tool + 当前/后续 attempt 判定修复关系。

- [ ] **Step 4: 写失败的 graph 路由测试**

断言一次通过为 `planner, verifier, final`；一次失败后为 `planner, verifier, planner, verifier, final`；达到上限仍进入 Final；Actor 永不注册或调用。

- [ ] **Step 5: 运行并确认旧 graph 失败**

Run: `uv run pytest tests/unit/test_workflow.py -q`
Expected: FAIL because current graph still contains Actor.

- [ ] **Step 6: 实现新 graph 与路由**

移除 tools 参数和 Actor 节点：`START → planner → verifier`，条件边到 planner/final，`final → END`。

- [ ] **Step 7: 运行节点与 graph 测试**

Run: `uv run pytest tests/unit/test_nodes.py tests/unit/test_workflow.py -q`
Expected: PASS.

- [ ] **Step 8: 更新进程记录并提交**

```powershell
git add src/xiliumini/graph/nodes.py src/xiliumini/graph/workflow.py tests/unit/test_nodes.py tests/unit/test_workflow.py 项目进程.md
git commit -m "feat: 接通 Supervisor 验证回环"
```

### Task 9: Runtime、事件、CLI 与文档迁移

**Files:**
- Modify: `src/xiliumini/events.py`
- Modify: `src/xiliumini/core/agent.py`
- Modify: `src/xiliumini/runtime.py`
- Modify: `src/xiliumini/cli/__init__.py`
- Modify: `tests/unit/test_agent.py`
- Modify: `tests/integration/test_runtime.py`
- Modify: `tests/integration/test_cli.py`
- Modify: `README.md`
- Modify: `SPEC.md`

**Interfaces:**
- `PlannerEvent` carries `todos`, `summary`, `attempt`; `ActorEvent` is removed。
- `ProgressEvent.stage` carries `planner`, `search_agent`, `code_agent`, `todo`。
- `stream_agent(model, inputs, checkpointer=None)` and `build_workflow(model, checkpointer=...)` no longer receive Actor tools。

- [ ] **Step 1: 写失败的事件/runtime/CLI 测试**

更新测试以断言完整新 GraphState 初始化、PlannerEvent 映射、嵌套 Agent custom events、无 Actor 输出、`--max-attempts` 文案为 Supervisor、`--no-stream` 只显示 Final、相同 session 复用 workspace。

- [ ] **Step 2: 运行并确认旧事件契约失败**

Run: `uv run pytest tests/unit/test_agent.py tests/integration/test_runtime.py tests/integration/test_cli.py -q`
Expected: FAIL on ActorEvent imports or new PlannerEvent fields.

- [ ] **Step 3: 迁移事件适配、Runtime 和 CLI**

`create_runtime()` 只创建主 Planner/Verifier model；移除 AnalysisAgent/delegate 注册；Runtime 初始化新列表字段。保留 provider error 分类与 workspace 创建错误行为。

- [ ] **Step 4: 更新 README 与 SPEC**

记录 Supervisor 图、Tavily key、search/code Agent、todo/notepad 位置、受限 Bash 非 OS 沙箱、CLI 新输出；SPEC 追加本项目 Task2，不改写旧历史任务定义。

- [ ] **Step 5: 运行迁移聚焦测试**

Run: `uv run pytest tests/unit/test_agent.py tests/integration/test_runtime.py tests/integration/test_cli.py -q`
Expected: PASS.

- [ ] **Step 6: 更新进程记录并提交**

```powershell
git add src/xiliumini/events.py src/xiliumini/core/agent.py src/xiliumini/runtime.py src/xiliumini/cli/__init__.py tests/unit/test_agent.py tests/integration/test_runtime.py tests/integration/test_cli.py README.md SPEC.md 项目进程.md
git commit -m "feat: 完成 Task2 Supervisor 运行时迁移"
```

### Task 10: 全量验证与真实 smoke test

**Files:**
- Modify: `项目进程.md`
- Modify only if verification exposes a defect: affected source and regression test files

**Interfaces:**
- Consumes the complete Task2 implementation。
- Produces evidence recorded in `项目进程.md`; no new production API。

- [ ] **Step 1: 运行全量自动化测试**

Run: `uv run pytest -q`
Expected: all tests pass; Windows symlink skips may remain only where already documented.

- [ ] **Step 2: 运行格式、lint、类型检查和 diff 检查**

Run: `uv run ruff format --check .`
Expected: all files already formatted.

Run: `uv run ruff check .`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors`.

Run: `git diff --check`
Expected: no whitespace errors.

- [ ] **Step 3: 运行 CLI 静态 smoke test**

Run: `uv run xiliumini ask --help`
Expected: command exits 0 and describes Supervisor attempts.

Run: `uv run xiliumini doctor`
Expected: with existing valid local model configuration, command exits 0 without exposing secrets.

- [ ] **Step 4: 有凭据时运行真实 searchAgent smoke test**

只有本机已有 `TAVILY_API_KEY` 时执行一个官方资料查询；不得打印 key。若未配置，记录“未执行：缺少 TAVILY_API_KEY”，并用缺 key 自动化测试作为已验证行为，不能伪称真实搜索通过。

- [ ] **Step 5: 运行真实 Supervisor coding smoke test**

使用新的 session，让 codeAgent 创建一个最小 Python 模块和测试、运行 pytest、完成 todo，并确认 Verifier passed；另运行一次需要研究的任务（仅当 Tavily key 可用），确认来源进入 codeAgent 上下文。

- [ ] **Step 6: 修复 smoke test 暴露的问题时执行新的 red-green 循环**

每个缺陷先在对应测试文件增加能重现问题的失败测试，确认 red 后再修改生产代码并运行全量验证。

- [ ] **Step 7: 更新最终进程记录并提交验证结果**

重新读取 `AGENTS.md` 和完整 `项目进程.md`；把所有实际 pass/skip/fail 数量、未执行原因和 smoke test 结果写入 Task2，不写预期结果。

```powershell
git add 项目进程.md
git commit -m "test: 记录 Task2 Supervisor 验收结果"
```
