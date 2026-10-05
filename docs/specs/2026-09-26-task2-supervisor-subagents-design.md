# Task2 Supervisor 与专业子 Agent 设计

## 背景与目标

Task1 当前使用 `Planner → Actor → Verifier → Final`：Planner 只生成静态 todo，Actor
负责一次性生成并执行动作。Task2 将 Planner 升级为 Supervisor，使其能够在同一轮推理中维护
todo、按需委派搜索、委派代码实现，并在获得子 Agent 结果后继续决策。

目标工作流为：

```text
START → Planner(Supervisor) → Verifier
             ↑                   │
             ├─ searchAgent      ├─ passed → Final → END
             └─ codeAgent        └─ failed 且有剩余次数 → Planner
```

旧 Actor 不再位于执行路径中。代码修改由 `codeAgent` 完成；外部资料搜索由
`searchAgent` 完成；Planner 只负责拆解、委派、汇总与根据验证反馈继续调度。

仓库实际 Python 包为 `src/xiliumini/`，因此用户描述中的
`src/mokioclaw/tools/web_search_tool.py` 对应实现为
`src/xiliumini/tools/web_search_tool.py`。

## 核心原则

- Planner、searchAgent、codeAgent 各自拥有独立消息上下文，避免子 Agent 的完整消息污染
  Supervisor 上下文。
- 子 Agent 通过结构化返回值向 Planner 汇报；共享状态只保存可序列化的摘要、todo、来源和
  工具事件，不保存模型、工具实例或 API 密钥。
- Planner 的委派是嵌套 ReAct 工具调用，不把每次子 Agent 调用建成独立 LangGraph 节点。
- 所有文件和命令操作固定在当前 session workspace。进程限制不是操作系统沙箱，文档和 CLI
  不得宣称可以抵御恶意生成代码。
- todo、notepad 和后续 memory 接口使用稳定边界，方便后续阶段替换持久化实现。

## 共享状态

`GraphState` 调整为适合 Supervisor 循环的状态：

- `task: str`：原始用户任务。
- `todos: list[TodoItem]`：结构化 todo；每项包含稳定 `id`、`content`、`status` 和可选
  `note`。状态只能是 `pending`、`in_progress`、`completed`、`blocked`。
- `research_notes: list[ResearchNote]`：搜索摘要、query 与来源 URL。
- `agent_results: list[AgentResult]`：每次 search/code 委派的可序列化结果摘要。
- `tool_events: list[ToolEvent]`：子 Agent 工具调用证据，包含 agent、tool、args 摘要、结果、
  `ok` 和当前 Supervisor attempt。
- `result: str`：Planner 当前轮总结。
- `graph_state`：`planning`、`verifying`、`passed` 或 `failed`。
- `verification`、`attempt`、`max_attempts`、`final_answer`、`session_id`、`workspace`：沿用
  Task1 的验证、重试和会话语义。

Planner 每开始一轮 Supervisor 调度便把 `attempt` 加一。Verifier 失败且
`attempt < max_attempts` 时返回 Planner；历史结果保留，但确定性校验以最新 attempt 的活动和
当前 todo 状态为主。

## Todo 与持久化工具

Planner 绑定 `TodoWriteTool`，用于创建或替换有稳定 ID 的 todo 计划。codeAgent 绑定
`TodoUpdateTool`，只允许更新已存在 todo 的状态与说明。两个工具共享相同的 todo store，并将
快照原子写入 session workspace 下的 `.xiliumini/todos.json`。

约束如下：

- Planner 必须在委派代码任务前建立 todo。
- codeAgent 开始某项工作前把它更新为 `in_progress`。
- 完成后更新为 `completed`；无法完成则更新为 `blocked` 并提供原因。
- 不允许从 `pending` 直接变成 `completed`，避免伪造未执行的完成状态。
- 工具返回更新后的完整 todo 快照，Planner 最终也从 store 读取快照，避免仅依赖模型文本。
- 持久化失败返回结构化工具错误，不把磁盘路径或异常详情暴露给模型。

## searchAgent

公开入口保持用户指定接口：

```python
def run_search_agent(state, instruction, *, writer=None, max_loops=4) -> dict: ...
```

执行流程：

1. 通过统一 Provider 工厂创建独立 chat model，并调用
   `bind_tools([WebSearchTool()])`。
2. 初始消息为 `SystemMessage(SEARCH_AGENT_PROMPT)` 与包含原始任务、Planner 指令、已有
   research notes 的 `HumanMessage`。
3. 最多执行 `max_loops` 轮。每轮把 AIMessage 加入上下文；若含 tool calls，则逐个调用
   `WebSearchTool`，追加对应 `ToolMessage` 并继续；若无 tool calls，则把当前文本作为总结并
   结束。
4. 收集去重后的 `queries`、`sources`、标准化 answer 和 `tool_events`。writer 存在时写入
   `type="tool_call"` 与 `type="search_results"` 事件。
5. 返回 `{ok, summary, queries, sources, messages, tool_events}`。达到循环上限、模型异常或
   所有搜索均失败时 `ok=False`，同时返回脱敏摘要，不抛出第三方异常。

`SEARCH_AGENT_PROMPT` 使用用户提供的原文，集中定义于
`src/xiliumini/prompts/task1.py` 并从 prompts 包导出。

### WebSearchTool

`WebSearchTool` 是 LangChain `BaseTool`，输入至少包含非空 `query`，可选 bounded
`max_results`。它按调用时环境读取 `TAVILY_API_KEY`：

- 缺失或空白时直接返回
  `{ok: False, error: "missing TAVILY_API_KEY", query: ...}`，不进行网络调用。
- 存在密钥时创建 `tavily.TavilyClient` 并调用 `search()`。
- 标准化输出为
  `{ok, query, answer, results: [{title, url, content, score}]}`。
- 第三方返回缺字段时使用安全默认值；异常转换为不含密钥和内部 traceback 的
  `{ok: False, query, error}`。

项目增加 `tavily-python` 运行时依赖，并在 `.env.example` 与 README 中记录
`TAVILY_API_KEY`。单元测试使用假 client，不访问真实网络。

## codeAgent

公开入口保持用户指定接口：

```python
def run_code_agent(state, instruction, *, writer=None, max_loops=10) -> dict: ...
```

执行流程：

1. 创建独立 chat model，并绑定 `build_tools(state) + [TodoUpdateTool]`。
2. 调用 `build_memory_snapshot(state)` 生成 layered memory 快照。当前实现包含 session 标识、
   todo、research notes 和 notepad 内容；接口保持独立，后续可增加压缩记忆层而不改变 Agent
   入口。
3. 初始消息为 `SystemMessage(CODE_AGENT_PROMPT)` 与包含原始任务、Planner 指令、session
   上下文、research notes 和 memory snapshot 的 `HumanMessage`。
4. 最多执行 `max_loops` 轮标准 ReAct：执行全部 tool calls，逐个追加 ToolMessage，并通过
   writer 发出 `tool_call`、`tool_result` 和 `todo_update` 事件。
5. 返回 `{ok, summary, todos, messages, tool_events}`。最终 `todos` 必须从持久化 store
   重读；存在 `blocked` 或仍处于 `in_progress` 的本轮 todo 时，结果不能标记成功。

`CODE_AGENT_PROMPT` 使用用户提供的原文并补上闭合三引号，集中定义于 `task1.py`。

### codeAgent 工具集

`build_tools(state)` 根据 `state["workspace"]` 创建：

- `FileReadTool`、`FileWriteTool`、`FileEditTool`、`GrepTool`。
- `BashTool`：名称保留用户要求，但使用 argv 与 `shell=False`，工作目录固定为 workspace，
  设置超时和合并输出上限。首版只开放 Python 脚本、pytest、Ruff 和 Pyright 等项目检查所需
  命令；拒绝 cwd、环境变量注入、重定向、管道和 workspace 外目标。
- `NotepadReadTool`、`NotepadAppendTool`：读写 `.xiliumini/notepad.md`，使用 UTF-8、大小上限
  和 workspace 路径校验。Append 只追加 durable context，不覆盖已有记录。

`TodoUpdateTool` 单独追加到列表，强调它是进度协议的一部分而非普通 workspace 工具。

## Planner Supervisor

Planner 绑定三个闭包式工具：

- `TodoWriteTool`：创建或调整计划。
- `CallSearchAgentTool`：接收 `instruction`，调用 `run_search_agent`，把摘要和来源写入
  `research_notes` 与 `agent_results`。
- `CallCodeAgentTool`：接收 `instruction`，调用 `run_code_agent`，把 summary、todos 与事件
  写回本轮状态。

`planner_node` 自身执行 bounded ReAct 循环。初始消息包含任务、当前 todo、已有 research、
Verifier 反馈、attempt 与可用委派规则；每个子 Agent 工具结果以 ToolMessage 返回，Planner
随后继续推理。没有 tool calls 时，Planner 必须返回严格 JSON 总结，例如
`{"summary":"...","ready_for_verification":true}`。非法 JSON 允许一次格式修复；仍非法则返回
受控节点错误。

Planner prompt 要求：

- 先用 TodoWriteTool 建立可验证计划。
- 只在需要外部事实或最新资料时调用 searchAgent。
- 代码、文件和测试工作必须委派给 codeAgent，Planner 自身不写文件。
- 研究型实现先搜索，再把研究摘要和来源传给 codeAgent。
- Verifier 失败重试时只处理反馈指出的缺口，不重复已完成工作。
- 只有 todo 状态和子 Agent 证据支持时才能请求验证。

## Verifier 与 Final

Verifier 接收 task、todos、research notes、agent results、tool events 和 Planner summary。模型仍
返回严格的 `{"status":"passed"|"failed","reason":"..."}`，但通过前执行确定性校验：

- 不能存在 `pending`、`in_progress` 或 `blocked` todo。
- 实现型任务必须有成功的 codeAgent 结果。
- 用户明确要求搜索或研究时，必须有成功搜索及至少一个来源 URL。
- 用户明确要求测试、TDD、demo、Ruff 或 Pyright 时，最新相关检查必须有成功工具证据。
- 超时、截断、被拒绝或未由后续成功结果修复的必要动作会导致失败。

Final 不调用模型，只根据最终状态、attempt、Planner summary 和 verifier reason 格式化稳定答案。
达到最大次数仍失败时也必须经过 Final，向用户说明未完成项和 blocked todo。

## Graph、事件与 CLI

LangGraph 节点改为 `planner`、`verifier`、`final`：

- `START → planner → verifier`
- `verifier(passed) → final`
- `verifier(failed && attempt < max_attempts) → planner`
- `verifier(failed && attempt >= max_attempts) → final`
- `final → END`

删除旧 Actor 图节点和 Actor prompt。运行时事件更新为：

- Planner 完成一轮时输出 todo 快照、summary 与 attempt。
- 子 Agent writer 事件通过 LangGraph custom stream 变成 ProgressEvent，stage 分别为
  `search_agent`、`code_agent` 或 `todo`。
- Verifier 与 Final 沿用明确的成功/失败展示。
- `--no-stream` 继续隐藏所有中间事件，只显示 Final。

CLI 的 `--max-attempts` 文案从 “Actor attempts” 改为 “Supervisor attempts”。

## 错误处理

- 缺少 Tavily key 是普通工具失败，不阻止不需要搜索的代码任务运行。
- 模型 tool call 参数由各工具 Pydantic schema 校验；错误作为 ToolMessage 返回，使 Agent 有机会
  修正。
- 子 Agent 达到 max loops 返回 `ok=False` 和已完成的部分证据，Planner 可在下一 Supervisor
  attempt 中决定重试或阻塞 todo。
- 文件、todo、notepad 和命令异常统一脱敏；不得把 API key、绝对外部路径或 traceback 写入
  event/state。
- writer 缺失时使用 no-op，不影响直接单元测试或非流式调用。

## 测试策略

严格按 TDD 分阶段实现：

1. `WebSearchTool`：缺 key、标准化成功结果、第三方异常脱敏。
2. todo/notepad/memory：状态转换、原子持久化、无效 ID、blocked 原因、大小和路径边界。
3. searchAgent：绑定唯一工具、多 tool call、来源去重、writer 事件、循环上限。
4. codeAgent：工具绑定、先读后改提示协议、todo 状态持久化、blocked 和循环上限。
5. Planner：TodoWrite、search/code 委派、子 Agent 结果回灌、JSON 修复和 Verifier 反馈重试。
6. Verifier 与 graph：通过路径、失败回到 Planner、最大次数进入 Final、确定性证据拒绝。
7. Runtime/CLI：新状态初始化、custom events、stage 展示、`--no-stream` 与错误脱敏。
8. 全量 pytest、Ruff format check、Ruff lint、Pyright；具备本地有效配置时再执行真实模型与
   Tavily smoke test，不能用 smoke test 替代自动化测试。

## 文档与兼容性

- README、`.env.example`、SPEC 和项目进程同步更新新工作流、Tavily 配置、工具边界和 CLI
  输出。
- `CommandTool` 可暂时保留为兼容接口，但不再由旧 Actor 节点驱动；`BashTool` 是 codeAgent
  的统一检查入口。
- 已存在的 `AnalysisAgent` 和 `delegate_analysis` 不进入新 Supervisor 路径；若无其他调用者，
  实现阶段只移除注册，不进行与 Task2 无关的历史清理。

## 验收标准

- Planner 能在一轮中建立 todo、调用 searchAgent 或 codeAgent，并基于返回结果继续调用或结束。
- codeAgent 能在隔离 workspace 写文件、编辑文件、运行允许的检查并持久更新 todo。
- searchAgent 能通过 Tavily 返回标准化研究摘要和来源；缺 key 时得到指定结构化错误。
- Verifier 失败能回到 Planner，成功或耗尽次数后进入 Final。
- 自动化测试不发起真实网络请求，不读取开发者真实密钥，并覆盖所有新增失败边界。
- 全量测试、格式检查、lint 和类型检查通过；所有实际结果写入 `项目进程.md` 后方可提交实现。
