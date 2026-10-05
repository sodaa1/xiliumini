# Task3 分层 Memory 与上下文压缩设计

## 背景与目标

Task2 已提供 todo、notepad 和最小 memory 快照，但 Planner 的 ReAct 消息会随工具调用增长，
且当前快照只包含 session、todo、research notes 和完整 notepad。Task3 将其升级为由 Runtime
统一组装的三层 Memory，并在 Planner 即将接近上下文上限时主动压缩历史。

目标如下：

- Planner 在每次模型调用前估算输入 token，在达到配置阈值时自动压缩旧消息。
- 关键决策通过 `NOTEPAD.md` 跨压缩、跨图重试和同 session 后续任务保存。
- 用户明确表达的长期偏好保存为项目级规则，跨 session、跨任务注入每次 Rules Layer。
- `src/xiliumini/graph/memory.py` 提供 Rules、Working Memory、History Summary 三层结构。
- Agent 只消费 Runtime 组装的 Memory 快照；持久化、裁剪和压缩由 Runtime memory service 负责。
- 保持现有 `START → Planner → Verifier → Final` 图结构和 Supervisor 委派方式不变。

## 方案选择

采用 **Runtime 管理 Memory、Planner 触发压缩** 的方案。

`Runtime` 为每次 session 执行创建 `MemoryManager`，并通过显式参数依次传给
`stream_agent`、workflow 和节点。Planner 负责在模型调用边界请求 token 检查，但不直接读取或
写入持久文件。`MemoryManager` 返回已组装的层级快照和需要替换的消息列表。

不新增独立 LangGraph 压缩节点。压缩是 Planner 内部消息生命周期的一部分，独立节点会增加
图路由、attempt 和事件恢复语义，却不能覆盖同一 Planner ReAct 循环内连续工具调用造成的增长。
也不等待 Provider 返回 context overflow 后再补救，因为那会失去为模型输出预留 token 的机会。

## 持久文件与兼容迁移

Memory 控制文件位于 session workspace 根目录，路径均为相对路径：

- `TODO.md`：结构化工作计划和状态。
- `NOTEPAD.md`：Agent 通过受控工具追加的持久笔记。
- `HISTORY_SUMMARY.md`：Runtime 写入的压缩历史。

现有 session 可能包含 `.xiliumini/todos.json` 和 `.xiliumini/notepad.md`。首次访问新路径且旧路径
存在时，store 执行一次兼容迁移：先完整校验旧内容，再原子写入新文件；旧文件保留，避免迁移
过程中造成不可恢复的数据丢失。新旧文件同时存在时只读取新文件，不自动合并有歧义的内容。

`TODO.md` 使用带版本字段的 JSON fenced block 保存 Todo 列表，保持 Markdown 可读性和严格
结构化解析。`NOTEPAD.md` 保持逐条追加文本格式。`HISTORY_SUMMARY.md` 保存当前累计摘要，不保存
API key、模型对象或完整工具参数。

所有写入继续使用同目录临时文件和 `os.replace`，并遵守现有 UTF-8 文件大小上限与 workspace
路径边界。

项目级长期偏好单独保存在 Runtime `data_dir` 根目录的 `USER_PREFERENCES.md`，默认位置为
`.xiliumini/USER_PREFERENCES.md`。它不属于任一 session workspace，因此普通文件工具无法读取或
修改；只有 Runtime 注入的 `UserPreferenceStore` 和专用 Planner 工具能够访问。该文件使用带
schema version 的 JSON fenced block，记录稳定 key、偏好文本和更新时间，使相同偏好可以更新、
删除和去重，而不是无限追加。

## 三层 Memory

### Rules Layer

Rules 是进程内固定、不可由 Agent 修改的规则列表：

- 只在当前 workspace 内操作。
- 文件与命令参数使用 workspace 相对路径。
- `TODO.md` 是工作计划状态的唯一新格式来源。
- `NOTEPAD.md` 保存需要跨压缩保留的关键信息。
- `HISTORY_SUMMARY.md` 保存 Runtime 生成的压缩历史。
- Agent 不直接组装或持久化 Memory；Runtime 是 Memory 快照的所有者。

Rules Layer 分为 `fixed_rules` 与 `user_preferences`。固定规则每次组装时由代码重新生成，不从
磁盘反序列化，防止工作区文件改变安全边界；用户偏好由 Runtime 从项目级
`USER_PREFERENCES.md` 读取，并在每个新任务开始以及 Memory 重新组装时注入。

规则优先级固定为：平台与代码安全边界 > 当前用户任务中的明确指令 > 已保存用户偏好。偏好不能
放宽 workspace、路径、命令、secret 或 Memory 所有权限制；当前任务与旧偏好冲突时只在本次任务
采用当前指令，除非用户明确要求更新长期偏好。

Planner 绑定 `PreferenceWriteTool`，仅在用户明确表达“以后、总是、默认、记住、忘记”等长期
偏好意图时调用。工具支持按稳定 key 执行 upsert 和 remove，通过 Runtime-owned store 原子更新
文件；不得把普通任务要求、推测出的习惯、secret、凭证或一次性内容保存为偏好。工具返回更新后
的有界偏好列表，当前 Planner 可立即看到结果，后续任务则从 Rules Layer 自动加载。

### Working Memory

Working Memory 只描述当前任务和当前图执行：

- `current_node`
- `task`、`session_id`
- `plan_summary`
- `todos`
- `acceptance_criteria`
- `research_notes`、去重后的 `sources`
- `agent_handoffs`，只保留最近 6 条
- `code_agent_summary`
- `verifier_summary`
- `last_error`
- `attempts`，包含 current 和 max

Planner 最终结构化输出扩展为 `summary`、`plan_summary`、`acceptance_criteria` 和
`ready_for_verification`。Supervisor 每次委派后记录一条 handoff；codeAgent 结果更新最新代码摘要；
Verifier 更新验证摘要。失败的委派、工具证据或验证结果更新 `last_error`，成功不会抹除尚未被后续
成功证据修复的错误。

### History Summary Layer

History Summary 包含：

- `history_summary`：`HISTORY_SUMMARY.md` 的有界内容。
- `notepad_summary`：`NOTEPAD.md` 最近有效条目的有界摘要，不注入无限增长的全文。
- `context_summary`：上一轮压缩结果。
- `compression_events`：最近 3 次压缩记录。

每条 compression event 只保存时间、压缩前后 token 估算、被压缩消息数和 attempt，不保存完整
消息。层级快照必须可 JSON 序列化，并明确排除模型、工具实例、secret 和绝对 workspace 路径。

## Token 预算与压缩算法

新增配置：

- `XILIUMINI_CONTEXT_WINDOW_TOKENS`：默认 `64000`，必须大于 0。
- `XILIUMINI_COMPRESSION_TRIGGER_RATIO`：默认 `0.8`，范围为大于 0 且小于 1。
- `XILIUMINI_COMPRESSION_KEEP_TOKENS`：默认 `8000`，必须小于触发 token 数。

token 估算使用项目的直接依赖 `tiktoken`。已知模型优先使用对应 encoding，OpenAI-compatible
未知模型回退到 `cl100k_base`。估算计入 message role、content、tool calls 和固定消息开销。

Planner 将一个 `before_model` hook 传给通用 `run_react`。hook 在每次 `model.invoke` 前执行：

1. 估算当前消息 token。
2. 未达到 `context_window_tokens * trigger_ratio` 时原样返回。
3. 达到阈值时始终保留首个 Planner system message、当前任务消息和不超过 keep budget 的最近
   AI/Tool 消息。
4. 将更早消息、已有 `context_summary` 和当前 Memory 摘要交给未绑定工具的模型生成压缩摘要。
5. 用一个明确标记为历史摘要的 system message 替换被压缩消息。
6. 原子更新 `HISTORY_SUMMARY.md`，并更新 state 中的 `context_summary` 与最近 3 条事件。
7. 重新估算；若仍超过触发阈值，则进一步缩短最近消息内容到安全预算，但永不删除当前任务或
   固定 Rules。

压缩摘要调用发生在触发阈值以内，因此仍为摘要响应保留窗口空间。摘要 prompt 限制输出长度，
只要求保留目标、决定、文件、todo 状态、验证证据、失败原因和后续动作。

## Runtime 与节点数据流

1. `Runtime.stream` 从项目 `data_dir` 加载长期偏好，再创建或恢复 session workspace、迁移持久
   文件并重置新任务的 Todo。
2. Runtime 以 preference store 和 session workspace 创建 `MemoryManager`，将固定规则与长期偏好
   合并为 Rules Layer，组装初始三层快照并放入 `GraphState.memory`。
3. Planner 以完整 Memory 快照替代当前手工拼接的零散状态作为 HumanMessage 上下文。
4. Planner ReAct 每次模型调用前执行 token hook；压缩产生的 state 更新随 Planner 节点结果返回。
5. `SupervisorContext` 维护 handoff、研究来源和 codeAgent 摘要；codeAgent 只读取传入的 Memory
   快照，Notepad 工具仍是其记录长期信息的唯一入口。
6. Verifier 使用原有确定性证据和模型判断，并返回更新后的 `verifier_summary`、`last_error`、
   `current_node` 与 Memory 快照。
7. Final 保持确定性格式化，不调用模型；最终 state 保留最新 Memory，供 LangGraph checkpointer
   和同 session 后续执行恢复。

显式参数传递优先于新增全局 ContextVar，使 Runtime 所有权、测试注入和并发 session 隔离可见。

## 状态与接口

`GraphState` 增加以下 JSON-safe 字段：

- `memory: LayeredMemory`
- `current_node: Literal["planner", "verifier", "final"]`
- `plan_summary: str`
- `acceptance_criteria: list[str]`
- `agent_handoffs: list[AgentHandoff]`
- `code_agent_summary: str`
- `verifier_summary: str`
- `last_error: str`
- `context_summary: str`
- `compression_events: list[CompressionEvent]`

`LayeredMemory.rules` 包含 `fixed_rules` 和有界的 `user_preferences`；偏好文件本身不进入
`GraphState`，state 只保存当前任务已装配的 JSON-safe 快照。

现有 `attempt` 和 `max_attempts` 保持图路由契约不变；Memory 中的 `attempts` 只是这两个字段的
只读投影。`agent_results` 和 `tool_events` 继续作为 Verifier 的完整结构化证据，Working Memory
只暴露其有界摘要。

主要接口为：

```python
class MemoryManager:
    def assemble(self, state: Mapping[str, Any], *, current_node: str) -> LayeredMemory: ...
    def prepare_planner_messages(
        self,
        messages: list[BaseMessage],
        state: MutableMapping[str, Any],
        *,
        model: Any,
    ) -> list[BaseMessage]: ...


class UserPreferenceStore:
    def read(self) -> list[UserPreference]: ...
    def upsert(self, key: str, content: str) -> list[UserPreference]: ...
    def remove(self, key: str) -> list[UserPreference]: ...
```

`run_react` 新增可选 `before_model` callback；不传时行为与 Task2 完全一致。

## 错误处理

- token encoding 识别失败：使用 `cl100k_base`，不终止任务。
- Todo/Notepad/History 文件格式损坏：返回 workspace/memory 错误，不覆盖损坏文件。
- 摘要模型失败或返回空文本：保留当前消息，不写 History，设置脱敏 `last_error`，由正常 Planner
  调用继续；若 Provider 随后仍因上下文失败，则沿用现有 Provider 错误映射。
- History 写入失败：不替换内存消息，防止出现“消息已丢弃但摘要未持久化”。
- 偏好文件损坏：不覆盖原文件，Runtime 在进入图之前返回脱敏 `memory_error`，避免任务在用户以为
  长期偏好仍生效时静默运行。
- 非法、空白、超长或疑似 secret 的偏好：专用工具拒绝写入并返回结构化错误。
- 迁移失败：保留旧文件并返回脱敏错误。
- Memory 中所有列表均复制并裁剪，节点不能通过修改快照意外修改原始 state 集合。

## 测试策略

实现遵循测试先行，并覆盖：

- Rules 不可被 workspace 内容覆盖，所有层均可 JSON 序列化且不泄漏 secret/绝对路径。
- 项目级偏好跨 session 和任务恢复、按 key 更新/删除，并在每次任务的 Rules Layer 中出现。
- 固定规则优先于偏好，当前明确指令覆盖冲突偏好但不会隐式改写长期记录。
- Planner 只记录明确的长期偏好意图，并拒绝保存普通任务要求和疑似 secret。
- Working Memory 字段投影、source 去重、handoff 最近 6 条和 compression event 最近 3 条。
- 精确低于、等于和高于阈值的 token 行为；未知模型 encoding 回退。
- 压缩保留 system、当前任务和最近工具消息，旧消息被单一摘要替换。
- 摘要成功后原子持久化及同 session 后续 Runtime 恢复。
- 摘要模型失败、空摘要、损坏文件和写入失败时不丢消息。
- 旧 Todo/Notepad 的兼容迁移以及新旧文件同时存在时的新格式优先级。
- Planner 每轮模型调用前检查 token，而不是只在节点入口检查。
- Planner、Supervisor handoff、codeAgent、Verifier 和 Runtime 的集成状态更新。
- 现有 workflow、事件、CLI、Verifier 证据语义和全量测试保持通过。

最终验证包括聚焦 pytest、全量 pytest、Ruff format check、Ruff lint、Pyright 和
`git diff --check`。真实模型 smoke 在本地配置可用时验证至少一次触发压缩后的 Planner 继续执行。

## 文档范围

更新 README 与 SPEC，说明三层 Memory、三个 session 持久文件、项目级
`USER_PREFERENCES.md`、规则优先级、默认 token 配置、压缩触发语义和 Agent 不能直接组装
Memory 的边界。`项目进程.md` 追加 Task3 设计、实现和实际验证记录，不覆盖历史。
