# Task1 LangGraph Plan-Act-Verify 设计

## 目标

将当前 `模型 → 工具 → 模型` 的 ReAct 主循环替换为显式的 LangGraph
`Planner → Actor → Verifier → Final` 工作流。新工作流必须能在每个 session 的隔离工作区中
真实创建文件、运行测试和执行 demo，并把每个节点的结果以统一事件交给 CLI 展示。

Task1 的验收任务是：

```text
帮我实现一个 Conway's Game of Life，要求 TDD：先写测试，再写实现，最后跑 demo
```

使用 `--max-attempts 3` 时，Actor 必须在隔离工作区中先创建测试，再创建实现，实际运行测试，
最后实际运行 demo。Verifier 只能根据真实执行证据判定通过或失败。

## 范围

本次包含：

- 新增 `graph/state.py`、`graph/nodes.py`、`graph/workflow.py` 和包入口。
- 将 Task1 的 Planner、Actor、Verifier、Final 提示词集中到 `prompts/task1.py`。
- 删除旧 ReAct 主提示词及其导出。
- 将 `core/agent.py` 改为 LangGraph 流事件适配器，不再实现 ReAct 循环。
- 新增受限的工作区命令工具，使 Actor 能真实运行 Python 测试和 demo。
- 更新 Runtime 和 CLI 的事件协议及 `--max-attempts` 参数。
- 更新自动化测试、README、SPEC 和项目进程记录。

本次不包含：

- 任意 PowerShell、cmd 或 Bash 字符串执行。
- 管道、重定向、命令拼接、网络命令或工作区外命令。
- 会话持久化、Trace 持久化、浏览器工具或 MCP 工具。
- 对 Task3、Task4 的提前实现。

## 架构

```text
Typer CLI
    ↓
Runtime
    ↓
core/agent.py  ── 解析 updates/custom ──→ 统一 RuntimeEvent
    ↓
graph/workflow.py
    START → Planner → Actor → Verifier ── passed ─→ Final → END
                         ↑        │
                         └ failed 且还有尝试次数 ┘
                                  │
                                  └ failed 且次数耗尽 → Final → END
```

`graph/` 负责状态、节点和路由；`core/agent.py` 只负责构建图、调用
`build_workflow().stream(inputs, stream_mode=["updates", "custom"])`，并把图事件转换为稳定的
应用事件。Runtime 继续负责 session 工作区、Provider 错误映射和 CLI 边界。

## 共享状态

`graph/state.py` 定义 `GraphState`，至少包含：

- `task: str`：用户原始任务。
- `todo: list[str]`：Planner 生成的有序任务清单。
- `result: str`：Actor 最近一次执行摘要及真实命令输出。
- `execution: list[ActionResult]`：按实际发生顺序保存文件和命令动作的结构化证据。
- `graph_state: Literal["planning", "acting", "passed", "failed"]`：当前图状态。
- `verification: str`：Verifier 的判定理由和下一次修正建议。
- `attempt: int`：已执行的 Actor 尝试次数，从 0 开始、每次 Actor 执行后加 1。
- `max_attempts: int`：最大 Actor 尝试次数，CLI 默认传入 3，必须至少为 1。
- `final_answer: str`：Final 节点格式化后的最终文本。
- `session_id: str`：当前 session UUID。
- `workspace: Path`：当前 session 的隔离工作区。

节点依赖的模型、文件执行器和命令执行器通过 `build_workflow()` 的闭包显式注入，不把不可序列化
对象写入共享状态。

## 节点设计

### Planner

`planner_node` 调用模型，把用户任务拆成简短、有序、可验证的 `todo`。提示词要求：

- 识别测试、实现、验证和 demo 的先后依赖。
- 对明确要求 TDD 的任务，测试步骤必须排在实现步骤之前。
- 每个条目包含可观察的完成条件。
- 只输出可解析的 JSON 对象，节点验证后写入 `todo`。

Planner 只在图开始时运行一次。Actor 重试时复用原始 todo，并结合 Verifier 反馈修正实现。

### Actor

`actor_node(state)` 调用模型，根据 `task`、`todo`、上次结果和验证反馈生成本次受限执行方案。
执行方案是结构化 JSON，包含按顺序排列的文件写入和命令动作。节点负责校验动作并执行，而不是
让模型直接获得任意 Shell：

1. 文件动作复用工作区路径边界和原子写入逻辑。
2. 命令动作只接受参数数组，不解释 Shell 字符串。
3. 每个动作按生成顺序执行并记录结果；显式 TDD 任务必须出现“写测试 → 运行并观察预期失败 →
   写实现 → 运行并通过测试 → 运行 demo”的证据顺序。
4. 命令结果记录退出码、标准输出、标准错误、是否超时和截断状态。
5. 任一动作失败不会泄漏工作区外路径或 Provider 细节；失败证据写入 `result` 交给 Verifier。

Actor 每运行一次将 `attempt` 加 1，并通过 custom stream 发送动作开始/结束进度。Actor 完成后通过
updates stream 暴露本次状态更新。

### Verifier

`verifier_node(state)` 调用模型，根据 todo、Actor 摘要、文件动作结果和命令结果输出结构化判定：

- `passed`：所有用户要求都有证据，必要的测试和 demo 命令均真实执行且退出码为 0。
- `failed`：缺少证据、测试失败、demo 失败、动作被拒绝或实现不完整。

模型不能把失败的命令判为通过。节点先执行确定性前置检查；存在超时、最终测试或 demo 非零退出、
被拒绝动作，或任务要求运行测试/demo 而没有对应成功记录时，最终状态强制为 `failed`。对于显式
TDD 任务，预实现测试的非零退出码是预期 red 证据，不直接判定最终失败；但缺少该 red 证据时必须
判定失败。Verifier 的理由必须提供给下一次 Actor 尝试使用。

### Final

Final 不调用模型。`FINAL_PROMPT` 只是一个格式化模板，把 `passed/failed`、尝试次数、Actor 结果和
Verifier 理由组合成 `final_answer`。Final 节点不会补写文件、重跑命令或伪造成功。

## 受限命令执行

新增绑定到单一 session 工作区、工具名固定为 `command` 的 `CommandTool`：

- 使用 `asyncio.create_subprocess_exec` 或等价的 argv API，禁止 `shell=True`。
- 固定 `cwd=workspace`，拒绝调用方传入工作目录。
- 允许 Python 解释器及其模块模式，例如 `python -m pytest` 和 `python demo.py`。
- 拒绝 PowerShell、cmd、bash、路径限定的外部可执行文件、管道、重定向和命令连接符。
- 使用当前项目可用的 Python 解释器，避免从 `PATH` 解析到任意同名程序。
- 拒绝 `python -c`、`python -m` 的任意模块；首期只允许相对路径 Python 脚本和
  `python -m pytest`。
- 每条命令有固定超时和输出字节上限；超时后终止子进程。
- stdout/stderr 以 UTF-8 安全解码，并在事件和错误中隐藏工作区绝对路径。

该能力只用于当前隔离工作区内的开发验证，不等价于通用 Shell。工作目录限制和 argv 执行只能约束
进程启动方式，不能把生成的 Python 代码变成 OS 沙箱；被执行的 Python 文件仍拥有当前 xiliumini
进程的用户权限。README 和 CLI 必须明确这一风险，不宣称能抵御恶意生成代码。

## 工作流与重试

`build_workflow()` 校验 `max_attempts >= 1`，构建并编译以下边：

- `START → planner`
- `planner → actor`
- `actor → verifier`
- Verifier 返回 `passed`：`verifier → final`
- Verifier 返回 `failed` 且 `attempt < max_attempts`：`verifier → actor`
- Verifier 返回 `failed` 且 `attempt >= max_attempts`：`verifier → final`
- `final → END`

只有 Actor 执行计入 attempt。Planner、Verifier 和 Final 不消耗尝试次数。即使最大次数耗尽，Final
也必须产生明确的 failed 答案。

## 事件协议

应用层新增或调整为以下统一事件：

- `PlannerEvent(todo)`
- `ActorEvent(result, attempt)`
- `VerifierEvent(passed, reason, attempt)`
- `FinalEvent(text, session_id)`
- `ProgressEvent(stage, message)`：由 custom stream 传递动作级进度。
- `ErrorEvent(code, message)`

`core/agent.py` 同时消费：

- `updates`：识别完成的节点及其状态更新，生成 Planner/Actor/Verifier/Final 事件。
- `custom`：识别 Actor 发出的受限文件或命令进度，生成 ProgressEvent。

未知或不完整图事件被安全忽略或转换为脱敏错误，不能让 CLI 依赖 LangGraph 的原始元组结构。

## CLI

`ask` 增加 `--max-attempts`，默认 3；交互式 `chat` 使用同一默认值。CLI 按统一事件显示：

```text
📋 Planner: ...
🔧 Actor (attempt 1/3): ...
✅ Verifier: ...
❌ Verifier: ...
📝 Final: ...
```

`--max-attempts` 小于 1 时由 Typer 参数校验直接拒绝。原有 `--no-stream` 保留，但含义改为隐藏中间
节点和进度，只显示 Final；默认模式显示每个核心节点，不再输出旧的 token/tool 生命周期格式。

## 错误处理

- Provider、工作区和命令异常继续映射为脱敏 `ErrorEvent`。
- Planner 返回无效 JSON 时执行一次格式修复调用，仍无效则产生脱敏 ErrorEvent 并结束；Actor 的
  无效执行方案计为一次失败尝试并交给 Verifier；Verifier 的无效输出按 failed 处理。Actor/Verifier
  可在剩余次数内重试，无法继续时进入 failed Final。
- 写文件或执行命令被安全策略拒绝时，证据进入 Actor 结果，由 Verifier 判定失败。
- 命令超时和输出截断必须显式记录，不能被误判为成功。
- Ctrl+C 沿用现有安全退出行为，并确保正在运行的子进程被终止。

## 测试策略

严格使用 TDD，先写并观察失败测试，再实现最小代码：

- 状态与路由：Planner 只运行一次；failed 重试 Actor；passed 和次数耗尽进入 Final。
- 节点：提示词输入完整；结构化输出有效和无效分支；attempt 计数正确。
- 执行安全：命令固定在工作区；允许 Python/pytest；拒绝 Shell、外部路径、连接符；覆盖超时、
  非零退出码和输出截断。
- TDD 证据：显式 TDD 任务缺少预实现 red 测试时失败，最终 green 测试和 demo 都成功时才可通过。
- 事件适配：覆盖多模式 stream 的 `updates` 和 `custom`，确认输出稳定 RuntimeEvent。
- CLI：覆盖默认值 3、显式 `--max-attempts 3`、四类图标、`--no-stream` 和非法参数。
- 回归：工作区隔离、配置脱敏、Provider 错误、现有安全文件工具保持通过。
- 真实验收：使用本地已配置模型运行指定 Game of Life 命令，检查 session 工作区中的测试、实现、
  demo 文件，确认测试和 demo 命令真实成功。真实 API 验收结果与自动化测试分开记录。

## 完成标准

- 仓库中不再存在旧 ReAct 主循环和 `ACTOR_PROMPT`。
- `graph/state.py`、`graph/nodes.py`、`graph/workflow.py` 职责清晰且有自动化覆盖。
- `core/agent.py` 使用指定的 `stream_mode=["updates", "custom"]` 并只输出统一事件。
- CLI 能显示 Planner、Actor、Verifier、Final，且 `--max-attempts` 默认 3。
- 受限执行能力能真实运行 Game of Life 的测试和 demo，同时不能作为任意 Shell 使用。
- 全量 pytest、Ruff、格式检查和 Pyright 通过；真实 Game of Life 验收结果写入项目进程记录。
