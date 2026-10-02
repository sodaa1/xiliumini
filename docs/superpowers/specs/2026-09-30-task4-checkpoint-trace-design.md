# Task4 Checkpoint 与 Trace 设计

## 背景与目标

现有 Runtime 使用 LangGraph `InMemorySaver` 保存进程内图状态，但进程中断后无法恢复；
`stream_agent` 又会把原始 `updates/custom` chunk 转换为少量 UI 事件，导致工具调用、节点更新和
审批等执行证据无法形成持久链路。Task4 第二步引入两个 Runtime-owned 基础设施：

- Checkpoint 保存可恢复状态和工作区 Git 快照，中断后从正确节点继续。
- Trace 持久记录运行开始、图更新、自定义事件、错误和结束统计。
- 所有控制文件使用当前项目名，位于 workspace 内的 `.xiliumini/`，恢复命令为
  `xiliumini --resume <workspace>`。
- Checkpoint/Trace 必须保持 session 隔离，不把模型、凭证或绝对 workspace 路径写入文件。

本任务不替换 LangGraph 自带 checkpointer；它继续承担单进程图执行职责。新增 Checkpoint 负责
跨进程恢复和 workspace 文件版本，Trace 负责可审计执行记录。

## 方案选择

采用在 `stream_agent` 原始事件边界接入 Checkpoint 与 Trace 的方案。

Runtime 在每次 `stream` 或 `resume` 调用中创建独立运行上下文，再构造 `CheckpointManager` 和
`TraceRecorder`。`stream_agent` 在转换 UI 事件前观察原始 `updates/custom` chunk，并维护一份由
初始 inputs 与节点 update 合并而成的最新 GraphState。两个 recorder 只接收 JSON-safe 快照或
事件，不拥有模型和图对象。

不在最终 `RuntimeEvent` 层记录，因为该层已经丢失工具事件字段和完整节点 update，无法支持
strict Checkpoint 或详细 Trace。也不把 LangGraph 磁盘 checkpointer 作为唯一恢复源，因为它不
覆盖 workspace 文件快照、人工恢复指南和实现无关的 Trace 格式。

## 配置与运行上下文

`Settings` 增加：

- `checkpoint_mode: Literal["light", "strict", "off"] = "light"`
- `trace_mode: Literal["full", "summary", "off"] = "full"`
- `trace_id: str | None = None`

对应环境变量为 `XILIUMINI_CHECKPOINT_MODE`、`XILIUMINI_TRACE_MODE` 和
`XILIUMINI_TRACE_ID`。模式规范化函数对 `None` 或非法值采用安全默认值：Checkpoint 回退
`light`，Trace 回退 `full`。

同一 Runtime 可以交错执行多个 session，因此不在共享 Runtime 实例上保存可变的当前 workspace。
每次运行创建一个私有、不可变的上下文，至少包含 `workspace`、三个模式/ID 和 session 信息；
CheckpointManager 与 TraceRecorder 仍通过 `runtime` 形参消费该 duck-typed 上下文，符合要求的
构造接口，同时避免跨 session 污染。

## CheckpointManager

`src/xiliumini/core/checkpoint.py` 提供：

```python
class CheckpointManager:
    def __init__(self, runtime, task: str = ""): ...
    @property
    def enabled(self) -> bool: ...
    def save(
        self,
        state,
        *,
        status: str = "running",
        latest_node: str | None = None,
        event=None,
    ): ...
    @classmethod
    def load_resume_inputs(cls, runtime, task=None, max_attempts=3): ...


def resume_command(workspace: Path) -> str: ...
def build_recovery_markdown(payload) -> str: ...
```

根目录固定为 `<workspace>/.xiliumini/checkpoints/`，包含：

- `checkpoint.json`：schema version、任务、状态、latest/next node、attempt、保存时间、manifest、
  Git commit 和 light 模式恢复所需的有界 `state_summary`。
- `state.json`：strict 模式完整 JSON-safe state。
- `events.jsonl`：strict 模式按事件追加的 Checkpoint 事件记录。
- `RECOVERY.md`：面向人的任务、状态、文件清单、commit 和恢复命令。
- `repo.git/`：独立 Git 仓库；workspace 根目录不创建或替换 `.git`。

### 保存顺序与原子性

启用模式下，`save` 按以下顺序执行：

1. 创建并验证 Checkpoint 根目录仍位于 workspace 内。
2. strict 模式将脱敏 event 追加到 `events.jsonl`，并原子写完整 `state.json`。
3. 生成 workspace manifest。
4. 使用内部 `repo.git` 对普通 workspace 文件创建 commit。
5. 原子写 `checkpoint.json`。
6. 原子写 `RECOVERY.md`。
7. 返回 `type="checkpoint_saved"` 的事件；off 模式不创建目录并返回 `None`。

JSON 文件使用同目录临时文件、flush、fsync 和 `os.replace`。JSONL 每行是一个独立 JSON 对象，
写入后 flush/fsync，避免半行被当成有效事件。

### Workspace manifest 与 Git 快照

manifest 按相对 POSIX 路径排序，每项包含 path、size 和 SHA-256。以下控制路径永远排除：

- `.xiliumini/**`，防止 Checkpoint/Trace 递归进入自身快照。
- `.git/**`，避免覆盖用户已有仓库元数据。

Git 通过显式 `--git-dir <checkpoint-root>/repo.git --work-tree <workspace>` 运行，使用仓库局部身份
创建允许空快照的 commit。所有 subprocess 调用均为 argv + `shell=False`，设置超时并限制输出；
底层错误转换为不含绝对路径的 `CheckpointError`。

### 模式语义

- `light`：每次节点完成后更新 checkpoint.json、RECOVERY.md、manifest 和 Git commit。
- `strict`：具有 light 的全部行为，并在每个 custom/update 事件保存 state.json、追加
  events.jsonl；同一节点内事件允许产生额外快照。
- `off`：不创建目录、不运行 Git、不返回 checkpoint 事件。

### 状态序列化

序列化器显式支持 JSON primitives、Path、tuple/list/dict、dataclass 和 LangChain message。
Path 不保存绝对 workspace；workspace 自身编码为 `"<workspace>"`。LangChain message 使用稳定的
message dict 结构并在恢复时重建。未知对象、模型、工具、callable 和 SecretStr 不落盘。

strict `state.json` 保存可恢复的完整 state。light `state_summary` 保存任务、session、图状态、
attempt/max_attempts、节点路由、todo、研究与 Agent 结果、工具证据、Memory 摘要、handoff、压缩
事件及最终/错误摘要；大文本按明确上限裁剪，但不删除路由和恢复所需字段。

## 恢复与节点路由

恢复遵循“先验证、后修改”：

1. 验证 workspace 是真实目录，且位于配置的 `data_dir/workspaces` 下。
2. 读取 checkpoint.json，验证 schema version、必需字段、模式和相对路径。
3. 验证 commit 可由内部 repo.git 解析，且状态 JSON 可完整反序列化。
4. 在修改 workspace 前创建 `pre-restore` 安全 commit。
5. 恢复目标 commit 中的普通文件，并删除快照后新增的普通文件。
6. 始终保留 `.xiliumini/**` 和 `.git/**`；每个删除/写入目标都重新解析并验证位于 workspace 内。
7. 若恢复过程失败，尽力回滚到 pre-restore commit，再抛出脱敏 CheckpointError。
8. strict 优先使用 state.json，light 使用 checkpoint.json.state_summary；重新注入 runtime 与
   workspace，并允许调用方 task/max_attempts 覆盖已保存值。
9. 返回 `(inputs, resume_event)`，resume_event 标记来源 commit、恢复节点和 trace 关联信息。

`runtime` 不写入文件，只在 `load_resume_inputs` 返回值中重新注入。当前 GraphState 没有顶层
messages，但恢复器兼容状态中存在的 LangChain messages；当前运行默认不新增无用消息字段。

为避免恢复后总是从 Planner 重跑，GraphState 增加 `resume_node`。workflow 的 START 改为条件
入口：

- 新任务：`resume_node="planner"`。
- Planner checkpoint：下一节点为 verifier。
- failed verifier 且 attempt 未耗尽：下一节点为 planner。
- passed verifier 或 attempt 已耗尽：下一节点为 final。
- final checkpoint：恢复为 final，确保确定性重建最终输出。

checkpoint 同时保存 `latest_node` 和计算后的 `next_node`，恢复时校验该组合，拒绝任意节点注入。

CLI 根 callback 增加 `--resume <workspace>`。该选项与普通子命令互斥，创建 Runtime 后调用专用
resume 流程，并复用现有 RuntimeEvent 展示和退出码。`resume_command()` 使用当前平台安全引用
包含空格的绝对 workspace 路径。

## TraceRecorder

`src/xiliumini/core/trace.py` 提供：

```python
class TraceRecorder:
    def __init__(self, runtime, task: str = ""): ...
    def start(self, inputs, *, resumed=False, resume_event=None): ...
    def record_custom_event(self, event): ...
    def record_graph_update(self, event): ...
    def end(self, *, status, latest_node, final_state) -> dict | None: ...
```

未配置 ID 时生成 `trace-<uuid hex>`，并拒绝含路径分隔符或越界字符的外部 trace_id。根目录为
`<workspace>/.xiliumini/traces/<trace_id>/`：

- `events.jsonl`：稳定序号、UTC 时间、类型和脱敏 payload。
- `trace.json`：运行元数据、统计和有界 timeline head/tail。
- `timeline.md`：人类可读时间线，不复制完整工具输出。

`start` 记录 run_start，并包含 task、session、模式、resumed 和脱敏 resume_event。重复 start 或 end
被拒绝，防止同一 recorder 生成矛盾时序。`end` 先写 run_end，再原子生成汇总文件；持续时间使用
单调时钟，展示时间使用 UTC ISO 8601。

统计更新规则：

- `tool_call`：`tool_calls += 1`
- `tool_result` 且 `ok is False`：`failed_tool_calls += 1`
- `tool_result` 且 `requires_approval is True`：`approval_count += 1`
- `handoff`：`handoff_count += 1`
- `checkpoint_saved`：`checkpoint_count += 1`
- graph update：对应 node 的 `node_visits[node] += 1`

现有 `run_react` 在生成 `tool_result` custom event 时，将解析后的
`requires_approval` 以严格布尔值提升到事件顶层，避免 Trace 重新解释任意 output 文本。
`SupervisorContext.delegate` 在一次委派完成后额外发出规范 `type="handoff"` 事件；该事件只含
Agent 名、attempt、ok 和有界 summary，不重复完整 instruction 或工具输出。

`trace.json` 保存 timeline 前 20 条和后 80 条；总数超过 100 时设置精确
`timeline_omitted = total - 100`，否则为 0 且不重复事件。

### Trace 模式

- `full`：写全部 custom、graph、run、error、resume 和 checkpoint 事件。
- `summary`：仅写 run、resume、节点、错误、handoff、approval、checkpoint 和结束等关键事件，
  但仍生成 events.jsonl、trace.json、timeline.md。
- `off`：不创建目录，所有记录方法和 end 返回 None。

统计只基于实际写入/接收的规范事件更新；同一个事件不会因同时用于 Checkpoint 而重复计数。

## 脱敏与有界输出

Checkpoint 与 Trace 共用递归 JSON-safe 脱敏器：

- key 名包含 token、secret、password、authorization、api_key 时写为 `[REDACTED]`。
- SecretStr 始终写为 `[REDACTED]`。
- workspace 绝对路径替换为 `<workspace>`。
- 工具 output、message content 和异常文本使用固定字符上限；截断写入显式标记。
- 不序列化模型、工具实例、callback、环境变量或任意对象 repr。

恢复只接受该序列化器支持的类型标签和已知 LangChain message 类型；未知标签导致验证失败，
不会动态导入或执行对象。

## Runtime 与事件数据流

1. Runtime 创建/验证 workspace 和 run context，构造 Memory、Checkpoint、Trace。
2. 新任务组装初始 state；恢复任务通过 CheckpointManager 重建 inputs 和 resume_event。
3. `stream_agent` 调用 Trace.start。
4. custom chunk 先进入 Trace；strict Checkpoint 保存当前 state/event；随后转换为 UI 事件。
5. updates chunk 按 node 合并 state，Trace 记录 node update，Checkpoint 保存并把返回的
   checkpoint_saved 事件交给 Trace；随后转换为 UI 事件。
6. 正常耗尽图流时保存 completed checkpoint 并 Trace.end(status="completed")。
7. 已映射错误保存 failed checkpoint、记录 error 并 Trace.end(status="failed")。
8. GeneratorExit、KeyboardInterrupt 或调用方关闭流时保存 interrupted checkpoint 并以
   interrupted 结束 Trace；原中断继续向上层传播或由现有 CLI 处理。

两个 recorder 的状态只属于本次 iterator。ContextVar 模型工厂仍在每次 `next()` 周围设置/重置，
Checkpoint/Trace 不改变其隔离语义。

## 错误处理

新增 `CheckpointError` 和 `TraceError`，继承 XiliuminiError 并提供稳定 code。错误消息不包含绝对
路径、commit 命令输出、工具参数或原始异常。

- Checkpoint 启用后任一关键步骤失败：停止运行并映射为 CheckpointError；不能宣称存在可恢复点。
- Trace full/summary 任一写入失败：停止运行并映射为 TraceError；启用追踪不能静默退化。
- off 模式不做 I/O，因此不会因目标目录不可写而失败。
- 损坏 JSON、未知 schema、非法 commit、workspace 越界：恢复前失败，不修改 workspace。
- pre-restore 后失败：尽力回滚；若回滚也失败，仍只返回稳定恢复错误并保留控制目录供人工检查。
- Git 不可用、超时或返回非零：CheckpointError，不回显 stderr。
- 结束阶段多次调用：第二次不重复写 run_end 或统计。

## 测试策略

实现遵循 TDD，分层覆盖：

### Checkpoint 单元与 Git 集成

- mode 的有效值、默认值和非法值 fallback。
- off 完全不创建目录且 save/load 行为明确。
- light 只生成规定文件；strict 额外生成 state.json/events.jsonl。
- manifest 排序、size/hash、控制目录排除和符号链接/越界防护。
- snapshot commit 可解析，workspace 根不新增 `.git`。
- 精确恢复修改/删除/新增文件，并保留 `.xiliumini` 与已有 `.git`。
- task/max_attempts 覆盖、Path/messages 往返、runtime 重新注入和 next node 计算。
- 损坏 checkpoint、未知 schema、非法 commit、越界 workspace 在修改文件前失败。
- 中途恢复失败回滚到 pre-restore commit。
- RECOVERY.md 包含任务、状态、manifest、commit 和安全引用的恢复命令。

### Trace 单元测试

- trace_id 生成与非法 ID 拒绝。
- full/summary/off 文件集合与事件过滤。
- 每类统计只增加一次；严格布尔 `ok is False` 和 `requires_approval is True`。
- node_visits、开始/结束状态、UTC 时间、单调 duration。
- 少于/等于/超过 100 条事件时 head 20、tail 80 和 omitted 精确无重复。
- 敏感 key、SecretStr、绝对 workspace、长 output 和异常内容脱敏/裁剪。
- timeline.md 不包含完整工具输出或 secret。

### Runtime、Graph 与 CLI 集成

- 原始 custom/update 的记录顺序、状态合并和 checkpoint 保存频率。
- light 每节点、strict 每事件、off 零调用。
- resume_node 的 planner/verifier/final 路由，以及 attempt 上限。
- completed/failed/interrupted 对应最终 checkpoint 与 Trace 状态。
- 同一 Runtime 交错 session 的 manager、trace_id、workspace 和统计隔离。
- `xiliumini --resume` 成功、workspace 不存在、越界、损坏 checkpoint 的退出码和无 traceback 输出。
- 现有 ask/chat、MemoryManager、InMemorySaver、RuntimeEvent 和模型 ContextVar 回归保持通过。

最终验证运行聚焦 pytest、全量 pytest、Ruff format check、Ruff lint、Pyright 和
`git diff --check`。Git 测试只使用临时目录和本机 Git，不访问网络、不修改开发仓库。

## 文档与范围

实现完成后更新 `.env.example`、README、SPEC 和 `项目进程.md`，说明模式、默认值、文件布局、
恢复命令、精确恢复语义、错误边界和 Trace 统计。

本步骤不实现远程 checkpoint、增量对象存储、多个命名 checkpoint 的选择 UI、Trace 上传或
跨机器 Git 同步。每个 workspace 只维护“最新可恢复 checkpoint”，内部 Git 历史保留较早 commit
供 pre-restore 回滚和人工诊断。
