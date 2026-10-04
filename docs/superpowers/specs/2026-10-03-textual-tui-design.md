# xiliumini Textual TUI 设计

**日期：** 2026-10-03  
**状态：** 已确认  
**范围：** 多轮会话 TUI、结构化运行事件、同步审批弹窗、Rich 启动 Logo

## 背景

xiliumini 当前提供 Typer 驱动的逐行 CLI，并已经具备持久化多轮会话入口
`stream_session_events`。新的 TUI 需要在不破坏 `ask`、`doctor` 等命令的前提下，提供：

- 可持续多轮输入的全屏界面；
- Plan、工具调用、Agent 交接、检查点和最终回复的实时展示；
- Bash 风险命令的图形化同步审批；
- 使用 Rich renderable 呈现的 xiliumini 启动 Logo 和短动画。

仓库的实际 Python 包名是 `xiliumini`，因此需求示例中的 `mokioclaw` 路径统一映射到
`src/xiliumini`。应用类保留需求指定的 `MokioClawTuiApp` 名称，用户可见品牌统一为
`xiliumini`。

## 目标

1. 直接运行 `xiliumini` 时进入 Textual TUI。
2. 后台线程调用同步生成器 `stream_session_events`，UI 线程不执行模型、工具或文件操作。
3. 通过 Textual `Message` 将后台事件安全传回 UI 线程。
4. 同一 session 内支持连续多轮输入，并避免同一 session 的并发写入。
5. 风险 Bash 命令通过 `ApprovalGate` 阻塞工具线程，由 TUI 弹窗完成批准或拒绝。
6. 保持现有 Runtime 事件消费者兼容，并提供 TUI 所需的结构化事件元数据。
7. TUI 异常、审批拒绝或单轮失败后仍可继续下一轮。

## 非目标

- 不实现多个 turn 并发执行或输入排队。
- 不改变 BashTool 的风险分类规则。
- 不从 trace/checkpoint 文件反向轮询 UI 事件。
- 不在本任务中实现主题编辑器、历史 session 选择器或鼠标专属交互。
- 不替换 `xiliumini ask`、`xiliumini doctor` 等已有非交互命令。

## 方案选择

### 方案 A：UI 解析现有事件文案

改动最小，但工具名、交接方向、检查点状态都依赖字符串格式，文案变化就可能破坏界面。

### 方案 B：扩展稳定 ProgressEvent（采用）

为 `ProgressEvent` 增加带默认值的 `event_type` 和 `details`，保留原有 `stage`、`message`
字段。底层 LangGraph custom event 的结构化数据随稳定事件向上传递，已有消费者无需修改。

### 方案 C：TUI 读取 trace 文件

能减少 Runtime API 改动，但会产生事件延迟、重复、落盘模式依赖和生命周期竞态，不采用。

## 总体架构

```text
Input.Submitted
      │
      ▼
Textual thread worker（单 turn、exclusive）
      │
      ▼
stream_session_events
      │
      ├── AgentEventMessage ─────────────► UI 线程更新 Plan / EventLog / Final
      │
      └── approval_handler
              │
              ▼
         ApprovalGate.wait()
              ▲
              │
      ApprovalRequestedMessage
              │
              ▼
      ApprovalModal ──► gate.resolve(True / False)
```

Textual 的 `post_message` 支持从 thread worker 调用，因此 worker 只发布消息，所有 widget
修改均由消息处理器在 UI 线程完成。

## 文件结构

现有 `src/xiliumini/cli/tui.py` 是占位文件，迁移为同名包：

```text
src/xiliumini/cli/tui/
├── __init__.py       # run_tui 公共入口
├── app.py            # MokioClawTuiApp、消息类型、worker 和事件分发
├── approval.py       # ApprovalModal、ApprovalGate
├── logo.py           # Rich Logo renderable 和动画帧
└── styles.tcss       # 主界面、状态、事件和弹窗样式
```

测试按职责拆分到 `tests/unit/test_tui_*.py`，CLI 入口回归测试继续放在
`tests/integration/test_cli.py`。

## 界面设计

主界面自上而下由以下区域构成：

1. **Header / 状态栏**：显示 `🐾 xiliumini`、session ID 简写以及
   `idle`、`running`、`approval` 状态。
2. **Conversation**：每轮包含用户输入、默认折叠的“思考过程”和始终可见的最终结果。
   Plan、工具、handoff、checkpoint 与 Verifier 事件进入思考过程；完成后折叠标题显示耗时。
3. **Input**：提交一轮输入；运行期间禁用，结束或失败后恢复并重新聚焦。

布局目标：Conversation 占据剩余空间并保留各轮独立内容；窄终端下减少装饰和边距，但不隐藏
审批信息、折叠控件或最终回复。

## Logo

默认视觉：

```text
              🐾  <large block xiliumini wordmark>
        ━━━━━━━━━━━━━━  你的专属智能 Agent  ━━━━━━━━━━━━━━
```

- 欢迎页 Logo 使用五行块状 ASCII 字标并整体居中；爪印为暖橙色，字标使用青色到蓝色的
  离散字符颜色，副标题“你的专属智能 Agent”使用灰蓝色。
- 启动时约 0.8 秒：爪印轻微明暗变化，扫描线从中心或左侧展开。
- Logo 由 `rich.text.Text` / Rich renderable 构造，Textual `Static` 通过定时器更新帧。
- `NO_COLOR`、非交互环境和测试模式直接显示静态纯文本版本。
- 窄终端按可用宽度裁剪分隔线，不裁剪品牌名或副标题的语义内容。
- 动画结束后 Logo 在首次欢迎页保持；用户提交第一条消息后隐藏，并将品牌区收缩为单行
  session 状态，不持续占用主要事件区域。

## 事件协议

`ProgressEvent` 扩展为兼容结构：

```python
@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: str
    message: str
    event_type: str = "progress"
    details: dict[str, Any] = field(default_factory=dict)
```

对 custom payload：

- `event_type` 取可信字符串 `payload["type"]`，无效或缺失时回退 `progress`；
- `details` 保存结构化副本，供 TUI 展示工具、参数、结果状态、Agent 和检查点信息；
- `stage`、`message` 保持当前校验和语义；
- 对 UI 展示的长字段做边界裁剪，但不修改底层事件或持久化数据。

检查点保存目前只记录到 trace。实现时由 `stream_agent` 在保存成功后向运行时事件流发布
`checkpoint_saved` ProgressEvent，同时继续执行现有 trace 记录。终止阶段的检查点若生成器正在正常
消费则发送；若生成器被强制关闭，优先完成清理，不为发送 UI 事件阻塞退出。

事件到 UI 的映射：

| 事件 | UI 行为 |
|---|---|
| `PlannerEvent` | 写入当前轮次折叠的思考过程 |
| `ProgressEvent: tool_call` | 在思考过程中显示工具名和调用目标 |
| `ProgressEvent: tool_result` | 在思考过程中显示成功/失败、结果摘要和审批标记 |
| `ProgressEvent: search_results` | 在思考过程中显示搜索摘要 |
| `ProgressEvent: handoff` | 在思考过程中显示 Agent 交接和结果状态 |
| `ProgressEvent: checkpoint_saved` | 在思考过程中显示检查点状态和 latest node |
| 其他 `ProgressEvent` | 在思考过程中作为普通进度消息显示 |
| `VerifierEvent` | 在思考过程中显示验证通过/失败和原因 |
| `FinalEvent` | 在折叠区外追加最终回复并结束当前 turn |
| `ErrorEvent` | 在折叠区外追加错误并结束当前 turn |

`stream_session_events` 继续返回当前字典封装：`custom_event` 或 `graph_event`，避免破坏已有调用者。
TUI 根据 dataclass 序列化结果中的 `event_type` 和 `details` 做分发。

## 后台执行与多轮会话

- 每次 `Input.Submitted` 启动一个 `thread=True` 且互斥的 Textual worker。
- worker 使用同一个 `session_workspace` 调用 `stream_session_events`，因此复用磁盘 session。
- `run_tui` 在启动 Textual event loop 前调用 `load_or_create_session`，把 session ID 传给
  App；因此 Header 可立即显示 ID，同时 UI 线程不承担磁盘 I/O。
- turn 运行期间禁用输入框，避免 `session.json` 并发更新。
- worker 逐个将字典事件封装为 `AgentEventMessage` 并调用 `post_message`。
- worker 的 `finally` 必须关闭生成器并发送 turn 完成消息；UI 收到后恢复输入。
- 单轮异常转换为安全错误消息，不结束 App，也不向界面泄露未清洗异常细节。

## 审批模型

### ApprovalGate

`ApprovalGate` 内部使用 `threading.Event` 和锁：

- `wait()` 阻塞 BashTool 所在 worker，直至得到决策；
- `resolve(approved=True|False)` 只接受第一次决策并唤醒等待线程；
- 重复 resolve 不覆盖首个结果；
- App 关闭时将所有未完成 gate 解析为拒绝，防止线程永久等待。

TUI 的 approval handler：

1. 为 `ApprovalRequest` 创建 gate；
2. 发布包含 request、session workspace 和 gate 的 `ApprovalRequestedMessage`；
3. 调用 `gate.wait()`；
4. 将布尔结果转换为现有 `ApprovalDecision` 返回给 BashTool。

### ApprovalModal

弹窗显示：

- 工具名（默认 `BashTool`）；
- 风险原因；
- session workspace 的绝对路径；
- 完整命令，支持换行和滚动；
- Approve / Deny 按钮。

键位：

- `Y` 或 `Enter`：批准；
- `N` 或 `Escape`：拒绝。

Modal dismiss 回调调用 `gate.resolve`。Modal 激活期间主输入不可操作，Header 状态显示
`approval`。

## CLI 集成

- 裸命令 `xiliumini` 启动 TUI。
- `ask`、`doctor`、resume 和其他已有显式子命令保持原行为。
- `chat` 作为 TUI 的显式别名，session 参数语义后续可扩展；本任务不实现 session 浏览器。
- 配置加载失败时，在进入 alternate screen 前使用现有 Typer 错误风格退出。
- Textual 依赖添加到项目运行时依赖并锁定兼容版本范围。

## 错误处理与关闭

- 单个 turn 的 Runtime ErrorEvent 只结束该 turn，输入框恢复。
- worker 自身异常转换为通用错误事件，详细异常不直接显示给用户。
- App 退出时先拒绝全部待审批 gate，并标记 worker 停止接收 UI 更新；事件生成器在 worker
  重新取得控制权后由其自身 `finally` 关闭，避免跨线程关闭正在执行的生成器。
- thread worker 无法被强制安全终止；因此退出路径以解除审批等待、关闭生成器和避免新的 UI
  更新为目标。
- EventLog 对命令和结果设置展示长度上限，并明确标记截断。
- Unicode 或颜色能力不足时退化为纯文本图标和无颜色 Logo。

## 测试策略

### 单元测试

- Logo 精确文案、颜色关闭、窄宽度和动画帧边界。
- ApprovalGate 的等待、批准、拒绝、首个决策获胜和退出释放。
- ApprovalModal 的按钮以及 Y/Enter/N/Escape 键位。
- custom payload 到扩展 ProgressEvent 的映射和兼容默认值。
- 默认折叠、用户展开、耗时更新、多轮隔离，以及 Plan、工具、handoff、checkpoint、final、
  error 的显示映射。

### Textual 集成测试

- 使用 `App.run_test()` / pilot 提交输入并断言 worker 消息更新 UI。
- 多轮输入复用 session ID，运行期间输入禁用，结束后恢复焦点。
- 模拟审批请求，确认 modal 结果解除 gate 并恢复运行状态。
- 关闭 App 时未决审批被拒绝。

### 回归与静态检查

- 现有 CLI `ask`、`doctor`、resume 行为不变。
- 现有 RuntimeEvent 和 session 测试继续通过。
- 全量 `pytest`、Ruff lint/format、Pyright 和 `git diff --check`。

## 验收标准

1. `xiliumini` 可启动 TUI，显示确认过的 Logo 动画和 session 状态。
2. 连续提交至少两轮输入时使用同一 session，并能正确恢复输入状态。
3. Plan、工具调用/结果、handoff、checkpoint 和 verifier 默认收纳在可展开的思考过程；final
   和 error 直接可见。
4. 风险 Bash 请求可通过弹窗批准或拒绝，键盘和按钮均有效。
5. 退出 TUI 不留下等待审批的线程。
6. 非 TUI 命令和现有事件消费者保持兼容。
7. 本任务新增测试与全量质量检查通过。

## 参考

- [Textual Workers](https://textual.textualize.io/guide/workers/)
- [Textual Screens and Modal Results](https://textual.textualize.io/guide/screens/)
