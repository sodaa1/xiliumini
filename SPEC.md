# xiliumini SPEC

## 1. 技术栈

- Python 3.12+
- uv
- Typer + Rich
- LangGraph + LangChain
- langchain-openai
- Pydantic Settings + python-dotenv
- pytest + pytest-asyncio + Ruff
- Pyright

## 2. 目录

~~~text
xiliumini/
├── .env.example
├── .gitignore
├── README.md
├── PRD.md
├── SPEC.md
├── pyproject.toml
├── src/xiliumini/
│   ├── __init__.py
│   ├── cli/
│   │   ├── tui
│   ├── config.py
│   ├── events.py
│   ├── runtime.py
│   ├── prompts/
│   │   └── task1.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── memory.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   ├── core/
│   │   ├── agent.py
│   │   ├── approval.py
│   │   ├── checkpoint.py
│   │   ├── harness_io.py
│   │   └── trace.py
│   ├── agents/
│   │   └── analysis.py
│   ├── providers/openai_compatible.py
│   ├── tools/
│   │   ├── calculator.py
│   │   ├── current_time.py
│   │   ├── delegate_analysis.py
│   │   ├── file_read.py
│   │   ├── file_write.py
│   │   ├── file_edit.py
│   │   ├── grep.py
│   │   ├── workspace.py
│   │   └── bash_tool.py
│   └── storage/
│       ├── sessions.py
│       └── traces.py
└── tests/
    ├── unit/
    └── integration/
~~~

## 3. 架构

~~~text
Typer CLI → Runtime → core/agent.py → LangGraph → Provider
                    │                      │
                    │                      └→ bounded workspace tools
                    └→ per-session workspace
~~~

参考 [MokioAgent](https://github.com/Wood-Q/MokioAgent) 的 CLI、Runtime、Graph、Provider、Tool 和 Trace 分层。

## 4. 核心约定

### Provider

- Settings 读取并校验 XILIUMINI_* 配置。
- create_chat_model(settings) 创建支持流式输出和工具绑定的 ChatOpenAI。
- 错误统一映射为配置、鉴权、限流、超时和通用 Provider 错误。

### Runtime 事件

- PlannerEvent(todo)
- ActorEvent(result, attempt)
- VerifierEvent(passed, reason, attempt)
- ProgressEvent(stage, message)
- FinalEvent(text, session_id)
- ErrorEvent(code, message)

### LangGraph

`GraphState` 包含 task、todos、research/tool evidence、result、graph_state、verification、
attempt、max_attempts、final_answer、session_id、workspace、resume_node，以及 Runtime 组装的三层
Memory 和节点摘要字段。

流程为 START → Planner → Verifier。Verifier 通过或次数耗尽时进入 Final，否则返回
Planner；Planner 通过专业子 Agent 工具完成研究和实现。Final 不调用模型，只格式化验证状态。
`core/agent.py` 同时消费 `stream_mode=["updates", "custom"]`，前者映射节点完成
事件，后者映射动作级进度。

### 工具

- calculator(expression)：使用 ast.parse 和运算符白名单；禁止 eval、名称、属性和函数调用；限制长度、幂指数和结果大小。
- current_time(timezone)：使用 zoneinfo.ZoneInfo；时钟可注入；无效时区返回可读错误。
- delegate_analysis(question)：调用无工具的 analysis_agent；设置独立超时和输出上限；禁止递归。
- file_read(path, offset, limit)：读取当前 session 工作区内的 UTF-8 文本。
- file_write(path, content)：在当前 session 工作区内原子写入 UTF-8 文本。
- file_edit(path, old_string, new_string, replace_all)：在工作区内执行受控文本替换。
- grep(pattern, path, glob, max_results)：在工作区内执行有界正则搜索。
- bash(argv, cwd, phase)：安全白名单包括工作区 Python 脚本、pytest、Ruff 只读检查、Pyright、
  compileall、`pip check`；安装/下载/开发服务器由 Task4 审批门控制。使用受约束的相对
  `cwd`，禁止 shell、`python -c`、未知模块与命令连接符，并限制超时和输出。
- 文件工具只接受相对路径，并拒绝目录穿越、绝对路径、盘符、UNC 和符号链接逃逸。
- 命令限制不是 OS 沙箱；生成的 Python 仍拥有当前进程的用户权限。

### 会话（历史规划）与当前 Trace

- 会话保存到 .xiliumini/sessions/SESSION_ID.json。
- 字段包含 schema_version、session_id、时间、model 和 messages。
- 使用临时文件和 os.replace 原子写入。
- 会话 transcript 持久化仍为历史规划；当前 Trace 由 Task4 的 workspace 内 recorder 实现，
  文件布局和脱敏规则见本文末尾，不使用旧 SESSION_ID.jsonl 方案。

## 5. CLI 行为

- doctor：检查 Python 版本、配置、数据目录和模型初始化。
- ask：创建会话并显示 Planner、专业 Agent 进度、Verifier、Final，支持 `--max-attempts`
  （默认 3）和 `--no-stream`。
- chat：启动内存对话；支持 /help、/status、/new、/exit。持久 transcript 恢复仍为历史规划。
- sessions：会话列表占位命令。
- 根参数：`--workspace/-w`、`--max-attempts`、`--approval-mode inline|auto|deny`、
  `--checkpoint-mode light|strict|off`、`--trace-mode on|off`；位于子命令之前并作用于
  ask/chat/resume。ask 保留原有子命令级 `--max-attempts`，显式局部值优先。
- --resume：恢复 workspace 最新 checkpoint；与子命令和 `--workspace` 互斥，见 Task4 补充。
- 退出码：成功 0、运行错误 1、配置错误 2。
- Ctrl+C 尝试保存 interrupted checkpoint 并结束 Trace 后退出，不显示 traceback。

## 6. 测试

- 单元测试不访问真实网络。
- Fake Chat Model 验证流式事件和工具循环。
- mock 验证 Provider 配置映射和错误转换。
- Typer CliRunner 验证命令、退出码和会话恢复。
- 真实 API 只做手动冒烟测试。

## 7. 执行 Task

每个 Task 完成后运行对应测试并单独提交。

### Task 0：项目骨架与 CLI

目标：建立可安装、可测试的 Typer 项目。

1. 创建 pyproject.toml、src 和 tests 目录。
2. 添加 CLI 命令骨架、配置类和异常类型。
3. 添加 .env.example 与最小 README。
4. 编写 CLI 和配置测试。
5. 运行 pytest、Ruff 和 CLI help 检查。
6. 编写两个常用子agent与几个tools补充prompts
7. 写好api接口完成最小的react形式的循环

提交：chore: scaffold xiliumini typer cli

### Task 0.1：隔离工作区、主 Actor 与文件工具

目标：补齐安全的运行时工作目录，并明确主循环与子 Agent 的职责边界。

1. 每个 session 自动创建并复用 `.xiliumini/workspaces/<session_id>/`。
2. `RuntimeState` 增加 `workspace: Path`。
3. 主 ReAct 循环迁移到 `core/agent.py`，`agents/` 只保留功能型子 Agent。
4. 注册 FileReadTool、FileWriteTool、FileEditTool 和 GrepTool。
5. 所有文件操作限制在 session workspace 内。
6. 自动测试覆盖路径逃逸、工具调用、工作区复用与 session 隔离。
7. 完成验证后由用户确认，再创建 Task 0.1 补充提交。

### Task 1：基础模型对话

目标：完成“用户输入 → 大模型 → 流式输出”。

1. 实现 OpenAI-compatible Provider。
2. 定义 Runtime 事件并转发流式 Token。
3. 接通 ask 和无工具的 chat。
4. 处理超时、中断和 Provider 错误。
5. 使用 Fake Model 与 mock 完成测试。

提交：feat: add openai-compatible streaming chat

### Task 2：LangGraph 与工具循环

目标：完成最小 ReAct 循环。

1. 实现状态、agent 节点、tools 节点和条件边。
2. 增加最大步数限制。
3. 实现并注册 calculator 与 current_time。
4. 将工具状态事件接入 CLI。
5. 测试直接回答、工具调用、工具失败和步数上限。

提交：feat: add langgraph react tool loop

### Task 3：会话与分析子 Agent

目标：支持会话恢复和一次受控委派。

1. 实现会话创建、原子保存、读取和列表。
2. 接通 chat --session、sessions 和斜杠命令。
3. 实现无工具的 analysis_agent。
4. 实现并注册 delegate_analysis。
5. 测试会话恢复、损坏文件、委派、超时和禁止递归。

提交：feat: add sessions and analysis subagent

### Task 4：Trace 与交付

目标：完成可调试、可运行的 MVP。

1. 添加模型、工具、错误和结束事件边界。
2. 实现脱敏 JSONL Trace。
3. 完善 Ctrl+C、错误信息和退出码。
4. 完成 README 和手动冒烟步骤。
5. 运行全量测试、静态检查和真实 API 验收。

提交：feat: harden and document xiliumini mvp

## 8. 完成定义

- 5 个 Task 分别提交。
- 自动测试和静态检查通过。
- CLI、模型对话、工具循环、分析子 Agent、会话和 Trace 可用。
- README 可指导首次运行。
# Task2 实现补充：Supervisor 与专业子 Agent

当前执行图为 `START → Planner → Verifier → Final → END`，验证失败且仍有次数时回到
Planner。Planner 通过 `TodoWriteTool`、`CallSearchAgentTool`、`CallCodeAgentTool` 调度，
不直接操作代码。旧 Actor 不再注册。

- `run_search_agent(state, instruction, *, writer=None, max_loops=4)`：仅绑定 Tavily
  `WebSearchTool`，返回摘要、queries、sources、messages、tool_events；没有
  `TAVILY_API_KEY` 时返回明确工具错误。环境变量和 `.env` 均支持。
- `run_code_agent(state, instruction, *, writer=None, max_loops=10)`：绑定工作区文件工具、
  argv 形式的受限 Bash、TodoUpdate、Notepad；显式维护 todo 状态。Bash 不是操作系统沙箱。
- 受限 Bash 支持工作区相对 `cwd`，并将拒绝、超时、输出截断和非零退出码分别映射为
  可诊断的 CLI 进度；不通过 Python subprocess 包装器绕过命令白名单。
- todo/notepad 已在 Task3 迁移到 session workspace 根目录的 canonical Markdown 文件；
  新问题重置 todo，重试保留进度。
- Verifier 同时检查 todo、委派结果、来源和实际检查证据；不能用模型声称成功替代失败、
  超时或截断的必要检查。耗尽尝试也经过 Final，展示未完成项。
- CLI 展示 Planner attempt 和子 Agent 工具进度；`--no-stream` 只显示最终结果。

以下原有阶段定义保留作为历史规划。

# Task3 实现补充：分层 Memory 与上下文压缩

Runtime 是 Memory 的唯一组装者，并为每个 stream/session 创建独立 MemoryManager，显式
传递到 workflow 与所有图节点，不使用全局变量或 ContextVar。三层结构为：

1. Rules Layer：固定安全规则与项目级长期偏好。
2. Working Memory：当前节点/任务/session、plan、todo、验收条件、研究与来源、最近 6 条
   handoff、code/verifier summary、last error 和 attempts。
3. History Summary：`HISTORY_SUMMARY.md`、`NOTEPAD.md` 摘要、上一轮 context summary 和
   最近 3 次 compression event。

固定安全规则 > 当前显式任务指令 > 已保存用户偏好。只有用户明确要求“记住/更新/忘记”
时 Planner 才调用 `PreferenceWriteTool`；项目偏好存放在
`<XILIUMINI_DATA_DIR>/USER_PREFERENCES.md`，跨 task/session 生效。

session workspace 的 canonical 文件为 `TODO.md`、`NOTEPAD.md`、
`HISTORY_SUMMARY.md`。合法旧 `.xiliumini/todos.json` 与 `.xiliumini/notepad.md` 仅在
canonical 文件不存在时迁移，旧文件保留；损坏输入不覆盖。

Planner 每次模型调用前用 tiktoken 估算 role/content/tool calls。默认在 64,000 token
窗口的 80% 触发压缩，保留最新 8,000 token；固定规则、完整当前任务和最近消息不由初次
裁剪移除。摘要持久化成功后才替换消息并记录事件；摘要或写入失败保持原消息。损坏 store
或不可压缩预算由 Runtime 映射为脱敏 `memory_error`。

# Task4 实现补充：审批、Checkpoint 与 Trace

`core/approval.py` 提供风险正则分类、不可变 ApprovalRequest/ApprovalDecision 和
inline/auto/deny 规范化（默认及非法值回退 inline）。BashTool 以实际 argv 前缀识别风险：
pip install、python -m pip install、uv add/sync/pip install、npm/pnpm install、yarn
install/add、curl/wget、uvicorn、python -m http.server。inline 调用应用提供的 handler，
缺失、异常或无效决策均拒绝；auto 放行，deny 拒绝。风险结果标记 requires_approval=True。
审批模式是 BashTool/Runtime/CLI 接口，无对应 Settings 环境变量。CLI inline handler 显示风险
原因与准确命令并通过 `typer.confirm` 等待人类决定，默认拒绝；auto/deny 不提示。handler 通过
运行级 ContextVar 隔离，不进入 GraphState、Checkpoint 或 Trace。
放行仍是 shell=False argv、受约束 cwd、超时和输出上限，不能执行通用 shell。

Runtime 每次 stream/resume 创建冻结上下文和独立 CheckpointManager/TraceRecorder。
在 `core/agent.py` 原始 updates/custom 边界，先持久记录，再映射现有 RuntimeEvent。
节点更新逐节点合并深拷贝状态；活 runtime 不落盘。checkpoint_saved 仅交给 Trace，
避免循环保存。终止时尝试终态保存和 Trace.end 各一次，持久化失败不掩盖原始异常。

`Runtime.stream_workspace(task, workspace, max_attempts=3)` 支持显式命名工作区；路径必须
严格位于 `<data_dir>/workspaces` 下，不能指向该根、穿越、链接或外部目录。新任务重置 Todo，
保留其他工作区文件。`core/agent.py` 的 `stream_agent_events(...)` 是兼容入口，调用
`create_runtime` 后委托 `stream_workspace`/`resume`，将 ProgressEvent 一次性映射为
custom_event，其余 RuntimeEvent 映射为 graph_event；不重复 Trace/Checkpoint 记录。
该 API 从 `xiliumini.core` 延迟公开导出。实际 Typer 文件为 `cli/__init__.py`，没有 `cli/app.py`。

| Settings / 环境变量 | 默认与合法值 | 持久化语义 |
| --- | --- | --- |
| checkpoint_mode / XILIUMINI_CHECKPOINT_MODE | light（默认）、strict、off | light 在启动、每节点和终态保存；strict 额外每 custom/update 保存完整状态与事件；off 零 I/O，拒绝恢复 |
| trace_mode / XILIUMINI_TRACE_MODE | full（默认）、summary、off | full 全部 raw custom/node/lifecycle；summary 关键事件；off 零 I/O |
| trace_id / XILIUMINI_TRACE_ID | None | 自动生成 trace-UUID；指定安全唯一 ID，已有目录拒绝覆盖 |

Settings 未提供模式使用默认值，环境变量非法/空模式严格报 ConfigError；只有内部 normalizer
和直接 recorder/runtime 上下文采用 light/full 回退。空白 trace_id 规范化 None，直接 recorder
也自动生成 ID（off 同样处理）；非空非法 ID 继续拒绝。Checkpoint 启用要求本机 Git，无网络依赖。布局为：

```text
<data_dir>/workspaces/<session-id>/.xiliumini/
├── checkpoints/
│   ├── checkpoint.json       # schema、task/status、latest/next node、attempt、时间、manifest、commit、state_summary
│   ├── RECOVERY.md           # 任务、状态、文件清单、commit、平台安全引用的恢复命令
│   ├── repo.git/             # 独立裸仓库；不创建/替换 workspace 根 .git
│   ├── state.json           # strict 完整 JSON-safe state
│   └── events.jsonl         # strict 收到的 custom/update 事件
└── traces/<trace-id>/
    ├── events.jsonl         # sequence、UTC timestamp、type、脱敏 payload
    ├── trace.json           # 概览、统计、timeline_head/tail/omitted
    └── timeline.md          # 人类可读时间线摘要
```

Checkpoint manifest 从实际提交的不可变 raw blob 派生，是按 POSIX 相对路径排序的
path/size/SHA-256 清单，避免节点后台写入导致两次读取不一致。快照包含普通文件，
即使被 .gitignore 忽略；casefold 排除根 .git/.xiliumini 的全部大小写变体，拒绝链接、越界与 Windows 路径冲突。
Git plumbing 读取/写入原始 bytes，跳过 clean/smudge filters，使用显式 git-dir/work-tree、
本地身份、禁用 hooks、shell=False 和 timeout。持久文件先 staging，再原子替换且 metadata
最后发布；发布失败逐文件尽力回滚，失败备份保留在 recovery/save-*。

CLI `--trace-mode on` 映射 Runtime `full`，`off` 关闭；未显式给 checkpoint/trace 根参数时保留
Settings/环境值，显式参数才覆盖，Settings/Runtime 的 `summary` 仍保留。
`xiliumini --resume "<workspace>"` 是根选项，与子命令及 `--workspace` 互斥，复用事件展示；
CLI 保留 saved task，根 `--max-attempts` 默认 3 并支持覆盖。
`Runtime.resume(workspace, task=None, max_attempts=3)` 支持覆盖。
加载先检查 workspace 严格位于配置 data_dir/workspaces 下、schema/全部必需 GraphState
字段存在与类型（含 session_id）、strict 状态一致性、
manifest 与内部 commit 祖先关系，再创建 pre-restore commit 并恢复普通文件原始内容。
删除 checkpoint 后新增普通文件；保留根 .git/.xiliumini 和无关现有空目录。目录占据目标文件
路径时仅删除该冲突目录；Git 不记录空目录，因此恢复前已删除的 checkpoint-era 空目录无法
重建。失败尽力恢复原始文件与 mutation 前空目录拓扑；回滚也失败仍返回稳定错误。
这是单 recorder 协议，无跨进程锁；硬终止可能需要从内部历史/备份人工恢复。

GraphState.resume_node 仅允许 planner/verifier/final。新任务 planner；保存 Planner 后 verifier；
failed Verifier 且次数未耗尽回 planner；passed/耗尽进入 final；Final checkpoint 仍进入 final。
工具内部不作为恢复入口。恢复重新注入 workspace 和所属 Runtime，保留 Todo 与 Memory 文件。

Trace.end 生成 trace_id/task/status、started_at/ended_at、单调 duration_ms、node_visits、
tool_calls、failed_tool_calls、approval_count、checkpoint_count、handoff_count。
失败仅 `ok is False` 计数，审批仅 `requires_approval is True` 计数（风险结果数量，不是
审批提示数量）。summary 过滤成功普通工具事件，但所有收到的 canonical 事件仍更新计数。
timeline_head 保留前 20，tail 保留随后/最新 80；总持久事件超过 100 时 omitted=total-100，
否则不重复。成功的重复 end 返回首次摘要；持久化失败封闭 recorder，拒绝后续写入。

结束状态 completed/failed/interrupted：KeyboardInterrupt、GeneratorExit 和 close 是 interrupted，
SystemExit 是 failed 并原样传播。child graph chunks 显式关闭并等待后台 executor 清理后，
才进行终态 snapshot 与 trace.end；close 异常不覆盖已有运行异常，所有终态操作恰好尝试一次。
启用持久化失败停止运行，Runtime/CLI 使用稳定
checkpoint_error/trace_error；CLI 运行错误与中断退出 1，配置或选项错误退出 2，无 traceback。
损坏/越界恢复在 mutation 前失败，off 不触碰对应持久目录。

共享脱敏器显式支持 primitives、Path、集合、dataclass 和已知 LangChain message，不动态导入
或执行任意对象。token/secret/password/authorization/api_key 字段及 SecretStr 脱敏，workspace
绝对路径替换为 <workspace>，大文本有界裁剪。RECOVERY.md 为人提供绝对引用命令。
Git 保存原始普通文件字节，文件内容不因元数据脱敏而被清除，内部历史须按工作区隐私保护。
