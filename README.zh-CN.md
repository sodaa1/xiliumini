[English](README.md)

# xiliumini

> 一个本地优先、可恢复、可审计的 Python Agent CLI。

`xiliumini` 以 OpenAI 兼容模型为推理核心，通过 LangGraph 组织意图路由、规划、专业 Agent
执行与结果验证，并提供 Textual 全屏交互界面。它面向需要在本地工作区中持续对话、检索资料、
修改文件和运行受控命令的个人开发场景。


## 关于项目

`xiliumini` 是一个用于学习与实践 Agent 工程化的本地优先项目。

项目从最基础的 **ReAct + Tool Calling** 开始，逐步加入：

- LangGraph 工作流编排
- Plan-Act-Verify 执行闭环
- Supervisor 与专业 Agent
- 分层 Memory 与上下文压缩
- Human-in-the-loop Approval
- Checkpoint / Resume
- Trace 可观测性
- 持久化 Session
- Textual TUI

它并非简单封装一次模型调用，而是尝试把 Agent 从“能够调用工具”逐步构建为具备
**规划、执行、验证、记忆、恢复与持续交互能力**的完整系统。

> [!NOTE]
> 本项目在学习和实现过程中参考了 [Wood-Q/MokioAgent](https://github.com/Wood-Q/MokioAgent)
> 的 Agent 演进路线，并围绕工作区隔离、Memory、Session、Harness 与 TUI 进行了独立实现。

当前版本为 `0.1.0`，要求 Python `>=3.12`，使用 `uv` 管理依赖。

### 工程能力

- **本地工作区隔离**：每个 Session 使用独立目录，文件工具拒绝绝对路径、父目录穿越和链接逃逸。
- **持久多轮交互**：保存最近会话，并在后续请求中复用 Workspace、Todo、Memory 与运行记录。
- **分层记忆**：管理 Rules、Working Memory、History Summary、Notepad 与长期偏好。
- **安全执行**：BashTool 只接受白名单命令，高风险操作可要求人工审批。
- **可恢复运行**：Checkpoint 保存状态与文件快照，支持从安全图节点继续执行。
- **结构化追踪**：Trace 记录节点、工具、审批、handoff、耗时和失败信息，并脱敏敏感字段。
- **联网检索**：优先使用 Tavily API Key，认证失败或未配置时可尝试 keyless 模式。

## 功能特性

### 多 Agent 架构

系统采用意图路由、Supervisor 与专业 Agent 协作的执行模式。

| Agent / Node | Responsibility |
| --- | --- |
| `IntentRouter` | 判断输入应进入普通对话还是完整工作流 |
| `ChatResponder` | 处理普通聊天和简单问答 |
| `Supervisor` | 规划任务、维护 Todo 并调度专业 Agent |
| `searchAgent` | 搜索网络资料并整理来源与结论 |
| `codeAgent` | 操作工作区文件、执行受限命令并完成代码任务 |
| `Verifier` | 根据测试、静态检查和工具证据判断任务是否完成 |
| `Final` | 汇总状态并输出最终回答 |

复杂任务不会交给单个模型一次完成。Supervisor 会根据任务状态调用专业 Agent，收集结构化
证据，并在 Verifier 未通过时进行有界重试。

## 工作流

```text
                         ┌──────────────┐
                         │  User Input  │
                         └──────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Intent Router │
                        └──────┬────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌────────────────┐          ┌────────────────┐
        │ ChatResponder  │          │   Supervisor   │◄──────────┐
        └───────┬────────┘          └───────┬────────┘           │
                │                 ┌─────────┼─────────┐          │
                │                 │         │         │          │
                │                 ▼         ▼         ▼          │
                │          ┌───────────┐ ┌─────────┐ ┌──────┐   │
                │          │searchAgent│ │codeAgent│ │ Todo │   │
                │          └─────┬─────┘ └────┬────┘ └──────┘   │
                │                └──────┬──────┘                 │
                │                       │ results                │
                │                       ▼                        │
                │               ┌──────────────┐                 │
                │               │  Supervisor  │                 │
                │               └──────┬───────┘                 │
                │                      ▼                         │
                │               ┌──────────────┐                 │
                │               │   Verifier   │── retry ────────┘
                │               └──────┬───────┘
                │                      │ pass
                └──────────┬───────────┘
                           ▼
                    ┌────────────┐
                    │   Final    │
                    └────────────┘
```

普通对话会通过 `ChatResponder` 快速响应。创建或修改文件、执行命令、联网搜索、编写代码、
运行测试、修复错误以及继续工作区任务等请求会进入完整 Agent 工作流。
## 项目结构

```text
xiliumini/
├── src/xiliumini/
│   ├── agents/       # 专业 Agent 与共用 ReAct 循环
│   ├── cli/          # Typer CLI 与 Textual TUI
│   ├── core/         # Approval、Checkpoint、Session、Trace
│   ├── graph/        # LangGraph 状态、节点、路由与验证
│   ├── providers/    # OpenAI 兼容模型适配
│   ├── storage/      # Session 与 Trace 存储接口
│   └── tools/        # 文件、搜索、命令、Todo、Notepad 等工具
├── docs/             # 产品、规格、进程、设计与实施文档
├── .env.example      # 配置模板
├── pyproject.toml    # 项目与依赖配置
└── uv.lock           # 可复现依赖锁
```

详细架构请查看 [项目进程](docs/项目进程.md)。

## 快速开始

### 环境要求

- Python 3.12 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 可用的 OpenAI 兼容模型、API Key 与服务地址
- Git（默认 Checkpoint 模式需要；关闭 Checkpoint 时可不安装）

### 安装

```powershell
git clone https://github.com/sodaa1/xiliumini.git
cd xiliumini
uv sync --no-dev
Copy-Item .env.example .env
```

编辑本地 `.env`，至少填写模型配置：

```dotenv
XILIUMINI_API_KEY=your-provider-key
XILIUMINI_MODEL=your-model-name
XILIUMINI_BASE_URL=https://your-provider.example/v1
```

默认 OpenAI 服务可省略 `XILIUMINI_BASE_URL`。Tavily Key 是可选项：

```dotenv
TAVILY_API_KEY=your-tavily-key
```

`.env` 已被 Git 忽略，请勿提交真实密钥。

检查环境并启动 TUI：

```powershell
uv run xiliumini doctor
uv run xiliumini
```

也可以安装为独立命令：

```powershell
uv tool install --editable .
xiliumini doctor
xiliumini
```

## 使用说明

### 启动 TUI

```powershell
uv run xiliumini
uv run xiliumini chat
```

同一 Session 会持续复用对话、Workspace、Todo 与 Memory；每轮运行会在该工作区中记录对应的
Checkpoint 与 Trace，适合连续完成任务：

```text
创建一个 Todo API
给它加入 SQLite
运行测试
修复刚才的报错
```

### 单次任务

```powershell
uv run xiliumini ask "分析当前项目结构并给出改进建议"
uv run xiliumini ask "计算 (17 + 5) * 3" --no-stream
```

### 工作区与恢复

```powershell
uv run xiliumini --workspace ".xiliumini/workspaces/demo" `
  --checkpoint-mode strict --trace-mode on ask "完成这个任务"

uv run xiliumini --resume ".xiliumini/workspaces/<session-id>"
```

根选项必须写在 `ask` 或 `chat` 前。常用选项包括：

| Option | Values | Description |
| --- | --- | --- |
| `--max-attempts` | 正整数 | Supervisor 最大尝试次数，默认 `3` |
| `--approval-mode` | `inline` / `auto` / `deny` | 高风险命令审批策略 |
| `--checkpoint-mode` | `light` / `strict` / `off` | Checkpoint 持久化级别 |
| `--trace-mode` | `on` / `off` | 是否保存执行 Trace |
| `--workspace` | 数据目录下的路径 | 使用指定工作区 |
| `--resume` | Checkpoint 工作区路径 | 恢复已保存的运行 |

运行 `uv run xiliumini --help`、`uv run xiliumini chat --help` 或
`uv run xiliumini ask --help` 可查看完整参数。

## 数据与安全边界

运行数据默认写入：

```text
.xiliumini/
├── session/                         # 多轮会话
├── USER_PREFERENCES.md              # 项目级长期偏好
└── workspaces/<session-id>/
    ├── TODO.md
    ├── NOTEPAD.md
    ├── HISTORY_SUMMARY.md
    └── .xiliumini/
        ├── checkpoints/             # 状态、恢复说明与文件快照
        └── traces/<trace-id>/        # 事件、统计与时间线
```
- 文件操作只能发生在当前 Workspace 中。
- 高风险 Bash 命令可在真正执行前要求人工审批。
- `TODO.md`、`NOTEPAD.md`、`HISTORY_SUMMARY.md` 等 Runtime 控制文件受到保护。
- API Key 不写入 Agent State，事件、错误和 Trace 会对敏感字段脱敏。
- BashTool 使用参数数组、命令白名单、运行时限和输出上限，但它不是操作系统级沙箱。
- Checkpoint 的 Git 快照保存工作区原始字节，可能包含敏感数据，应将运行目录视为私有数据。
- 恢复 Checkpoint 会还原普通文件，并删除快照之后新增的普通文件；恢复前请备份重要内容。

## 设计原则

- **Explicit over Magic**：规划、工具调用、handoff、验证、Checkpoint 与 Trace 均通过明确事件呈现。
- **Evidence-based Verification**：Verifier 依据文件状态、测试、静态检查、命令输出和工具证据判断结果。
- **Workspace First**：所有文件和命令操作都从明确的 Workspace 边界开始。
- **Recoverable Execution**：长任务应支持保存和恢复，而不是在异常后完全重新开始。
- **Memory != Full History**：优先保留规则、当前任务、工作状态、长期笔记与压缩历史，而非无限堆积消息。


## 文档入口

- [产品需求（PRD）](docs/PRD.md)
- [技术规格（SPEC）](docs/SPEC.md)
- [项目进程](docs/项目进程.md)

## 当前限制

- 仅支持 Python 3.12 及以上版本。
- 命令执行与生成代码共享当前用户权限，不提供容器或操作系统级隔离。
- Checkpoint 恢复发生在图节点边界，不能从工具调用的中间状态继续。
- `chat --session` 尚未接入按名称选择 Session 的能力，可在.xiliumini下找到对话位置。
