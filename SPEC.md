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
│   │   ├── nodes.py
│   │   └── workflow.py
│   ├── core/
│   │   └── agent.py
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

`GraphState` 包含 task、todo、result、execution、graph_state、verification、
attempt、max_attempts、final_answer、session_id 和 workspace。

流程为 START → Planner → Actor → Verifier。Verifier 通过或次数耗尽时进入
Final，否则返回 Actor；Planner 只运行一次。Final 不调用模型，只格式化验证状态。
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
- command(argv)：仅允许 `python <相对脚本.py>` 或 `python -m pytest`，固定工作区、
  禁止 shell、`python -c`、任意模块、绝对/穿越路径与命令连接符，并限制超时和输出。
- 文件工具只接受相对路径，并拒绝目录穿越、绝对路径、盘符、UNC 和符号链接逃逸。
- 命令限制不是 OS 沙箱；生成的 Python 仍拥有当前进程的用户权限。

### 会话与 Trace

- 会话保存到 .xiliumini/sessions/SESSION_ID.json。
- 字段包含 schema_version、session_id、时间、model 和 messages。
- 使用临时文件和 os.replace 原子写入。
- Trace 保存到 .xiliumini/traces/SESSION_ID.jsonl。
- Trace 只记录事件类型、调用 ID、耗时、状态和错误码，不记录完整内容、工具参数或 API Key。

## 5. CLI 行为

- doctor：检查 Python 版本、配置、数据目录和模型初始化。
- ask：创建会话并显示 Planner、Actor、Verifier、Final，支持 `--max-attempts`
  （默认 3）和 `--no-stream`。
- chat：启动或恢复会话；支持 /help、/status、/new、/exit。
- sessions：按更新时间倒序显示会话。
- 退出码：成功 0、运行错误 1、配置错误 2。
- Ctrl+C 保存当前会话后退出，不显示 traceback。

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
