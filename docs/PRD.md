# xiliumini PRD

## 1. 产品目标

xiliumini 是一个基于 Python 的最小 Agent CLI。用户输入消息后，Agent 调用大模型，按需执行工具或委派分析子 Agent，并返回最终结果。

MVP 只验证 Agent 核心闭环，不开发图形界面或业务模块。

## 2. MVP 功能

### CLI

- xiliumini doctor：检查运行配置。
- xiliumini ask "问题"：执行一次对话。
- xiliumini chat：进入连续对话。
- xiliumini chat --session ID：恢复会话。
- xiliumini sessions：列出本地会话。
- xiliumini --version：显示版本。

### 模型与 Agent

- 支持一个 OpenAI-compatible 接口和流式输出
- 使用 LangGraph 实现“模型 → 工具 → 模型”循环。
设置检查点，断联重新连接后可从检查点重启任务
- 设置最大循环步数。
- 内置 calculator 和 current_time 工具。
- 主 Agent 可通过 delegate_analysis 调用隔离的 analysis_agent。
- 子 Agent 有独立上下文，不允许递归委派。

### 会话与 Trace

- 本地保存、列出和恢复多轮会话。
- 记录模型调用、工具调用、耗时和错误事件。
- API Key 不得写入会话、日志或错误信息。

## 3. 配置

必填：

~~~dotenv
XILIUMINI_API_KEY=
XILIUMINI_MODEL=
~~~

可选：

~~~dotenv
XILIUMINI_BASE_URL=
XILIUMINI_TEMPERATURE=0
XILIUMINI_MAX_STEPS=8
XILIUMINI_TIMEOUT_SECONDS=60
XILIUMINI_DATA_DIR=.xiliumini

目标：
TUI
文件、Shell、浏览器、搜索和 MCP 工具。
长期记忆和多级子 Agent

## 4. 非目标

- 桌面端、Web 端或 。
- 用户系统、云同步和后台任务。

## 5. 验收标准

- 新环境按 README 可完成安装和配置。
- doctor 能发现缺失或无效配置。
- ask 和 chat 能流式返回模型内容。
- Agent 能正确调用两个内置工具。
- 主 Agent 能调用分析子 Agent。
- 会话可保存、列出和恢复。
- Ctrl+C 能安全退出。
- 自动测试和静态检查通过。
