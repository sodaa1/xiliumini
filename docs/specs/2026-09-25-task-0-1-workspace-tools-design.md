# Task 0.1：隔离工作区、主 Agent 循环与文件工具设计

## 1. 目标

Task 0.1 补齐一个具备明确执行边界的主 Agent 运行时：每个 session 拥有独立工作区，主 ReAct 循环位于 `core` 层，`agents` 目录只包含按需调用的功能型子 Agent，所有文件工具只能访问当前 session 的工作区。

本阶段完成后先运行自动验证并展示 diff，不创建 Git 提交；提交必须由用户确认。

## 2. 已确认的关键决策

- 每个 session 的工作区自动创建在：

  ```text
  <data_dir>/workspaces/<session_id>/
  ```

- 工作区跨同一 session 的多轮调用保留，不在进程退出时自动删除，以便后续会话恢复功能复用。
- 主 ReAct 循环迁移到 `src/xiliumini/core/agent.py`。
- `src/xiliumini/agents/` 只保留 `AnalysisAgent` 等功能型子 Agent，不再放置主循环包装器。
- 主循环初始消息明确为：

  ```python
  [
      SystemMessage(content=ACTOR_PROMPT),
      HumanMessage(content=task),
  ]
  ```

- ClaudeCode 工具源码仅用于理解行为和安全边界。由于其许可证为“© Anthropic PBC. All rights reserved”，本项目不直接复制源码，而使用 Python 独立实现。

## 3. 方案比较

### 方案 A：工具接收模型传入的 workspace

实现最简单，但模型能够尝试更改根目录，安全边界不可信，因此不采用。

### 方案 B：Runtime 为每个 session 创建工具实例并绑定 workspace

Runtime 根据 session ID 创建工作区，再构建绑定该工作区的工具实例。模型只能传递工作区内部的相对路径，无法改变根目录。安全边界清晰，测试也容易隔离。

这是采用的方案。

### 方案 C：通过全局当前目录或上下文变量隐式传递 workspace

调用接口较短，但异步并发 session 容易相互污染，也难以定位测试失败，因此不采用。

## 4. 状态与工作区

`RuntimeState` 是主循环的 LangGraph 状态，至少包含：

```python
class RuntimeState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    workspace: Path
    step_count: int
```

Runtime 接收 task 与 session ID 后执行：

1. 校验 session ID，现阶段只接受 UUID 字符串。
2. 将 `settings.data_dir` 规范化为绝对路径。
3. 创建 `<data_dir>/workspaces/<session_id>/`。
4. 创建 `RuntimeState(workspace=...)`。
5. 为该 workspace 创建文件工具。
6. 构建并执行主 ReAct 图。

工作区创建失败时返回稳定、脱敏的 Runtime 错误，不向终端输出底层路径之外的敏感异常信息。

## 5. 主 Agent ReAct 循环

主循环实现从 `graph/workflow.py` 迁移到 `core/agent.py`，职责包括：

- 使用 `ACTOR_PROMPT` 和用户 task 构建初始消息。
- 调用绑定工具的模型。
- 将流式模型 chunk 转换为 `TokenEvent`。
- 检测 Tool Call，并执行当前 session 绑定的工具。
- 将 `ToolMessage` 加回消息列表后再次调用模型。
- 发出 ToolStarted/ToolFinished 事件。
- 强制执行 `max_steps`。
- 在没有 Tool Call 时结束。

`runtime.py` 只负责运行时编排：创建 workspace、Provider、子 Agent、工具、主图以及对外事件流；不再包含 ReAct 路由细节。

`agents/analysis.py` 继续作为隔离的功能型子 Agent。`agents/main.py` 删除，避免把主循环误解为子 Agent。

## 6. 文件工具

四个工具都绑定固定 workspace，只接受相对路径。

### 6.1 公共路径边界

新增统一的 workspace 路径解析器：

1. 拒绝空路径、绝对路径、Windows 盘符和 UNC 路径。
2. 将目标路径与 workspace 拼接并执行规范化解析。
3. 解析后目标必须仍是 workspace 的后代或 workspace 本身。
4. 已存在路径及其父目录中的符号链接不得把目标引出 workspace。
5. 写入不存在的文件时，先校验最近的已存在父目录，再创建目录并重新校验。

所有工具返回稳定的可读错误，不返回 Python traceback。

### 6.2 FileReadTool

输入：`path`、可选 `offset`、可选 `limit`。

- 只读取 UTF-8 文本文件。
- 默认限制返回行数，防止大文件占满上下文。
- 拒绝目录、二进制或无法解码的文件。
- 输出包含相对路径、行范围和文本内容。

### 6.3 FileWriteTool

输入：`path`、`content`。

- 可创建父目录。
- 使用同目录临时文件加 `os.replace` 原子写入。
- 可创建或整体覆盖 UTF-8 文本文件。
- 返回相对路径和写入字符数。

### 6.4 FileEditTool

输入：`path`、`old_string`、`new_string`、`replace_all=False`。

- 文件必须存在且是 UTF-8 文本。
- `old_string` 不存在时失败。
- 默认要求 `old_string` 只匹配一次；多次匹配时提示增加上下文或显式启用 `replace_all`。
- 写入沿用 FileWriteTool 的原子替换能力。
- 返回替换次数和相对路径。

### 6.5 GrepTool

输入：`pattern`、可选 `path`、可选 `glob`、可选 `max_results`。

- 使用 Python 正则表达式独立实现，不依赖外部 `rg` 可执行文件。
- 只遍历 workspace 内文件，不跟随目录符号链接。
- 默认排除 `.git`、`.xiliumini`、`.venv` 和 `__pycache__`。
- 对单个文件大小、匹配行长度和总结果数设置上限。
- 输出 workspace 相对路径、行号和匹配文本。

## 7. 预期文件调整

新增：

- `src/xiliumini/core/agent.py`
- `src/xiliumini/core/state.py`
- `src/xiliumini/tools/workspace.py`
- `src/xiliumini/tools/file_read.py`
- `src/xiliumini/tools/file_write.py`
- `src/xiliumini/tools/file_edit.py`
- `src/xiliumini/tools/grep.py`
- 对应单元测试与 Runtime 集成测试

修改：

- `src/xiliumini/runtime.py`
- `src/xiliumini/tools/__init__.py`
- `src/xiliumini/prompts/main.py`
- `src/xiliumini/prompts/__init__.py`
- 现有 workflow/runtime/CLI 测试
- `SPEC.md` 和项目进程文档中的 Task 0.1 记录

删除：

- `src/xiliumini/agents/main.py`
- `src/xiliumini/graph/state.py`
- `src/xiliumini/graph/workflow.py`
- 无实际职责的 `graph` 包残留文件

## 8. 测试策略

实现遵循先失败测试、再最小实现：

- RuntimeState 保存规范化 workspace。
- 相同 session 重用同一工作区，不同 session 完全隔离。
- 主循环初始消息顺序是 `ACTOR_PROMPT` 后接用户 task。
- 普通回答、流式回答、工具调用和 max_steps 行为保持正确。
- 四个文件工具的成功路径与错误路径。
- `..`、绝对路径、盘符、UNC 和符号链接逃逸全部失败。
- FileEditTool 的零匹配、单匹配、多匹配和 replace_all。
- GrepTool 的 glob、结果限制、无匹配和非法正则。
- 现有测试不回归。

最终验证命令：

```powershell
uv run ruff check .
uv run pyright
uv run pytest -q
uv run xiliumini --help
```

## 9. 非目标

- Task 0.1 不实现持久化 session 元数据或 Trace。
- 不实现 Shell/Bash 工具。
- 不支持读取图片、PDF、Notebook 或任意二进制文件。
- 不实现工具权限询问 UI。
- 不允许文件工具访问 session workspace 以外的位置。

## 10. 提交边界

Task 0.1 的代码、测试和必要文档作为一个补充提交。完成实现和验证后，仅展示状态、diff 摘要和测试证据；收到用户明确确认后才创建提交。
