# Task 0.1 Workspace Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every session a persistent isolated workspace, move the main ReAct loop into `core/agent.py`, and add four workspace-confined file tools.

**Architecture:** `Runtime` derives `<data_dir>/workspaces/<session_id>` from a validated UUID, builds tools bound to that immutable root, and invokes a LangGraph actor whose state includes the workspace. The actor owns model/tool routing; `agents/` contains only optional specialist agents. A shared path resolver rejects absolute, UNC, drive-qualified, traversal, and symlink-escape paths before any file operation.

**Tech Stack:** Python 3.12, pathlib, tempfile/os.replace, Pydantic 2, LangChain Core, LangGraph, pytest, pytest-asyncio, Ruff, Pyright.

**Spec:** `docs/superpowers/specs/2026-09-25-task-0-1-workspace-tools-design.md`

## Global Constraints

- Each session workspace is exactly `<data_dir>/workspaces/<uuid-session-id>/` and persists across turns.
- File tools accept workspace-relative paths only and never expose a parameter that can replace the bound workspace.
- ClaudeCode sources are behavioral references only; implementation must be original Python because `E:/Learning/ClaudeCode/package/LICENSE.md` reserves all rights.
- Text tools support UTF-8 text only; images, PDFs, notebooks, binaries, Shell, and permission UI remain out of scope.
- Existing streaming, tool lifecycle events, max-step behavior, calculator, current-time, and analysis delegation must not regress.
- Do not create any Git commit during Tasks 1–6. After verification, show the user the diff and wait for explicit commit approval.

## Review Focus

- A path containing a Windows drive or UNC prefix must fail even when tests run on another operating system; Task 1 pins this with `PureWindowsPath`-style inputs.
- A symlink inside the workspace that points outside must fail for reads, writes, edits, and grep; Tasks 1–3 include escape tests and skip only when the host cannot create symlinks.
- A write below a newly created directory must remain inside the workspace after directory creation; Task 2 revalidates the destination before `os.replace`.
- A non-unique `old_string` must not edit an arbitrary occurrence; Task 3 requires either a unique match or explicit `replace_all=True`.
- An invalid regex, oversized file, undecodable text, or exhausted result limit must return a bounded readable tool result rather than a traceback; Tasks 2–3 pin each behavior.

---

### Task 1: RuntimeState and workspace boundary

**Files:**
- Modify: `.gitignore`
- Modify: `src/xiliumini/errors.py`
- Create: `src/xiliumini/core/state.py`
- Create: `src/xiliumini/tools/workspace.py`
- Create: `tests/unit/test_workspace.py`

**Interfaces:**
- Consumes: `ToolExecutionError` from `xiliumini.errors`.
- Produces: `RuntimeState`, `WorkspaceError`, `create_session_workspace(data_dir, session_id) -> Path`, `resolve_workspace_path(workspace, relative_path, allow_root=False) -> Path`, `read_utf8_text(path, max_bytes) -> str`, and `atomic_write_utf8(path, content) -> None`.

- [ ] **Step 1: Make new tests and plan artifacts trackable**

Remove the repository-wide `tests/` and `docs/superpowers/` ignore entries. Keep `.superpowers/` ignored because it is runtime scratch data, not project documentation.

```diff
 __pycache__/
 *.py[cod]
 .superpowers/
-tests/
-docs/superpowers/
```

- [ ] **Step 2: Write failing workspace tests**

Create `tests/unit/test_workspace.py` with exact cases for UUID layout, stable reuse, invalid session IDs, traversal, POSIX absolute paths, Windows drives, UNC paths, and symlink escape:

```python
from pathlib import Path
from uuid import UUID

import pytest

from xiliumini.errors import WorkspaceError
from xiliumini.tools.workspace import create_session_workspace, resolve_workspace_path

SESSION_ID = "11111111-1111-4111-8111-111111111111"


def test_create_session_workspace_uses_uuid_directory(tmp_path: Path) -> None:
    first = create_session_workspace(tmp_path, SESSION_ID)
    second = create_session_workspace(tmp_path, str(UUID(SESSION_ID)))
    assert first == second == (tmp_path / "workspaces" / SESSION_ID).resolve()
    assert first.is_dir()


@pytest.mark.parametrize("session_id", ["", "session-1", "../escape", "not-a-uuid"])
def test_create_session_workspace_rejects_invalid_session_id(
    tmp_path: Path, session_id: str
) -> None:
    with pytest.raises(WorkspaceError, match="valid UUID"):
        create_session_workspace(tmp_path, session_id)


@pytest.mark.parametrize(
    "path",
    ["../outside.txt", "/etc/passwd", r"C:\\outside.txt", r"\\server\\share\\x"],
)
def test_resolve_workspace_path_rejects_escape_forms(tmp_path: Path, path: str) -> None:
    workspace = create_session_workspace(tmp_path, SESSION_ID)
    with pytest.raises(WorkspaceError, match="workspace"):
        resolve_workspace_path(workspace, path)


def test_resolve_workspace_path_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = create_session_workspace(tmp_path, SESSION_ID)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit symlink creation")
    with pytest.raises(WorkspaceError, match="workspace"):
        resolve_workspace_path(workspace, "link/secret.txt")
```

- [ ] **Step 3: Run the tests and verify the missing-module failure**

Run: `uv run pytest tests/unit/test_workspace.py -q`

Expected: collection fails because `xiliumini.tools.workspace` and `WorkspaceError` do not exist.

- [ ] **Step 4: Add state and workspace primitives**

Add to `errors.py`:

```python
class WorkspaceError(ToolExecutionError):
    """Raised when a workspace or workspace-relative path is invalid."""

    code = "workspace_error"
```

Create `core/state.py`:

```python
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class RuntimeState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    workspace: Path
    step_count: int
```

Create `tools/workspace.py` around these rules:

```python
import os
from pathlib import Path, PureWindowsPath
from tempfile import mkstemp
from uuid import UUID

from xiliumini.errors import WorkspaceError


def create_session_workspace(data_dir: Path, session_id: str) -> Path:
    try:
        canonical_id = str(UUID(session_id))
    except (ValueError, AttributeError, TypeError):
        raise WorkspaceError("session_id must be a valid UUID") from None
    if canonical_id != session_id.lower():
        raise WorkspaceError("session_id must be a canonical UUID")
    root = data_dir.expanduser().resolve() / "workspaces" / canonical_id
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve(strict=True)


def resolve_workspace_path(
    workspace: Path, relative_path: str, *, allow_root: bool = False
) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise WorkspaceError("path must be a non-empty workspace-relative path")
    windows_path = PureWindowsPath(relative_path)
    if (
        Path(relative_path).is_absolute()
        or windows_path.drive
        or relative_path.startswith(("//", "\\\\"))
    ):
        raise WorkspaceError("path must stay inside the workspace")
    root = workspace.resolve(strict=True)
    candidate = (root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        raise WorkspaceError("path must stay inside the workspace") from None
    if candidate == root and not allow_root:
        raise WorkspaceError("path must name an item inside the workspace")
    return candidate


def read_utf8_text(path: Path, max_bytes: int) -> str:
    if not path.is_file():
        raise WorkspaceError("path is not a readable file")
    if path.stat().st_size > max_bytes:
        raise WorkspaceError(f"file exceeds the {max_bytes}-byte limit")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise WorkspaceError("file is not valid UTF-8 text") from None


def atomic_write_utf8(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = mkstemp(prefix=".xiliumini-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
```

- [ ] **Step 5: Run workspace tests**

Run: `uv run pytest tests/unit/test_workspace.py -q`

Expected: all cases pass or only the symlink case is reported as skipped on a restricted host.

- [ ] **Step 6: Review checkpoint without committing**

Run: `git diff -- .gitignore src/xiliumini/errors.py src/xiliumini/core/state.py src/xiliumini/tools/workspace.py tests/unit/test_workspace.py`

Expected: only Task 1 boundary/state changes; do not stage or commit.

### Task 2: FileReadTool and FileWriteTool

**Files:**
- Create: `src/xiliumini/tools/file_read.py`
- Create: `src/xiliumini/tools/file_write.py`
- Create: `tests/unit/test_file_tools.py`

**Interfaces:**
- Consumes: `resolve_workspace_path`, `read_utf8_text`, and `atomic_write_utf8` from Task 1.
- Produces: `FileReadTool(workspace: Path)` and `FileWriteTool(workspace: Path)`, both LangChain `BaseTool` subclasses with names `file_read` and `file_write`.

- [ ] **Step 1: Write failing read/write tests**

Create `tests/unit/test_file_tools.py` with these initial cases:

```python
from pathlib import Path

import pytest

from xiliumini.tools.file_read import FileReadTool
from xiliumini.tools.file_write import FileWriteTool


def test_file_write_then_read_round_trip(tmp_path: Path) -> None:
    writer = FileWriteTool(workspace=tmp_path)
    reader = FileReadTool(workspace=tmp_path)
    result = writer.invoke({"path": "notes/item.txt", "content": "alpha\nbeta\n"})
    assert "notes/item.txt" in result
    output = reader.invoke({"path": "notes/item.txt", "offset": 2, "limit": 1})
    assert "2: beta" in output
    assert "1: alpha" not in output


@pytest.mark.parametrize("path", ["../escape.txt", r"C:\\escape.txt", r"\\server\\share\\x"])
def test_file_tools_reject_paths_outside_workspace(tmp_path: Path, path: str) -> None:
    assert (
        FileWriteTool(workspace=tmp_path)
        .invoke({"path": path, "content": "x"})
        .startswith("Error:")
    )
    assert FileReadTool(workspace=tmp_path).invoke({"path": path}).startswith("Error:")


def test_file_read_rejects_oversized_and_invalid_utf8_files(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_bytes(b"x" * 1_000_001)
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe")
    reader = FileReadTool(workspace=tmp_path)
    assert "limit" in reader.invoke({"path": "large.txt"})
    assert "UTF-8" in reader.invoke({"path": "binary.txt"})


def test_file_write_revalidates_parent_after_creation(tmp_path: Path) -> None:
    writer = FileWriteTool(workspace=tmp_path)
    assert not writer.invoke({"path": "new/child.txt", "content": "safe"}).startswith("Error:")
    assert (tmp_path / "new" / "child.txt").read_text(encoding="utf-8") == "safe"


def test_read_and_write_reject_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-file.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("host does not permit symlink creation")
    assert FileReadTool(workspace=tmp_path).invoke({"path": "linked.txt"}).startswith("Error:")
    assert (
        FileWriteTool(workspace=tmp_path)
        .invoke({"path": "linked.txt", "content": "changed"})
        .startswith("Error:")
    )
    assert outside.read_text(encoding="utf-8") == "secret"
```

- [ ] **Step 2: Run focused tests and verify missing-module failure**

Run: `uv run pytest tests/unit/test_file_tools.py -q`

Expected: collection fails because both tool modules are missing.

- [ ] **Step 3: Implement FileReadTool**

Use a strict Pydantic input schema and convert controlled failures into `Error:` results:

```python
from pathlib import Path
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from xiliumini.errors import WorkspaceError
from xiliumini.tools.workspace import read_utf8_text, resolve_workspace_path

MAX_FILE_BYTES = 1_000_000


class FileReadInput(BaseModel):
    path: str = Field(min_length=1)
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=200, ge=1, le=2_000)


class FileReadTool(BaseTool):
    name: str = "file_read"
    description: str = "Read a bounded range of lines from a UTF-8 file in the session workspace."
    args_schema: Type[BaseModel] = FileReadInput
    workspace: Path

    def _run(self, path: str, offset: int = 1, limit: int = 200) -> str:
        try:
            target = resolve_workspace_path(self.workspace, path)
            text = read_utf8_text(target, MAX_FILE_BYTES)
            lines = text.splitlines()
            selected = lines[offset - 1 : offset - 1 + limit]
            numbered = "\n".join(
                f"{number}: {line}" for number, line in enumerate(selected, start=offset)
            )
            end = offset + len(selected) - 1
            return f"File: {path}\nLines: {offset}-{max(offset - 1, end)}/{len(lines)}\n{numbered}"
        except (OSError, WorkspaceError) as exc:
            return f"Error: {exc}"
```

- [ ] **Step 4: Implement FileWriteTool with post-mkdir revalidation**

```python
from pathlib import Path
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from xiliumini.errors import WorkspaceError
from xiliumini.tools.workspace import atomic_write_utf8, resolve_workspace_path


class FileWriteInput(BaseModel):
    path: str = Field(min_length=1)
    content: str = Field(max_length=1_000_000)


class FileWriteTool(BaseTool):
    name: str = "file_write"
    description: str = "Create or replace a UTF-8 file inside the session workspace."
    args_schema: Type[BaseModel] = FileWriteInput
    workspace: Path

    def _run(self, path: str, content: str) -> str:
        try:
            target = resolve_workspace_path(self.workspace, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target = resolve_workspace_path(self.workspace, path)
            atomic_write_utf8(target, content)
            return f"Wrote {len(content)} characters to {path}"
        except (OSError, WorkspaceError) as exc:
            return f"Error: {exc}"
```

- [ ] **Step 5: Run read/write tests**

Run: `uv run pytest tests/unit/test_file_tools.py -q`

Expected: all read/write cases pass.

- [ ] **Step 6: Review checkpoint without committing**

Run: `git diff -- src/xiliumini/tools/file_read.py src/xiliumini/tools/file_write.py tests/unit/test_file_tools.py`

Expected: only read/write implementation and tests; do not stage or commit.

### Task 3: FileEditTool and GrepTool

**Files:**
- Create: `src/xiliumini/tools/file_edit.py`
- Create: `src/xiliumini/tools/grep.py`
- Modify: `tests/unit/test_file_tools.py`

**Interfaces:**
- Consumes: Task 1 workspace helpers and Task 2 size limit.
- Produces: `FileEditTool(workspace: Path)` named `file_edit` and `GrepTool(workspace: Path)` named `grep`.

- [ ] **Step 1: Add failing edit and grep tests**

Append cases covering zero/one/multiple replacements, replace-all, invalid regex, glob filtering, bounded results, oversized/invalid UTF-8 skipping, and symlink escape:

```python
from xiliumini.tools.file_edit import FileEditTool
from xiliumini.tools.grep import GrepTool


def test_file_edit_requires_unique_match_by_default(tmp_path: Path) -> None:
    target = tmp_path / "item.txt"
    target.write_text("old old", encoding="utf-8")
    editor = FileEditTool(workspace=tmp_path)
    assert "2 matches" in editor.invoke(
        {"path": "item.txt", "old_string": "old", "new_string": "new"}
    )
    assert target.read_text(encoding="utf-8") == "old old"
    result = editor.invoke(
        {"path": "item.txt", "old_string": "old", "new_string": "new", "replace_all": True}
    )
    assert "2 replacements" in result
    assert target.read_text(encoding="utf-8") == "new new"


def test_file_edit_reports_missing_text(tmp_path: Path) -> None:
    (tmp_path / "item.txt").write_text("alpha", encoding="utf-8")
    result = FileEditTool(workspace=tmp_path).invoke(
        {"path": "item.txt", "old_string": "beta", "new_string": "gamma"}
    )
    assert result.startswith("Error:")
    assert "not found" in result


def test_grep_filters_by_glob_and_limits_results(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("needle\nneedle\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")
    result = GrepTool(workspace=tmp_path).invoke(
        {"pattern": "needle", "glob": "*.py", "max_results": 1}
    )
    assert "a.py:1:needle" in result
    assert "b.txt" not in result
    assert "truncated" in result


def test_grep_returns_readable_invalid_regex_error(tmp_path: Path) -> None:
    result = GrepTool(workspace=tmp_path).invoke({"pattern": "["})
    assert result.startswith("Error: invalid regular expression")


def test_edit_and_grep_reject_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("needle", encoding="utf-8")
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit symlink creation")
    edit_result = FileEditTool(workspace=tmp_path).invoke(
        {"path": "linked/secret.txt", "old_string": "needle", "new_string": "changed"}
    )
    grep_result = GrepTool(workspace=tmp_path).invoke({"pattern": "needle", "path": "linked"})
    assert edit_result.startswith("Error:")
    assert grep_result.startswith("Error:")


@pytest.mark.parametrize("glob", ["../*.py", r"C:\\*.py", r"\\server\\share\\*"])
def test_grep_rejects_glob_escape_forms(tmp_path: Path, glob: str) -> None:
    result = GrepTool(workspace=tmp_path).invoke({"pattern": "x", "glob": glob})
    assert result.startswith("Error:")
```

- [ ] **Step 2: Run new tests and verify missing-module failure**

Run: `uv run pytest tests/unit/test_file_tools.py -q`

Expected: collection fails for `file_edit` or `grep`.

- [ ] **Step 3: Implement FileEditTool**

Implement a synchronous read-count-replace-atomic-write sequence:

```python
class FileEditInput(BaseModel):
    path: str = Field(min_length=1)
    old_string: str = Field(min_length=1)
    new_string: str = Field(default="", max_length=1_000_000)
    replace_all: bool = False


class FileEditTool(BaseTool):
    name: str = "file_edit"
    description: str = "Replace exact text in a UTF-8 file inside the session workspace."
    args_schema: Type[BaseModel] = FileEditInput
    workspace: Path

    def _run(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        try:
            target = resolve_workspace_path(self.workspace, path)
            original = read_utf8_text(target, MAX_FILE_BYTES)
            matches = original.count(old_string)
            if matches == 0:
                return "Error: old_string was not found"
            if matches > 1 and not replace_all:
                return f"Error: found {matches} matches; add context or set replace_all"
            updated = original.replace(old_string, new_string, -1 if replace_all else 1)
            atomic_write_utf8(target, updated)
            replacements = matches if replace_all else 1
            return f"Updated {path} with {replacements} replacements"
        except (OSError, WorkspaceError) as exc:
            return f"Error: {exc}"
```

- [ ] **Step 4: Implement bounded pure-Python GrepTool**

Use schema defaults `path="."`, `glob=None`, and `max_results=200` with a maximum of 1,000. Compile the regex once, resolve the search root with `allow_root=True`, skip symlinks and ignored directory parts, skip files over 1 MB or invalid UTF-8, and cap every displayed line at 500 characters:

```python
IGNORED_PARTS = {".git", ".xiliumini", ".venv", "__pycache__"}


def _validate_glob(pattern: str | None) -> str:
    if pattern is None:
        return "*"
    windows_pattern = PureWindowsPath(pattern)
    if Path(pattern).is_absolute() or windows_pattern.drive or ".." in windows_pattern.parts:
        raise WorkspaceError("glob must stay inside the workspace")
    return pattern


def _candidate_files(root: Path, glob: str | None):
    if root.is_file():
        yield root
        return
    pattern = _validate_glob(glob)
    for candidate in root.rglob(pattern):
        if candidate.is_file() and not candidate.is_symlink():
            yield candidate


def _run(
    self, pattern: str, path: str = ".", glob: str | None = None, max_results: int = 200
) -> str:
    try:
        expression = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regular expression: {exc.msg}"
    try:
        root = resolve_workspace_path(self.workspace, path, allow_root=True)
        matches: list[str] = []
        truncated = False
        for candidate in _candidate_files(root, glob):
            relative = candidate.relative_to(self.workspace)
            if any(part in IGNORED_PARTS for part in relative.parts):
                continue
            try:
                text = read_utf8_text(candidate, MAX_FILE_BYTES)
            except WorkspaceError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if expression.search(line):
                    if len(matches) == max_results:
                        truncated = True
                        break
                    matches.append(f"{relative.as_posix()}:{line_number}:{line[:500]}")
            if truncated:
                break
        if not matches:
            return "No matches found"
        suffix = "\n[results truncated]" if truncated else ""
        return "\n".join(matches) + suffix
    except (OSError, WorkspaceError) as exc:
        return f"Error: {exc}"
```

- [ ] **Step 5: Run all file-tool tests**

Run: `uv run pytest tests/unit/test_file_tools.py tests/unit/test_workspace.py -q`

Expected: all cases pass, with only host-restricted symlink tests skipped.

- [ ] **Step 6: Review checkpoint without committing**

Run: `git diff -- src/xiliumini/tools/file_edit.py src/xiliumini/tools/grep.py tests/unit/test_file_tools.py`

Expected: only edit/grep implementation and tests; do not stage or commit.

### Task 4: Move the main ReAct loop into core/agent.py

**Files:**
- Create: `src/xiliumini/core/agent.py`
- Modify: `src/xiliumini/core/__init__.py`
- Modify: `src/xiliumini/prompts/main.py`
- Modify: `src/xiliumini/prompts/__init__.py`
- Rename/replace test: `tests/unit/test_workflow.py` -> `tests/unit/test_agent.py`
- Delete: `src/xiliumini/agents/main.py`
- Delete: `src/xiliumini/graph/state.py`
- Delete: `src/xiliumini/graph/workflow.py`
- Delete: `src/xiliumini/graph/__init__.py`

**Interfaces:**
- Consumes: `RuntimeState`, `ACTOR_PROMPT`, runtime events, LangChain model and tools.
- Produces: `build_actor(model, tools, max_steps, checkpointer=None, event_sink=None)` and `MAX_STEPS_MESSAGE`.

- [ ] **Step 1: Rewrite workflow tests against the actor interface**

Move the existing direct-answer, tool-loop, tool-error, and max-step tests to `tests/unit/test_agent.py`. Change imports to:

```python
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from xiliumini.core.agent import MAX_STEPS_MESSAGE, build_actor
from xiliumini.prompts import ACTOR_PROMPT
```

Pass state containing `workspace=tmp_path`, and add a prompt-order assertion:

```python
@pytest.mark.asyncio
async def test_actor_builds_actor_prompt_then_user_task(tmp_path: Path) -> None:
    model = ScriptedModel([AIMessage(content="answer")])
    graph = build_actor(model, [], max_steps=2)
    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="inspect project")],
            "session_id": "11111111-1111-4111-8111-111111111111",
            "workspace": tmp_path,
            "step_count": 0,
        }
    )
    first_call = model.calls[0]
    assert isinstance(first_call[0], SystemMessage)
    assert first_call[0].content == ACTOR_PROMPT
    assert isinstance(first_call[1], HumanMessage)
    assert first_call[1].content == "inspect project"
```

- [ ] **Step 2: Run actor tests and verify import failure**

Run: `uv run pytest tests/unit/test_agent.py -q`

Expected: collection fails because `core.agent` and `ACTOR_PROMPT` do not exist.

- [ ] **Step 3: Rename the prompt and implement the actor**

Rename `MAIN_SYSTEM_PROMPT` to `ACTOR_PROMPT`, update exports, and move the full implementation from `graph/workflow.py` into `core/agent.py` with these deliberate changes:

```python
from xiliumini.core.state import RuntimeState
from xiliumini.prompts import ACTOR_PROMPT


async def _invoke_model(bound_model, messages, event_sink):
    if event_sink is None or not hasattr(bound_model, "astream"):
        response = await bound_model.ainvoke(messages)
        if event_sink is not None:
            text = _content_text(response.content)
            if text:
                await event_sink(TokenEvent(text=text))
        return response
    aggregate: AIMessageChunk | None = None
    async for chunk in bound_model.astream(messages):
        text = _content_text(chunk.content)
        if text:
            await event_sink(TokenEvent(text=text))
        aggregate = chunk if aggregate is None else aggregate + chunk
    if aggregate is None:
        return AIMessage(content="")
    return AIMessage(
        content=aggregate.content,
        additional_kwargs=aggregate.additional_kwargs,
        response_metadata=aggregate.response_metadata,
        tool_calls=aggregate.tool_calls,
        invalid_tool_calls=aggregate.invalid_tool_calls,
        id=aggregate.id,
    )


def build_actor(model, tools, max_steps, checkpointer=None, event_sink=None):
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    tool_list = list(tools)
    tools_by_name = {tool.name: tool for tool in tool_list}
    bound_model = model.bind_tools(tool_list)

    async def call_actor(state: RuntimeState) -> dict[str, Any]:
        model_messages = [SystemMessage(content=ACTOR_PROMPT), *state["messages"]]
        response = await _invoke_model(bound_model, model_messages, event_sink)
        step_count = state["step_count"] + 1
        messages = [response]
        if response.tool_calls and step_count >= max_steps:
            messages.append(AIMessage(content=MAX_STEPS_MESSAGE))
        return {"messages": messages, "step_count": step_count}

    def route_after_actor(state: RuntimeState) -> str:
        latest = state["messages"][-1]
        if isinstance(latest, AIMessage) and latest.tool_calls:
            return "tools"
        return END

    async def call_tools(state: RuntimeState) -> dict[str, list[ToolMessage]]:
        request = state["messages"][-1]
        results: list[ToolMessage] = []
        for call in request.tool_calls:
            started = perf_counter()
            if event_sink is not None:
                await event_sink(ToolStartedEvent(name=call["name"], call_id=call["id"]))
            selected = tools_by_name.get(call["name"])
            try:
                if selected is None:
                    raise ValueError("unknown tool")
                content = str(await selected.ainvoke(call["args"]))
                ok = not content.startswith("Error:")
            except Exception:
                content = "Error: tool execution failed"
                ok = False
            duration_ms = (perf_counter() - started) * 1000
            results.append(
                ToolMessage(
                    content=content,
                    name=call["name"],
                    tool_call_id=call["id"],
                    additional_kwargs={"duration_ms": duration_ms, "ok": ok},
                )
            )
            if event_sink is not None:
                await event_sink(
                    ToolFinishedEvent(
                        name=call["name"],
                        call_id=call["id"],
                        duration_ms=duration_ms,
                        ok=ok,
                    )
                )
        return {"messages": results}

    graph = StateGraph(RuntimeState)
    graph.add_node("actor", call_actor)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "actor")
    graph.add_conditional_edges("actor", route_after_actor, {"tools": "tools", END: END})
    graph.add_edge("tools", "actor")
    return graph.compile(checkpointer=checkpointer)
```

Keep the system prompt ephemeral: it is prepended to each model call but not added to checkpointed state, preventing duplicate system messages across turns.

- [ ] **Step 4: Remove misleading main-agent and graph modules**

Delete `agents/main.py` and the old `graph` package only after all imports reference `core.agent` and `core.state`. Export `RuntimeState`, `build_actor`, and `MAX_STEPS_MESSAGE` from `core/__init__.py`.

- [ ] **Step 5: Run actor and analysis-agent tests**

Run: `uv run pytest tests/unit/test_agent.py tests/unit/test_delegate_analysis.py -q`

Expected: all tests pass; `AnalysisAgent` remains isolated and never binds tools.

- [ ] **Step 6: Search for stale architecture imports**

Run: `rg -n "graph\.workflow|graph\.state|MAIN_SYSTEM_PROMPT|agents\.main|build_workflow" src tests`

Expected: no matches.

- [ ] **Step 7: Review checkpoint without committing**

Run: `git diff -- src/xiliumini/core src/xiliumini/graph src/xiliumini/agents src/xiliumini/prompts tests/unit/test_agent.py tests/unit/test_workflow.py`

Expected: one ownership move with no duplicate main loop; do not stage or commit.

### Task 5: Bind session workspaces and file tools in Runtime

**Files:**
- Modify: `src/xiliumini/runtime.py`
- Modify: `src/xiliumini/tools/__init__.py`
- Modify: `tests/integration/test_runtime.py`
- Modify: `tests/unit/test_current_time.py`

**Interfaces:**
- Consumes: `build_actor`, `create_session_workspace`, four workspace-bound tools, existing stateless tools, and `Settings.data_dir`.
- Produces: `get_workspace_tools(workspace: Path) -> list[BaseTool]`; `Runtime(..., data_dir: Path, workspace_tool_factory=...)`; Runtime state initialized with `workspace`.

- [ ] **Step 1: Add failing tool-registry and workspace integration tests**

Use a canonical UUID constant throughout runtime tests:

```python
SESSION_ID = "11111111-1111-4111-8111-111111111111"
SECOND_SESSION_ID = "22222222-2222-4222-8222-222222222222"
```

Add registry and workspace assertions:

```python
def test_workspace_tools_are_bound_to_one_root(tmp_path: Path) -> None:
    tools = get_workspace_tools(tmp_path)
    assert [tool.name for tool in tools] == ["file_read", "file_write", "file_edit", "grep"]
    assert all(tool.workspace == tmp_path for tool in tools)


@pytest.mark.asyncio
async def test_runtime_creates_and_reuses_session_workspace(tmp_path: Path) -> None:
    captured: list[Path] = []

    def workspace_tools(workspace: Path):
        captured.append(workspace)
        return []

    runtime = Runtime(
        DirectModel(), data_dir=tmp_path, workspace_tool_factory=workspace_tools, max_steps=2
    )
    await collect(runtime.astream("first", SESSION_ID))
    await collect(runtime.astream("second", SESSION_ID))
    assert captured[0] == captured[1] == (tmp_path / "workspaces" / SESSION_ID).resolve()


@pytest.mark.asyncio
async def test_runtime_isolates_different_sessions(tmp_path: Path) -> None:
    captured: list[Path] = []
    runtime = Runtime(
        DirectModel(),
        data_dir=tmp_path,
        workspace_tool_factory=lambda workspace: captured.append(workspace) or [],
        max_steps=2,
    )
    await collect(runtime.astream("one", SESSION_ID))
    await collect(runtime.astream("two", SECOND_SESSION_ID))
    assert captured[0] != captured[1]


@pytest.mark.asyncio
async def test_runtime_reports_invalid_session_without_traceback(tmp_path: Path) -> None:
    runtime = Runtime(DirectModel(), data_dir=tmp_path, max_steps=2)
    events = await collect(runtime.astream("question", "../escape"))
    assert events == [ErrorEvent(code="workspace_error", message="session_id must be a valid UUID")]
```

Define the local helper exactly once:

```python
async def collect(stream):
    return [event async for event in stream]
```

- [ ] **Step 2: Run focused tests and confirm failures**

Run: `uv run pytest tests/integration/test_runtime.py tests/unit/test_current_time.py -q`

Expected: failures for the missing registry/factory parameters and non-UUID legacy fixtures.

- [ ] **Step 3: Add workspace tool registry**

Update `tools/__init__.py`:

```python
def get_workspace_tools(workspace: Path) -> list[BaseTool]:
    return [
        FileReadTool(workspace=workspace),
        FileWriteTool(workspace=workspace),
        FileEditTool(workspace=workspace),
        GrepTool(workspace=workspace),
    ]
```

Keep `get_builtin_tools()` returning calculator and current_time only so its existing safety contract remains explicit.

- [ ] **Step 4: Refactor Runtime orchestration**

Change Runtime construction and per-call setup:

```python
ToolFactory = Callable[[Path], Sequence[BaseTool]]


class Runtime:
    def __init__(
        self,
        model: Any,
        tools: Sequence[BaseTool] = (),
        max_steps: int = 8,
        checkpointer: Any | None = None,
        data_dir: Path = Path(".xiliumini"),
        workspace_tool_factory: ToolFactory | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._workspace_tool_factory = workspace_tool_factory or (lambda _workspace: ())
        self._model = model
        self._tools = list(tools)
        self._max_steps = max_steps
        self._checkpointer = checkpointer or InMemorySaver()

    async def astream(self, task: str, session_id: str) -> AsyncIterator[RuntimeEvent]:
        try:
            workspace = create_session_workspace(self._data_dir, session_id)
            tools = [*self._tools, *self._workspace_tool_factory(workspace)]
        except WorkspaceError as exc:
            yield ErrorEvent(code=exc.code, message=str(exc))
            return
        queue: asyncio.Queue[RuntimeEvent | object] = asyncio.Queue()
        finished = object()
        result_holder: dict[str, Any] = {}

        async def emit(event: RuntimeEvent) -> None:
            await queue.put(event)

        actor = build_actor(
            self._model,
            tools,
            self._max_steps,
            checkpointer=self._checkpointer,
            event_sink=emit,
        )
        initial_state: RuntimeState = {
            "messages": [HumanMessage(content=task)],
            "session_id": session_id,
            "workspace": workspace,
            "step_count": 0,
        }

        async def execute() -> None:
            try:
                result_holder["result"] = await actor.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": session_id}},
                )
            except Exception as exc:
                error = classify_provider_error(exc)
                await emit(ErrorEvent(code=error.code, message=str(error)))
            finally:
                await queue.put(finished)

        task_handle = asyncio.create_task(execute())
        try:
            while True:
                event = await queue.get()
                if event is finished:
                    break
                yield event
            await task_handle
        finally:
            if not task_handle.done():
                task_handle.cancel()
                with suppress(asyncio.CancelledError):
                    await task_handle

        result = result_holder.get("result")
        if result is None:
            return
        final = next(
            (
                message
                for message in reversed(result["messages"])
                if isinstance(message, AIMessage) and not message.tool_calls
            ),
            AIMessage(content=""),
        )
        yield FinalEvent(text=_message_text(final.content), session_id=session_id)
```

Update `create_runtime(settings)` to pass `data_dir=settings.data_dir` and `workspace_tool_factory=get_workspace_tools` while keeping calculator, current_time, and delegate_analysis as stateless tools.

- [ ] **Step 5: Update legacy runtime test session IDs and assertions**

Replace `session-1`, `session-stream`, and `session-tool` with canonical UUIDs. Add `tmp_path: Path` to every test that directly constructs `Runtime`, and pass `data_dir=tmp_path` so tests never create `.xiliumini` in the repository. Keep all existing first-token-before-finish, tool-start-before-tool-finish, error redaction, and final-event assertions. A representative conversion is:

```python
@pytest.mark.asyncio
async def test_runtime_emits_token_then_final_event(tmp_path: Path) -> None:
    runtime = Runtime(DirectModel(), tools=[], max_steps=2, data_dir=tmp_path)
    events = [event async for event in runtime.astream("hello", SESSION_ID)]
    assert events == [
        TokenEvent(text="answer"),
        FinalEvent(text="answer", session_id=SESSION_ID),
    ]
```

- [ ] **Step 6: Run all runtime and tool tests**

Run: `uv run pytest tests/integration/test_runtime.py tests/unit/test_agent.py tests/unit/test_file_tools.py tests/unit/test_workspace.py tests/unit/test_current_time.py -q`

Expected: all cases pass, with only unsupported symlink tests skipped.

- [ ] **Step 7: Review checkpoint without committing**

Run: `git diff -- src/xiliumini/runtime.py src/xiliumini/tools tests/integration/test_runtime.py tests/unit/test_current_time.py`

Expected: Runtime owns workspace orchestration while `core.agent` owns the loop; do not stage or commit.

### Task 6: Documentation, type checking, and full verification

**Files:**
- Modify: `SPEC.md`
- Modify: `README.md`
- Modify: `项目进程.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Verify: all source and tests from Tasks 1–5

**Interfaces:**
- Consumes: completed Task 0.1 behavior.
- Produces: user-facing workspace/tool documentation, tracked design/plan, type-check dependency, and final evidence for commit approval.

- [ ] **Step 1: Document Task 0.1 boundaries**

Add a Task 0.1 section to `SPEC.md` and `项目进程.md` stating the exact workspace layout, `core/agent.py` ownership, four tool names, relative-path-only rule, and user-confirmed commit boundary. Update README architecture and examples without claiming session persistence or Trace is complete.

- [ ] **Step 2: Add the missing Pyright development dependency**

Run: `uv add --dev "pyright>=1.1,<2"`

Expected: only `pyproject.toml` and `uv.lock` dependency metadata change.

- [ ] **Step 3: Run formatter/linter checks**

Run: `uv run ruff format --check .`

Expected: exit 0. If it reports files, run `uv run ruff format .`, inspect the formatting diff, then rerun the check.

Run: `uv run ruff check .`

Expected: exit 0 with no diagnostics.

- [ ] **Step 4: Run static type checking**

Run: `uv run pyright`

Expected: zero errors. Fix type errors at their source; do not add blanket ignores.

- [ ] **Step 5: Run the complete test suite**

Run: `uv run pytest -q`

Expected: all tests pass, with any symlink skips explicitly reported and explained by host permissions.

- [ ] **Step 6: Verify the CLI still starts**

Run: `uv run xiliumini --help`

Expected: exit 0 and the `doctor`, `ask`, `chat`, and `sessions` commands remain listed.

- [ ] **Step 7: Verify architecture and repository hygiene**

Run: `rg -n "graph\.workflow|graph\.state|MAIN_SYSTEM_PROMPT|agents\.main|build_workflow" src tests`

Expected: no matches.

Run: `git status --short`

Expected: only Task 0.1 source, tests, docs, dependency metadata, and the reviewed spec/plan are changed; no `.xiliumini`, `.env`, cache, or temporary files.

- [ ] **Step 8: Present evidence and wait for commit confirmation**

Run: `git diff --stat` and `git diff --check`.

Expected: `git diff --check` exits 0. Report the test, Ruff, Pyright, CLI, and diff results to the user. Do not run `git add` or `git commit` until the user explicitly approves the final Task 0.1 commit.
