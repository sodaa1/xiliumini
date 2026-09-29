from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from xiliumini.tools.bash_tool import BashTool
from xiliumini.tools.calculator import calculator
from xiliumini.tools.command import CommandTool
from xiliumini.tools.current_time import current_time
from xiliumini.tools.file_edit import FileEditTool
from xiliumini.tools.file_read import FileReadTool
from xiliumini.tools.file_write import FileWriteTool
from xiliumini.tools.grep import GrepTool
from xiliumini.tools.notepad import NotepadAppendTool, NotepadReadTool
from xiliumini.tools.todo import TodoStore, TodoUpdateTool, TodoWriteTool


def get_builtin_tools() -> list[BaseTool]:
    """Return the safe tools registered on the main Agent by default."""

    return [calculator, current_time]


def get_workspace_tools(workspace: Path) -> list[BaseTool]:
    """Return file tools permanently bound to one session workspace."""

    return [
        FileReadTool(workspace=workspace),
        FileWriteTool(workspace=workspace),
        FileEditTool(workspace=workspace),
        GrepTool(workspace=workspace),
        CommandTool(workspace=workspace),
    ]


def build_tools(state: Mapping[str, Any]) -> list[BaseTool]:
    workspace = state["workspace"]
    return [
        FileReadTool(workspace=workspace),
        FileWriteTool(workspace=workspace),
        FileEditTool(workspace=workspace),
        GrepTool(workspace=workspace),
        BashTool(workspace=workspace),
        NotepadReadTool(workspace=workspace),
        NotepadAppendTool(workspace=workspace),
    ]


__all__ = [
    "BashTool",
    "build_tools",
    "calculator",
    "CommandTool",
    "current_time",
    "get_builtin_tools",
    "get_workspace_tools",
    "NotepadAppendTool",
    "NotepadReadTool",
    "TodoStore",
    "TodoUpdateTool",
    "TodoWriteTool",
]
