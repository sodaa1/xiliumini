from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool

from xiliumini.tools.calculator import calculator
from xiliumini.tools.command import CommandTool
from xiliumini.tools.current_time import current_time
from xiliumini.tools.file_edit import FileEditTool
from xiliumini.tools.file_read import FileReadTool
from xiliumini.tools.file_write import FileWriteTool
from xiliumini.tools.grep import GrepTool


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


__all__ = [
    "calculator",
    "CommandTool",
    "current_time",
    "get_builtin_tools",
    "get_workspace_tools",
]
