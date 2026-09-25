from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from xiliumini.tools import get_builtin_tools, get_workspace_tools
from xiliumini.tools.current_time import current_time_value


def test_current_time_converts_an_injected_utc_clock() -> None:
    def fixed_clock() -> datetime:
        return datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    result = current_time_value("Asia/Shanghai", clock=fixed_clock)

    assert result == "2026-01-02 11:04:05 CST+0800"


def test_current_time_rejects_invalid_timezone() -> None:
    assert current_time_value("Mars/Olympus").startswith("Error: invalid timezone")


def test_registered_builtin_tools_exclude_bash() -> None:
    assert [tool.name for tool in get_builtin_tools()] == ["calculator", "current_time"]


def test_workspace_tools_are_bound_to_one_root(tmp_path: Path) -> None:
    tools = get_workspace_tools(tmp_path)

    assert [tool.name for tool in tools] == ["file_read", "file_write", "file_edit", "grep"]
    assert all(getattr(tool, "workspace", None) == tmp_path for tool in tools)
