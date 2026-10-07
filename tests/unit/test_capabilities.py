from __future__ import annotations

import json
from pathlib import Path

import pytest

from xiliumini.capabilities.manager import BUILTIN_PERMISSIONS, CapabilityManager
from xiliumini.capabilities.models import CapabilitySpec
from xiliumini.capabilities.skills import SkillRegistry
from xiliumini.execution.gateway import ExecutionGateway, RunContext
from xiliumini.tools.file_read import FileReadTool
from xiliumini.tools.file_write import FileWriteTool


def test_search_limits_results_and_keeps_namespaces():
    manager = CapabilityManager()
    manager.register(
        CapabilitySpec(
            id="builtin:search", kind="builtin", name="search",
            description="Search files", source_id="builtin",
        )
    )
    manager.register(
        CapabilitySpec(
            id="mcp:server:search", kind="mcp", name="search",
            description="Search remote news", source_id="server",
        )
    )
    assert [item.id for item in manager.search("search", limit=1)] == ["builtin:search"]
    assert [item.id for item in manager.search("search", kind="mcp")] == ["mcp:server:search"]
    with pytest.raises(ValueError):
        manager.register(
            CapabilitySpec(
                id="builtin:search", kind="builtin", name="search",
                description="Duplicate", source_id="builtin",
            )
        )


def test_gateway_preserves_tool_interface_and_output(tmp_path):
    base = FileWriteTool(workspace=tmp_path)
    events = []
    context = RunContext(
        run_id="run-1", session_id="session-1", workspace=tmp_path, event_sink=events.append
    )
    wrapped = ExecutionGateway().wrap_tool(base, context)
    assert wrapped.name == base.name
    assert wrapped.description == base.description
    assert wrapped.args_schema is base.args_schema
    assert (
        wrapped.invoke({"path": "demo.txt", "content": "hello"})
        == "Wrote 5 characters to demo.txt"
    )
    assert (tmp_path / "demo.txt").read_text() == "hello"
    assert events[-1]["success"] is True
    assert events[-1]["capability_id"] == "builtin:file_write"
    assert "hello" not in str(events)


def test_gateway_records_failure_without_leaking_arguments(tmp_path):
    events = []
    context = RunContext(
        run_id="run-2", session_id="session-2", workspace=tmp_path, event_sink=events.append
    )
    wrapped = ExecutionGateway().wrap_tool(FileReadTool(workspace=tmp_path), context)
    output = wrapped.invoke({"path": "missing-secret.txt"})
    assert output.startswith("Error:")
    assert events[-1]["success"] is False
    assert "missing-secret" not in str(events)


def test_builtin_permission_map_covers_registered_tools():
    from xiliumini.tools import build_tools
    from xiliumini.tools.approval_context import ApprovalConfig, approval_config

    token = approval_config.set(ApprovalConfig("deny", None))
    try:
        names = {tool.name for tool in build_tools({"workspace": Path(".")})}
    finally:
        approval_config.reset(token)
    assert names <= BUILTIN_PERMISSIONS.keys()


def test_gateway_marks_structured_tool_error_as_failure(tmp_path):
    from xiliumini.tools.bash_tool import BashTool

    events = []
    context = RunContext(
        run_id="run-3", session_id="session-3", workspace=tmp_path, event_sink=events.append
    )
    wrapped = ExecutionGateway().wrap_tool(
        BashTool(workspace=tmp_path, approval_mode="deny"), context
    )
    result = wrapped.invoke({"argv": ["unsupported"]})
    assert json.loads(result)["ok"] is False
    assert events[-1]["success"] is False


def test_manager_loads_skill_on_demand(tmp_path):
    skill = tmp_path / "skills" / "debugging"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: debugging\ndescription: Use for bugs.\n---\n\nInvestigate first.\n",
        encoding="utf-8",
    )
    manager = CapabilityManager(skill_registry=SkillRegistry(tmp_path / "skills"))
    assert [item.id for item in manager.search("bugs")] == ["skill:debugging"]
    assert "Investigate first" in manager.load_skill("skill:debugging")


def test_gateway_registers_builtin_from_actual_tool(tmp_path):
    manager = CapabilityManager()
    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        capability_manager=manager,
    )
    ExecutionGateway().wrap_tool(FileWriteTool(workspace=tmp_path), context)
    spec = manager.get("builtin:file_write")
    assert spec is not None
    assert spec.name == "file_write"
    assert spec.schema_digest


def test_ai_news_alias_finds_latest_mcp_tool():
    manager = CapabilityManager()
    manager.register(CapabilitySpec(
        id="mcp:aihot:aihot_get_latest", kind="mcp", name="aihot_get_latest",
        description="Get the latest items", source_id="aihot",
    ))
    assert [spec.id for spec in manager.search("AI 新闻")] == ["mcp:aihot:aihot_get_latest"]
