from __future__ import annotations

import json
import sqlite3
from typing import Any

from tests.agent_fakes import state
from xiliumini.graph.verification import requirements_failure


def test_pure_automation_accepts_persisted_tool_evidence(tmp_path):
    workspace = tmp_path / "workspaces" / "session"
    workspace.mkdir(parents=True)
    with sqlite3.connect(tmp_path / "automation.sqlite3") as db:
        db.execute("CREATE TABLE automations (id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO automations VALUES ('daily-brief')")
    current: dict[str, Any] = dict(state(workspace, task="创建每天 09:00 的简报自动任务"))
    current.update(
        supervisor_ok=True, todos=[], agent_results=[], research_notes=[],
        tool_events=[{
            "agent": "planner", "tool": "automation_manage", "ok": True,
            "output": json.dumps({"ok": True, "id": "daily-brief", "persisted": True}),
            "args": {"action": "add"}, "phase": None,
        }],
    )
    assert requirements_failure(current) is None


def test_automation_event_without_database_record_is_not_evidence(tmp_path):
    current: dict[str, Any] = dict(state(tmp_path, task="创建每天 09:00 的自动任务"))
    current.update(
        supervisor_ok=True, todos=[], agent_results=[], research_notes=[],
        tool_events=[{
            "agent": "planner", "tool": "automation_manage", "ok": True,
            "output": json.dumps({"ok": True, "id": "missing", "persisted": True}),
            "args": {"action": "add"}, "phase": None,
        }],
    )
    assert requirements_failure(current) == "Missing persisted automation evidence"


def test_mcp_research_requires_source_url(tmp_path):
    current: dict[str, Any] = dict(state(tmp_path, task="通过 AIHot MCP 搜索 AI 新闻"))
    current.update(supervisor_ok=True, todos=[{"id": "research", "status": "completed"}],
                   agent_results=[{"agent": "search_agent", "ok": True}],
                   research_notes=[{"sources": []}], tool_events=[])
    assert requirements_failure(current) == "Missing successful research and source URLs"


def test_code_plus_automation_still_requires_code_agent(tmp_path):
    current: dict[str, Any] = dict(state(tmp_path, task="创建定时任务并编写 Python 代码"))
    current.update(
        supervisor_ok=True, todos=[{"id": "task", "status": "completed"}],
        agent_results=[], research_notes=[],
        tool_events=[{
            "agent": "planner", "tool": "automation_manage", "ok": True,
            "output": json.dumps({"ok": True, "id": "daily-brief", "persisted": True}),
            "args": {"action": "add"}, "phase": None,
        }],
    )
    assert requirements_failure(current) == "Missing successful codeAgent result"
