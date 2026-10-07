from __future__ import annotations

import json

from xiliumini.capabilities.langchain_tools import build_meta_tools
from xiliumini.capabilities.manager import CapabilityManager
from xiliumini.execution.gateway import ExecutionGateway, RunContext, active_run
from xiliumini.policy.engine import PolicyEngine


def test_planner_automation_tool_persists_and_background_denies(tmp_path):
    manager = CapabilityManager()
    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        data_dir=tmp_path, capability_manager=manager,
        policy_engine=PolicyEngine(tmp_path / "policy.json"),
    )
    token = active_run.set(context)
    try:
        tools = build_meta_tools(role="planner")
    finally:
        active_run.reset(token)
    selected = next(tool for tool in tools if tool.name == "automation_manage")
    payload = {
        "action": "add", "task_id": "daily-brief",
        "definition": {
            "id": "daily-brief", "name": "Brief", "prompt": "Read news",
            "trigger_type": "cron", "schedule": {"hour": 9, "minute": 0},
            "timezone": "Asia/Shanghai",
        },
    }
    result = ExecutionGateway().wrap_tool(selected, context).invoke(payload)
    assert json.loads(result)["persisted"] is True
    background = RunContext(
        run_id="run-2", session_id="session", workspace=tmp_path, data_dir=tmp_path,
        policy_profile="background", actor_type="automation",
        policy_engine=PolicyEngine(tmp_path / "policy.json"),
    )
    result = ExecutionGateway().wrap_tool(selected, background).invoke(
        {**payload, "task_id": "other", "definition": {**payload["definition"], "id": "other"}}
    )
    assert json.loads(result)["error"] == "policy_denied"
