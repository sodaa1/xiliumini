from __future__ import annotations

import json

from langchain_core.tools import StructuredTool

from xiliumini.capabilities.manager import CapabilityManager
from xiliumini.capabilities.mcp import MCPManager
from xiliumini.execution.gateway import ExecutionGateway, RunContext
from xiliumini.policy.engine import PolicyEngine


def test_unknown_capability_denied_by_default(tmp_path):
    policy = PolicyEngine(tmp_path / "policy.json")
    decision = policy.evaluate(
        actor="interactive", source="mcp", capability_id="mcp:unknown:write",
        permission="remote.write", arguments={}, profile="interactive",
    )
    assert decision.effect == "deny"


def test_background_ask_becomes_deny(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"default": "deny", "profiles": {
        "background": {"remote.write": "ask"},
    }}), encoding="utf-8")
    policy = PolicyEngine(path, trusted_mcp_servers={"news"})
    decision = policy.evaluate(
        actor="automation", source="mcp", capability_id="mcp:news:write",
        permission="remote.write", arguments={}, profile="background",
    )
    assert decision.effect == "deny"
    assert "background" in decision.reason


def test_gateway_denies_remote_write_before_side_effect(tmp_path):
    calls = []

    def mcp_call(capability_id: str, arguments: dict) -> str:
        """Call remote tool."""
        calls.append((capability_id, arguments))
        return "called"

    tool = StructuredTool.from_function(mcp_call)
    events = []
    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        policy_profile="background", actor_type="automation", event_sink=events.append,
        policy_engine=PolicyEngine(tmp_path / "policy.json"),
    )
    output = ExecutionGateway().wrap_tool(tool, context).invoke({
        "capability_id": "mcp:news:write", "arguments": {},
    })
    assert json.loads(output)["error"] == "policy_denied"
    assert calls == []
    assert events[-1]["decision"] == "deny"


def test_background_denies_risky_bash_before_underlying_tool(tmp_path):
    calls = []

    def bash(argv: list[str]) -> str:
        """Run a restricted command."""
        calls.append(argv)
        return "ran"

    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        policy_profile="background", actor_type="automation",
        policy_engine=PolicyEngine(tmp_path / "policy.json"),
    )
    output = ExecutionGateway().wrap_tool(StructuredTool.from_function(bash), context).invoke(
        {"argv": ["python", "-m", "pip", "install", "example"]}
    )
    assert json.loads(output)["error"] == "policy_denied"
    assert calls == []


def test_stdio_discovery_needs_process_permission_before_client_creation(tmp_path):
    path = tmp_path / "mcp" / "local" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"servers": {"local": {
        "enabled": True, "transport": "stdio", "command": str(tmp_path / "server.exe"),
        "trust": "explicit", "permissions": ["network.read"],
        "tool_allowlist": ["read"],
    }}}), encoding="utf-8")
    calls = []
    manager = CapabilityManager(
        mcp_manager=MCPManager(tmp_path / "mcp", client_factory=lambda target: calls.append(target))
    )
    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        capability_manager=manager,
        policy_engine=PolicyEngine(tmp_path / "policy.json", trusted_mcp_servers={"local"}),
    )
    result = ExecutionGateway().discover_mcp("local", context)
    assert result["error"] == "policy_denied"
    assert calls == []


def test_direct_stdio_mcp_call_needs_process_permission(tmp_path):
    path = tmp_path / "mcp" / "local" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"servers": {"local": {
        "enabled": True, "transport": "stdio", "command": str(tmp_path / "server.exe"),
        "trust": "explicit", "permissions": ["network.read"],
        "tool_allowlist": ["read"],
    }}}), encoding="utf-8")
    connections = []
    manager = CapabilityManager(mcp_manager=MCPManager(
        tmp_path / "mcp", client_factory=lambda target: connections.append(target)
    ))
    invocations = []

    def mcp_call(capability_id: str, arguments: dict) -> str:
        """Call a configured MCP tool."""
        invocations.append(capability_id)
        return "called"

    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        capability_manager=manager,
        policy_engine=PolicyEngine(tmp_path / "policy.json", trusted_mcp_servers={"local"}),
    )
    output = ExecutionGateway().wrap_tool(StructuredTool.from_function(mcp_call), context).invoke({
        "capability_id": "mcp:local:read", "arguments": {},
    })
    assert json.loads(output)["error"] == "policy_denied"
    assert invocations == connections == []
