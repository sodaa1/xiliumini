from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from xiliumini.capabilities.mcp import MCPManager


class FakeClient:
    calls = []
    connects = 0

    def __init__(self, target):
        self.target = target

    async def __aenter__(self):
        type(self).connects += 1
        return self

    async def __aexit__(self, *_):
        pass

    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(
            name="get_latest", description="Latest news",
            input_schema={
                "type": "object", "properties": {"limit": {"type": "integer"}},
                "required": ["limit"],
            },
        )])

    async def call_tool(self, name, arguments):
        type(self).calls.append((name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="https://example.com/story")],
            structured_content=None, is_error=False,
        )


def config(tmp_path):
    path = tmp_path / "mcp" / "aihot" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"servers": {"aihot": {
        "enabled": True, "transport": "http", "url": "https://example.com/mcp",
        "trust": "explicit", "permissions": ["network.read"],
        "tool_allowlist": ["get_latest"],
    }}}), encoding="utf-8")
    return tmp_path / "mcp"


def test_mcp_is_lazy_and_validates_args_before_call(tmp_path):
    FakeClient.calls = []
    FakeClient.connects = 0
    manager = MCPManager(config(tmp_path), client_factory=FakeClient)
    assert FakeClient.connects == 0
    specs = manager.tools("aihot")
    assert [spec.id for spec in specs] == ["mcp:aihot:get_latest"]
    with pytest.raises(ValueError):
        manager.call("mcp:aihot:get_latest", {"limit": "bad"})
    assert FakeClient.calls == []
    result = manager.call("mcp:aihot:get_latest", {"limit": 5})
    assert result["ok"] is True
    assert "https://example.com/story" in result["text"]
    assert FakeClient.calls == [("get_latest", {"limit": 5})]
    manager.close()


def test_mcp_unlisted_tool_cannot_connect(tmp_path):
    FakeClient.connects = 0
    manager = MCPManager(config(tmp_path), client_factory=FakeClient)
    with pytest.raises(ValueError):
        manager.call("mcp:aihot:write_story", {})
    assert FakeClient.connects == 0
    manager.close()


def test_mcp_rejects_malformed_lists(tmp_path):
    config(tmp_path)
    path = tmp_path / "mcp" / "aihot" / "mcp.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["servers"]["aihot"]["permissions"] = "network.read"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid MCP server list"):
        MCPManager(tmp_path / "mcp")


def test_mcp_error_result_is_not_reported_as_success(tmp_path):
    class ErrorClient(FakeClient):
        async def call_tool(self, name, arguments):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="remote failure")],
                structured_content={"error": "failed"}, is_error=True,
            )

    manager = MCPManager(config(tmp_path), client_factory=ErrorClient)
    result = manager.call("mcp:aihot:get_latest", {"limit": 1})
    assert result["ok"] is False
    assert result["is_error"] is True
    assert result["text"] == "remote failure"
    manager.close()


def test_mcp_timeout_is_bounded_without_retry(tmp_path):
    class SlowClient(FakeClient):
        calls = 0

        async def call_tool(self, name, arguments):
            type(self).calls += 1
            await asyncio.sleep(1)
            return SimpleNamespace(content=[], structured_content=None, is_error=False)

    manager = MCPManager(
        config(tmp_path), client_factory=SlowClient, timeout_seconds=0.05
    )
    manager.tools("aihot")
    with pytest.raises(TimeoutError, match="remote side effects may have occurred"):
        manager.call("mcp:aihot:get_latest", {"limit": 1})
    assert SlowClient.calls == 1
    manager.close()
