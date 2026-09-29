import importlib
import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from tests.agent_fakes import ScriptedModel, call, state


@tool("web_search")
def fake_search(query: str) -> str:
    """Return a controlled research response."""
    return json.dumps(
        {
            "ok": True,
            "query": query,
            "answer": "official answer",
            "results": [{"url": "https://example.test/docs"}],
        }
    )


def install(monkeypatch, responses):
    module = importlib.import_module("xiliumini.agents.search_agent")
    model = ScriptedModel(responses)
    monkeypatch.setattr(module, "create_agent_model", lambda: model)
    monkeypatch.setattr(module, "WebSearchTool", lambda: fake_search)
    return module, model


def test_search_collects_sources_and_returns_tool_messages(monkeypatch, tmp_path):
    first = AIMessage(
        content="",
        tool_calls=[
            {"name": "web_search", "args": {"query": "official docs"}, "id": "one"},
            {"name": "web_search", "args": {"query": "official docs"}, "id": "two"},
        ],
    )
    module, model = install(monkeypatch, [first, "Use the official docs."])
    events = []
    result = module.run_search_agent(state(tmp_path), "find official sources", writer=events.append)
    assert result["ok"] is True
    assert result["queries"] == ["official docs"]
    assert result["sources"] == ["https://example.test/docs"]
    assert [m.tool_call_id for m in model.calls[1] if isinstance(m, ToolMessage)] == ["one", "two"]
    assert {t.name for t in model.tools} == {"web_search"}
    assert "find official sources" in model.calls[0][1].content
    assert any(e["type"] == "search_results" for e in events)


def test_search_loop_limit_retains_evidence(monkeypatch, tmp_path):
    module, _ = install(monkeypatch, [call("web_search", {"query": "docs"})])
    result = module.run_search_agent(state(tmp_path), "research", max_loops=1)
    assert result["ok"] is False
    assert result["sources"] == ["https://example.test/docs"]
    assert result["tool_events"]


def test_search_unknown_tool_and_bad_arguments_return_errors(monkeypatch, tmp_path):
    module, model = install(
        monkeypatch, [call("file_write", {}), call("web_search", {}), "cannot research"]
    )
    result = module.run_search_agent(state(tmp_path), "research")
    assert not result["ok"]
    assert len(result["tool_events"]) == 2
    assert all(not e["ok"] for e in result["tool_events"])
    assert isinstance(model.calls[-1][-1], ToolMessage)


def test_search_provider_error_is_redacted(monkeypatch, tmp_path):
    module, _ = install(monkeypatch, [RuntimeError("private-secret")])
    result = module.run_search_agent(state(tmp_path), "research")
    assert not result["ok"]
    assert "private-secret" not in result["summary"]
