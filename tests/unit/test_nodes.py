import json

from langchain_core.messages import ToolMessage

from tests.agent_fakes import ScriptedModel, call, state
from xiliumini.graph.nodes import final_node, planner_node, verifier_node
from xiliumini.graph.state import ToolEvent
from xiliumini.tools.todo import TodoStore


def test_planner_delegates_search_then_code_and_continues(monkeypatch, tmp_path):
    import xiliumini.tools.subagent_tools as delegates

    observed = []

    def search(s, instruction, **kwargs):
        observed.append(("search", instruction))
        return {
            "ok": True,
            "summary": "docs",
            "queries": ["docs"],
            "sources": ["https://example.test"],
            "messages": [],
            "tool_events": [],
        }

    def code(s, instruction, **kwargs):
        observed.append(("code", s["research_notes"][0]["sources"]))
        store = TodoStore(s["workspace"])
        store.update("impl", "in_progress")
        todos = store.update("impl", "completed")
        return {
            "ok": True,
            "summary": "implemented",
            "todos": todos,
            "messages": [],
            "tool_events": [],
        }

    monkeypatch.setattr(delegates, "run_search_agent", search)
    monkeypatch.setattr(delegates, "run_code_agent", code)
    model = ScriptedModel(
        [
            call("todo_write", {"todos": [{"id": "impl", "content": "implement"}]}),
            call("call_search_agent", {"instruction": "find docs"}),
            call("call_code_agent", {"instruction": "implement"}),
            '{"summary":"implemented with docs","ready_for_verification":true}',
        ]
    )
    result = planner_node(state(tmp_path, attempt=0), model=model)
    assert result["attempt"] == 1
    assert result["todos"][0]["status"] == "completed"
    assert observed == [("search", "find docs"), ("code", ["https://example.test"])]
    assert len(result["agent_results"]) == 2
    assert isinstance(model.calls[-1][-1], ToolMessage)


def test_planner_repair_and_retry_feedback(tmp_path):
    TodoStore(tmp_path).write([{"id": "impl", "content": "implement"}])
    model = ScriptedModel(["bad", '{"summary":"ready","ready_for_verification":true}'])
    result = planner_node(state(tmp_path, verification="missing demo"), model=model)
    assert result["attempt"] == 2
    assert "missing demo" in model.calls[0][1].content
    assert result["result"] == "ready"


def test_planner_loop_limit_is_failed_round(tmp_path):
    model = ScriptedModel([call("unknown", {})])
    result = planner_node(state(tmp_path), model=model, max_loops=1)
    assert not result["supervisor_ok"]
    assert not result["tool_events"][0]["ok"]


def event(phase, exit_code=0, *, tool="bash", attempt=1, path="impl.py", **extra) -> ToolEvent:
    output = {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "timed_out": False,
        "truncated": False,
        "phase": phase,
        **extra,
    }
    return {
        "agent": "code_agent",
        "tool": tool,
        "args": {"argv": ["python", "-m", "pytest", "-q"], "path": path},
        "output": json.dumps(output),
        "ok": output["ok"],
        "attempt": attempt,
        "phase": phase,
    }


def verified_state(tmp_path, **changes):
    return state(
        tmp_path,
        todos=[{"id": "impl", "content": "implement", "status": "completed", "note": ""}],
        agent_results=[
            {
                "agent": "code_agent",
                "instruction": "implement",
                "summary": "done",
                "ok": True,
                "attempt": 1,
            }
        ],
        tool_events=[event(None, tool="file_write"), event("test_green")],
        supervisor_ok=True,
        **changes,
    )


def verify(s):
    return verifier_node(s, model=ScriptedModel(['{"status":"passed","reason":"checked"}']))


def test_verifier_accepts_completed_todos_and_tests(tmp_path):
    assert verify(verified_state(tmp_path))["graph_state"] == "passed"


def test_verifier_rejects_incomplete_todo(tmp_path):
    s = verified_state(tmp_path)
    s["todos"][0]["status"] = "blocked"
    assert verify(s)["graph_state"] == "failed"


def test_verifier_rejects_failed_latest_check(tmp_path):
    s = verified_state(tmp_path)
    s["tool_events"].append(event("test_green", 1))
    assert verify(s)["graph_state"] == "failed"


def test_verifier_rejects_tests_before_final_edit(tmp_path):
    s = verified_state(tmp_path)
    s["tool_events"].append(event(None, tool="file_edit"))
    assert verify(s)["graph_state"] == "failed"


def test_verifier_retry_can_repair_previous_failure(tmp_path):
    s = verified_state(tmp_path)
    s["tool_events"] += [event("test_green", 1), event("test_green", attempt=2)]
    s["attempt"] = 2
    assert verify(s)["graph_state"] == "passed"


def test_verifier_rejects_latest_failed_code_delegation(tmp_path):
    s = verified_state(tmp_path)
    s["task"] = "Add a calculator"
    s["attempt"] = 2
    s["agent_results"].append(
        {
            "agent": "code_agent",
            "instruction": "repair verifier feedback",
            "summary": "agent loop limit reached",
            "ok": False,
            "attempt": 2,
        }
    )
    assert verify(s)["graph_state"] == "failed"


def test_tdd_requires_ordered_red_write_green(tmp_path):
    s = verified_state(tmp_path)
    s["task"] = "implement with TDD"
    assert verify(s)["graph_state"] == "failed"
    s["tool_events"] = [event("test_red", 1), event(None, tool="file_write"), event("test_green")]
    assert verify(s)["graph_state"] == "passed"


def test_research_requires_sources_and_successful_search(tmp_path):
    s = verified_state(tmp_path)
    s["task"] = "research official docs and implement"
    assert verify(s)["graph_state"] == "failed"
    s["research_notes"] = [
        {"summary": "docs", "queries": ["docs"], "sources": ["https://example.test"], "attempt": 1}
    ]
    s["agent_results"].append(
        {
            "agent": "search_agent",
            "instruction": "docs",
            "summary": "found",
            "ok": True,
            "attempt": 1,
        }
    )
    assert verify(s)["graph_state"] == "passed"


def test_explicitly_unneeded_research_does_not_require_tavily(tmp_path):
    for instruction in ("No research is needed.", "不需要联网搜索。", "Do not search the web."):
        s = verified_state(tmp_path)
        s["task"] = "Implement arithmetic and run tests. " + instruction
        assert verify(s)["graph_state"] == "passed"


def test_invalid_verifier_and_failed_final_report_remaining(tmp_path):
    s = verified_state(tmp_path)
    result = verifier_node(s, model=ScriptedModel(["invalid"]))
    assert result["graph_state"] == "failed"
    s["graph_state"] = result["graph_state"]
    s["verification"] = result["verification"]
    s["todos"][0]["status"] = "blocked"
    assert "blocked" in final_node(s)["final_answer"]
