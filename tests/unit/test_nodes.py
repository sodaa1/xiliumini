import json

import pytest
from langchain_core.messages import ToolMessage

from tests.agent_fakes import ScriptedModel, call, state
from xiliumini.errors import MemoryBudgetError, MemorySystemError, WorkspaceError
from xiliumini.graph.memory import MemoryLimits, MemoryManager
from xiliumini.graph.nodes import (
    chat_responder_node,
    final_node,
    intent_route_fn,
    intent_router_node,
    planner_node,
    verifier_node,
)
from xiliumini.graph.state import ToolEvent
from xiliumini.tools.preferences import UserPreferenceStore
from xiliumini.tools.subagent_tools import SupervisorContext
from xiliumini.tools.todo import TodoStore


def memory_manager(tmp_path, data_dir=None):
    return MemoryManager(
        tmp_path,
        UserPreferenceStore(data_dir or (tmp_path / "project")),
        MemoryLimits(64_000, 0.8, 8_000),
        model_name="test-model",
        token_counter=lambda _messages, _model: 1,
    )


def test_intent_router_returns_high_confidence_chat_route_with_reason(tmp_path):
    model = ScriptedModel(['{"route":"chat","reason":"ordinary greeting","confidence":0.92}'])

    result = intent_router_node(
        state(tmp_path, task="你好", context_summary="Earlier coding work is complete."),
        model=model,
    )

    assert result == {
        "intent_route": "chat",
        "intent_reason": "ordinary greeting",
        "intent_confidence": 0.92,
    }
    payload = json.loads(model.calls[0][1].content)
    assert payload == {
        "user_input": "你好",
        "session_context": "Earlier coding work is complete.",
    }


@pytest.mark.parametrize(
    ("response", "confidence"),
    [
        ('{"route":"chat","reason":"uncertain","confidence":0.54}', 0.54),
        ('{"route":"other","reason":"invalid route","confidence":0.99}', 0.0),
        ('{"route":"chat","reason":"invalid confidence","confidence":1.1}', 0.0),
        ("not json", 0.0),
        (RuntimeError("provider unavailable"), 0.0),
    ],
)
def test_intent_router_defaults_uncertain_or_invalid_results_to_workflow(
    tmp_path, response, confidence
):
    result = intent_router_node(state(tmp_path, task="继续"), model=ScriptedModel([response]))

    assert result["intent_route"] == "workflow"
    assert result["intent_confidence"] == confidence
    assert result["intent_reason"]


def test_intent_route_fn_only_selects_chat_for_explicit_chat_route(tmp_path):
    assert intent_route_fn(state(tmp_path, intent_route="chat")) == "chat_responder"
    assert intent_route_fn(state(tmp_path, intent_route="workflow")) == "planner"
    assert intent_route_fn(state(tmp_path)) == "planner"


def test_chat_responder_returns_model_text_as_chat_and_final_answer(tmp_path):
    model = ScriptedModel(["我是 MokioClaw 的轻量聊天节点。"])

    result = chat_responder_node(
        state(tmp_path, task="你是谁？", context_summary="User asked about capabilities."),
        model=model,
    )

    assert result == {
        "chat_response": "我是 MokioClaw 的轻量聊天节点。",
        "final_answer": "我是 MokioClaw 的轻量聊天节点。",
    }
    payload = json.loads(model.calls[0][1].content)
    assert payload == {
        "user_input": "你是谁？",
        "session_context": "User asked about capabilities.",
    }


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
            (
                '{"summary":"implemented with docs","plan_summary":"done",'
                '"acceptance_criteria":["tests pass"],"ready_for_verification":true}'
            ),
        ]
    )
    result = planner_node(
        state(tmp_path, attempt=0), model=model, memory_manager=memory_manager(tmp_path)
    )
    assert result["attempt"] == 1
    assert result["todos"][0]["status"] == "completed"
    assert observed == [("search", "find docs"), ("code", ["https://example.test"])]
    assert len(result["agent_results"]) == 2
    assert isinstance(model.calls[-1][-1], ToolMessage)


def test_planner_repair_and_retry_feedback(tmp_path):
    TodoStore(tmp_path).write([{"id": "impl", "content": "implement"}])
    model = ScriptedModel(
        [
            "bad",
            (
                '{"summary":"ready","plan_summary":"repair",'
                '"acceptance_criteria":["demo passes"],"ready_for_verification":true}'
            ),
        ]
    )
    result = planner_node(
        state(
            tmp_path,
            verification="missing demo",
            verifier_summary="missing demo",
        ),
        model=model,
        memory_manager=memory_manager(tmp_path),
    )
    assert result["attempt"] == 2
    assert "missing demo" in model.calls[0][1].content
    assert result["result"] == "ready"


def test_planner_loop_limit_is_failed_round(tmp_path):
    model = ScriptedModel([call("unknown", {})])
    result = planner_node(
        state(tmp_path), model=model, memory_manager=memory_manager(tmp_path), max_loops=1
    )
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
    return verifier_node(
        s,
        model=ScriptedModel(['{"status":"passed","reason":"checked"}']),
        memory_manager=memory_manager(s["workspace"]),
    )


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
    result = verifier_node(
        s, model=ScriptedModel(["invalid"]), memory_manager=memory_manager(tmp_path)
    )
    assert result["graph_state"] == "failed"
    s["graph_state"] = result["graph_state"]
    s["verification"] = result["verification"]
    s["todos"][0]["status"] = "blocked"
    assert "blocked" in final_node(s)["final_answer"]


def test_planner_uses_one_memory_object_tools_hook_and_expanded_output(tmp_path):
    manager = memory_manager(tmp_path)
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "summary": "ready",
                    "plan_summary": "implement then test",
                    "acceptance_criteria": ["tests pass"],
                    "ready_for_verification": True,
                }
            )
        ]
    )

    result = planner_node(state(tmp_path, attempt=0), model=model, memory_manager=manager)

    payload = json.loads(model.calls[0][1].content)
    assert set(payload) == {"memory"}
    assert payload["memory"]["working"]["current_node"] == "planner"
    assert {tool.name for tool in model.tools} >= {
        "todo_write",
        "call_search_agent",
        "call_code_agent",
        "preference_write",
    }
    assert result["result"] == "ready"
    assert result["plan_summary"] == "implement then test"
    assert result["acceptance_criteria"] == ["tests pass"]
    assert result["memory"]["working"]["plan_summary"] == "implement then test"


def test_planner_records_and_forgets_only_explicit_preferences(tmp_path):
    data_dir = tmp_path / "project"
    manager = memory_manager(tmp_path, data_dir)
    upsert = ScriptedModel(
        [
            call(
                "preference_write",
                {"action": "upsert", "key": "test_style", "content": "Always use pytest."},
            ),
            '{"summary":"saved","plan_summary":"","acceptance_criteria":[],"ready_for_verification":false}',
        ]
    )
    planner_node(state(tmp_path), model=upsert, memory_manager=manager)
    assert manager.preference_store.read()[0]["key"] == "test_style"
    refreshed = json.loads(upsert.calls[1][1].content)["memory"]
    assert refreshed["rules"]["user_preferences"][0]["key"] == "test_style"

    remove = ScriptedModel(
        [
            call("preference_write", {"action": "remove", "key": "test_style"}),
            '{"summary":"forgot","plan_summary":"","acceptance_criteria":[],"ready_for_verification":false}',
        ]
    )
    planner_node(state(tmp_path), model=remove, memory_manager=manager)
    assert manager.preference_store.read() == []

    preference_path = data_dir / "USER_PREFERENCES.md"
    before = preference_path.read_bytes()
    for task in ("Use tabs in this task", "Override the saved style just this once"):
        ordinary = ScriptedModel(
            [
                (
                    '{"summary":"not saved","plan_summary":"",'
                    '"acceptance_criteria":[],"ready_for_verification":false}'
                )
            ]
        )
        planner_node(state(tmp_path, task=task), model=ordinary, memory_manager=manager)
        assert preference_path.read_bytes() == before


def test_verifier_accepts_successful_preference_only_task_without_completed_todos(
    tmp_path,
):
    for task in (
        "Remember that I prefer concise answers for future tasks.",
        "Remember that I prefer Python and pytest for future coding tasks.",
    ):
        s = state(
            tmp_path,
            task=task,
            supervisor_ok=True,
            todos=[
                {
                    "id": "preference",
                    "content": "save preference",
                    "status": "pending",
                    "note": "",
                }
            ],
            tool_events=[
                {
                    "agent": "planner",
                    "tool": "preference_write",
                    "args": {"action": "upsert", "key": "answer_style"},
                    "output": '{"ok":true}',
                    "ok": True,
                    "attempt": 1,
                    "phase": None,
                }
            ],
        )

        result = verifier_node(
            s,
            model=ScriptedModel(['{"status":"passed","reason":"preference saved"}']),
            memory_manager=memory_manager(tmp_path),
        )

        assert result["graph_state"] == "passed"


def test_preference_write_does_not_bypass_incomplete_combined_task(tmp_path):
    s = state(
        tmp_path,
        task="Remember that I prefer tabs and then implement a calculator.",
        supervisor_ok=True,
        todos=[
            {
                "id": "implement",
                "content": "implement calculator",
                "status": "pending",
                "note": "",
            }
        ],
        tool_events=[
            {
                "agent": "planner",
                "tool": "preference_write",
                "args": {"action": "upsert", "key": "indentation"},
                "output": '{"ok":true}',
                "ok": True,
                "attempt": 1,
                "phase": None,
            }
        ],
    )

    result = verifier_node(
        s,
        model=ScriptedModel([]),
        memory_manager=memory_manager(tmp_path),
    )

    assert result["graph_state"] == "failed"
    assert result["verification"] == "Todos are missing or incomplete"


def test_verifier_updates_layered_memory_and_preserves_unrelated_error(tmp_path):
    s = verified_state(tmp_path, last_error="code_agent: build failed")
    manager = memory_manager(tmp_path)

    result = verifier_node(
        s,
        model=ScriptedModel(['{"status":"passed","reason":"checked"}']),
        memory_manager=manager,
    )

    assert result["current_node"] == "verifier"
    assert result["verifier_summary"] == "checked"
    assert result["last_error"] == "code_agent: build failed"
    assert result["memory"]["working"]["current_node"] == "verifier"
    assert result["memory"]["working"]["verifier_summary"] == "checked"


def test_supervisor_context_bounds_handoffs_and_updates_agent_status(monkeypatch, tmp_path):
    import xiliumini.tools.subagent_tools as delegates

    TodoStore(tmp_path).write([{"id": "impl", "content": "implement"}])
    calls = 0

    def code(_state, instruction, **_kwargs):
        nonlocal calls
        calls += 1
        ok = calls != 6
        return {
            "ok": ok,
            "summary": f"result {instruction}",
            "todos": TodoStore(tmp_path).read(),
            "messages": [],
            "tool_events": [],
        }

    monkeypatch.setattr(delegates, "run_code_agent", code)
    s = state(tmp_path)
    context = SupervisorContext(s, memory_manager(tmp_path))

    for index in range(7):
        context.delegate("code_agent", f"step {index}")

    assert [item["instruction"] for item in s["agent_handoffs"]] == [
        f"step {index}" for index in range(1, 7)
    ]
    assert s["code_agent_summary"] == "result step 6"
    assert s["last_error"] == ""
    assert s["memory"]["working"]["agent_handoffs"] == s["agent_handoffs"]


def test_canonical_handoff_is_once_after_result_and_safe(monkeypatch, tmp_path):
    import xiliumini.tools.subagent_tools as delegates

    events = []

    def search(_state, instruction, **kwargs):
        kwargs["writer"]({"type": "tool_result"})
        return {"ok": False, "summary": "secret sk-test " + str(tmp_path) + "x" * 10000}

    monkeypatch.setattr(delegates, "run_search_agent", search)
    context = SupervisorContext(state(tmp_path), memory_manager(tmp_path))
    context.delegate("search_agent", "private instruction", events.append)
    assert [event["type"] for event in events] == ["tool_result", "handoff"]
    handoff = events[-1]
    assert handoff["agent"] == "search_agent"
    assert handoff["attempt"] == 1
    assert handoff["ok"] is False
    assert len(handoff["summary"]) <= 512
    assert "secret" not in json.dumps(handoff)
    assert str(tmp_path) not in json.dumps(handoff)
    assert "private instruction" not in json.dumps(handoff)


@pytest.mark.parametrize("outcome", ["success", "exception", "missing_todos"])
def test_handoff_for_every_delegation_outcome(monkeypatch, tmp_path, outcome):
    import xiliumini.tools.subagent_tools as delegates

    if outcome != "missing_todos":
        TodoStore(tmp_path).write([{"id": "impl", "content": "implement"}])

    def code(*args, **kwargs):
        if outcome == "exception":
            raise RuntimeError("private failure")
        return {"ok": True, "summary": "done", "tool_events": []}

    monkeypatch.setattr(delegates, "run_code_agent", code)
    events = []
    context = SupervisorContext(state(tmp_path), memory_manager(tmp_path))
    result = json.loads(context.delegate("code_agent", "task", events.append))
    if outcome == "missing_todos":
        assert events == []
        assert result["ok"] is False
        return
    assert len(events) == 1
    assert events[0]["type"] == "handoff"
    assert events[0]["ok"] is result["ok"]


@pytest.mark.parametrize("error_type", [MemoryBudgetError, MemorySystemError])
@pytest.mark.parametrize("runner_fails", [False, True])
def test_handoff_precedes_failed_memory_assembly(monkeypatch, tmp_path, error_type, runner_fails):
    import xiliumini.tools.subagent_tools as delegates

    events = []

    def search(_state, instruction, **kwargs):
        kwargs["writer"]({"type": "tool_result"})
        if runner_fails:
            raise RuntimeError("private secret")
        return {"ok": True, "summary": "private " + str(tmp_path), "tool_events": []}

    class BrokenMemory:
        def assemble(self, current, **kwargs):
            assert len(current["agent_results"]) == 1
            raise error_type("private secret")

    monkeypatch.setattr(delegates, "run_search_agent", search)
    context = SupervisorContext(state(tmp_path), BrokenMemory())
    with pytest.raises(error_type):
        context.delegate("search_agent", "private instruction", events.append)
    assert [event["type"] for event in events] == ["tool_result", "handoff"]
    assert events[-1] == {
        "type": "handoff",
        "agent": "search_agent",
        "attempt": 1,
        "ok": not runner_fails,
        "summary": "Delegation failed" if runner_fails else "Delegation completed",
    }


def test_real_code_runner_corrupt_todo_still_emits_handoff(monkeypatch, tmp_path):
    import xiliumini.agents.code_agent as code

    TodoStore(tmp_path).write([{"id": "impl", "content": "implement"}])
    model = ScriptedModel(
        [
            call("file_write", {"path": "TODO.md", "content": "invalid private secret"}),
            "done",
        ]
    )
    monkeypatch.setattr(code, "create_agent_model", lambda: model)
    events = []
    context = SupervisorContext(state(tmp_path), memory_manager(tmp_path))
    with pytest.raises(WorkspaceError, match="todo data is invalid"):
        context.delegate("code_agent", "private instruction", events.append)
    assert len(model.calls) == 2
    assert [event["type"] for event in events] == ["tool_call", "tool_result", "handoff"]
    assert events[-1] == {
        "type": "handoff",
        "agent": "code_agent",
        "attempt": 1,
        "ok": False,
        "summary": "Delegation failed",
    }


def test_search_handoff_precedes_failed_state_postprocessing(monkeypatch, tmp_path):
    import xiliumini.tools.subagent_tools as delegates

    events = []

    def search(*args, **kwargs):
        kwargs["writer"]({"type": "tool_result"})
        return {"ok": True, "summary": "private secret", "sources": []}

    class BrokenList(list):
        def append(self, value):
            raise MemorySystemError("controlled state failure")

    monkeypatch.setattr(delegates, "run_search_agent", search)
    current = state(tmp_path)
    current["agent_results"] = BrokenList()
    with pytest.raises(MemorySystemError):
        SupervisorContext(current, memory_manager(tmp_path)).delegate(
            "search_agent", "private instruction", events.append
        )
    assert [event["type"] for event in events] == ["tool_result", "handoff"]
    assert events[-1]["summary"] == "Delegation completed"
    assert "private" not in json.dumps(events[-1])
