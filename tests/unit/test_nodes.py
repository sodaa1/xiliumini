import json

from langchain_core.messages import ToolMessage

from tests.agent_fakes import ScriptedModel, call, state
from xiliumini.graph.memory import MemoryLimits, MemoryManager
from xiliumini.graph.nodes import final_node, planner_node, verifier_node
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
