from __future__ import annotations

from pathlib import Path

import xiliumini.runtime as runtime_module
from xiliumini.errors import NodeOutputError
from xiliumini.events import ErrorEvent, FinalEvent, PlannerEvent
from xiliumini.graph.state import GraphState
from xiliumini.runtime import Runtime

SESSION_ID = "11111111-1111-4111-8111-111111111111"
SECOND_SESSION_ID = "22222222-2222-4222-8222-222222222222"


def test_new_task_resets_todos_but_preserves_workspace(monkeypatch, tmp_path):
    from xiliumini.tools.todo import TodoStore

    def capture(model, inputs, **kwargs):
        store = TodoStore(inputs["workspace"])
        assert store.read() == []
        store.write([{"id": "old", "content": "previous task"}])
        store.update("old", "in_progress")
        (inputs["workspace"] / "keep.txt").write_text("keep")
        return iter(())

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    runtime = Runtime(object(), data_dir=tmp_path)
    assert list(runtime.stream("first", SESSION_ID)) == []
    assert list(runtime.stream("second", SESSION_ID)) == []
    assert (tmp_path / "workspaces" / SESSION_ID / "keep.txt").read_text() == "keep"


def test_runtime_scopes_model_factory(monkeypatch, tmp_path):
    from xiliumini.agents.react import create_agent_model, model_factory

    fake = object()

    def capture(*args, **kwargs):
        assert create_agent_model() is fake
        return iter(())

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    assert (
        list(
            Runtime(object(), data_dir=tmp_path, agent_model_factory=lambda: fake).stream(
                "task", SESSION_ID
            )
        )
        == []
    )
    assert model_factory.get() is None


def test_runtime_does_not_leak_factories_between_stream_events(monkeypatch, tmp_path):
    from xiliumini.agents.react import create_agent_model, model_factory

    fake = object()

    def capture(model, inputs, **kwargs):
        assert create_agent_model() is fake
        yield FinalEvent(text="one", session_id=inputs["session_id"])
        assert create_agent_model() is fake
        yield FinalEvent(text="two", session_id=inputs["session_id"])

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    events = Runtime(object(), data_dir=tmp_path, agent_model_factory=lambda: fake).stream(
        "task", SESSION_ID
    )
    assert next(events) == FinalEvent(text="one", session_id=SESSION_ID)
    assert model_factory.get() is None
    assert next(events) == FinalEvent(text="two", session_id=SESSION_ID)
    assert model_factory.get() is None


def test_runtime_delegates_complete_fresh_graph_state(monkeypatch, tmp_path: Path) -> None:
    captured: list[GraphState] = []

    def fake_stream_agent(model, inputs: GraphState, checkpointer=None):
        captured.append(inputs.copy())
        yield PlannerEvent(todos=[], summary="one", attempt=1)
        yield FinalEvent(text="done", session_id=inputs["session_id"])

    monkeypatch.setattr(runtime_module, "stream_agent", fake_stream_agent)
    runtime = Runtime(object(), data_dir=tmp_path)

    first = list(runtime.stream("first", SESSION_ID, max_attempts=2))
    second = list(runtime.stream("second", SESSION_ID, max_attempts=3))

    assert first == [
        PlannerEvent(todos=[], summary="one", attempt=1),
        FinalEvent(text="done", session_id=SESSION_ID),
    ]
    assert second[-1] == FinalEvent(text="done", session_id=SESSION_ID)
    assert captured[0]["task"] == "first"
    assert captured[1]["task"] == "second"
    for state, maximum in zip(captured, [2, 3], strict=True):
        assert state["todos"] == []
        assert state["research_notes"] == []
        assert state["agent_results"] == []
        assert state["tool_events"] == []
        assert state["supervisor_ok"] is False
        assert state["result"] == ""
        assert state["graph_state"] == "planning"
        assert state["verification"] == ""
        assert state["attempt"] == 0
        assert state["max_attempts"] == maximum
        assert state["final_answer"] == ""


def test_runtime_reuses_workspace_for_same_session(monkeypatch, tmp_path: Path) -> None:
    workspaces: list[Path] = []

    def capture(model, inputs, **kwargs):
        workspaces.append(inputs["workspace"])
        return iter(())

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    runtime = Runtime(object(), data_dir=tmp_path)

    list(runtime.stream("first", SESSION_ID))
    list(runtime.stream("second", SESSION_ID))

    assert workspaces[0] == workspaces[1] == (tmp_path / "workspaces" / SESSION_ID).resolve()


def test_runtime_isolates_different_sessions(monkeypatch, tmp_path: Path) -> None:
    workspaces: list[Path] = []

    def capture(model, inputs, **kwargs):
        workspaces.append(inputs["workspace"])
        return iter(())

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    runtime = Runtime(
        object(),
        data_dir=tmp_path,
    )

    list(runtime.stream("one", SESSION_ID))
    list(runtime.stream("two", SECOND_SESSION_ID))

    assert workspaces[0] != workspaces[1]


def test_runtime_reports_invalid_session_without_traceback(tmp_path: Path) -> None:
    runtime = Runtime(object(), data_dir=tmp_path)

    assert list(runtime.stream("question", "../escape")) == [
        ErrorEvent(code="workspace_error", message="session_id must be a valid UUID")
    ]


def test_runtime_redacts_workspace_creation_failure(tmp_path: Path) -> None:
    data_file = tmp_path / "not-a-directory"
    data_file.write_text("occupied", encoding="utf-8")
    runtime = Runtime(object(), data_dir=data_file)

    events = list(runtime.stream("question", SESSION_ID))

    assert events == [
        ErrorEvent(code="workspace_error", message="could not create session workspace")
    ]
    assert isinstance(events[0], ErrorEvent)
    assert str(data_file) not in events[0].message


def test_runtime_maps_node_error(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise NodeOutputError("planner returned invalid output")
        yield

    monkeypatch.setattr(runtime_module, "stream_agent", fail)

    assert list(Runtime(object(), data_dir=tmp_path).stream("question", SESSION_ID)) == [
        ErrorEvent(code="node_output_error", message="planner returned invalid output")
    ]


def test_runtime_maps_provider_error_without_leaking_details(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("must-not-leak")
        yield

    monkeypatch.setattr(runtime_module, "stream_agent", fail)

    events = list(Runtime(object(), data_dir=tmp_path).stream("question", SESSION_ID))

    assert events == [ErrorEvent(code="provider_error", message="Provider request failed")]
    assert isinstance(events[0], ErrorEvent)
    assert "must-not-leak" not in events[0].message
