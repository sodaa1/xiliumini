from __future__ import annotations

from pathlib import Path

import xiliumini.runtime as runtime_module
from xiliumini.errors import NodeOutputError
from xiliumini.events import ErrorEvent, FinalEvent, PlannerEvent
from xiliumini.graph.state import GraphState
from xiliumini.runtime import Runtime

SESSION_ID = "11111111-1111-4111-8111-111111111111"
SECOND_SESSION_ID = "22222222-2222-4222-8222-222222222222"


def test_runtime_delegates_complete_fresh_graph_state(monkeypatch, tmp_path: Path) -> None:
    captured: list[GraphState] = []

    def fake_stream_agent(model, tools, inputs: GraphState, checkpointer=None):
        captured.append(inputs.copy())
        yield PlannerEvent(todo=["one"])
        yield FinalEvent(text="done", session_id=inputs["session_id"])

    monkeypatch.setattr(runtime_module, "stream_agent", fake_stream_agent)
    runtime = Runtime(object(), data_dir=tmp_path)

    first = list(runtime.stream("first", SESSION_ID, max_attempts=2))
    second = list(runtime.stream("second", SESSION_ID, max_attempts=3))

    assert first == [PlannerEvent(todo=["one"]), FinalEvent(text="done", session_id=SESSION_ID)]
    assert second[-1] == FinalEvent(text="done", session_id=SESSION_ID)
    assert captured[0]["task"] == "first"
    assert captured[1]["task"] == "second"
    for state, maximum in zip(captured, [2, 3], strict=True):
        assert state["todo"] == []
        assert state["result"] == ""
        assert state["execution"] == []
        assert state["graph_state"] == "planning"
        assert state["verification"] == ""
        assert state["attempt"] == 0
        assert state["max_attempts"] == maximum
        assert state["final_answer"] == ""


def test_runtime_reuses_workspace_for_same_session(monkeypatch, tmp_path: Path) -> None:
    workspaces: list[Path] = []

    def tools_for(workspace: Path):
        workspaces.append(workspace)
        return []

    monkeypatch.setattr(runtime_module, "stream_agent", lambda *args, **kwargs: iter(()))
    runtime = Runtime(object(), data_dir=tmp_path, workspace_tool_factory=tools_for)

    list(runtime.stream("first", SESSION_ID))
    list(runtime.stream("second", SESSION_ID))

    assert workspaces[0] == workspaces[1] == (tmp_path / "workspaces" / SESSION_ID).resolve()


def test_runtime_isolates_different_sessions(monkeypatch, tmp_path: Path) -> None:
    workspaces: list[Path] = []
    monkeypatch.setattr(runtime_module, "stream_agent", lambda *args, **kwargs: iter(()))
    runtime = Runtime(
        object(),
        data_dir=tmp_path,
        workspace_tool_factory=lambda workspace: workspaces.append(workspace) or [],
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
