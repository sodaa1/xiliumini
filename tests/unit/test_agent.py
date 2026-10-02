from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import xiliumini.core.agent as agent_module
from xiliumini.core.approval import ApprovalDecision
from xiliumini.errors import CheckpointError, TraceError
from xiliumini.events import (
    ErrorEvent,
    FinalEvent,
    PlannerEvent,
    ProgressEvent,
    VerifierEvent,
)
from xiliumini.graph.state import GraphState

SESSION_ID = "11111111-1111-4111-8111-111111111111"


def test_stream_agent_events_builds_overridden_runtime_and_adapts_events(
    monkeypatch, tmp_path: Path
) -> None:
    import xiliumini.config as config_module
    import xiliumini.runtime as runtime_module

    def handler(request):
        return ApprovalDecision(True, request.id)

    calls = []

    class FakeRuntime:
        def stream_workspace(self, task, workspace, max_attempts=3):
            calls.append((task, workspace, max_attempts))
            yield ProgressEvent(stage="planner", message="working")
            yield FinalEvent(text="done", session_id=SESSION_ID)

    monkeypatch.setattr(config_module, "load_settings", lambda: object())

    def create(settings, **overrides):
        assert overrides == {
            "approval_mode": "deny",
            "approval_handler": handler,
            "checkpoint_mode": "strict",
            "trace_mode": "full",
        }
        return FakeRuntime()

    monkeypatch.setattr(runtime_module, "create_runtime", create)

    events = list(
        agent_module.stream_agent_events(
            "task",
            workspace=tmp_path,
            max_attempts=5,
            approval_mode="deny",
            approval_handler=handler,
            checkpoint_mode="strict",
            trace_mode="on",
        )
    )

    assert calls == [("task", tmp_path, 5)]
    assert events == [
        {
            "type": "custom_event",
            "event": {"stage": "planner", "message": "working"},
        },
        {
            "type": "graph_event",
            "event": {"text": "done", "session_id": SESSION_ID},
        },
    ]


def test_stream_agent_events_rejects_conflicting_resume_workspace(
    monkeypatch, tmp_path: Path
) -> None:
    import xiliumini.config as config_module
    import xiliumini.runtime as runtime_module

    monkeypatch.setattr(config_module, "load_settings", lambda: object())
    monkeypatch.setattr(
        runtime_module,
        "create_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be created"),
    )
    workspace = tmp_path / "one"
    resume = tmp_path / "two"

    events = list(
        agent_module.stream_agent_events("task", workspace=workspace, resume_workspace=resume)
    )

    assert events == [
        {
            "type": "graph_event",
            "event": {"code": "workspace_error", "message": "workspace conflicts with resume"},
        }
    ]
    assert str(workspace) not in str(events)
    assert str(resume) not in str(events)


def test_stream_agent_events_delegates_resume_when_paths_match(monkeypatch, tmp_path: Path) -> None:
    import xiliumini.config as config_module
    import xiliumini.runtime as runtime_module

    calls = []

    class FakeRuntime:
        def resume(self, workspace, *, task=None, max_attempts=3):
            calls.append((workspace, task, max_attempts))
            yield ErrorEvent(code="checkpoint_error", message="missing")

    monkeypatch.setattr(config_module, "load_settings", lambda: object())
    monkeypatch.setattr(runtime_module, "create_runtime", lambda *args, **kwargs: FakeRuntime())
    workspace = tmp_path / "same"

    events = list(
        agent_module.stream_agent_events(
            "task", workspace=workspace, resume_workspace=workspace, trace_mode="off"
        )
    )

    assert calls == [(workspace, "task", 3)]
    assert events == [
        {
            "type": "graph_event",
            "event": {"code": "checkpoint_error", "message": "missing"},
        }
    ]


def test_stream_agent_events_close_propagates_to_runtime_generator(monkeypatch, tmp_path: Path):
    import xiliumini.config as config_module
    import xiliumini.runtime as runtime_module

    closed = []

    class ClosableEvents:
        def __init__(self):
            self.remaining = iter([ProgressEvent(stage="planner", message="working")])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.remaining)

        def close(self):
            closed.append(True)

    class FakeRuntime:
        events = ClosableEvents()

        def stream_workspace(self, task, workspace, max_attempts=3):
            return self.events

    monkeypatch.setattr(config_module, "load_settings", lambda: object())
    monkeypatch.setattr(runtime_module, "create_runtime", lambda *args, **kwargs: FakeRuntime())
    events = agent_module.stream_agent_events("task", workspace=tmp_path)

    assert next(events)["type"] == "custom_event"
    events.close()

    assert closed == [True]


def inputs(tmp_path: Path, *, max_attempts: int = 3) -> GraphState:
    return {
        "task": "build it",
        "todos": [],
        "research_notes": [],
        "agent_results": [],
        "tool_events": [],
        "supervisor_ok": False,
        "result": "",
        "graph_state": "planning",
        "verification": "",
        "attempt": 0,
        "max_attempts": max_attempts,
        "final_answer": "",
        "session_id": SESSION_ID,
        "workspace": tmp_path,
        "memory": {
            "rules": {"fixed_rules": [], "user_preferences": []},
            "working": {
                "current_node": "planner",
                "task": "build it",
                "session_id": SESSION_ID,
                "plan_summary": "",
                "todos": [],
                "acceptance_criteria": [],
                "research_notes": [],
                "sources": [],
                "agent_handoffs": [],
                "code_agent_summary": "",
                "verifier_summary": "",
                "last_error": "",
                "attempts": {"current": 0, "max": max_attempts},
            },
            "history": {
                "history_summary": "",
                "notepad_summary": "",
                "context_summary": "",
                "compression_events": [],
            },
        },
        "current_node": "planner",
        "resume_node": "planner",
        "plan_summary": "",
        "acceptance_criteria": [],
        "agent_handoffs": [],
        "code_agent_summary": "",
        "verifier_summary": "",
        "last_error": "",
        "context_summary": "",
        "compression_events": [],
    }


class FakeWorkflow:
    def __init__(self, chunks: list[object]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[dict, dict, list[str]]] = []

    def stream(self, graph_inputs, *, config, stream_mode):
        self.calls.append((graph_inputs, config, stream_mode))
        yield from self.chunks


def test_stream_agent_calls_workflow_with_both_stream_modes(monkeypatch, tmp_path: Path) -> None:
    workflow = FakeWorkflow([])
    manager = object()
    built = []

    def build(*args, **kwargs):
        built.append((args, kwargs))
        return workflow

    monkeypatch.setattr(agent_module, "build_workflow", build)
    graph_inputs = inputs(tmp_path)

    assert list(agent_module.stream_agent(object(), graph_inputs, memory_manager=manager)) == []
    assert built[0][0][1] is manager
    assert workflow.calls == [
        (
            graph_inputs,
            {"configurable": {"thread_id": SESSION_ID}, "recursion_limit": 11},
            ["updates", "custom"],
        )
    ]


def test_stream_agent_maps_updates_and_custom_events(monkeypatch, tmp_path: Path) -> None:
    workflow = FakeWorkflow(
        [
            ("updates", {"planner": {"todos": [], "result": "implemented", "attempt": 1}}),
            ("custom", {"stage": "actor", "message": "Starting: tests"}),
            (
                "updates",
                {
                    "verifier": {
                        "graph_state": "failed",
                        "verification": "green missing",
                        "attempt": 1,
                    }
                },
            ),
            (
                "updates",
                {
                    "final": {
                        "final_answer": "failed after 1",
                        "session_id": "ignored-node-value",
                    }
                },
            ),
        ]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: workflow)

    events = list(agent_module.stream_agent(object(), inputs(tmp_path), memory_manager=object()))

    assert events == [
        PlannerEvent(todos=[], summary="implemented", attempt=1),
        ProgressEvent(stage="actor", message="Starting: tests"),
        VerifierEvent(passed=False, reason="green missing", attempt=1),
        FinalEvent(text="failed after 1", session_id=SESSION_ID),
    ]


def test_stream_agent_ignores_unknown_and_incomplete_chunks(monkeypatch, tmp_path: Path) -> None:
    workflow = FakeWorkflow(
        [
            ("debug", {"planner": {"todo": ["ignored"]}}),
            ("updates", {"unknown": {"value": 1}}),
            ("updates", {"actor": {"attempt": 1}}),
            ("custom", {"stage": "actor"}),
            "malformed",
        ]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: workflow)

    assert (
        list(agent_module.stream_agent(object(), inputs(tmp_path), memory_manager=object())) == []
    )


def test_stream_agent_rejects_zero_attempts_before_building(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        agent_module,
        "build_workflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        list(
            agent_module.stream_agent(
                object(), inputs(tmp_path, max_attempts=0), memory_manager=object()
            )
        )


class Harness:
    def __init__(self, calls, mode="strict"):
        self.calls = calls
        self.mode = mode

    def start(self, state, **kwargs):
        self.calls.append(("start", deepcopy(kwargs)))

    def record_custom_event(self, event):
        self.calls.append(("custom", deepcopy(event)))

    def record_graph_update(self, event):
        self.calls.append(("update", deepcopy(event)))

    def save(self, state, **kwargs):
        self.calls.append(("save", deepcopy(state), deepcopy(kwargs)))
        return {"type": "checkpoint_saved"}

    def end(self, **kwargs):
        self.calls.append(("end", deepcopy(kwargs)))


@pytest.mark.parametrize("mode, saves", [("strict", 4), ("light", 3), ("off", 0)])
def test_harness_normal_order_and_state(monkeypatch, tmp_path, mode, saves):
    calls = []
    harness = Harness(calls, mode)
    custom = {"type": "tool_call", "stage": "code", "message": "working"}
    update = {"planner": {"todos": [], "result": "done", "attempt": 1}}
    workflow = FakeWorkflow([("custom", custom), ("updates", update)])
    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: workflow)
    state = inputs(tmp_path)
    events = list(
        agent_module.stream_agent(
            object(),
            state,
            memory_manager=object(),
            checkpoint_manager=harness,
            trace_recorder=harness,
            resumed=True,
            resume_event={"type": "resume"},
        )
    )
    assert events == [
        ProgressEvent(stage="code", message="working"),
        PlannerEvent(todos=[], summary="done", attempt=1),
    ]
    assert calls[0] == ("start", {"resumed": True, "resume_event": {"type": "resume"}})
    assert ("custom", custom) in calls
    saved = [call for call in calls if call[0] == "save"]
    assert len(saved) == saves
    if saved:
        assert saved[0][2] == {"status": "started", "latest_node": None, "event": None}
        assert saved[-1][1]["attempt"] == 1
        assert saved[-1][2]["status"] == "completed"
        assert calls[-2] == ("custom", {"type": "checkpoint_saved"})
    assert calls[-1][0] == "end"
    assert calls[-1][1]["status"] == "completed"
    assert state["attempt"] == 0


def test_harness_mutating_observers_cannot_change_state(monkeypatch, tmp_path):
    calls = []
    harness = Harness(calls)
    original = harness.record_graph_update

    def mutate(event):
        original(event)
        event["planner"]["todos"].append({"bad": True})

    harness.record_graph_update = mutate
    workflow = FakeWorkflow(
        [("updates", {"planner": {"todos": [], "result": "done", "attempt": 1}})]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: workflow)
    state = inputs(tmp_path)
    list(
        agent_module.stream_agent(
            object(),
            state,
            memory_manager=object(),
            checkpoint_manager=harness,
            trace_recorder=harness,
        )
    )
    assert state["todos"] == []
    assert [call for call in calls if call[0] == "save"][-1][1]["todos"] == []


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("workflow"),
        CheckpointError("checkpoint"),
        TraceError("trace"),
        KeyboardInterrupt(),
        GeneratorExit(),
        SystemExit(7),
    ],
)
def test_harness_failure_and_interrupt_preserve_original(monkeypatch, tmp_path, error):
    calls = []
    harness = Harness(calls)

    class BrokenWorkflow:
        def stream(self, *args, **kwargs):
            raise error
            yield

    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: BrokenWorkflow())
    with pytest.raises(type(error)) as caught:
        list(
            agent_module.stream_agent(
                object(),
                inputs(tmp_path),
                memory_manager=object(),
                checkpoint_manager=harness,
                trace_recorder=harness,
            )
        )
    assert caught.value is error
    # SystemExit remains a failed execution; caller-controlled close is interrupted.
    status = "interrupted" if isinstance(error, (KeyboardInterrupt, GeneratorExit)) else "failed"
    assert [call[2]["status"] for call in calls if call[0] == "save"] == [
        "started",
        status,
    ]
    assert [call[1]["status"] for call in calls if call[0] == "end"] == [status]


def test_harness_close_finalizes_once_despite_persistence_failure(monkeypatch, tmp_path):
    calls = []
    harness = Harness(calls)
    original = harness.save

    def broken(state, **kwargs):
        result = original(state, **kwargs)
        if kwargs["status"] == "interrupted":
            raise RuntimeError("save failure")
        return result

    harness.save = broken
    workflow = FakeWorkflow([("custom", {"stage": "code", "message": "working"})])
    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: workflow)
    stream = agent_module.stream_agent(
        object(),
        inputs(tmp_path),
        memory_manager=object(),
        checkpoint_manager=harness,
        trace_recorder=harness,
    )
    next(stream)
    stream.close()
    stream.close()
    assert [call[1]["status"] for call in calls if call[0] == "end"] == ["interrupted"]


@pytest.mark.parametrize(
    "method", ["start", "record_custom_event", "record_graph_update", "save", "end"]
)
@pytest.mark.parametrize("error_type", [RuntimeError, CheckpointError, TraceError])
def test_harness_recorder_failure_is_not_masked(monkeypatch, tmp_path, method, error_type):
    calls = []
    harness = Harness(calls)
    error = error_type(method)

    def broken(*args, **kwargs):
        calls.append(("broken", method))
        raise error

    setattr(harness, method, broken)
    workflow = FakeWorkflow(
        [
            ("custom", {"stage": "code", "message": "working"}),
            ("updates", {"planner": {"attempt": 1}}),
        ]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: workflow)
    with pytest.raises(error_type) as caught:
        list(
            agent_module.stream_agent(
                object(),
                inputs(tmp_path),
                memory_manager=object(),
                checkpoint_manager=harness,
                trace_recorder=harness,
            )
        )
    assert caught.value is error
    end_attempts = [call for call in calls if call[0] == "end" or call == ("broken", "end")]
    assert len(end_attempts) == 1


def test_core_public_export_remains_canonical():
    from xiliumini.core import stream_agent, stream_agent_events

    assert stream_agent is agent_module.stream_agent
    assert stream_agent_events is agent_module.stream_agent_events


def test_harness_workflow_failure_survives_broken_finalizers(monkeypatch, tmp_path):
    harness = Harness([])
    error = ValueError("original")

    original_save = harness.save

    def broken_save(*args, **kwargs):
        if kwargs["status"] == "started":
            return original_save(*args, **kwargs)
        raise RuntimeError("persistence")

    def broken_end(*args, **kwargs):
        raise RuntimeError("persistence")

    harness.save = broken_save
    harness.end = broken_end

    class BrokenWorkflow:
        def stream(self, *args, **kwargs):
            raise error
            yield

    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: BrokenWorkflow())
    with pytest.raises(ValueError) as caught:
        list(
            agent_module.stream_agent(
                object(),
                inputs(tmp_path),
                memory_manager=object(),
                checkpoint_manager=harness,
                trace_recorder=harness,
            )
        )
    assert caught.value is error


def test_harness_each_node_merged_before_checkpoint(monkeypatch, tmp_path):
    calls = []
    harness = Harness(calls, "light")
    workflow = FakeWorkflow(
        [("updates", {"planner": {"attempt": 1}, "verifier": {"verification": "pass"}})]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: workflow)
    list(
        agent_module.stream_agent(
            object(),
            inputs(tmp_path),
            memory_manager=object(),
            checkpoint_manager=harness,
            trace_recorder=harness,
        )
    )
    saves = [call for call in calls if call[0] == "save"]
    assert [call[2]["latest_node"] for call in saves] == [
        None,
        "planner",
        "verifier",
        "verifier",
    ]
    assert saves[0][1]["attempt"] == 0
    assert saves[1][1]["attempt"] == 1
    assert saves[1][1]["verification"] == ""
    assert saves[2][1]["verification"] == "pass"
