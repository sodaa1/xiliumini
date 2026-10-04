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


def test_stream_session_events_persists_chat_turns_and_is_public(
    monkeypatch, tmp_path: Path
) -> None:
    import xiliumini.config as config_module
    import xiliumini.runtime as runtime_module
    from xiliumini.core.session import load_or_create_session

    observed = []

    class FakeRuntime:
        def stream_session(self, task, session_id, context_summary, max_attempts=3):
            current = load_or_create_session((tmp_path / ".xiliumini").resolve())
            observed.append((task, session_id, context_summary, max_attempts, current.copy()))
            yield FinalEvent(text="你好！", session_id=session_id)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "load_settings", lambda: object())

    def create(settings, **overrides):
        assert overrides == {
            "approval_mode": "inline",
            "approval_handler": None,
            "checkpoint_mode": "light",
            "trace_mode": "full",
            "data_dir": (tmp_path / ".xiliumini").resolve(),
        }
        return FakeRuntime()

    monkeypatch.setattr(runtime_module, "create_runtime", create)

    events = list(agent_module.stream_session_events("你好"))

    assert events == [
        {
            "type": "graph_event",
            "event": {"text": "你好！", "session_id": observed[0][1]},
        }
    ]
    task, session_id, context, maximum, before_answer = observed[0]
    assert task == "你好"
    assert maximum == 3
    assert before_answer["turn_index"] == 1
    assert before_answer["recent_turns"][0]["role"] == "user"
    assert f"Session ID: {session_id}" in context
    assert "Turn 1 user: 你好" in context

    saved = load_or_create_session((tmp_path / ".xiliumini").resolve())
    assert saved["turn_index"] == 2
    assert saved["recent_turns"][1] == {
        "turn": 2,
        "role": "assistant",
        "route": "chat",
        "content": "你好！",
        "summary": "",
    }

    from xiliumini.core import stream_session_events

    assert stream_session_events is agent_module.stream_session_events


def test_stream_session_events_forwards_workflow_and_saves_summary(monkeypatch, tmp_path):
    import xiliumini.config as config_module
    import xiliumini.runtime as runtime_module
    from xiliumini.core.session import load_or_create_session

    workspace = tmp_path / ".xiliumini"

    class FakeRuntime:
        def stream_session(self, task, session_id, context_summary, max_attempts=3):
            yield ProgressEvent(stage="code_agent", message="working")
            yield PlannerEvent(todos=[], summary="implemented", attempt=1)
            yield VerifierEvent(passed=True, reason="checks passed", attempt=1)
            yield FinalEvent(text="workflow complete", session_id=session_id)

    monkeypatch.setattr(config_module, "load_settings", lambda: object())
    monkeypatch.setattr(runtime_module, "create_runtime", lambda *args, **kwargs: FakeRuntime())

    events = list(agent_module.stream_session_events("build it", session_workspace=workspace))

    assert [event["type"] for event in events] == [
        "custom_event",
        "graph_event",
        "graph_event",
        "graph_event",
    ]
    assert events[-1]["event"]["text"] == "workflow complete"
    saved = load_or_create_session(workspace.resolve())
    assistant = saved["recent_turns"][-1]
    assert assistant["route"] == "workflow"
    assert assistant["content"] == "workflow complete"
    assert "implemented" in assistant["summary"]
    assert "checks passed" in assistant["summary"]


def test_workflow_summary_keeps_latest_planner_and_final_verifier(monkeypatch, tmp_path):
    import xiliumini.config as config_module
    import xiliumini.runtime as runtime_module
    from xiliumini.core.session import MAX_TURN_CONTENT, load_or_create_session

    workspace = tmp_path / ".xiliumini"

    class FakeRuntime:
        def stream_session(self, task, session_id, context_summary, max_attempts=3):
            yield PlannerEvent(todos=[], summary="obsolete failure", attempt=1)
            yield VerifierEvent(passed=False, reason="retry needed", attempt=1)
            yield PlannerEvent(todos=[], summary="P" * MAX_TURN_CONTENT, attempt=2)
            yield VerifierEvent(passed=True, reason="FINAL VERIFIED", attempt=2)
            yield FinalEvent(text="done", session_id=session_id)

    monkeypatch.setattr(config_module, "load_settings", lambda: object())
    monkeypatch.setattr(runtime_module, "create_runtime", lambda *args, **kwargs: FakeRuntime())

    list(agent_module.stream_session_events("build it", session_workspace=workspace))

    summary = load_or_create_session(workspace.resolve())["recent_turns"][-1]["summary"]
    assert len(summary) <= MAX_TURN_CONTENT
    assert "FINAL VERIFIED" in summary
    assert "obsolete failure" not in summary
    assert "retry needed" not in summary


def test_stream_session_events_second_turn_receives_prior_conversation(monkeypatch, tmp_path):
    import xiliumini.config as config_module
    import xiliumini.runtime as runtime_module

    contexts = []
    workspace = tmp_path / ".xiliumini"

    class FakeRuntime:
        def stream_session(self, task, session_id, context_summary, max_attempts=3):
            contexts.append(context_summary)
            yield FinalEvent(text=f"answer to {task}", session_id=session_id)

    monkeypatch.setattr(config_module, "load_settings", lambda: object())
    monkeypatch.setattr(runtime_module, "create_runtime", lambda *args, **kwargs: FakeRuntime())

    list(agent_module.stream_session_events("first question", session_workspace=workspace))
    list(agent_module.stream_session_events("second question", session_workspace=workspace))

    assert "Turn 1 user: first question" in contexts[0]
    assert "Turn 2 assistant [chat]: answer to first question" in contexts[1]
    assert "Turn 3 user: second question" in contexts[1]


def test_stream_session_events_error_preserves_only_saved_user_turn(monkeypatch, tmp_path):
    import xiliumini.config as config_module
    import xiliumini.runtime as runtime_module
    from xiliumini.core.session import load_or_create_session

    workspace = tmp_path / ".xiliumini"

    class FakeRuntime:
        def stream_session(self, task, session_id, context_summary, max_attempts=3):
            yield ErrorEvent(code="provider_error", message="Provider request failed")

    monkeypatch.setattr(config_module, "load_settings", lambda: object())
    monkeypatch.setattr(runtime_module, "create_runtime", lambda *args, **kwargs: FakeRuntime())

    events = list(agent_module.stream_session_events("keep me", session_workspace=workspace))

    assert events == [
        {
            "type": "graph_event",
            "event": {"code": "provider_error", "message": "Provider request failed"},
        }
    ]
    saved = load_or_create_session(workspace.resolve())
    assert saved["turn_index"] == 1
    assert [turn["role"] for turn in saved["recent_turns"]] == ["user"]


def test_stream_session_events_close_closes_runtime_without_assistant_turn(monkeypatch, tmp_path):
    import xiliumini.config as config_module
    import xiliumini.runtime as runtime_module
    from xiliumini.core.session import load_or_create_session

    workspace = tmp_path / ".xiliumini"
    closed = []

    class ClosableEvents:
        def __init__(self):
            self.remaining = iter(
                [
                    ProgressEvent(stage="planner", message="working"),
                    FinalEvent(text="too late", session_id=SESSION_ID),
                ]
            )

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.remaining)

        def close(self):
            closed.append(True)

    class FakeRuntime:
        def stream_session(self, task, session_id, context_summary, max_attempts=3):
            return ClosableEvents()

    monkeypatch.setattr(config_module, "load_settings", lambda: object())
    monkeypatch.setattr(runtime_module, "create_runtime", lambda *args, **kwargs: FakeRuntime())

    events = agent_module.stream_session_events("unfinished", session_workspace=workspace)
    assert next(events)["type"] == "custom_event"
    events.close()

    assert closed == [True]
    saved = load_or_create_session(workspace.resolve())
    assert saved["turn_index"] == 1
    assert [turn["role"] for turn in saved["recent_turns"]] == ["user"]


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
            "event": {
                "stage": "planner",
                "message": "working",
                "event_type": "progress",
                "details": {},
            },
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
        ProgressEvent(
            stage="actor",
            message="Starting: tests",
            details={"stage": "actor", "message": "Starting: tests"},
        ),
        VerifierEvent(passed=False, reason="green missing", attempt=1),
        FinalEvent(text="failed after 1", session_id=SESSION_ID),
    ]


def test_chunk_events_preserves_structured_progress_payload_independently() -> None:
    payload = {
        "type": "tool_call",
        "stage": "code_agent",
        "message": "code_agent: file_write",
        "tool": "file_write",
        "args": {"path": "app.py"},
    }

    events = list(agent_module._chunk_events(("custom", payload), SESSION_ID))
    payload["args"]["path"] = "changed.py"

    assert events == [
        ProgressEvent(
            stage="code_agent",
            message="code_agent: file_write",
            event_type="tool_call",
            details={
                "type": "tool_call",
                "stage": "code_agent",
                "message": "code_agent: file_write",
                "tool": "file_write",
                "args": {"path": "app.py"},
            },
        )
    ]


@pytest.mark.parametrize("event_type", [None, 7, False])
def test_chunk_events_malformed_custom_type_falls_back_to_progress(event_type) -> None:
    payload = {"stage": "planner", "message": "working", "type": event_type}

    event = next(agent_module._chunk_events(("custom", payload), SESSION_ID))

    assert isinstance(event, ProgressEvent)
    assert event.event_type == "progress"
    assert event.details == payload


def test_progress_event_compatibility_defaults_structured_fields() -> None:
    event = ProgressEvent(stage="planner", message="working")

    assert event.event_type == "progress"
    assert event.details == {}


def test_chunk_events_synthesizes_display_fields_for_handoff() -> None:
    payload = {
        "type": "handoff",
        "agent": "code_agent",
        "attempt": 1,
        "ok": True,
        "summary": "Delegation completed",
    }

    event = next(agent_module._chunk_events(("custom", payload), SESSION_ID))

    assert event == ProgressEvent(
        stage="planner",
        message="planner → code_agent: Delegation completed",
        event_type="handoff",
        details=payload,
    )


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
        return {
            "type": "checkpoint_saved",
            "status": kwargs["status"],
            "latest_node": kwargs["latest_node"],
        }

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
    assert [
        event
        for event in events
        if not isinstance(event, ProgressEvent) or event.event_type != "checkpoint_saved"
    ] == [
        ProgressEvent(
            stage="code",
            message="working",
            event_type="tool_call",
            details=custom,
        ),
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
        assert calls[-2] == (
            "custom",
            {
                "type": "checkpoint_saved",
                "status": "completed",
                "latest_node": "planner",
            },
        )
    assert calls[-1][0] == "end"
    assert calls[-1][1]["status"] == "completed"
    assert state["attempt"] == 0


@pytest.mark.parametrize(
    ("mode", "expected_statuses"),
    [
        ("light", ["running"]),
        ("strict", ["running", "running"]),
    ],
)
def test_checkpoint_events_are_streamed_once_per_successful_save(
    monkeypatch, tmp_path, mode, expected_statuses
):
    calls = []
    harness = Harness(calls, mode)
    workflow = FakeWorkflow(
        [
            ("custom", {"type": "tool_call", "stage": "code", "message": "working"}),
            ("updates", {"planner": {"todos": [], "result": "done", "attempt": 1}}),
        ]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: workflow)

    events = list(
        agent_module.stream_agent(
            object(),
            inputs(tmp_path),
            memory_manager=object(),
            checkpoint_manager=harness,
            trace_recorder=harness,
        )
    )

    checkpoint_events = [
        event
        for event in events
        if isinstance(event, ProgressEvent) and event.event_type == "checkpoint_saved"
    ]
    assert [event.details["status"] for event in checkpoint_events] == expected_statuses
    assert checkpoint_events[-1].details["latest_node"] == "planner"
    trace_checkpoints = [
        call[1]
        for call in calls
        if call[0] == "custom" and call[1].get("type") == "checkpoint_saved"
    ]
    assert [event["status"] for event in trace_checkpoints] == [
        "started",
        *expected_statuses,
        "completed",
    ]


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
