from __future__ import annotations

from pathlib import Path

import pytest

import xiliumini.runtime as runtime_module
from tests.agent_fakes import state as complete_state
from xiliumini.core.approval import ApprovalDecision
from xiliumini.errors import NodeOutputError
from xiliumini.events import ErrorEvent, FinalEvent, PlannerEvent, ProgressEvent
from xiliumini.graph.memory import MemoryLimits
from xiliumini.graph.state import GraphState
from xiliumini.runtime import Runtime
from xiliumini.tools.approval_context import approval_config
from xiliumini.tools.preferences import UserPreferenceStore

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


def test_stream_workspace_reuses_contained_directory_and_resets_todos(monkeypatch, tmp_path):
    from xiliumini.tools.todo import TodoStore

    workspace = tmp_path / "workspaces" / "manual"
    observed = []

    def capture(model, inputs, **kwargs):
        observed.append((inputs["workspace"], inputs["session_id"], TodoStore(workspace).read()))
        TodoStore(workspace).write([{"id": "old", "content": "previous task"}])
        return iter(())

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    runtime = Runtime(object(), data_dir=tmp_path)

    assert list(runtime.stream_workspace("first", workspace)) == []
    assert list(runtime.stream_workspace("second", workspace)) == []
    assert observed == [(workspace.resolve(), "manual", []), (workspace.resolve(), "manual", [])]


def test_stream_workspace_rejects_foreign_path_without_leaking_it(tmp_path):
    foreign = tmp_path.parent / "private-workspace"

    events = list(Runtime(object(), data_dir=tmp_path).stream_workspace("task", foreign))

    assert events == [
        ErrorEvent(code="workspace_error", message="workspace must stay inside the data directory")
    ]
    assert isinstance(events[0], ErrorEvent)
    assert str(foreign) not in events[0].message


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


def test_runtime_keeps_approval_configuration_isolated_across_interleaved_streams(
    monkeypatch, tmp_path
):
    def first_handler(request):
        return ApprovalDecision(True, request.id)

    def second_handler(request):
        return ApprovalDecision(False, request.id)

    observed = []

    def capture(model, inputs, **kwargs):
        try:
            observed.append((inputs["session_id"], approval_config.get()))
            yield FinalEvent(text="one", session_id=inputs["session_id"])
            observed.append((inputs["session_id"], approval_config.get()))
            yield FinalEvent(text="two", session_id=inputs["session_id"])
        finally:
            observed.append((f"{inputs['session_id']}:close", approval_config.get()))

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    first = Runtime(
        object(),
        data_dir=tmp_path,
        approval_mode="auto",
        approval_handler=first_handler,
    ).stream("first", SESSION_ID)
    second = Runtime(
        object(),
        data_dir=tmp_path,
        approval_mode="deny",
        approval_handler=second_handler,
    ).stream("second", SECOND_SESSION_ID)

    next(first)
    assert approval_config.get().mode == "inline"
    next(second)
    assert approval_config.get().mode == "inline"
    next(first)
    first.close()
    second.close()

    first_configs = [config for session, config in observed if session.startswith(SESSION_ID)]
    second_configs = [
        config for session, config in observed if session.startswith(SECOND_SESSION_ID)
    ]
    assert {(config.mode, config.handler) for config in first_configs} == {("auto", first_handler)}
    assert {(config.mode, config.handler) for config in second_configs} == {
        ("deny", second_handler)
    }
    assert approval_config.get().mode == "inline"


def test_runtime_keeps_memory_managers_isolated_when_streams_interleave(monkeypatch, tmp_path):
    observed = []

    def capture(_model, inputs, *, memory_manager, **_kwargs):
        observed.append((inputs["session_id"], memory_manager))
        yield FinalEvent(text="one", session_id=inputs["session_id"])
        observed.append((inputs["session_id"], memory_manager))
        yield FinalEvent(text="two", session_id=inputs["session_id"])

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    runtime = Runtime(object(), data_dir=tmp_path)
    first = runtime.stream("first", SESSION_ID)
    second = runtime.stream("second", SECOND_SESSION_ID)

    next(first)
    next(second)
    next(first)
    next(second)

    first_managers = [manager for session, manager in observed if session == SESSION_ID]
    second_managers = [manager for session, manager in observed if session == SECOND_SESSION_ID]
    assert first_managers[0] is first_managers[1]
    assert second_managers[0] is second_managers[1]
    assert first_managers[0] is not second_managers[0]


def test_runtime_delegates_complete_fresh_graph_state(monkeypatch, tmp_path: Path) -> None:
    captured: list[GraphState] = []

    managers = []

    def fake_stream_agent(
        model, inputs: GraphState, checkpointer=None, memory_manager=None, **kwargs
    ):
        captured.append(inputs.copy())
        managers.append(memory_manager)
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
        assert state["current_node"] == "planner"
        assert state["plan_summary"] == ""
        assert state["acceptance_criteria"] == []
        assert state["agent_handoffs"] == []
        assert state["code_agent_summary"] == ""
        assert state["verifier_summary"] == ""
        assert state["last_error"] == ""
        assert state["context_summary"] == ""
        assert state["compression_events"] == []
        assert state["memory"]["working"]["task"] == state["task"]
    assert managers[0] is not managers[1]


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


def test_stream_session_chat_uses_context_without_starting_full_workflow(
    monkeypatch, tmp_path: Path
) -> None:
    observed = []

    class EntryWorkflow:
        def invoke(self, inputs, **kwargs):
            observed.append(inputs.copy())
            return {
                **inputs,
                "intent_route": "chat",
                "intent_reason": "greeting",
                "intent_confidence": 0.95,
                "chat_response": "你好！",
                "final_answer": "你好！",
            }

    monkeypatch.setattr(
        runtime_module, "build_entry_workflow", lambda model: EntryWorkflow(), raising=False
    )
    monkeypatch.setattr(
        runtime_module,
        "stream_agent",
        lambda *args, **kwargs: pytest.fail("full workflow must not start for chat"),
    )
    monkeypatch.setattr(
        runtime_module.TodoStore,
        "start_task",
        lambda self: pytest.fail("chat must not reset Todo state"),
    )

    events = list(
        Runtime(object(), data_dir=tmp_path).stream_session(
            "你好", SESSION_ID, "prior conversation", max_attempts=4
        )
    )

    assert events == [FinalEvent(text="你好！", session_id=SESSION_ID)]
    assert observed[0]["context_summary"] == "prior conversation"
    assert observed[0]["task"] == "你好"
    assert observed[0]["max_attempts"] == 4
    assert observed[0]["workspace"] == (tmp_path / "workspaces" / SESSION_ID).resolve()


def test_stream_session_workflow_handoff_preserves_route_and_uses_harness(
    monkeypatch, tmp_path: Path
) -> None:
    captured = []
    started = []

    class EntryWorkflow:
        def invoke(self, inputs, **kwargs):
            return {
                **inputs,
                "intent_route": "workflow",
                "intent_reason": "needs workspace files",
                "intent_confidence": 0.98,
            }

    def capture(model, inputs, **kwargs):
        captured.append((inputs.copy(), kwargs))
        yield FinalEvent(text="workflow complete", session_id=inputs["session_id"])

    original_start = runtime_module.TodoStore.start_task

    def record_start(store):
        started.append(store.workspace)
        return original_start(store)

    monkeypatch.setattr(runtime_module, "build_entry_workflow", lambda model: EntryWorkflow())
    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    monkeypatch.setattr(runtime_module.TodoStore, "start_task", record_start)

    events = list(
        Runtime(object(), data_dir=tmp_path).stream_session(
            "inspect files", SESSION_ID, "earlier turn summary"
        )
    )

    workspace = (tmp_path / "workspaces" / SESSION_ID).resolve()
    assert events == [FinalEvent(text="workflow complete", session_id=SESSION_ID)]
    assert started == [workspace]
    assert len(captured) == 1
    routed, kwargs = captured[0]
    assert routed["intent_route"] == "workflow"
    assert routed["intent_reason"] == "needs workspace files"
    assert routed["intent_confidence"] == 0.98
    assert routed["context_summary"] == "earlier turn summary"
    assert kwargs["memory_manager"] is not None
    assert kwargs["checkpoint_manager"] is not None
    assert kwargs["trace_recorder"] is not None


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


def test_preferences_recover_across_tasks_and_sessions(monkeypatch, tmp_path: Path) -> None:
    captured = []

    def capture(_model, inputs, **_kwargs):
        captured.append(inputs["memory"]["rules"]["user_preferences"])
        return iter(())

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    store = UserPreferenceStore(tmp_path)
    store.upsert("test_style", "Always use pytest.")
    runtime = Runtime(object(), data_dir=tmp_path)

    list(runtime.stream("one", SESSION_ID))
    list(runtime.stream("two", SECOND_SESSION_ID))
    assert [[item["key"] for item in rules] for rules in captured] == [
        ["test_style"],
        ["test_style"],
    ]

    store.remove("test_style")
    list(runtime.stream("three", SESSION_ID))
    list(runtime.stream("four", SECOND_SESSION_ID))
    assert captured[-2:] == [[], []]


def test_corrupt_preference_and_history_map_to_memory_error(tmp_path: Path) -> None:
    preference_path = tmp_path / "USER_PREFERENCES.md"
    preference_path.write_bytes(b"not valid")
    runtime = Runtime(object(), data_dir=tmp_path)

    preference_events = list(runtime.stream("task", SESSION_ID))
    assert preference_events == [ErrorEvent(code="memory_error", message="memory data is invalid")]
    assert preference_path.read_bytes() == b"not valid"

    preference_path.unlink()
    history_path = tmp_path / "workspaces" / SESSION_ID / "HISTORY_SUMMARY.md"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_bytes(b"\xff")
    history_events = list(runtime.stream("task", SESSION_ID))
    assert history_events == [ErrorEvent(code="memory_error", message="memory data is invalid")]
    assert history_path.read_bytes() == b"\xff"


def test_impossible_memory_budget_fails_before_provider_call(tmp_path: Path) -> None:
    class CountingModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            self.calls += 1
            raise AssertionError("provider must not be called")

    model = CountingModel()
    runtime = Runtime(
        model,
        data_dir=tmp_path,
        memory_limits=MemoryLimits(10, 0.8, 1),
        token_counter=lambda messages, _model: 8 if len(messages) >= 2 else 1,
    )

    events = list(runtime.stream("task", SESSION_ID))

    assert events == [
        ErrorEvent(
            code="memory_error",
            message="required planner context exceeds the token budget",
        )
    ]
    assert model.calls == 0


def test_runtime_checkpoint_trace_interleave(monkeypatch, tmp_path):
    observed = []

    def capture(model, inputs, **kwargs):
        checkpoint = kwargs["checkpoint_manager"]
        trace = kwargs["trace_recorder"]
        observed.append((inputs, checkpoint, trace))
        trace.tool_calls += 1
        yield FinalEvent(text="one", session_id=inputs["session_id"])
        assert trace.tool_calls == 1
        assert checkpoint.workspace == inputs["workspace"]

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    runtime = Runtime(object(), data_dir=tmp_path)
    first = runtime.stream("one", SESSION_ID)
    second = runtime.stream("two", SECOND_SESSION_ID)
    next(first)
    next(second)
    list(first)
    list(second)
    assert observed[0][1] is not observed[1][1]
    assert observed[0][2].trace_id != observed[1][2].trace_id
    assert not hasattr(runtime, "workspace")


def test_runtime_resume_preserves_todos_and_restored_inputs(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from xiliumini.core.checkpoint import CheckpointManager
    from xiliumini.tools.todo import TodoStore

    workspace = tmp_path / "workspaces" / SESSION_ID
    workspace.mkdir(parents=True)
    TodoStore(workspace).write([{"id": "keep", "content": "unfinished"}])
    state = {
        **complete_state(workspace),
        "task": "saved",
        "session_id": SESSION_ID,
        "workspace": workspace,
        "attempt": 1,
        "max_attempts": 3,
        "graph_state": "planning",
    }
    CheckpointManager(SimpleNamespace(workspace=workspace, checkpoint_mode="light")).save(
        state, latest_node="planner"
    )

    def capture(model, inputs, **kwargs):
        assert "runtime" not in inputs
        assert inputs["workspace"] == workspace.resolve()
        assert inputs["task"] == "override"
        assert inputs["max_attempts"] == 5
        assert inputs["resume_node"] == "verifier"
        assert TodoStore(workspace).read()[0]["id"] == "keep"
        assert kwargs["resumed"] is True
        assert kwargs["resume_event"]["type"] == "resume"
        yield FinalEvent(text="resumed", session_id=SESSION_ID)

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    assert list(
        Runtime(object(), data_dir=tmp_path).resume(workspace, task="override", max_attempts=5)
    ) == [FinalEvent(text="resumed", session_id=SESSION_ID)]


def test_runtime_resume_invalid_workspace_is_redacted(tmp_path):
    runtime = Runtime(object(), data_dir=tmp_path)
    for workspace in (tmp_path / "missing", tmp_path):
        events = list(runtime.resume(workspace))
        assert len(events) == 1 and isinstance(events[0], ErrorEvent)
        assert events[0].code == "checkpoint_error"
        assert str(tmp_path) not in events[0].message


def test_create_runtime_passes_harness_settings(monkeypatch, tmp_path):
    from xiliumini.config import Settings

    settings = Settings.from_env(
        {
            "XILIUMINI_API_KEY": "secret",
            "XILIUMINI_MODEL": "fake",
            "XILIUMINI_DATA_DIR": str(tmp_path),
            "XILIUMINI_CHECKPOINT_MODE": "strict",
            "XILIUMINI_TRACE_MODE": "summary",
            "XILIUMINI_TRACE_ID": "configured",
        }
    )
    monkeypatch.setattr(runtime_module, "create_chat_model", lambda _: object())
    observed = []

    def capture(model, inputs, **kwargs):
        observed.append(
            (
                kwargs["checkpoint_manager"].mode,
                kwargs["trace_recorder"].mode,
                kwargs["trace_recorder"].trace_id,
            )
        )
        return iter(())

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    list(runtime_module.create_runtime(settings).stream("task", SESSION_ID))
    assert observed == [("strict", "summary", "configured")]


def test_create_runtime_applies_harness_overrides_and_on_alias(monkeypatch, tmp_path):
    from xiliumini.config import Settings

    settings = Settings.from_env(
        {
            "XILIUMINI_API_KEY": "secret",
            "XILIUMINI_MODEL": "fake",
            "XILIUMINI_DATA_DIR": str(tmp_path),
            "XILIUMINI_CHECKPOINT_MODE": "light",
            "XILIUMINI_TRACE_MODE": "summary",
        }
    )

    def handler(request):
        return ApprovalDecision(True, request.id)

    monkeypatch.setattr(runtime_module, "create_chat_model", lambda _: object())
    observed = []

    def capture(model, inputs, **kwargs):
        active = approval_config.get()
        observed.append(
            (
                kwargs["checkpoint_manager"].mode,
                kwargs["trace_recorder"].mode,
                active.mode,
                active.handler,
            )
        )
        return iter(())

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    runtime = runtime_module.create_runtime(
        settings,
        approval_mode="deny",
        approval_handler=handler,
        checkpoint_mode="strict",
        trace_mode="on",
    )

    list(runtime.stream("task", SESSION_ID))

    assert observed == [("strict", "full", "deny", handler)]


def test_runtime_close_restores_contextvars(monkeypatch, tmp_path):
    from pydantic import SecretStr

    from xiliumini.agents.react import model_factory
    from xiliumini.tools.web_search_tool import search_api_key

    def factory():
        return object()

    secret = SecretStr("private")
    closed = []

    def capture(model, inputs, **kwargs):
        try:
            assert model_factory.get() is factory
            assert search_api_key.get() is secret
            yield FinalEvent(text="one", session_id=SESSION_ID)
        finally:
            assert model_factory.get() is factory
            assert search_api_key.get() is secret
            closed.append(True)

    monkeypatch.setattr(runtime_module, "stream_agent", capture)
    events = Runtime(
        object(), data_dir=tmp_path, agent_model_factory=factory, tavily_api_key=secret
    ).stream("task", SESSION_ID)
    next(events)
    assert model_factory.get() is None and search_api_key.get() is None
    events.close()
    assert closed == [True]
    assert model_factory.get() is None and search_api_key.get() is None


def test_runtime_resume_rejects_foreign_runtime_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runtime_module.CheckpointManager,
        "load_resume_inputs",
        lambda *args, **kwargs: ({"runtime": object()}, {}),
    )
    events = list(Runtime(object(), data_dir=tmp_path).resume(tmp_path))
    assert events == [ErrorEvent(code="checkpoint_error", message="Invalid resume runtime.")]


@pytest.mark.parametrize(
    "failure", [TimeoutError("private"), RuntimeError("private"), NodeOutputError("invalid node")]
)
def test_resume_execution_errors_match_stream(monkeypatch, tmp_path, failure):
    from xiliumini.events import ProgressEvent

    def load(context, **kwargs):
        return {
            "runtime": context.owner,
            "workspace": tmp_path,
            "session_id": SESSION_ID,
            "task": "task",
            "max_attempts": 3,
        }, {"type": "resume"}

    def fail(model, inputs, **kwargs):
        yield ProgressEvent(stage="planner", message="running")
        raise failure

    monkeypatch.setattr(runtime_module.CheckpointManager, "load_resume_inputs", load)
    monkeypatch.setattr(runtime_module, "stream_agent", fail)
    runtime = Runtime(object(), data_dir=tmp_path)
    assert list(runtime.resume(tmp_path)) == list(runtime.stream("task", SESSION_ID))


def test_resume_corrupt_preflight_and_interrupt_identity(monkeypatch, tmp_path):
    def corrupt(*args, **kwargs):
        raise ValueError("private path")

    monkeypatch.setattr(runtime_module.CheckpointManager, "load_resume_inputs", corrupt)
    runtime = Runtime(object(), data_dir=tmp_path)
    assert list(runtime.resume(tmp_path)) == [
        ErrorEvent(code="checkpoint_error", message="Could not resume checkpoint.")
    ]
    interruption = KeyboardInterrupt("identity")

    def interrupted(*args, **kwargs):
        raise interruption

    monkeypatch.setattr(runtime_module.CheckpointManager, "load_resume_inputs", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        list(runtime.resume(tmp_path))
    assert caught.value is interruption

    def load(context, **kwargs):
        return {
            "runtime": context.owner,
            "workspace": tmp_path,
            "session_id": SESSION_ID,
            "task": "task",
            "max_attempts": 3,
        }, {}

    def interrupted_run(*args, **kwargs):
        yield FinalEvent(text="before interruption", session_id=SESSION_ID)
        raise interruption

    monkeypatch.setattr(runtime_module.CheckpointManager, "load_resume_inputs", load)
    monkeypatch.setattr(runtime_module, "stream_agent", interrupted_run)
    with pytest.raises(KeyboardInterrupt) as caught:
        list(runtime.resume(tmp_path))
    assert caught.value is interruption


def test_real_resume_interleave_close_and_complete(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import xiliumini.core.agent as agent_module
    from xiliumini.core.checkpoint import CheckpointManager
    from xiliumini.tools.todo import TodoStore

    workspaces = []
    for session in (SESSION_ID, SECOND_SESSION_ID):
        workspace = tmp_path / "workspaces" / session
        workspace.mkdir(parents=True)
        TodoStore(workspace).write([{"id": session, "content": "keep"}])
        state = {
            **complete_state(workspace),
            "task": session,
            "session_id": session,
            "workspace": workspace,
            "attempt": 1,
            "max_attempts": 3,
            "graph_state": "planning",
        }
        CheckpointManager(SimpleNamespace(workspace=workspace, checkpoint_mode="light")).save(
            state, latest_node="planner"
        )
        workspaces.append(workspace)

    class Workflow:
        def stream(self, inputs, **kwargs):
            assert TodoStore(inputs["workspace"]).read()[0]["id"] == inputs["session_id"]
            yield "custom", {"type": "tool_call", "stage": "code_agent", "message": "work"}
            yield (
                "updates",
                {"final": {"final_answer": inputs["session_id"], "graph_state": "passed"}},
            )

    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: Workflow())
    runtime = Runtime(object(), data_dir=tmp_path)
    first = runtime.resume(workspaces[0])
    second = runtime.resume(workspaces[1])
    next(first)
    next(second)
    first.close()
    remaining = list(second)
    assert remaining[0] == FinalEvent(text=SECOND_SESSION_ID, session_id=SECOND_SESSION_ID)
    assert isinstance(remaining[1], ProgressEvent)
    assert remaining[1].event_type == "checkpoint_saved"
    assert remaining[1].details["latest_node"] == "final"
    traces = []
    for workspace, status, latest in zip(
        workspaces, ["interrupted", "completed"], [None, "final"], strict=True
    ):
        trace_path = next((workspace / ".xiliumini" / "traces").glob("*/trace.json"))
        trace = json.loads(trace_path.read_text())
        traces.append(trace)
        assert trace["status"] == status
        assert trace["tool_calls"] == 1
        assert trace["node_visits"] == ({} if latest is None else {"final": 1})
        checkpoint = json.loads((workspace / ".xiliumini/checkpoints/checkpoint.json").read_text())
        assert checkpoint["status"] == status
        assert checkpoint["latest_node"] == latest
        assert TodoStore(workspace).read()[0]["id"] == workspace.name
    assert traces[0]["trace_id"] != traces[1]["trace_id"]
