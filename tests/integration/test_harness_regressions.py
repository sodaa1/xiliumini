"""Public-boundary regressions for the Task4 whole-change review."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from xiliumini.config import Settings
from xiliumini.core.trace import TraceRecorder
from xiliumini.errors import CheckpointError, TraceError
from xiliumini.graph.state import GraphState


@pytest.mark.parametrize("mode", ["light", "strict"])
@pytest.mark.parametrize("missing", sorted(GraphState.__required_keys__))
def test_missing_resume_field_rejected_before_workspace_mutation(tmp_path, mode, missing):
    import json

    from tests.agent_fakes import state
    from xiliumini.core.checkpoint import CheckpointManager
    from xiliumini.errors import CheckpointError
    from xiliumini.events import ErrorEvent
    from xiliumini.runtime import Runtime

    workspace = tmp_path / "workspaces" / "11111111-1111-4111-8111-111111111111"
    workspace.mkdir(parents=True)
    context = SimpleNamespace(
        workspace=workspace, data_dir=tmp_path, checkpoint_mode=mode, trace_id=None
    )
    manager = CheckpointManager(context)
    (workspace / "keep.txt").write_bytes(b"saved")
    manager.save(state(workspace), latest_node="planner")
    metadata = manager.root / "checkpoint.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    del payload["state_summary"][missing]
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    if mode == "strict":
        (manager.root / "state.json").write_text(
            json.dumps(payload["state_summary"]), encoding="utf-8"
        )
    (workspace / "keep.txt").write_bytes(b"current")
    (workspace / "new.txt").write_bytes(b"new")
    (workspace / "empty").mkdir()
    head = manager.root / "repo.git/refs/heads/checkpoint"
    before = head.read_bytes()
    with pytest.raises(CheckpointError):
        manager.load_resume_inputs(context)
    events = list(Runtime(object(), data_dir=tmp_path, checkpoint_mode=mode).resume(workspace))
    assert len(events) == 1 and isinstance(events[0], ErrorEvent)
    assert events[0].code == "checkpoint_error"
    assert (workspace / "keep.txt").read_bytes() == b"current"
    assert (workspace / "new.txt").read_bytes() == b"new"
    assert (workspace / "empty").is_dir()
    assert head.read_bytes() == before


@pytest.mark.parametrize("identifier", ["", " ", "\t\n"])
@pytest.mark.parametrize("mode", ["full", "off"])
def test_blank_optional_trace_id_settings_and_direct_recorder(tmp_path, identifier, mode):
    settings = Settings.from_env(
        {
            "XILIUMINI_API_KEY": "test",
            "XILIUMINI_MODEL": "test",
            "XILIUMINI_TRACE_ID": identifier,
            "XILIUMINI_TRACE_MODE": mode,
        }
    )
    assert settings.trace_id is None
    context = SimpleNamespace(workspace=tmp_path, trace_mode=mode, trace_id=identifier)
    recorder = TraceRecorder(context)
    assert recorder.trace_id.startswith("trace-")
    recorder.start({})
    recorder.end(status="completed", latest_node=None, final_state={})
    assert recorder.root.exists() is (mode == "full")


def test_example_env_runs_default_trace_through_runtime(tmp_path, monkeypatch):
    from dotenv import dotenv_values

    import xiliumini.core.agent as agent
    import xiliumini.runtime as runtime_module
    from xiliumini.runtime import create_runtime

    values = {
        key: value
        for key, value in dotenv_values(Path(".env.example")).items()
        if value is not None
    }
    values.update(
        XILIUMINI_API_KEY="test", XILIUMINI_MODEL="test", XILIUMINI_DATA_DIR=str(tmp_path)
    )
    settings = Settings.from_env(values)
    assert settings.trace_id is None
    monkeypatch.setattr(runtime_module, "create_chat_model", lambda *args, **kwargs: object())

    class EmptyWorkflow:
        def stream(self, *args, **kwargs):
            return iter(())

    monkeypatch.setattr(agent, "build_workflow", lambda *args, **kwargs: EmptyWorkflow())
    runtime = create_runtime(settings)
    assert list(runtime.stream("test", "11111111-1111-4111-8111-111111111111")) == []
    assert len(list(tmp_path.glob("workspaces/*/.xiliumini/traces/*/trace.json"))) == 1


@pytest.mark.parametrize("mode", ["full", "off"])
def test_invalid_nonblank_trace_identifier_is_rejected(tmp_path, mode):
    with pytest.raises(TraceError):
        TraceRecorder(SimpleNamespace(workspace=tmp_path, trace_mode=mode, trace_id="../bad"))


def test_real_graph_writer_checkpoint_manifest_matches_committed_bytes(tmp_path, monkeypatch):
    import hashlib
    import json
    from threading import Event

    from langgraph.config import get_stream_writer
    from langgraph.graph import END, START, StateGraph

    import xiliumini.core.agent as agent
    import xiliumini.core.checkpoint as checkpoint
    from tests.agent_fakes import state

    workspace = tmp_path / "workspaces" / "session"
    workspace.mkdir(parents=True)
    target = workspace / "changing.txt"
    target.write_bytes(b"before")
    mutate, changed = Event(), Event()
    captures = []

    class CapturingCheckpoint(checkpoint.CheckpointManager):
        def save(self, state, **kwargs):
            result = super().save(state, **kwargs)
            captures.append(json.loads((self.root / "checkpoint.json").read_text(encoding="utf-8")))
            return result

    context = SimpleNamespace(
        workspace=workspace, data_dir=tmp_path, checkpoint_mode="strict", trace_id=None
    )
    manager = CapturingCheckpoint(context)

    def node(state: GraphState):
        get_stream_writer()({"type": "tool_call", "stage": "planner", "message": "writing"})
        assert mutate.wait(10)
        target.write_bytes(b"after\r\n\x00\xff")
        changed.set()
        return {}

    graph = StateGraph(GraphState)
    graph.add_node("planner", node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    monkeypatch.setattr(agent, "build_workflow", lambda *args, **kwargs: graph.compile())
    real_git = checkpoint._git
    read_tree_calls = 0

    def synchronized_git(*args, **kwargs):
        nonlocal read_tree_calls
        if args[2:3] == ("read-tree",):
            read_tree_calls += 1
            if read_tree_calls == 2:
                mutate.set()
                assert changed.wait(10)
        return real_git(*args, **kwargs)

    monkeypatch.setattr(checkpoint, "_git", synchronized_git)
    try:
        list(
            agent.stream_agent(
                object(), state(workspace), memory_manager=object(), checkpoint_manager=manager
            )
        )
    finally:
        mutate.set()
    assert captures
    for payload in captures:
        tree = checkpoint._read_tree(workspace, manager.root, payload["git_commit"])
        assert payload["workspace_manifest"] == [
            {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in sorted(tree.items())
        ]
    # Recover the checkpoint published during the actual background mutation.
    first = captures[1]
    (manager.root / "checkpoint.json").write_text(json.dumps(first), encoding="utf-8")
    (manager.root / "state.json").write_text(json.dumps(first["state_summary"]), encoding="utf-8")
    target.write_bytes(b"later")
    manager.load_resume_inputs(context)
    assert target.read_bytes() == b"after\r\n\x00\xff"


def test_real_graph_close_waits_for_background_write_before_terminal_save(tmp_path, monkeypatch):
    import json
    from threading import Event, Thread

    from langgraph.config import get_stream_writer
    from langgraph.graph import END, START, StateGraph

    import xiliumini.core.agent as agent
    from tests.agent_fakes import state
    from xiliumini.core.checkpoint import CheckpointManager

    workspace = tmp_path / "workspaces" / "session"
    workspace.mkdir(parents=True)
    release, stopped, boundary, terminal = Event(), Event(), Event(), Event()
    errors = []
    order = []

    def node(state: GraphState):
        get_stream_writer()({"stage": "planner", "message": "running"})
        assert release.wait(10)
        (workspace / "late.txt").write_bytes(b"late")
        order.append("node_stopped")
        stopped.set()
        return {}

    graph = StateGraph(GraphState)
    graph.add_node("planner", node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    compiled = graph.compile()

    class Workflow:
        child = None

        def stream(self, *args, **kwargs):
            self.child = compiled.stream(*args, **kwargs)
            return self

        def __iter__(self):
            return self

        def __next__(self):
            assert self.child is not None
            return next(self.child)

        def close(self):
            boundary.set()
            assert self.child is not None
            method_name = "close"
            close = getattr(self.child, method_name)
            close()
            order.append("child_closed")

    workflow = Workflow()
    monkeypatch.setattr(agent, "build_workflow", lambda *args, **kwargs: workflow)
    context = SimpleNamespace(
        workspace=workspace,
        data_dir=tmp_path,
        checkpoint_mode="light",
        trace_mode="full",
        trace_id=None,
    )

    class TerminalCheckpoint(CheckpointManager):
        def save(self, state, **kwargs):
            if kwargs["status"] in {"completed", "failed", "interrupted"}:
                terminal.set()
                boundary.set()
                order.append("terminal_save")
            return super().save(state, **kwargs)

    class OrderedTrace(TraceRecorder):
        def end(self, **kwargs):
            order.append("trace_end")
            return super().end(**kwargs)

    manager = TerminalCheckpoint(context)
    trace = OrderedTrace(context)
    outer = agent.stream_agent(
        object(),
        state(workspace),
        memory_manager=object(),
        checkpoint_manager=manager,
        trace_recorder=trace,
    )
    next(outer)

    def close_outer():
        try:
            outer.close()
        except BaseException as error:
            errors.append(error)

    closing = Thread(target=close_outer)
    closing.start()
    try:
        assert boundary.wait(10)
        assert not terminal.is_set(), "terminal snapshot happened before child close"
    finally:
        release.set()
        closing.join(10)
        if workflow.child is not None:
            method_name = "close"
            getattr(workflow.child, method_name)()
    assert not closing.is_alive() and not errors
    assert stopped.is_set()
    assert order == ["node_stopped", "child_closed", "terminal_save", "trace_end"]
    payload = json.loads((manager.root / "checkpoint.json").read_text(encoding="utf-8"))
    assert [entry["path"] for entry in payload["workspace_manifest"]] == ["late.txt"]
    assert payload["status"] == "interrupted"
    summary = json.loads((trace.root / "trace.json").read_text(encoding="utf-8"))
    assert summary["status"] == "interrupted"


@pytest.mark.parametrize("primary", [None, RuntimeError("node"), SystemExit(3), GeneratorExit()])
@pytest.mark.parametrize(
    "close_error",
    [
        RuntimeError("close"),
        CheckpointError("close"),
        TraceError("close"),
        KeyboardInterrupt("close"),
        GeneratorExit(),
        SystemExit(4),
    ],
)
def test_child_close_failure_preserves_primary_and_finalizes_once(
    tmp_path, monkeypatch, primary, close_error
):
    import xiliumini.core.agent as agent
    from tests.agent_fakes import state
    from tests.unit.test_agent import Harness

    calls = []

    class Child:
        def __iter__(self):
            return self

        def __next__(self):
            if primary is not None:
                raise primary
            raise StopIteration

        def close(self):
            calls.append(("child_close",))
            raise close_error

    class Workflow:
        def stream(self, *args, **kwargs):
            return Child()

    monkeypatch.setattr(agent, "build_workflow", lambda *args, **kwargs: Workflow())
    harness = Harness(calls, "light")
    expected = primary if primary is not None else close_error
    with pytest.raises(type(expected)) as caught:
        list(
            agent.stream_agent(
                object(),
                state(tmp_path),
                memory_manager=object(),
                checkpoint_manager=harness,
                trace_recorder=harness,
            )
        )
    assert caught.value is expected
    assert len([call for call in calls if call[0] == "child_close"]) == 1
    assert len([call for call in calls if call[0] == "end"]) == 1
    expected_status = (
        "interrupted" if isinstance(expected, (KeyboardInterrupt, GeneratorExit)) else "failed"
    )
    assert [call[1]["status"] for call in calls if call[0] == "end"] == [expected_status]
    assert calls.index(("child_close",)) < next(
        i for i, call in enumerate(calls) if call[0] == "save" and call[2]["status"] != "started"
    )


@pytest.mark.parametrize("control", [".GIT", ".XILIUMINI", ".GiT", ".XiLiUmInI"])
def test_control_directory_case_variants_excluded_and_preserved(tmp_path, control):
    import json

    from tests.agent_fakes import state
    from xiliumini.core.checkpoint import CheckpointManager, workspace_manifest

    workspace = tmp_path / "workspaces" / "session"
    workspace.mkdir(parents=True)
    (workspace / control).mkdir()
    sentinel = workspace / control / "keep"
    sentinel.write_bytes(b"control")
    ordinary = workspace / "ordinary.txt"
    ordinary.write_bytes(b"saved")
    context = SimpleNamespace(
        workspace=workspace, data_dir=tmp_path, checkpoint_mode="light", trace_id=None
    )
    manager = CheckpointManager(context)
    assert [entry["path"] for entry in workspace_manifest(workspace)] == ["ordinary.txt"]
    manager.save(state(workspace), latest_node="planner")
    payload = json.loads((manager.root / "checkpoint.json").read_text(encoding="utf-8"))
    assert [entry["path"] for entry in payload["workspace_manifest"]] == ["ordinary.txt"]
    ordinary.write_bytes(b"modified")
    (workspace / "extra.txt").write_bytes(b"remove")
    manager.load_resume_inputs(context)
    assert sentinel.read_bytes() == b"control"
    assert ordinary.read_bytes() == b"saved"
    assert not (workspace / "extra.txt").exists()


@pytest.mark.parametrize("name", ["XILIUMINI_CHECKPOINT_MODE", "XILIUMINI_TRACE_MODE"])
@pytest.mark.parametrize("mode", ["", " ", "invalid"])
def test_settings_env_modes_fail_strictly_instead_of_fallback(name, mode):
    from xiliumini.errors import ConfigError

    with pytest.raises(ConfigError, match=name):
        Settings.from_env({"XILIUMINI_API_KEY": "test", "XILIUMINI_MODEL": "test", name: mode})
