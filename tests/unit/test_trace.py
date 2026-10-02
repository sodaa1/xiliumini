import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from xiliumini.core import trace as trace_module
from xiliumini.core.trace import TraceRecorder
from xiliumini.errors import TraceError


def recorder(tmp_path: Path, mode="full", trace_id=None):
    return TraceRecorder(
        SimpleNamespace(
            workspace=tmp_path, trace_mode=mode, trace_id=trace_id, session_id="session"
        ),
        "task",
    )


def events(trace):
    return [json.loads(line) for line in (trace.root / "events.jsonl").read_text().splitlines()]


def test_off_mode_has_zero_io(tmp_path):
    trace = recorder(tmp_path, "off")
    trace.start({})
    trace.record_custom_event({"type": "tool_call"})
    trace.record_graph_update({"planner": {}})
    assert trace.end(status="completed", latest_node="final", final_state={}) is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", ["full", "summary"])
def test_mode_lifecycle_and_filter(tmp_path, mode):
    trace = recorder(tmp_path, mode)
    trace.start({}, resumed=True, resume_event={"type": "resume"})
    trace.start({})
    trace.record_custom_event({"type": "tool_call"})
    trace.record_custom_event({"type": "handoff"})
    trace.record_graph_update({"planner": {"attempt": 1}})
    summary = trace.end(status="completed", latest_node="final", final_state={})
    artifacts = {path.name: path.read_bytes() for path in trace.root.iterdir()}
    assert trace.end(status="failed", latest_node=None, final_state={"changed": True}) == summary
    assert {path.name: path.read_bytes() for path in trace.root.iterdir()} == artifacts
    assert {path.name for path in trace.root.iterdir()} == {
        "trace.json",
        "events.jsonl",
        "timeline.md",
    }
    kinds = [event["type"] for event in events(trace)]
    assert kinds == (
        ["run_start", "tool_call", "handoff", "graph_update", "run_end"]
        if mode == "full"
        else ["run_start", "handoff", "graph_update", "run_end"]
    )
    assert [event["sequence"] for event in events(trace)] == list(range(1, len(kinds) + 1))
    assert all(event["timestamp"].endswith("+00:00") for event in events(trace))


@pytest.mark.parametrize("trace_id", ["../escape", "a/b", "a\\b", "CON", "a.", "a b"])
def test_invalid_trace_id_rejected_before_io(tmp_path, trace_id):
    with pytest.raises(TraceError):
        recorder(tmp_path, trace_id=trace_id)
    assert list(tmp_path.iterdir()) == []


def test_trace_id_collision_preserves_existing_artifacts(tmp_path):
    first = recorder(tmp_path, trace_id="trace-fixed")
    first.start({})
    before = (first.root / "events.jsonl").read_bytes()
    second = recorder(tmp_path, trace_id="trace-fixed")
    with pytest.raises(TraceError):
        second.start({})
    assert (first.root / "events.jsonl").read_bytes() == before


@pytest.mark.parametrize("mode", ["full", "summary"])
def test_count_strict_bools_and_node_visits(tmp_path, mode):
    trace = recorder(tmp_path, mode)
    trace.start({})
    trace.record_custom_event({"type": "tool_call"})
    for event in [
        {"type": "tool_result", "ok": False, "requires_approval": True},
        {"type": "tool_result", "ok": "false", "requires_approval": 1},
        {"type": "tool_result", "ok": 0},
        {"type": "tool_result"},
        {"type": "handoff"},
        {"type": "checkpoint_saved"},
    ]:
        trace.record_custom_event(event)
    trace.record_graph_update({"planner": {}})
    trace.record_graph_update({"planner": {}, "verifier": {}})
    result = trace.end(status="failed", latest_node="verifier", final_state={})
    assert result is not None
    assert result["tool_calls"] == 1
    assert result["failed_tool_calls"] == 1
    assert result["approval_count"] == 1
    assert result["handoff_count"] == 1
    assert result["checkpoint_count"] == 1
    assert result["node_visits"] == {"planner": 2, "verifier": 1}


@pytest.mark.parametrize("total,tail,omitted", [(99, 79, 0), (100, 80, 0), (101, 80, 1)])
def test_timeline_boundary(tmp_path, total, tail, omitted):
    trace = recorder(tmp_path)
    trace.start({})
    for i in range(total - 2):
        trace.record_custom_event({"type": "tool_call", "index": i})
    result = trace.end(status="completed", latest_node="final", final_state={})
    assert result is not None
    assert len(result["timeline_head"]) == 20
    assert len(result["timeline_tail"]) == tail
    assert result["timeline_omitted"] == omitted
    selected = result["timeline_head"] + result["timeline_tail"]
    assert len({item["sequence"] for item in selected}) == len(selected)
    assert selected[-1]["sequence"] == total


def test_redact_all_artifacts(tmp_path):
    trace = recorder(tmp_path)
    trace.task = str(tmp_path) + " task"
    trace.start({}, resume_event={"password": "hidden", "value": SecretStr("secret-value")})
    trace.record_custom_event(
        {
            "type": "tool_result",
            "output": "Z" * 30_000,
            "api_key": "key-value",
            "cwd": str(tmp_path),
        }
    )
    result = trace.end(
        status="failed", latest_node=str(tmp_path), final_state={"password": "hidden"}
    )
    assert result == json.loads((trace.root / "trace.json").read_text())
    for path in trace.root.iterdir():
        text = path.read_text()
        assert str(tmp_path) not in text
        assert "hidden" not in text and "secret-value" not in text and "key-value" not in text
        assert "Z" * 2001 not in text
    assert "[TRUNCATED]" in (trace.root / "events.jsonl").read_text()


def test_enabled_write_failure_is_safe_trace_error(tmp_path):
    trace = recorder(tmp_path)
    trace.start({})
    (trace.root / "trace.json").mkdir()
    with pytest.raises(TraceError) as error:
        trace.end(status="completed", latest_node=None, final_state={})
    assert str(tmp_path) not in str(error.value)
    before = (trace.root / "events.jsonl").read_bytes()
    with pytest.raises(TraceError):
        trace.end(status="failed", latest_node="planner", final_state={"changed": True})
    assert (trace.root / "events.jsonl").read_bytes() == before
    assert [item["type"] for item in events(trace)].count("run_end") == 1


def test_start_append_failure_seals_recorder(tmp_path, monkeypatch):
    trace = recorder(tmp_path)
    original_append = trace_module.append_jsonl

    def fail_append(*args, **kwargs):
        raise OSError("sensitive failure")

    monkeypatch.setattr(trace_module, "append_jsonl", fail_append)
    with pytest.raises(TraceError):
        trace.start({})
    monkeypatch.setattr(trace_module, "append_jsonl", original_append)
    with pytest.raises(TraceError):
        trace.start({})
    with pytest.raises(TraceError):
        trace.record_custom_event({"type": "tool_call"})
    with pytest.raises(TraceError):
        trace.end(status="completed", latest_node=None, final_state={})
    assert not (trace.root / "events.jsonl").exists()


@pytest.mark.parametrize("method", ["custom", "graph"])
def test_record_append_failure_seals_recorder(tmp_path, monkeypatch, method):
    trace = recorder(tmp_path)
    trace.start({})
    original_append = trace_module.append_jsonl
    before = (trace.root / "events.jsonl").read_bytes()

    def fail_append(*args, **kwargs):
        raise OSError("sensitive failure")

    monkeypatch.setattr(trace_module, "append_jsonl", fail_append)
    with pytest.raises(TraceError):
        if method == "custom":
            trace.record_custom_event({"type": "tool_call"})
        else:
            trace.record_graph_update({"planner": {}})
    monkeypatch.setattr(trace_module, "append_jsonl", original_append)
    with pytest.raises(TraceError):
        trace.record_custom_event({"type": "handoff"})
    with pytest.raises(TraceError):
        trace.record_graph_update({"verifier": {}})
    with pytest.raises(TraceError):
        trace.end(status="completed", latest_node="final", final_state={})
    assert (trace.root / "events.jsonl").read_bytes() == before
    assert trace.tool_calls == 0 and trace.handoff_count == 0 and trace.node_visits == {}


def test_run_end_append_failure_seals_recorder(tmp_path, monkeypatch):
    trace = recorder(tmp_path)
    trace.start({})
    original_append = trace_module.append_jsonl
    before = (trace.root / "events.jsonl").read_bytes()

    def fail_append(*args, **kwargs):
        raise OSError("sensitive failure")

    monkeypatch.setattr(trace_module, "append_jsonl", fail_append)
    with pytest.raises(TraceError):
        trace.end(status="failed", latest_node="planner", final_state={"attempt": 1})
    monkeypatch.setattr(trace_module, "append_jsonl", original_append)
    with pytest.raises(TraceError):
        trace.end(status="completed", latest_node="final", final_state={"attempt": 2})
    with pytest.raises(TraceError):
        trace.record_custom_event({"type": "tool_call"})
    assert (trace.root / "events.jsonl").read_bytes() == before
    assert not (trace.root / "trace.json").exists()


def test_timeline_failure_seals_and_preserves_partial_artifacts(tmp_path, monkeypatch):
    trace = recorder(tmp_path)
    trace.start({})
    original_write = trace_module.atomic_write_utf8

    def fail_timeline(*args, **kwargs):
        raise OSError("sensitive failure")

    monkeypatch.setattr(trace_module, "atomic_write_utf8", fail_timeline)
    with pytest.raises(TraceError):
        trace.end(status="failed", latest_node="planner", final_state={"attempt": 1})
    monkeypatch.setattr(trace_module, "atomic_write_utf8", original_write)
    before_events = (trace.root / "events.jsonl").read_bytes()
    before_summary = (trace.root / "trace.json").read_bytes()
    with pytest.raises(TraceError):
        trace.end(status="completed", latest_node="final", final_state={"attempt": 2})
    with pytest.raises(TraceError):
        trace.record_custom_event({"type": "tool_call"})
    assert (trace.root / "events.jsonl").read_bytes() == before_events
    assert (trace.root / "trace.json").read_bytes() == before_summary
    assert not (trace.root / "timeline.md").exists()


def test_message_payload_remains_serializable_in_summary(tmp_path):
    trace = recorder(tmp_path)
    trace.start({})
    trace.record_custom_event({"type": "custom", "message": HumanMessage(content="hello")})
    result = trace.end(status="completed", latest_node=None, final_state={})
    assert result is not None
    assert result["timeline_head"][1]["payload"]["message"]["value"]["data"]["content"] == "hello"


def test_duration_uses_monotonic_clock(tmp_path, monkeypatch):
    values = iter([100.0, 102.5])
    monkeypatch.setattr("xiliumini.core.trace.time.monotonic", lambda: next(values))
    trace = recorder(tmp_path)
    trace.start({})
    result = trace.end(status="interrupted", latest_node=None, final_state={})
    assert result is not None
    assert result["duration_ms"] == 2500


def test_failed_end_prevents_more_events(tmp_path):
    trace = recorder(tmp_path)
    trace.start({})
    (trace.root / "trace.json").mkdir()
    with pytest.raises(TraceError):
        trace.end(status="completed", latest_node=None, final_state={})
    with pytest.raises(TraceError):
        trace.record_custom_event({"type": "tool_call"})
    assert [item["type"] for item in events(trace)] == ["run_start", "run_end"]


def test_summary_keeps_tool_errors_and_approvals(tmp_path):
    trace = recorder(tmp_path, "summary")
    trace.start({})
    trace.record_custom_event({"type": "tool_result", "ok": False})
    trace.record_custom_event({"type": "tool_result", "requires_approval": True})
    trace.end(status="failed", latest_node=None, final_state={})
    assert [item["type"] for item in events(trace)] == [
        "run_start",
        "tool_result",
        "tool_result",
        "run_end",
    ]


@pytest.mark.parametrize("bad", [None, [], {"type": []}, {"type": 1}])
def test_malformed_event_is_controlled(tmp_path, bad):
    trace = recorder(tmp_path)
    trace.start({})
    with pytest.raises(TraceError):
        trace.record_custom_event(bad)


def test_event_file_symlink_cannot_write_outside_workspace(tmp_path):
    trace = recorder(tmp_path)
    trace.start({})
    outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
    outside.write_text("untouched")
    path = trace.root / "events.jsonl"
    try:
        link = trace.root / "probe-link"
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    path.unlink()
    link.rename(path)
    with pytest.raises(TraceError):
        trace.record_custom_event({"type": "tool_call"})
    assert outside.read_text() == "untouched"
