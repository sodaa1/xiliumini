from __future__ import annotations

import xiliumini.core.agent as agent_module
from tests.agent_fakes import state
from xiliumini.execution.gateway import RunContext, active_run
from xiliumini.hooks.engine import HookEngine


def test_run_lifecycle_hook_events_reach_trace(monkeypatch, tmp_path):
    class FakeWorkflow:
        def stream(self, *_args, **_kwargs):
            yield "updates", {"verifier": {"graph_state": "passed", "verification": "ok", "attempt": 1}}
            yield "updates", {"final": {"final_answer": "done"}}

    class FakeTrace:
        def __init__(self):
            self.events = []

        def start(self, *_args, **_kwargs):
            pass

        def record_custom_event(self, event):
            self.events.append(event)

        def record_graph_update(self, *_args):
            pass

        def end(self, **_kwargs):
            pass

    monkeypatch.setattr(agent_module, "build_workflow", lambda *_args, **_kwargs: FakeWorkflow())
    trace = FakeTrace()
    context = RunContext(
        run_id="run", session_id="session", workspace=tmp_path,
        event_sink=trace.record_custom_event, hook_engine=HookEngine(tmp_path / "hooks.json"),
    )
    token = active_run.set(context)
    try:
        list(agent_module.stream_agent(
            object(), state(tmp_path), memory_manager=object(), trace_recorder=trace,
        ))
    finally:
        active_run.reset(token)
    assert [event["event"] for event in trace.events if event["type"] == "hook_lifecycle"] == [
        "run.start", "run.success",
    ]
