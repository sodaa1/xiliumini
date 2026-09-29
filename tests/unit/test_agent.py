from __future__ import annotations

from pathlib import Path

import pytest

import xiliumini.core.agent as agent_module
from xiliumini.events import (
    FinalEvent,
    PlannerEvent,
    ProgressEvent,
    VerifierEvent,
)
from xiliumini.graph.state import GraphState

SESSION_ID = "11111111-1111-4111-8111-111111111111"


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
    monkeypatch.setattr(agent_module, "build_workflow", lambda *args, **kwargs: workflow)
    graph_inputs = inputs(tmp_path)

    assert list(agent_module.stream_agent(object(), graph_inputs)) == []
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

    events = list(agent_module.stream_agent(object(), inputs(tmp_path)))

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

    assert list(agent_module.stream_agent(object(), inputs(tmp_path))) == []


def test_stream_agent_rejects_zero_attempts_before_building(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        agent_module,
        "build_workflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        list(agent_module.stream_agent(object(), inputs(tmp_path, max_attempts=0)))
