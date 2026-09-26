from __future__ import annotations

from pathlib import Path

import xiliumini.graph.workflow as workflow_module
from xiliumini.graph.state import GraphState


def initial_state(tmp_path: Path, *, max_attempts: int = 3) -> GraphState:
    return {
        "task": "build it",
        "todo": [],
        "result": "",
        "execution": [],
        "graph_state": "planning",
        "verification": "",
        "attempt": 0,
        "max_attempts": max_attempts,
        "final_answer": "",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "workspace": tmp_path,
    }


def install_nodes(monkeypatch, *, pass_on_attempt: int) -> list[str]:
    calls: list[str] = []

    def planner(state, *, model):
        calls.append("planner")
        return {"todo": ["do work"], "graph_state": "planning"}

    def actor(state, *, model, tools):
        calls.append("actor")
        attempt = state["attempt"] + 1
        return {
            "attempt": attempt,
            "result": f"attempt {attempt}",
            "execution": state["execution"],
            "graph_state": "acting",
        }

    def verifier(state, *, model):
        calls.append("verifier")
        passed = state["attempt"] >= pass_on_attempt
        return {
            "graph_state": "passed" if passed else "failed",
            "verification": "done" if passed else "retry",
            "attempt": state["attempt"],
        }

    def final(state):
        calls.append("final")
        return {"final_answer": f"{state['graph_state']}:{state['attempt']}"}

    monkeypatch.setattr(workflow_module, "planner_node", planner)
    monkeypatch.setattr(workflow_module, "actor_node", actor)
    monkeypatch.setattr(workflow_module, "verifier_node", verifier)
    monkeypatch.setattr(workflow_module, "final_node", final)
    return calls


def test_workflow_finishes_after_one_passing_attempt(monkeypatch, tmp_path: Path) -> None:
    calls = install_nodes(monkeypatch, pass_on_attempt=1)

    result = workflow_module.build_workflow(object(), []).invoke(initial_state(tmp_path))

    assert calls == ["planner", "actor", "verifier", "final"]
    assert result["final_answer"] == "passed:1"


def test_workflow_retries_only_actor_and_verifier(monkeypatch, tmp_path: Path) -> None:
    calls = install_nodes(monkeypatch, pass_on_attempt=2)

    result = workflow_module.build_workflow(object(), []).invoke(initial_state(tmp_path))

    assert calls == ["planner", "actor", "verifier", "actor", "verifier", "final"]
    assert calls.count("planner") == 1
    assert result["final_answer"] == "passed:2"


def test_workflow_returns_failed_final_at_attempt_limit(monkeypatch, tmp_path: Path) -> None:
    calls = install_nodes(monkeypatch, pass_on_attempt=99)

    result = workflow_module.build_workflow(object(), []).invoke(
        initial_state(tmp_path, max_attempts=2)
    )

    assert calls == ["planner", "actor", "verifier", "actor", "verifier", "final"]
    assert result["final_answer"] == "failed:2"


def test_route_after_verifier_rejects_invalid_attempt_limit(tmp_path: Path) -> None:
    state = initial_state(tmp_path, max_attempts=0)

    try:
        workflow_module.route_after_verifier(state)
    except ValueError as exc:
        assert str(exc) == "max_attempts must be at least 1"
    else:
        raise AssertionError("expected invalid max_attempts to be rejected")
