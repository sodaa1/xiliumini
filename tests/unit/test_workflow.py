import pytest

import xiliumini.graph.workflow as workflow
from tests.agent_fakes import state


def install(monkeypatch, pass_on, manager):
    calls = []

    def planner(s, **kwargs):
        assert kwargs["memory_manager"] is manager
        calls.append("planner")
        return {"attempt": s["attempt"] + 1, "result": "done", "graph_state": "verifying"}

    def verifier(s, **kwargs):
        assert kwargs["memory_manager"] is manager
        calls.append("verifier")
        return {
            "graph_state": "passed" if s["attempt"] >= pass_on else "failed",
            "verification": "checked",
        }

    def final(s, **kwargs):
        assert kwargs["memory_manager"] is manager
        calls.append("final")
        return {"final_answer": s["graph_state"]}

    monkeypatch.setattr(workflow, "planner_node", planner)
    monkeypatch.setattr(workflow, "verifier_node", verifier)
    monkeypatch.setattr(workflow, "final_node", final)
    return calls


def test_workflow_finishes_after_one_passing_attempt(monkeypatch, tmp_path):
    manager = object()
    calls = install(monkeypatch, 1, manager)
    result = workflow.build_workflow(object(), manager).invoke(state(tmp_path, attempt=0))
    assert calls == ["planner", "verifier", "final"]
    assert result["final_answer"] == "passed"


def test_workflow_retries_planner(monkeypatch, tmp_path):
    manager = object()
    calls = install(monkeypatch, 2, manager)
    workflow.build_workflow(object(), manager).invoke(state(tmp_path, attempt=0))
    assert calls == ["planner", "verifier", "planner", "verifier", "final"]


def test_workflow_exhaustion_still_reaches_final(monkeypatch, tmp_path):
    manager = object()
    calls = install(monkeypatch, 99, manager)
    result = workflow.build_workflow(object(), manager).invoke(
        state(tmp_path, attempt=0, max_attempts=2)
    )
    assert calls == ["planner", "verifier", "planner", "verifier", "final"]
    assert result["final_answer"] == "failed"


@pytest.mark.parametrize(
    "entry,attempt,status,expected",
    [
        ("verifier", 1, "verifying", ["verifier", "final"]),
        ("planner", 1, "failed", ["planner", "verifier", "final"]),
        ("final", 1, "passed", ["final"]),
        ("final", 3, "failed", ["final"]),
    ],
)
def test_resume_entry(monkeypatch, tmp_path, entry, attempt, status, expected):
    manager = object()
    calls = install(monkeypatch, 1, manager)
    workflow.build_workflow(object(), manager).invoke(
        state(tmp_path, resume_node=entry, attempt=attempt, graph_state=status)
    )
    assert calls == expected


def test_resume_entry_rejects_unknown_node(tmp_path):
    with pytest.raises(ValueError, match="resume node"):
        workflow.build_workflow(object(), object()).invoke(state(tmp_path, resume_node="actor"))
