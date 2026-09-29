import xiliumini.graph.workflow as workflow
from tests.agent_fakes import state


def install(monkeypatch, pass_on):
    calls = []

    def planner(s, **kwargs):
        calls.append("planner")
        return {"attempt": s["attempt"] + 1, "result": "done", "graph_state": "verifying"}

    def verifier(s, **kwargs):
        calls.append("verifier")
        return {
            "graph_state": "passed" if s["attempt"] >= pass_on else "failed",
            "verification": "checked",
        }

    def final(s):
        calls.append("final")
        return {"final_answer": s["graph_state"]}

    monkeypatch.setattr(workflow, "planner_node", planner)
    monkeypatch.setattr(workflow, "verifier_node", verifier)
    monkeypatch.setattr(workflow, "final_node", final)
    return calls


def test_workflow_finishes_after_one_passing_attempt(monkeypatch, tmp_path):
    calls = install(monkeypatch, 1)
    result = workflow.build_workflow(object()).invoke(state(tmp_path, attempt=0))
    assert calls == ["planner", "verifier", "final"]
    assert result["final_answer"] == "passed"


def test_workflow_retries_planner(monkeypatch, tmp_path):
    calls = install(monkeypatch, 2)
    workflow.build_workflow(object()).invoke(state(tmp_path, attempt=0))
    assert calls == ["planner", "verifier", "planner", "verifier", "final"]


def test_workflow_exhaustion_still_reaches_final(monkeypatch, tmp_path):
    calls = install(monkeypatch, 99)
    result = workflow.build_workflow(object()).invoke(state(tmp_path, attempt=0, max_attempts=2))
    assert calls == ["planner", "verifier", "planner", "verifier", "final"]
    assert result["final_answer"] == "failed"
