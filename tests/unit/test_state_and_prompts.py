from pathlib import Path

from xiliumini.graph.state import GraphState
from xiliumini.prompts import (
    ACTOR_NODE_PROMPT,
    FINAL_PROMPT,
    PLANNER_NODE_PROMPT,
    VERIFIER_NODE_PROMPT,
)


def test_graph_state_accepts_complete_initial_contract(tmp_path: Path) -> None:
    state: GraphState = {
        "task": "build life",
        "todo": [],
        "result": "",
        "execution": [],
        "graph_state": "planning",
        "verification": "",
        "attempt": 0,
        "max_attempts": 3,
        "final_answer": "",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "workspace": tmp_path,
    }
    assert state["attempt"] == 0
    assert state["workspace"] == tmp_path


def test_task1_prompts_define_each_node_contract() -> None:
    assert '"todo"' in PLANNER_NODE_PROMPT
    assert '"actions"' in ACTOR_NODE_PROMPT
    assert '"phase"' in ACTOR_NODE_PROMPT
    assert '"status"' in VERIFIER_NODE_PROMPT
    assert "{status}" in FINAL_PROMPT
