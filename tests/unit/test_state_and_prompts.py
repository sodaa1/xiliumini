from pathlib import Path
from typing import get_args

import xiliumini.graph.state as state_module
import xiliumini.prompts as prompts


def test_graph_state_accepts_complete_initial_contract(tmp_path: Path) -> None:
    state: state_module.GraphState = {
        "task": "build life",
        "supervisor_ok": False,
        "todos": [],
        "research_notes": [],
        "agent_results": [],
        "tool_events": [],
        "result": "",
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


def test_task2_state_and_prompts_define_supervisor_contract() -> None:
    assert hasattr(state_module, "TodoStatus")
    assert set(get_args(state_module.TodoStatus)) == {
        "pending",
        "in_progress",
        "completed",
        "blocked",
    }
    assert "TodoWriteTool" in prompts.PLANNER_NODE_PROMPT
    assert "CallSearchAgentTool" in prompts.PLANNER_NODE_PROMPT
    assert "CallCodeAgentTool" in prompts.PLANNER_NODE_PROMPT
    assert "WebSearchTool" in prompts.SEARCH_AGENT_PROMPT
    assert "Do not write files" in prompts.SEARCH_AGENT_PROMPT
    assert "TodoUpdateTool" in prompts.CODE_AGENT_PROMPT
    assert "FileReadTool" in prompts.CODE_AGENT_PROMPT
    assert "BashTool" in prompts.CODE_AGENT_PROMPT
    assert '"status"' in prompts.VERIFIER_NODE_PROMPT
    assert "{status}" in prompts.FINAL_PROMPT
    assert not hasattr(prompts, "ACTOR_NODE_PROMPT")
