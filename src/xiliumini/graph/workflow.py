from __future__ import annotations

from functools import partial
from typing import Any, Literal

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from xiliumini.graph.nodes import actor_node, final_node, planner_node, verifier_node
from xiliumini.graph.state import GraphState


def route_after_verifier(state: GraphState) -> Literal["actor", "final"]:
    """Retry Actor when verification fails and attempts remain."""

    if state["max_attempts"] < 1:
        raise ValueError("max_attempts must be at least 1")
    if state["graph_state"] == "passed" or state["attempt"] >= state["max_attempts"]:
        return "final"
    return "actor"


def build_workflow(
    model: Any,
    tools: list[BaseTool],
    checkpointer: Any | None = None,
):
    """Build the Planner → Actor → Verifier retry graph."""

    graph = StateGraph(GraphState)
    graph.add_node("planner", partial(planner_node, model=model))
    graph.add_node("actor", partial(actor_node, model=model, tools=tools))
    graph.add_node("verifier", partial(verifier_node, model=model))
    graph.add_node("final", final_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"actor": "actor", "final": "final"},
    )
    graph.add_edge("final", END)
    return graph.compile(checkpointer=checkpointer)
