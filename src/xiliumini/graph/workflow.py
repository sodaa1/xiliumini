from __future__ import annotations

from functools import partial
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from xiliumini.graph.nodes import (
    chat_responder_node,
    final_node,
    intent_route_fn,
    intent_router_node,
    planner_node,
    verifier_node,
)
from xiliumini.graph.state import GraphState


def route_entry(state: GraphState) -> Literal["planner", "verifier", "final"]:
    node = state.get("resume_node", "planner")
    if node not in {"planner", "verifier", "final"}:
        raise ValueError("invalid resume node")
    return node


def route_after_verifier(state: GraphState) -> Literal["planner", "final"]:
    if state["max_attempts"] < 1:
        raise ValueError("max_attempts must be at least 1")
    if state["graph_state"] == "passed" or state["attempt"] >= state["max_attempts"]:
        return "final"
    return "planner"


def build_entry_workflow(model: Any):
    """Build the intent-routing graph that precedes the main workflow."""
    graph = StateGraph(GraphState)
    graph.add_node("intent_router", partial(intent_router_node, model=model))
    graph.add_node("chat_responder", partial(chat_responder_node, model=model))
    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        intent_route_fn,
        {"chat_responder": "chat_responder", "planner": END},
    )
    graph.add_edge("chat_responder", END)
    return graph.compile()


def build_workflow(model: Any, memory_manager: Any, checkpointer: Any | None = None):
    graph = StateGraph(GraphState)
    graph.add_node("planner", partial(planner_node, model=model, memory_manager=memory_manager))
    graph.add_node("verifier", partial(verifier_node, model=model, memory_manager=memory_manager))
    graph.add_node("final", partial(final_node, memory_manager=memory_manager))
    graph.add_conditional_edges(
        START, route_entry, {"planner": "planner", "verifier": "verifier", "final": "final"}
    )
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges(
        "verifier", route_after_verifier, {"planner": "planner", "final": "final"}
    )
    graph.add_edge("final", END)
    return graph.compile(checkpointer=checkpointer)


build_complex_workflow = build_workflow
