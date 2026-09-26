from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from xiliumini.events import (
    ActorEvent,
    FinalEvent,
    PlannerEvent,
    ProgressEvent,
    RuntimeEvent,
    VerifierEvent,
)
from xiliumini.graph.state import GraphState
from xiliumini.graph.workflow import build_workflow


def _update_event(node: str, update: Any, session_id: str) -> RuntimeEvent | None:
    if not isinstance(update, dict):
        return None
    if node == "planner":
        todo = update.get("todo")
        if isinstance(todo, list) and all(isinstance(item, str) for item in todo):
            return PlannerEvent(todo=todo)
    elif node == "actor":
        result = update.get("result")
        attempt = update.get("attempt")
        if isinstance(result, str) and isinstance(attempt, int):
            return ActorEvent(result=result, attempt=attempt)
    elif node == "verifier":
        status = update.get("graph_state")
        reason = update.get("verification")
        attempt = update.get("attempt")
        if status in {"passed", "failed"} and isinstance(reason, str) and isinstance(attempt, int):
            return VerifierEvent(passed=status == "passed", reason=reason, attempt=attempt)
    elif node == "final":
        answer = update.get("final_answer")
        if isinstance(answer, str):
            return FinalEvent(text=answer, session_id=session_id)
    return None


def _chunk_events(chunk: Any, session_id: str) -> Iterator[RuntimeEvent]:
    if not isinstance(chunk, tuple) or len(chunk) != 2:
        return
    mode, payload = chunk
    if not isinstance(payload, dict):
        return
    if mode == "custom":
        stage = payload.get("stage")
        message = payload.get("message")
        if isinstance(stage, str) and isinstance(message, str):
            yield ProgressEvent(stage=stage, message=message)
        return
    if mode != "updates":
        return
    for node, update in payload.items():
        event = _update_event(node, update, session_id)
        if event is not None:
            yield event


def stream_agent(
    model: Any,
    tools: Sequence[BaseTool],
    inputs: GraphState,
    checkpointer: Any | None = None,
) -> Iterator[RuntimeEvent]:
    """Run the graph and translate LangGraph chunks into stable runtime events."""

    if inputs["max_attempts"] < 1:
        raise ValueError("max_attempts must be at least 1")
    workflow = build_workflow(model, list(tools), checkpointer=checkpointer)
    chunks = workflow.stream(
        inputs,
        config={"configurable": {"thread_id": inputs["session_id"]}},
        stream_mode=["updates", "custom"],
    )
    for chunk in chunks:
        yield from _chunk_events(chunk, inputs["session_id"])
