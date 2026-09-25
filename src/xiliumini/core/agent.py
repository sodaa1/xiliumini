from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from xiliumini.core.state import RuntimeState
from xiliumini.events import RuntimeEvent, TokenEvent, ToolFinishedEvent, ToolStartedEvent
from xiliumini.prompts import ACTOR_PROMPT

MAX_STEPS_MESSAGE = "Maximum Agent steps reached; stopped safely."
EventSink = Callable[[RuntimeEvent], Awaitable[None]]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return str(content) if content is not None else ""


async def _invoke_model(bound_model: Any, messages: list, event_sink: EventSink | None):
    if event_sink is None or not hasattr(bound_model, "astream"):
        response = await bound_model.ainvoke(messages)
        if event_sink is not None:
            text = _content_text(response.content)
            if text:
                await event_sink(TokenEvent(text=text))
        return response

    aggregate: AIMessageChunk | None = None
    async for chunk in bound_model.astream(messages):
        text = _content_text(chunk.content)
        if text:
            await event_sink(TokenEvent(text=text))
        aggregate = chunk if aggregate is None else aggregate + chunk
    if aggregate is None:
        return AIMessage(content="")
    return AIMessage(
        content=aggregate.content,
        additional_kwargs=aggregate.additional_kwargs,
        response_metadata=aggregate.response_metadata,
        tool_calls=aggregate.tool_calls,
        invalid_tool_calls=aggregate.invalid_tool_calls,
        id=aggregate.id,
    )


def build_actor(
    model: Any,
    tools: Sequence[BaseTool],
    max_steps: int,
    checkpointer: Any | None = None,
    event_sink: EventSink | None = None,
):
    """Build the main model-to-tools ReAct loop."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    tool_list = list(tools)
    tools_by_name = {tool.name: tool for tool in tool_list}
    bound_model = model.bind_tools(tool_list)

    async def call_actor(state: RuntimeState) -> dict[str, Any]:
        model_messages = [SystemMessage(content=ACTOR_PROMPT), *state["messages"]]
        response = await _invoke_model(bound_model, model_messages, event_sink)
        step_count = state["step_count"] + 1
        messages = [response]
        if response.tool_calls and step_count >= max_steps:
            messages.append(AIMessage(content=MAX_STEPS_MESSAGE))
        return {"messages": messages, "step_count": step_count}

    def route_after_actor(state: RuntimeState) -> str:
        latest = state["messages"][-1]
        if isinstance(latest, AIMessage) and latest.tool_calls:
            return "tools"
        return END

    async def call_tools(state: RuntimeState) -> dict[str, list[ToolMessage]]:
        request = state["messages"][-1]
        if not isinstance(request, AIMessage):
            raise TypeError("tools node requires an AIMessage")
        results: list[ToolMessage] = []
        for index, call in enumerate(request.tool_calls):
            raw_call_id = call.get("id")
            call_id = raw_call_id if isinstance(raw_call_id, str) else f"tool-call-{index}"
            started = perf_counter()
            if event_sink is not None:
                await event_sink(ToolStartedEvent(name=call["name"], call_id=call_id))
            selected = tools_by_name.get(call["name"])
            try:
                if selected is None:
                    raise ValueError("unknown tool")
                content = str(await selected.ainvoke(call["args"]))
                ok = not content.startswith("Error:")
            except Exception:
                content = "Error: tool execution failed"
                ok = False
            duration_ms = (perf_counter() - started) * 1000
            results.append(
                ToolMessage(
                    content=content,
                    name=call["name"],
                    tool_call_id=call_id,
                    additional_kwargs={"duration_ms": duration_ms, "ok": ok},
                )
            )
            if event_sink is not None:
                await event_sink(
                    ToolFinishedEvent(
                        name=call["name"],
                        call_id=call_id,
                        duration_ms=duration_ms,
                        ok=ok,
                    )
                )
        return {"messages": results}

    graph = StateGraph(RuntimeState)
    graph.add_node("actor", call_actor)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "actor")
    graph.add_conditional_edges(
        "actor",
        route_after_actor,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "actor")
    return graph.compile(checkpointer=checkpointer)
