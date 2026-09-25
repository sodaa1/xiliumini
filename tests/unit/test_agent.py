from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from xiliumini.core.agent import MAX_STEPS_MESSAGE, build_actor
from xiliumini.core.state import RuntimeState
from xiliumini.prompts import ACTOR_PROMPT
from xiliumini.tools.calculator import calculator

SESSION_ID = "11111111-1111-4111-8111-111111111111"


class ScriptedModel:
    def __init__(self, responses: Sequence[AIMessage], *, repeat: bool = False) -> None:
        self.responses = list(responses)
        self.repeat = repeat
        self.calls: list[list] = []
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if self.repeat:
            return self.responses[0]
        return self.responses.pop(0)


def tool_call(expression: str = "2 + 2") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": expression},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )


def initial_state(tmp_path: Path, task: str) -> RuntimeState:
    return {
        "messages": [HumanMessage(content=task)],
        "session_id": SESSION_ID,
        "workspace": tmp_path,
        "step_count": 0,
    }


@pytest.mark.asyncio
async def test_actor_builds_actor_prompt_then_user_task(tmp_path: Path) -> None:
    model = ScriptedModel([AIMessage(content="answer")])
    graph = build_actor(model, [], max_steps=2)

    await graph.ainvoke(initial_state(tmp_path, "inspect project"))

    first_call = model.calls[0]
    assert isinstance(first_call[0], SystemMessage)
    assert first_call[0].content == ACTOR_PROMPT
    assert isinstance(first_call[1], HumanMessage)
    assert first_call[1].content == "inspect project"


@pytest.mark.asyncio
async def test_actor_returns_a_direct_model_answer(tmp_path: Path) -> None:
    model = ScriptedModel([AIMessage(content="direct answer")])
    graph = build_actor(model, [calculator], max_steps=4)

    result = await graph.ainvoke(initial_state(tmp_path, "hello"))

    assert result["messages"][-1].content == "direct answer"
    assert result["step_count"] == 1
    assert result["workspace"] == tmp_path
    assert model.bound_tool_names == ["calculator"]


@pytest.mark.asyncio
async def test_actor_executes_a_tool_then_returns_to_the_model(tmp_path: Path) -> None:
    model = ScriptedModel([tool_call(), AIMessage(content="The result is 4")])
    graph = build_actor(model, [calculator], max_steps=4)

    result = await graph.ainvoke(initial_state(tmp_path, "2 + 2?"))

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.content for message in tool_messages] == ["4"]
    assert result["messages"][-1].content == "The result is 4"
    assert result["step_count"] == 2


@pytest.mark.asyncio
async def test_actor_returns_tool_errors_to_the_model(tmp_path: Path) -> None:
    model = ScriptedModel(
        [tool_call("open('secret')"), AIMessage(content="The expression was rejected")]
    )
    graph = build_actor(model, [calculator], max_steps=4)

    result = await graph.ainvoke(initial_state(tmp_path, "unsafe"))

    tool_message = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert str(tool_message.content).startswith("Error:")
    assert result["messages"][-1].content == "The expression was rejected"


@pytest.mark.asyncio
async def test_actor_stops_repeated_tool_calls_at_max_steps(tmp_path: Path) -> None:
    model = ScriptedModel([tool_call()], repeat=True)
    graph = build_actor(model, [calculator], max_steps=2)

    result = await graph.ainvoke(initial_state(tmp_path, "loop"))

    assert result["step_count"] == 2
    assert result["messages"][-1].content == MAX_STEPS_MESSAGE
    assert len(model.calls) == 2
