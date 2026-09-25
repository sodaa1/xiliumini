from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from xiliumini.events import (
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
)
from xiliumini.runtime import Runtime
from xiliumini.tools.calculator import calculator

SESSION_ID = "11111111-1111-4111-8111-111111111111"
SECOND_SESSION_ID = "22222222-2222-4222-8222-222222222222"


async def collect(stream):
    return [event async for event in stream]


class DirectModel:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        return AIMessage(content="answer")


class FailingModel:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        raise RuntimeError("must-not-leak")


class ToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "2 + 2"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="4")


class StreamingModel:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        return AIMessage(content="hello")

    async def astream(self, _messages):
        yield AIMessageChunk(content="hel")
        await asyncio.sleep(0)
        yield AIMessageChunk(content="lo")


@pytest.mark.asyncio
async def test_runtime_emits_token_then_final_event(tmp_path: Path) -> None:
    runtime = Runtime(DirectModel(), tools=[], max_steps=2, data_dir=tmp_path)

    events = [event async for event in runtime.astream("hello", SESSION_ID)]

    assert events == [
        TokenEvent(text="answer"),
        FinalEvent(text="answer", session_id=SESSION_ID),
    ]


@pytest.mark.asyncio
async def test_runtime_maps_model_failure_without_leaking_details(tmp_path: Path) -> None:
    runtime = Runtime(FailingModel(), tools=[], max_steps=2, data_dir=tmp_path)

    events = [event async for event in runtime.astream("hello", SESSION_ID)]

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].code == "provider_error"
    assert "must-not-leak" not in events[0].message


@pytest.mark.asyncio
async def test_runtime_emits_tool_lifecycle_events(tmp_path: Path) -> None:
    runtime = Runtime(ToolCallingModel(), tools=[calculator], max_steps=3, data_dir=tmp_path)

    events = [event async for event in runtime.astream("calculate", SESSION_ID)]

    assert events[0] == ToolStartedEvent(name="calculator", call_id="call-1")
    assert isinstance(events[1], ToolFinishedEvent)
    assert events[1].name == "calculator"
    assert events[1].call_id == "call-1"
    assert events[1].duration_ms >= 0
    assert events[1].ok is True
    assert events[2:] == [
        TokenEvent(text="4"),
        FinalEvent(text="4", session_id=SESSION_ID),
    ]


@pytest.mark.asyncio
async def test_runtime_yields_the_first_token_before_stream_completion(tmp_path: Path) -> None:
    runtime = Runtime(StreamingModel(), tools=[], max_steps=2, data_dir=tmp_path)
    stream = runtime.astream("hello", SESSION_ID)

    first = await anext(stream)

    assert first == TokenEvent(text="hel")
    assert [event async for event in stream] == [
        TokenEvent(text="lo"),
        FinalEvent(text="hello", session_id=SESSION_ID),
    ]


@pytest.mark.asyncio
async def test_runtime_yields_tool_started_before_the_tool_finishes(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    @tool
    async def slow_tool() -> str:
        """Wait until the test releases this controlled tool."""

        entered.set()
        await release.wait()
        return "done"

    class SlowToolModel:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "slow_tool",
                            "args": {},
                            "id": "slow-1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="finished")

    runtime = Runtime(SlowToolModel(), tools=[slow_tool], max_steps=3, data_dir=tmp_path)
    stream = runtime.astream("run", SESSION_ID)
    first_event = asyncio.ensure_future(anext(stream))
    await asyncio.wait_for(entered.wait(), timeout=1)

    try:
        assert await asyncio.wait_for(first_event, timeout=0.1) == ToolStartedEvent(
            name="slow_tool", call_id="slow-1"
        )
    finally:
        release.set()
    remaining = [event async for event in stream]
    assert isinstance(remaining[0], ToolFinishedEvent)
    assert remaining[-1] == FinalEvent(text="finished", session_id=SESSION_ID)


@pytest.mark.asyncio
async def test_runtime_creates_and_reuses_session_workspace(tmp_path: Path) -> None:
    captured: list[Path] = []

    def workspace_tools(workspace: Path):
        captured.append(workspace)
        return []

    runtime = Runtime(
        DirectModel(),
        data_dir=tmp_path,
        workspace_tool_factory=workspace_tools,
        max_steps=2,
    )

    await collect(runtime.astream("first", SESSION_ID))
    await collect(runtime.astream("second", SESSION_ID))

    assert captured[0] == captured[1] == (tmp_path / "workspaces" / SESSION_ID).resolve()


@pytest.mark.asyncio
async def test_runtime_isolates_different_sessions(tmp_path: Path) -> None:
    captured: list[Path] = []
    runtime = Runtime(
        DirectModel(),
        data_dir=tmp_path,
        workspace_tool_factory=lambda workspace: captured.append(workspace) or [],
        max_steps=2,
    )

    await collect(runtime.astream("one", SESSION_ID))
    await collect(runtime.astream("two", SECOND_SESSION_ID))

    assert captured[0] != captured[1]


@pytest.mark.asyncio
async def test_runtime_reports_invalid_session_without_traceback(tmp_path: Path) -> None:
    runtime = Runtime(DirectModel(), data_dir=tmp_path, max_steps=2)

    events = await collect(runtime.astream("question", "../escape"))

    assert events == [ErrorEvent(code="workspace_error", message="session_id must be a valid UUID")]


@pytest.mark.asyncio
async def test_runtime_redacts_workspace_creation_failure(tmp_path: Path) -> None:
    data_file = tmp_path / "not-a-directory"
    data_file.write_text("occupied", encoding="utf-8")
    runtime = Runtime(DirectModel(), data_dir=data_file, max_steps=2)

    events = await collect(runtime.astream("question", SESSION_ID))

    assert events == [
        ErrorEvent(code="workspace_error", message="could not create session workspace")
    ]
    assert str(data_file) not in events[0].message
