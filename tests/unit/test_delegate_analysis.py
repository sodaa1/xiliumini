from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, AnyMessage

from xiliumini.agents.analysis import AnalysisAgent
from xiliumini.tools.delegate_analysis import make_delegate_analysis


class NoToolsModel:
    def __init__(self, answer: str = "analysis result") -> None:
        self.answer = answer
        self.received_messages: list[AnyMessage] | None = None

    def bind_tools(self, _tools):
        raise AssertionError("analysis Agent must not bind tools")

    async def ainvoke(self, messages):
        self.received_messages = messages
        return AIMessage(content=self.answer)


@pytest.mark.asyncio
async def test_analysis_agent_uses_an_isolated_text_only_context() -> None:
    model = NoToolsModel()
    agent = AnalysisAgent(model)

    result = await agent.analyze("compare options")

    assert result == "analysis result"
    assert model.received_messages is not None
    assert len(model.received_messages) == 2
    assert model.received_messages[-1].content == "compare options"


@pytest.mark.asyncio
async def test_delegate_analysis_truncates_oversized_output() -> None:
    async def analyze(_question: str) -> str:
        return "abcdef"

    delegate = make_delegate_analysis(analyze, timeout_seconds=1, max_chars=4)

    assert await delegate.ainvoke({"question": "question"}) == "abcd"


@pytest.mark.asyncio
async def test_delegate_analysis_returns_readable_timeout() -> None:
    async def analyze(_question: str) -> str:
        await asyncio.sleep(0.05)
        return "late"

    delegate = make_delegate_analysis(analyze, timeout_seconds=0.001, max_chars=100)

    assert await delegate.ainvoke({"question": "question"}) == "Error: analysis timed out"


@pytest.mark.asyncio
async def test_delegate_analysis_rejects_blank_question() -> None:
    async def analyze(_question: str) -> str:
        raise AssertionError("blank questions must not reach the analysis Agent")

    delegate = make_delegate_analysis(analyze, timeout_seconds=1, max_chars=100)

    assert await delegate.ainvoke({"question": "   "}) == "Error: question is empty"
