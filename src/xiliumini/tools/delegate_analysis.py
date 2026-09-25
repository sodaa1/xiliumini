from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from langchain_core.tools import BaseTool, tool

Analyze = Callable[[str], Awaitable[str]]


def make_delegate_analysis(
    analyze: Analyze,
    timeout_seconds: float,
    max_chars: int,
) -> BaseTool:
    """Create a bounded tool that delegates one question to an isolated Agent."""

    @tool("delegate_analysis")
    async def delegate_analysis(question: str) -> str:
        """Delegate a focused question to an isolated analysis Agent."""

        if not question.strip():
            return "Error: question is empty"
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await analyze(question)
        except TimeoutError:
            return "Error: analysis timed out"
        return result[:max_chars]

    return delegate_analysis
