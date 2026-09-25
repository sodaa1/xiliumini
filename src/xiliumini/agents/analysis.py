from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from xiliumini.prompts import ANALYSIS_SYSTEM_PROMPT


class AnalysisAgent:
    """A text-only Agent with an isolated context and no tools."""

    def __init__(self, model: Any) -> None:
        self._model = model

    async def analyze(self, question: str) -> str:
        response = await self._model.ainvoke(
            [
                SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        )
        content = response.content
        if isinstance(content, str):
            return content
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
