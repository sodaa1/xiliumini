from __future__ import annotations

import json
import math
import os
from contextvars import ContextVar

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field, SecretStr
from tavily import TavilyClient

search_api_key: ContextVar[SecretStr | None] = ContextVar("search_api_key", default=None)


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    max_results: int = Field(default=5, ge=1, le=10)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "WebSearchTool: search reliable web sources using Tavily."
    args_schema: ArgsSchema | None = SearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        key = os.getenv("TAVILY_API_KEY", "").strip()
        configured = search_api_key.get()
        if "TAVILY_API_KEY" not in os.environ and configured is not None:
            key = configured.get_secret_value().strip()
        error = ""
        if not key:
            error = "missing TAVILY_API_KEY"
        elif not query.strip():
            error = "query must not be blank"
        if error:
            return json.dumps({"ok": False, "query": query, "error": error})
        try:
            raw = TavilyClient(api_key=key).search(
                query=query, max_results=max_results, include_answer=True, timeout=30
            )
            if not isinstance(raw, dict):
                raise ValueError("invalid search response")
            results = []
            for item in (raw.get("results") or []) if isinstance(raw.get("results"), list) else []:
                if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                    continue
                if not item["url"].startswith(("https://", "http://")):
                    continue
                score = item.get("score", 0.0)
                score = float(score) if isinstance(score, (float, int)) else 0.0
                results.append(
                    {
                        "title": str(item.get("title") or "")[:500],
                        "url": item["url"][:2_000],
                        "content": str(item.get("content") or "")[:6_000],
                        "score": score if math.isfinite(score) else 0.0,
                    }
                )
            return json.dumps(
                {
                    "ok": True,
                    "query": query,
                    "answer": str(raw.get("answer") or "")[:8_000],
                    "results": results[:max_results],
                },
                ensure_ascii=False,
            )
        except Exception:
            return json.dumps({"ok": False, "query": query, "error": "search failed"})
