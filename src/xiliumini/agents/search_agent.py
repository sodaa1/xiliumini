from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from xiliumini.agents.react import create_agent_model, payload, run_react
from xiliumini.capabilities.langchain_tools import build_meta_tools
from xiliumini.execution.gateway import wrap_tools
from xiliumini.prompts import SEARCH_AGENT_PROMPT
from xiliumini.tools.web_search_tool import WebSearchTool


def mcp_source_urls(data: dict) -> list[str]:
    if data.get("ok") is not True or data.get("is_error") is True:
        return []
    content = data.get("text", "")
    structured = data.get("structured_content")
    if structured is not None:
        content += " " + json.dumps(structured, ensure_ascii=False)
    return list(
        dict.fromkeys(
            url.rstrip(".,;)]}\"'") for url in re.findall(r"https?://[^\s<>\"']+", content)
        )
    )


def run_search_agent(state, instruction, *, writer=None, max_loops=4) -> dict:
    messages = [
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(
            content=json.dumps(
                {
                    "task": state["task"],
                    "instruction": instruction,
                    "research_notes": state.get("research_notes", []),
                },
                ensure_ascii=False,
            )
        ),
    ]
    try:
        result = run_react(
            create_agent_model(),
            wrap_tools([WebSearchTool(), *build_meta_tools(role="search_agent")]),
            messages,
            agent="search_agent",
            attempt=state.get("attempt", 0),
            writer=writer,
            max_loops=max_loops,
        )
    except Exception:
        result = {
            "ok": False,
            "summary": "Search agent initialization failed",
            "messages": messages,
            "tool_events": [],
        }
    queries, sources, answers = [], [], []
    for event in result["tool_events"]:
        if event["tool"] == "mcp_call":
            data = payload(event["output"])
            if event["ok"]:
                query = event["args"].get("capability_id")
                if query and query not in queries:
                    queries.append(query)
                answers.append(data.get("text", ""))
                for url in mcp_source_urls(data):
                    if url not in sources:
                        sources.append(url)
            continue
        if event["tool"] != "web_search":
            continue
        data = payload(event["output"])
        query = event["args"].get("query")
        if query and query not in queries:
            queries.append(query)
        if data.get("ok"):
            answers.append(data.get("answer", ""))
            for source in data.get("results", []):
                url = source.get("url")
                if url and url not in sources:
                    sources.append(url)
    result.update(queries=queries, sources=sources, answers=answers)
    if result["summary"] == "Agent loop limit reached" and sources:
        fallback = "\n\n".join(answer.strip() for answer in answers if answer.strip())
        result["summary"] = fallback[:8_000] or f"Research collected {len(sources)} source(s)."
        result["ok"] = True
    else:
        result["ok"] = result["ok"] and bool(sources)
    return result
