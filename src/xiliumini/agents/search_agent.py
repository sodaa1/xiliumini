from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from xiliumini.agents.react import create_agent_model, payload, run_react
from xiliumini.prompts import SEARCH_AGENT_PROMPT
from xiliumini.tools.web_search_tool import WebSearchTool


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
            [WebSearchTool()],
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
