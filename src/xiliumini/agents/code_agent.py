from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from xiliumini.agents.react import create_agent_model, run_react, unresolved_failures
from xiliumini.memory import build_memory_snapshot
from xiliumini.prompts import CODE_AGENT_PROMPT
from xiliumini.tools import build_tools
from xiliumini.tools.todo import TodoStore, TodoUpdateTool


def run_code_agent(state, instruction, *, writer=None, max_loops=10) -> dict:
    messages = []
    result = {
        "ok": False,
        "summary": "Code agent initialization failed",
        "messages": messages,
        "tool_events": [],
        "todos": state.get("todos", []),
    }
    try:
        store = TodoStore(state["workspace"])
        snapshot = build_memory_snapshot({**state, "todos": store.read()})
        messages.extend(
            [
                SystemMessage(content=CODE_AGENT_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "task": state["task"],
                            "instruction": instruction,
                            "session_id": state["session_id"],
                            "memory": snapshot,
                            "research_notes": state.get("research_notes", []),
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        result = run_react(
            create_agent_model(),
            [*build_tools(state), TodoUpdateTool(workspace=state["workspace"])],
            messages,
            agent="code_agent",
            attempt=state.get("attempt", 0),
            writer=writer,
            max_loops=max_loops,
        )
        todos = store.read()
        result["todos"] = todos
        result["ok"] = (
            result["ok"]
            and bool(todos)
            and not any(t["status"] in {"blocked", "in_progress"} for t in todos)
            and not unresolved_failures(result["tool_events"])
        )
        if not result["ok"] and unresolved_failures(result["tool_events"]):
            result["summary"] += "\nUnresolved tool failures; review tool evidence before retrying."
    except Exception:
        result["ok"] = False
        result["summary"] = "Code agent initialization or persistence failed"
        result.setdefault("todos", state.get("todos", []))
    return result
