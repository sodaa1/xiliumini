from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field, StrictBool

from xiliumini.agents.react import content_text, run_react
from xiliumini.graph.verification import requirements_failure
from xiliumini.prompts import FINAL_PROMPT, PLANNER_NODE_PROMPT, VERIFIER_NODE_PROMPT
from xiliumini.tools.subagent_tools import CallCodeAgentTool, CallSearchAgentTool, SupervisorContext
from xiliumini.tools.todo import TodoStore, TodoWriteTool


class PlannerOutput(BaseModel):
    summary: str = Field(min_length=1)
    ready_for_verification: StrictBool


class VerifierOutput(BaseModel):
    status: str = Field(pattern="^(passed|failed)$")
    reason: str = Field(min_length=1)


def _writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return None


def planner_node(state, *, model: Any, max_loops: int = 8) -> dict[str, Any]:
    current = {**state, "attempt": state["attempt"] + 1}
    for key in ("agent_results", "research_notes", "tool_events"):
        current[key] = list(state.get(key, []))
    current["todos"] = TodoStore(state["workspace"]).read()
    context = SupervisorContext(current)
    writer = _writer()
    messages = [
        SystemMessage(content=PLANNER_NODE_PROMPT),
        HumanMessage(
            content=json.dumps(
                {
                    "task": state["task"],
                    "todos": current["todos"],
                    "research_notes": current["research_notes"],
                    "prior_results": current["agent_results"],
                    "verification": state["verification"],
                    "attempt": current["attempt"],
                    "max_attempts": state["max_attempts"],
                },
                ensure_ascii=False,
            )
        ),
    ]
    result = run_react(
        model,
        [
            TodoWriteTool(workspace=state["workspace"]),
            CallSearchAgentTool(context=context, writer=writer),
            CallCodeAgentTool(context=context, writer=writer),
        ],
        messages,
        agent="planner",
        attempt=current["attempt"],
        writer=writer,
        max_loops=max_loops,
    )
    summary, ready = result["summary"], False
    if result["ok"]:
        for repair in range(2):
            try:
                parsed = PlannerOutput.model_validate_json(summary)
                summary, ready = parsed.summary, parsed.ready_for_verification
                break
            except ValueError:
                if repair:
                    summary = "Planner returned invalid output"
                    break
                messages.append(
                    HumanMessage(
                        content='Return exactly {"summary":"result","ready_for_verification":true}.'
                    )
                )
                try:
                    summary = content_text(model.invoke(messages).content)
                except Exception:
                    summary = "Planner model request failed"
                    break
    current["tool_events"].extend(result["tool_events"])
    return {
        "todos": TodoStore(state["workspace"]).read(),
        "research_notes": current["research_notes"],
        "agent_results": current["agent_results"],
        "tool_events": current["tool_events"],
        "result": summary,
        "supervisor_ok": ready,
        "attempt": current["attempt"],
        "graph_state": "verifying",
    }


def verifier_node(state, *, model: Any) -> dict[str, Any]:
    reason = requirements_failure(state)
    status = "failed"
    if reason is None:
        evidence = {
            key: state.get(key)
            for key in ("task", "todos", "research_notes", "agent_results", "tool_events", "result")
        }
        try:
            response = model.invoke(
                [
                    SystemMessage(content=VERIFIER_NODE_PROMPT),
                    HumanMessage(content=json.dumps(evidence, ensure_ascii=False)),
                ]
            )
            parsed = VerifierOutput.model_validate_json(content_text(response.content))
            status, reason = parsed.status, parsed.reason
        except Exception:
            reason = "Verifier returned invalid output or model request failed"
    return {"graph_state": status, "verification": reason, "attempt": state["attempt"]}


def final_node(state) -> dict[str, str]:
    remaining = "; ".join(
        f"{t['id']}: {t['status']} {t['note']}"
        for t in state["todos"]
        if t["status"] != "completed"
    )
    return {
        "final_answer": FINAL_PROMPT.format(
            status=state["graph_state"],
            attempt=state["attempt"],
            max_attempts=state["max_attempts"],
            result=state["result"],
            verification=state["verification"],
        )
        + ("\n未完成：" + remaining if remaining else "")
    }
