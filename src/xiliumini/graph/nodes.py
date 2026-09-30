from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field, StrictBool

from xiliumini.agents.react import content_text, run_react
from xiliumini.graph.memory import MemoryManager
from xiliumini.graph.verification import requirements_failure
from xiliumini.prompts import FINAL_PROMPT, PLANNER_NODE_PROMPT, VERIFIER_NODE_PROMPT
from xiliumini.tools.preferences import PreferenceWriteTool
from xiliumini.tools.subagent_tools import CallCodeAgentTool, CallSearchAgentTool, SupervisorContext
from xiliumini.tools.todo import TodoStore, TodoWriteTool


class PlannerOutput(BaseModel):
    summary: str = Field(min_length=1)
    plan_summary: str
    acceptance_criteria: list[str]
    ready_for_verification: StrictBool


class VerifierOutput(BaseModel):
    status: str = Field(pattern="^(passed|failed)$")
    reason: str = Field(min_length=1)


def _writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return None


def planner_node(
    state, *, model: Any, memory_manager: MemoryManager, max_loops: int = 8
) -> dict[str, Any]:
    current = {**state, "attempt": state["attempt"] + 1}
    for key in (
        "agent_results",
        "research_notes",
        "tool_events",
        "agent_handoffs",
        "compression_events",
    ):
        current[key] = list(state.get(key, []))
    current["todos"] = TodoStore(state["workspace"]).read()
    current["current_node"] = "planner"
    current["memory"] = memory_manager.assemble(current, current_node="planner")
    context = SupervisorContext(current, memory_manager)
    writer = _writer()
    messages = [
        SystemMessage(content=PLANNER_NODE_PROMPT),
        HumanMessage(
            content=json.dumps(
                {
                    "memory": current["memory"],
                },
                ensure_ascii=False,
            )
        ),
    ]

    def prepare_messages(pending):
        current["todos"] = TodoStore(state["workspace"]).read()
        current["memory"] = memory_manager.assemble(current, current_node="planner")
        refreshed = list(pending)
        if len(refreshed) >= 2 and isinstance(refreshed[1], HumanMessage):
            refreshed[1] = HumanMessage(
                content=json.dumps({"memory": current["memory"]}, ensure_ascii=False)
            )
        return memory_manager.prepare_planner_messages(refreshed, current, model=model)

    result = run_react(
        model,
        [
            TodoWriteTool(workspace=state["workspace"]),
            CallSearchAgentTool(context=context, writer=writer),
            CallCodeAgentTool(context=context, writer=writer),
            PreferenceWriteTool(store=memory_manager.preference_store),
        ],
        messages,
        agent="planner",
        attempt=current["attempt"],
        writer=writer,
        max_loops=max_loops,
        before_model=prepare_messages,
    )
    messages = result["messages"]
    summary, plan_summary, criteria, ready = result["summary"], "", [], False
    if result["ok"]:
        for repair in range(2):
            try:
                parsed = PlannerOutput.model_validate_json(summary)
                summary = parsed.summary
                plan_summary = parsed.plan_summary
                criteria = parsed.acceptance_criteria
                ready = parsed.ready_for_verification
                break
            except ValueError:
                if repair:
                    summary = "Planner returned invalid output"
                    break
                messages.append(
                    HumanMessage(
                        content=(
                            'Return exactly {"summary":"result","plan_summary":"plan",'
                            '"acceptance_criteria":["criterion"],'
                            '"ready_for_verification":true}.'
                        )
                    )
                )
                try:
                    prepared = prepare_messages(messages)
                    summary = content_text(model.invoke(prepared).content)
                except Exception:
                    summary = "Planner model request failed"
                    break
    current["tool_events"].extend(result["tool_events"])
    current["plan_summary"] = plan_summary
    current["acceptance_criteria"] = criteria
    current["result"] = summary
    current["supervisor_ok"] = ready
    current["graph_state"] = "verifying"
    current["memory"] = memory_manager.assemble(current, current_node="planner")
    return {
        "todos": TodoStore(state["workspace"]).read(),
        "research_notes": current["research_notes"],
        "agent_results": current["agent_results"],
        "tool_events": current["tool_events"],
        "result": summary,
        "supervisor_ok": ready,
        "attempt": current["attempt"],
        "graph_state": "verifying",
        "current_node": "planner",
        "plan_summary": plan_summary,
        "acceptance_criteria": criteria,
        "agent_handoffs": current["agent_handoffs"],
        "code_agent_summary": current.get("code_agent_summary", ""),
        "last_error": current.get("last_error", ""),
        "context_summary": current.get("context_summary", ""),
        "compression_events": current.get("compression_events", []),
        "memory": current["memory"],
    }


def verifier_node(state, *, model: Any, memory_manager: MemoryManager) -> dict[str, Any]:
    current = {**state, "current_node": "verifier"}
    reason = requirements_failure(state)
    status = "failed"
    if reason is None:
        current["memory"] = memory_manager.assemble(current, current_node="verifier")
        evidence = {"memory": current["memory"]}
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
    current["verifier_summary"] = reason
    if status == "failed":
        current["last_error"] = f"verifier: {reason}"
    elif str(current.get("last_error", "")).startswith("verifier:"):
        current["last_error"] = ""
    current["graph_state"] = status
    current["verification"] = reason
    current["memory"] = memory_manager.assemble(current, current_node="verifier")
    return {
        "graph_state": status,
        "verification": reason,
        "attempt": state["attempt"],
        "current_node": "verifier",
        "verifier_summary": reason,
        "last_error": current.get("last_error", ""),
        "memory": current["memory"],
    }


def final_node(state, *, memory_manager: MemoryManager | None = None) -> dict[str, Any]:
    remaining = "; ".join(
        f"{t['id']}: {t['status']} {t['note']}"
        for t in state["todos"]
        if t["status"] != "completed"
    )
    result: dict[str, Any] = {
        "final_answer": FINAL_PROMPT.format(
            status=state["graph_state"],
            attempt=state["attempt"],
            max_attempts=state["max_attempts"],
            result=state["result"],
            verification=state["verification"],
        )
        + ("\n未完成：" + remaining if remaining else "")
    }
    if memory_manager is not None:
        current = {**state, "current_node": "final"}
        result["current_node"] = "final"
        result["memory"] = memory_manager.assemble(current, current_node="final")
    return result
