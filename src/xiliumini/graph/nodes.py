from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field, StrictBool

from xiliumini.agents.react import content_text, run_react
from xiliumini.capabilities.langchain_tools import build_meta_tools
from xiliumini.execution.gateway import active_run, wrap_tools
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


class IntentRouterOutput(BaseModel):
    route: Literal["chat", "workflow"]
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


INTENT_ROUTER_PROMPT = """You are the intent router for MokioClaw.

Classify the user's latest input into exactly one route:
- chat: greetings, thanks, identity/help questions, ordinary conceptual Q&A,
  or conversational messages that do not need workspace access.
- workflow: any request that needs creating/editing/reading files, running commands,
  installing packages, searching the web, checking the current project, verifying a
  result, or producing a concrete deliverable.

When session context is provided, use it only to understand whether the latest
input is a continuation of prior coding work. A short follow-up like "继续",
"修一下", or "运行测试" should be workflow if it refers to prior workspace work.

Return only JSON with this shape:
{"route":"chat"|"workflow","reason":"brief reason","confidence":0.0}

If uncertain, choose workflow.
"""


CHAT_RESPONDER_PROMPT = """You are MokioClaw's lightweight chat node.

Answer the user directly and concisely. Do not claim that you read files,
searched the web, ran commands, edited files, or inspected the workspace.
If the user asks for work requiring tools or project context, say that it
should be handled by the workflow route.

If session context is provided, you may use the recent conversation summary to
answer conversational follow-ups, but do not invent workspace facts.
"""


def _writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return None


def _intent_payload(state) -> str:
    return json.dumps(
        {
            "user_input": state["task"],
            "session_context": state.get("context_summary", ""),
        },
        ensure_ascii=False,
    )


def intent_router_node(state, *, model: Any) -> dict[str, Any]:
    try:
        response = model.invoke(
            [
                SystemMessage(content=INTENT_ROUTER_PROMPT),
                HumanMessage(content=_intent_payload(state)),
            ]
        )
        parsed = IntentRouterOutput.model_validate_json(content_text(response.content))
    except Exception:
        return {
            "intent_route": "workflow",
            "intent_reason": "Intent router returned invalid output or model request failed",
            "intent_confidence": 0.0,
        }

    route = parsed.route if parsed.confidence >= 0.55 else "workflow"
    return {
        "intent_route": route,
        "intent_reason": parsed.reason,
        "intent_confidence": parsed.confidence,
    }


def chat_responder_node(state, *, model: Any) -> dict[str, str]:
    response = model.invoke(
        [
            SystemMessage(content=CHAT_RESPONDER_PROMPT),
            HumanMessage(content=_intent_payload(state)),
        ]
    )
    answer = content_text(response.content)
    return {"chat_response": answer, "final_answer": answer}


def intent_route_fn(state) -> str:
    return "chat_responder" if state.get("intent_route") == "chat" else "planner"


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
        wrap_tools(
            [
                TodoWriteTool(workspace=state["workspace"]),
                CallSearchAgentTool(context=context, writer=writer),
                CallCodeAgentTool(context=context, writer=writer),
                PreferenceWriteTool(store=memory_manager.preference_store),
                *build_meta_tools(role="planner"),
            ]
        ),
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
    run_context = active_run.get()
    if reason is None and run_context is not None and run_context.hook_failures:
        reason = "Blocking hook failed: " + ", ".join(run_context.hook_failures)
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
