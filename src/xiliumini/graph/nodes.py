from __future__ import annotations

import asyncio
import json
import re
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field, ValidationError, field_validator

from xiliumini.errors import NodeOutputError
from xiliumini.graph.state import ActionPhase, ActionResult, GraphState
from xiliumini.prompts import (
    ACTOR_NODE_PROMPT,
    FINAL_PROMPT,
    PLANNER_NODE_PROMPT,
    VERIFIER_NODE_PROMPT,
)
from xiliumini.tools.command import CommandResult, CommandTool


class PlannerOutput(BaseModel):
    todo: list[str] = Field(min_length=1)

    @field_validator("todo")
    @classmethod
    def reject_blank_items(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("todo items must not be blank")
        return cleaned


class ToolAction(BaseModel):
    kind: Literal["tool"]
    name: str = Field(min_length=1)
    args: dict[str, Any]
    phase: ActionPhase
    label: str = Field(min_length=1)


class CommandAction(BaseModel):
    kind: Literal["command"]
    argv: list[str] = Field(min_length=2)
    phase: ActionPhase
    label: str = Field(min_length=1)


class ActorOutput(BaseModel):
    actions: list[Annotated[ToolAction | CommandAction, Field(discriminator="kind")]] = Field(
        min_length=1
    )


class VerifierOutput(BaseModel):
    status: Literal["passed", "failed"]
    reason: str = Field(min_length=1)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
        )
    return str(content)


def _json_object(text: str) -> dict[str, Any]:
    if text.lstrip().startswith("```"):
        raise ValueError("Markdown fences are not valid JSON output")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


def _model_json(model: Any, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = model.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )
    return _json_object(_content_text(response.content))


def planner_node(state: GraphState, *, model: Any) -> dict[str, Any]:
    payload = {"task": state["task"]}
    try:
        parsed = PlannerOutput.model_validate(_model_json(model, PLANNER_NODE_PROMPT, payload))
    except (ValueError, ValidationError, json.JSONDecodeError):
        repair = {
            "task": state["task"],
            "instruction": 'Return exactly {"todo":["non-empty step"]}.',
        }
        try:
            parsed = PlannerOutput.model_validate(_model_json(model, PLANNER_NODE_PROMPT, repair))
        except (ValueError, ValidationError, json.JSONDecodeError):
            raise NodeOutputError("planner returned invalid output") from None
    return {"todo": parsed.todo, "graph_state": "planning"}


def _failed_evidence(label: str, output: str, attempt: int) -> ActionResult:
    return {
        "attempt": attempt,
        "kind": "tool",
        "phase": "other",
        "label": label,
        "ok": False,
        "output": output,
        "exit_code": None,
        "timed_out": False,
        "truncated": False,
    }


def _progress_writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


def _command_evidence(action: CommandAction, command: CommandTool, attempt: int) -> ActionResult:
    result: CommandResult = command.execute(action.argv)
    streams = []
    if result.stdout:
        streams.append(f"stdout:\n{result.stdout}")
    if result.stderr:
        streams.append(f"stderr:\n{result.stderr}")
    output = "\n".join(streams)
    return {
        "attempt": attempt,
        "kind": "command",
        "phase": action.phase,
        "label": action.label,
        "ok": result.ok,
        "output": output,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
    }


def _tool_evidence(action: ToolAction, tools: dict[str, BaseTool], attempt: int) -> ActionResult:
    selected = tools.get(action.name)
    if selected is None:
        return {
            **_failed_evidence(action.label, f"unknown tool: {action.name}", attempt),
            "phase": action.phase,
        }
    try:
        try:
            raw_output = selected.invoke(action.args)
        except NotImplementedError:
            raw_output = asyncio.run(selected.ainvoke(action.args))
        output = str(raw_output)
        ok = not output.startswith("Error:")
    except Exception:
        output = "tool execution failed"
        ok = False
    return {
        "attempt": attempt,
        "kind": "tool",
        "phase": action.phase,
        "label": action.label,
        "ok": ok,
        "output": output,
        "exit_code": None,
        "timed_out": False,
        "truncated": False,
    }


def actor_node(state: GraphState, *, model: Any, tools: list[BaseTool]) -> dict[str, Any]:
    current_attempt = state["attempt"] + 1
    tools_by_name = {tool.name: tool for tool in tools}
    payload = {
        "task": state["task"],
        "todo": state["todo"],
        "prior_result": state["result"],
        "verification": state["verification"],
        "attempt": state["attempt"],
        "max_attempts": state["max_attempts"],
        "tools": [{"name": tool.name, "description": tool.description} for tool in tools],
    }
    try:
        parsed = ActorOutput.model_validate(_model_json(model, ACTOR_NODE_PROMPT, payload))
    except (ValueError, ValidationError, json.JSONDecodeError):
        new_evidence = [
            _failed_evidence(
                "actor output validation", "Actor returned invalid output", current_attempt
            )
        ]
    else:
        writer = _progress_writer()
        new_evidence = []
        command = next((tool for tool in tools if isinstance(tool, CommandTool)), None)
        for action in parsed.actions:
            writer(
                {
                    "stage": "actor",
                    "message": f"Starting: {action.label}",
                    "phase": action.phase,
                    "status": "started",
                }
            )
            if isinstance(action, CommandAction):
                evidence = (
                    _command_evidence(action, command, current_attempt)
                    if command is not None
                    else _failed_evidence(action.label, "unknown tool: command", current_attempt)
                )
                evidence["phase"] = action.phase
            else:
                evidence = _tool_evidence(action, tools_by_name, current_attempt)
            new_evidence.append(evidence)
            writer(
                {
                    "stage": "actor",
                    "message": f"Finished: {action.label}",
                    "phase": action.phase,
                    "status": "finished",
                    "ok": evidence["ok"],
                }
            )
    result = "\n".join(
        f"{item['label']}: {'ok' if item['ok'] else 'failed'} — {item['output']}"
        for item in new_evidence
    )
    return {
        "attempt": current_attempt,
        "result": result,
        "execution": [*state["execution"], *new_evidence],
        "graph_state": "acting",
    }


def _requirements_failure(state: GraphState) -> str | None:
    task_text = " ".join([state["task"], *state["todo"]]).lower()
    tdd_required = any(marker in task_text for marker in ("tdd", "test-driven", "测试驱动"))
    test_required = (
        tdd_required or bool(re.search(r"\btests?\b|\bpytest\b", task_text)) or "测试" in task_text
    )
    demo_required = "demo" in task_text or "演示" in task_text
    evidence = state["execution"]
    green: int | None = None
    implementation: int | None = None
    if test_required:
        green = next(
            (
                index
                for index, item in reversed(list(enumerate(evidence)))
                if item["phase"] == "test_green"
            ),
            None,
        )
        if green is None:
            return "Missing successful test evidence"
        green_result = evidence[green]
        if (
            not green_result["ok"]
            or green_result["kind"] != "command"
            or green_result["exit_code"] != 0
            or green_result["timed_out"]
            or green_result["truncated"]
        ):
            return (
                "Latest TDD green test did not complete successfully"
                if tdd_required
                else "Latest required test did not complete successfully"
            )
        implementation = next(
            (
                index
                for index, item in reversed(list(enumerate(evidence)))
                if item["phase"] == "implementation"
            ),
            None,
        )
        if implementation is not None and (
            not evidence[implementation]["ok"] or implementation >= green
        ):
            return "Latest implementation was not followed by a successful green test"
    if tdd_required:
        assert green is not None
        if implementation is None:
            return "Missing ordered TDD red, implementation, and green evidence"
        red = next(
            (
                index
                for index, item in enumerate(evidence)
                if item["phase"] == "test_red"
                and item["kind"] == "command"
                and item["exit_code"] not in (None, 0)
                and not item["timed_out"]
                and not item["truncated"]
                and index < implementation
            ),
            None,
        )
        if red is None:
            return "Missing ordered TDD red, implementation, and green evidence"
    if demo_required:
        demo = next(
            (
                index
                for index, item in reversed(list(enumerate(evidence)))
                if item["phase"] == "demo"
            ),
            None,
        )
        if demo is None:
            return "Missing successful demo evidence after tests"
        demo_result = evidence[demo]
        if (
            demo_result["kind"] != "command"
            or not demo_result["ok"]
            or demo_result["exit_code"] != 0
            or demo_result["timed_out"]
            or demo_result["truncated"]
            or (green is not None and demo <= green)
        ):
            return "Latest demo did not complete successfully after tests"
    for index, item in enumerate(evidence):
        if item["attempt"] != state["attempt"]:
            continue
        if item["timed_out"] or item["truncated"]:
            return f"Action did not complete cleanly: {item['label']}"
        if item["ok"]:
            continue
        expected_red = (
            tdd_required
            and green is not None
            and item["phase"] == "test_red"
            and item["kind"] == "command"
            and item["exit_code"] not in (None, 0)
            and any(
                later["phase"] == "implementation" and later["ok"]
                for later in evidence[index + 1 : green]
            )
        )
        if not expected_red:
            return f"Failed or rejected action: {item['label']}"
    return None


def verifier_node(state: GraphState, *, model: Any) -> dict[str, Any]:
    payload = {
        "task": state["task"],
        "todo": state["todo"],
        "result": state["result"],
        "execution": state["execution"],
    }
    try:
        parsed = VerifierOutput.model_validate(_model_json(model, VERIFIER_NODE_PROMPT, payload))
    except (ValueError, ValidationError, json.JSONDecodeError):
        return {
            "graph_state": "failed",
            "verification": "Verifier returned invalid output",
            "attempt": state["attempt"],
        }
    guard = _requirements_failure(state)
    if guard is not None:
        return {
            "graph_state": "failed",
            "verification": guard,
            "attempt": state["attempt"],
        }
    return {
        "graph_state": parsed.status,
        "verification": parsed.reason,
        "attempt": state["attempt"],
    }


def final_node(state: GraphState) -> dict[str, str]:
    return {
        "final_answer": FINAL_PROMPT.format(
            status=state["graph_state"],
            attempt=state["attempt"],
            max_attempts=state["max_attempts"],
            result=state["result"],
            verification=state["verification"],
        )
    }
