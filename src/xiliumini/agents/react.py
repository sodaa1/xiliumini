from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from xiliumini.config import load_settings
from xiliumini.errors import MemoryBudgetError
from xiliumini.providers.openai_compatible import create_chat_model

model_factory: ContextVar[Callable[[], Any] | None] = ContextVar(
    "agent_model_factory", default=None
)

_SAFE_TOOL_ERRORS = {
    "missing TAVILY_API_KEY",
    "invalid TAVILY_API_KEY",
    "Tavily usage limit exceeded",
    "Tavily request forbidden",
    "Tavily rejected search request",
}


def create_agent_model() -> Any:
    factory = model_factory.get()
    return factory() if factory else create_chat_model(load_settings())


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def payload(text: str) -> dict[str, Any]:
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except (TypeError, ValueError):
        return {}


def emit(writer, event: dict[str, Any]) -> None:
    if writer is not None:
        writer(event)


def _tool_result_message(agent: str, name: str, ok: bool, data: dict[str, Any]) -> str:
    if ok:
        return f"{agent}: {name} ok"
    error = data.get("error")
    if name == "web_search" and error in _SAFE_TOOL_ERRORS:
        return f"{agent}: {error}"
    if name != "bash":
        return f"{agent}: {name} failed"
    if data.get("timed_out"):
        return f"{agent}: bash timed out"
    if data.get("truncated"):
        return f"{agent}: bash failed: output truncated"
    exit_code = data.get("exit_code")
    if isinstance(exit_code, int):
        return f"{agent}: bash failed: exit {exit_code}"
    stderr = str(data.get("stderr", ""))
    prefix = "command rejected:"
    if stderr.startswith(prefix):
        reason = stderr.removeprefix(prefix).strip()
        return f"{agent}: bash rejected: {reason}"
    return f"{agent}: bash failed"


def run_react(
    model,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    *,
    agent: str,
    attempt: int,
    writer=None,
    max_loops: int = 4,
    before_tool: Callable[[str, dict], str | None] | None = None,
    before_model: Callable[[list[BaseMessage]], list[BaseMessage]] | None = None,
) -> dict[str, Any]:
    if max_loops < 1:
        raise ValueError("max_loops must be at least 1")
    events: list[dict[str, Any]] = []
    by_name = {tool.name: tool for tool in tools}
    summary = "Agent loop limit reached"
    completed = False
    try:
        bound = model.bind_tools(tools)
        stop_reason: str | None = None
        stop_requested = False
        for _ in range(max_loops):
            if before_model is not None:
                messages = before_model(messages)
            response = bound.invoke(messages)
            if not isinstance(response, AIMessage):
                raise ValueError("expected AIMessage")
            messages.append(response)
            calls = response.tool_calls
            if response.invalid_tool_calls:
                summary = "Invalid model tool call"
                break
            if not calls:
                summary = content_text(response.content).strip()
                completed = bool(summary)
                break
            if len(calls) > 32 or any(not c.get("id") for c in calls):
                summary = "Invalid model tool call IDs or call count"
                break
            for call in calls:
                name, args = call["name"], call["args"]
                emit(
                    writer,
                    {
                        "type": "tool_call",
                        "stage": agent,
                        "message": f"{agent}: {name}",
                        "tool": name,
                        "args": args,
                    },
                )
                try:
                    selected = by_name.get(name)
                    if selected is None:
                        output = json.dumps({"ok": False, "error": "unknown tool"})
                    elif before_tool and (reason := before_tool(name, args)):
                        output = json.dumps({"ok": False, "error": reason})
                    else:
                        raw = selected.invoke(args)
                        output = (
                            raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                        )
                except Exception:
                    output = json.dumps(
                        {"ok": False, "error": "tool arguments or execution failed"}
                    )
                data = payload(output)
                ok = data.get("ok") is not False and not output.startswith("Error:")
                ok = ok and not data.get("timed_out", False) and not data.get("truncated", False)
                event = {
                    "agent": agent,
                    "tool": name,
                    "args": args,
                    "output": output,
                    "ok": bool(ok),
                    "attempt": attempt,
                    "phase": data.get("phase"),
                }
                events.append(event)
                messages.append(ToolMessage(content=output, tool_call_id=call["id"], name=name))
                event_type = (
                    "search_results"
                    if name == "web_search"
                    else ("todo_update" if name in {"todo_update", "todo_write"} else "tool_result")
                )
                emit(
                    writer,
                    {
                        **event,
                        "type": event_type,
                        "requires_approval": data.get("requires_approval") is True,
                        "stage": agent,
                        "message": _tool_result_message(agent, name, bool(ok), data),
                    },
                )
                if data.get("ok") is False and data.get("retryable") is False:
                    error = data.get("error")
                    stop_reason = (
                        error if isinstance(error, str) and error in _SAFE_TOOL_ERRORS else None
                    )
                    stop_requested = True
                    summary = stop_reason or "Non-retryable tool failure"
                    break
            if stop_requested:
                break
    except MemoryBudgetError:
        raise
    except Exception:
        summary = "Agent model request failed"
    return {"ok": completed, "summary": summary, "messages": messages, "tool_events": events}


def unresolved_failures(events: list[dict[str, Any]]) -> bool:
    latest: dict[str, dict] = {}
    rejected_exploration = False
    for index, event in enumerate(events):
        data = payload(event["output"])
        if event["tool"] == "bash":
            # No command ran: a later permitted invocation can repair an exploratory
            # request. Executed failures, timeouts and required phases remain evidence.
            if (
                data.get("exit_code") is None
                and str(data.get("stderr", "")).startswith("command rejected:")
                and event["phase"] in {"other", "check"}
            ):
                rejected_exploration = True
                continue
            if event["ok"]:
                rejected_exploration = False
        if event["tool"] == "bash" and event["phase"] == "test_red":
            expected_failure = (
                data.get("exit_code") in {1, 2}
                and not data.get("timed_out")
                and not data.get("truncated")
            )
            writes = [
                i
                for i, later in enumerate(events[index + 1 :], start=index + 1)
                if later["tool"] in {"file_write", "file_edit"} and later["ok"]
            ]
            repaired = bool(writes) and any(
                i > writes[0]
                and later["tool"] == "bash"
                and later["phase"] == "test_green"
                and later["ok"]
                and payload(later["output"]).get("exit_code") == 0
                for i, later in enumerate(events[index + 1 :], start=index + 1)
            )
            if expected_failure and repaired:
                continue
        args = event["args"]
        key = event["tool"] + str(args.get("path", args.get("todo_id", "")))
        if event["tool"] == "bash":
            argv = args.get("argv", [])
            if argv[:3] == ["python", "-m", "pytest"]:
                harmless = {
                    "-q",
                    "--quiet",
                    "-v",
                    "--verbose",
                    "-x",
                    "-rf",
                    "--exitfirst",
                    "--disable-warnings",
                }
                selection: list[str] = []
                tail = iter(argv[3:])
                for value in tail:
                    if value in harmless or value.startswith(("--maxfail=", "--tb=")):
                        continue
                    if value in {"--maxfail", "--tb"}:
                        next(tail, None)
                        continue
                    selection.append(value)
                family = "pytest" + json.dumps(selection, ensure_ascii=False)
            elif len(argv) >= 3 and argv[:2] == ["python", "-m"]:
                family = json.dumps(argv[2:], ensure_ascii=False)
            else:
                family = str(argv[1] if len(argv) > 1 else argv)
            key += family + str(event["phase"])
        latest[key] = event
    return rejected_exploration or any(not event["ok"] for event in latest.values())
