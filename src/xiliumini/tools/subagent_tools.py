from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from xiliumini.agents.code_agent import run_code_agent
from xiliumini.agents.search_agent import run_search_agent
from xiliumini.tools.todo import TodoStore


@dataclass
class SupervisorContext:
    state: Any
    memory_manager: Any

    def delegate(self, agent: str, instruction: str, writer=None) -> str:
        if agent == "code_agent" and not TodoStore(self.state["workspace"]).read():
            return json.dumps({"ok": False, "error": "create todos before delegating code"})
        self.state["todos"] = TodoStore(self.state["workspace"]).read()
        runner = run_code_agent if agent == "code_agent" else run_search_agent
        try:
            result = runner(self.state, instruction, writer=writer)
        except Exception:
            result = {"ok": False, "summary": "subagent failed", "tool_events": []}
        summary = {
            "agent": agent,
            "instruction": instruction,
            "ok": bool(result["ok"]),
            "summary": result["summary"],
            "attempt": self.state["attempt"],
        }
        self.state["agent_results"].append(summary.copy())
        self.state["agent_handoffs"] = [
            *self.state.get("agent_handoffs", []),
            summary.copy(),
        ][-6:]
        self.state["tool_events"].extend(result.get("tool_events", []))
        if agent == "search_agent":
            if result["ok"]:
                self.state["research_notes"].append(
                    {
                        "summary": result["summary"],
                        "queries": result.get("queries", []),
                        "sources": result.get("sources", []),
                        "attempt": self.state["attempt"],
                    }
                )
            summary["sources"] = result.get("sources", [])
        else:
            self.state["todos"] = TodoStore(self.state["workspace"]).read()
            summary["todos"] = self.state["todos"]
            self.state["code_agent_summary"] = result["summary"]
        error_prefix = f"{agent}:"
        if result["ok"]:
            if str(self.state.get("last_error", "")).startswith(error_prefix):
                self.state["last_error"] = ""
        else:
            self.state["last_error"] = f"{error_prefix} {result['summary']}"
        self.state["memory"] = self.memory_manager.assemble(self.state, current_node="planner")
        return json.dumps(summary, ensure_ascii=False)


class DelegateInput(BaseModel):
    instruction: str = Field(min_length=1, max_length=16_000)


class CallSearchAgentTool(BaseTool):
    name: str = "call_search_agent"
    description: str = (
        "CallSearchAgentTool: delegate factual research; returns summary and source URLs."
    )
    args_schema: ArgsSchema | None = DelegateInput
    context: Any = Field(exclude=True)
    writer: Any = Field(default=None, exclude=True)

    def _run(self, instruction: str) -> str:
        return self.context.delegate("search_agent", instruction, self.writer)


class CallCodeAgentTool(BaseTool):
    name: str = "call_code_agent"
    description: str = (
        "CallCodeAgentTool: delegate workspace implementation and checks after creating todos."
    )
    args_schema: ArgsSchema | None = DelegateInput
    context: Any = Field(exclude=True)
    writer: Any = Field(default=None, exclude=True)

    def _run(self, instruction: str) -> str:
        return self.context.delegate("code_agent", instruction, self.writer)
