from typing import cast

from langchain_core.messages import AIMessage

from xiliumini.graph.state import GraphState


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.tools = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, AIMessage) else AIMessage(content=result)


def call(name, args, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def state(workspace, **changes) -> GraphState:
    initial: GraphState = {
        "supervisor_ok": False,
        "task": "build a Python function and run tests",
        "todos": [],
        "research_notes": [],
        "agent_results": [],
        "tool_events": [],
        "result": "",
        "graph_state": "planning",
        "verification": "",
        "attempt": 1,
        "max_attempts": 3,
        "final_answer": "",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "workspace": workspace,
        "memory": {
            "rules": {"fixed_rules": [], "user_preferences": []},
            "working": {
                "current_node": "planner",
                "task": "build a Python function and run tests",
                "session_id": "11111111-1111-4111-8111-111111111111",
                "plan_summary": "",
                "todos": [],
                "acceptance_criteria": [],
                "research_notes": [],
                "sources": [],
                "agent_handoffs": [],
                "code_agent_summary": "",
                "verifier_summary": "",
                "last_error": "",
                "attempts": {"current": 1, "max": 3},
            },
            "history": {
                "history_summary": "",
                "notepad_summary": "",
                "context_summary": "",
                "compression_events": [],
            },
        },
        "current_node": "planner",
        "plan_summary": "",
        "acceptance_criteria": [],
        "agent_handoffs": [],
        "code_agent_summary": "",
        "verifier_summary": "",
        "last_error": "",
        "context_summary": "",
        "compression_events": [],
    }
    return cast(GraphState, {**initial, **changes})
