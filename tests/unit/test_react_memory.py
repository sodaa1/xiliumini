from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from tests.agent_fakes import ScriptedModel, call
from xiliumini.agents.react import run_react


def test_before_model_runs_before_every_model_invocation() -> None:
    model = ScriptedModel(
        [
            call("echo", {"value": "one"}, "call-1"),
            call("echo", {"value": "two"}, "call-2"),
            "finished",
        ]
    )
    tool = StructuredTool.from_function(
        func=lambda value: value,
        name="echo",
        description="Echo a value.",
    )
    messages = [SystemMessage(content="rules"), HumanMessage(content="task")]
    seen: list[int] = []

    def before_model(current):
        seen.append(len(current))
        return list(current)

    result = run_react(
        model,
        [tool],
        messages,
        agent="planner",
        attempt=1,
        before_model=before_model,
    )

    assert result["ok"] is True
    assert seen == [2, 4, 6]
    assert [len(call_messages) for call_messages in model.calls] == [2, 4, 6]


def test_omitting_before_model_keeps_existing_react_behavior() -> None:
    model = ScriptedModel(["finished"])
    messages = [SystemMessage(content="rules"), HumanMessage(content="task")]

    result = run_react(model, [], messages, agent="planner", attempt=1)

    assert result["ok"] is True
    assert result["summary"] == "finished"
