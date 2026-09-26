import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from xiliumini.graph.nodes import actor_node, final_node, planner_node, verifier_node
from xiliumini.graph.state import ActionResult, GraphState
from xiliumini.tools.command import CommandResult, CommandTool
from xiliumini.tools.delegate_analysis import make_delegate_analysis
from xiliumini.tools.file_write import FileWriteTool


class ScriptedModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=self.responses.pop(0))


@pytest.fixture
def base_state(tmp_path: Path) -> GraphState:
    return {
        "task": "build life with TDD and run demo",
        "todo": [],
        "result": "",
        "execution": [],
        "graph_state": "planning",
        "verification": "",
        "attempt": 0,
        "max_attempts": 3,
        "final_answer": "",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "workspace": tmp_path,
    }


def evidence(
    phase: str,
    ok: bool,
    *,
    kind: str = "command",
    exit_code: int | None = None,
    timed_out: bool = False,
    truncated: bool = False,
    attempt: int = 1,
) -> ActionResult:
    return {
        "attempt": attempt,
        "kind": kind,
        "phase": phase,
        "label": phase,
        "ok": ok,
        "output": phase,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "truncated": truncated,
    }  # type: ignore[return-value]


@tool("note")
def note_tool() -> str:
    """Return controlled evidence."""
    return "done"


@tool("async_note")
async def async_note_tool() -> str:
    """Return controlled evidence asynchronously."""

    return "async done"


def test_planner_returns_todo_and_repairs_invalid_json(base_state: GraphState) -> None:
    model = ScriptedModel(["bad", '{"todo":["write tests","implement"]}'])
    update = planner_node(base_state, model=model)
    assert update == {"todo": ["write tests", "implement"], "graph_state": "planning"}
    assert len(model.calls) == 2
    human_text = " ".join(
        str(message.content) for message in model.calls[0] if isinstance(message, HumanMessage)
    )
    assert base_state["task"] in human_text


def test_actor_executes_actions_in_order_and_emits_progress(
    monkeypatch, base_state: GraphState, tmp_path: Path
) -> None:
    progress = []
    monkeypatch.setattr("xiliumini.graph.nodes.get_stream_writer", lambda: progress.append)
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "actions": [
                        {
                            "kind": "tool",
                            "name": "file_write",
                            "args": {"path": "red.py", "content": "raise SystemExit(1)\n"},
                            "phase": "test_red",
                            "label": "write red test",
                        },
                        {
                            "kind": "command",
                            "argv": ["python", "red.py"],
                            "phase": "test_red",
                            "label": "run red test",
                        },
                    ]
                }
            )
        ]
    )
    update = actor_node(
        base_state,
        model=model,
        tools=[FileWriteTool(workspace=tmp_path), CommandTool(workspace=tmp_path)],
    )
    assert update["attempt"] == 1
    assert [item["label"] for item in update["execution"]] == [
        "write red test",
        "run red test",
    ]
    assert update["execution"][1]["exit_code"] == 1
    assert [item["status"] for item in progress] == [
        "started",
        "finished",
        "started",
        "finished",
    ]


def test_actor_invalid_json_and_unknown_tool_are_failed_evidence(
    base_state: GraphState,
) -> None:
    invalid = actor_node(base_state, model=ScriptedModel(["bad"]), tools=[])
    assert invalid["attempt"] == 1
    assert invalid["execution"][0]["ok"] is False
    prior = invalid["execution"]
    model = ScriptedModel(
        [
            '{"actions":[{"kind":"tool","name":"missing","args":{},'
            '"phase":"other","label":"missing"}]}'
        ]
    )
    retry_state: GraphState = {**base_state, "attempt": 1, "execution": prior}
    unknown = actor_node(retry_state, model=model, tools=[])
    assert unknown["attempt"] == 2
    assert len(unknown["execution"]) == 2
    assert "unknown tool" in unknown["execution"][-1]["output"]


def test_actor_executes_async_only_tool(base_state: GraphState) -> None:
    model = ScriptedModel(
        [
            '{"actions":[{"kind":"tool","name":"async_note","args":{},'
            '"phase":"other","label":"delegate"}]}'
        ]
    )

    update = actor_node(base_state, model=model, tools=[async_note_tool])

    assert update["execution"][0]["ok"] is True
    assert update["execution"][0]["output"] == "async done"


def test_actor_executes_delegate_analysis_tool(base_state: GraphState) -> None:
    async def analyze(question: str) -> str:
        return f"analysis: {question}"

    delegate = make_delegate_analysis(analyze, timeout_seconds=1, max_chars=100)
    model = ScriptedModel(
        [
            '{"actions":[{"kind":"tool","name":"delegate_analysis",'
            '"args":{"question":"compare"},"phase":"other","label":"delegate"}]}'
        ]
    )

    update = actor_node(base_state, model=model, tools=[delegate])

    assert update["execution"][0]["ok"] is True
    assert update["execution"][0]["output"] == "analysis: compare"


def test_actor_keeps_command_stdout_and_stderr(
    monkeypatch, base_state: GraphState, tmp_path: Path
) -> None:
    command = CommandTool(workspace=tmp_path)
    monkeypatch.setattr(
        CommandTool,
        "execute",
        lambda self, argv: CommandResult(
            ok=False,
            argv=argv,
            exit_code=1,
            stdout="before crash\n",
            stderr="traceback\n",
        ),
    )
    model = ScriptedModel(
        [
            '{"actions":[{"kind":"command","argv":["python","demo.py"],'
            '"phase":"other","label":"run"}]}'
        ]
    )

    update = actor_node(base_state, model=model, tools=[command])

    output = update["execution"][0]["output"]
    assert "stdout:\nbefore crash" in output
    assert "stderr:\ntraceback" in output


def test_verifier_accepts_ordered_tdd_evidence(base_state: GraphState) -> None:
    state: GraphState = {
        **base_state,
        "attempt": 1,
        "execution": [
            evidence("test_red", False, exit_code=1),
            evidence("implementation", True, kind="tool"),
            evidence("test_green", True, exit_code=0),
            evidence("demo", True, exit_code=0),
        ],
    }
    update = verifier_node(state, model=ScriptedModel(['{"status":"passed","reason":"verified"}']))
    assert update == {"graph_state": "passed", "verification": "verified", "attempt": 1}


@pytest.mark.parametrize(
    "execution",
    [
        [
            evidence("implementation", True, kind="tool"),
            evidence("test_green", True, exit_code=0),
            evidence("demo", True, exit_code=0),
        ],
        [
            evidence("test_red", False, exit_code=1),
            evidence("implementation", True, kind="tool"),
            evidence("test_green", False, timed_out=True),
            evidence("demo", True, exit_code=0),
        ],
        [
            evidence("test_red", False, exit_code=1),
            evidence("implementation", True, kind="tool"),
            evidence("test_green", True, exit_code=0),
            evidence("demo", False, exit_code=2),
        ],
    ],
)
def test_verifier_overrides_false_model_pass(
    base_state: GraphState, execution: list[ActionResult]
) -> None:
    state: GraphState = {**base_state, "attempt": 1, "execution": execution}
    update = verifier_node(
        state,
        model=ScriptedModel(['{"status":"passed","reason":"claimed"}']),
    )
    assert update["graph_state"] == "failed"


def test_verifier_rejects_failed_action_for_ordinary_task(base_state: GraphState) -> None:
    state: GraphState = {
        **base_state,
        "task": "write a summary",
        "todo": ["write it"],
        "attempt": 1,
        "execution": [evidence("other", False, kind="tool")],
    }

    update = verifier_node(state, model=ScriptedModel(['{"status":"passed","reason":"claimed"}']))

    assert update["graph_state"] == "failed"
    assert "Failed or rejected action" in update["verification"]


def test_verifier_uses_latest_green_result(base_state: GraphState) -> None:
    state: GraphState = {
        **base_state,
        "attempt": 1,
        "execution": [
            evidence("test_red", False, exit_code=1),
            evidence("implementation", True, kind="tool"),
            evidence("test_green", True, exit_code=0),
            evidence("demo", True, exit_code=0),
            evidence("test_green", False, exit_code=1),
        ],
    }

    update = verifier_node(state, model=ScriptedModel(['{"status":"passed","reason":"claimed"}']))

    assert update["graph_state"] == "failed"
    assert "Latest TDD green" in update["verification"]


def test_verifier_requires_green_after_latest_implementation(
    base_state: GraphState,
) -> None:
    state: GraphState = {
        **base_state,
        "task": "build life with TDD",
        "attempt": 1,
        "execution": [
            evidence("test_red", False, exit_code=1),
            evidence("implementation", True, kind="tool"),
            evidence("test_green", True, exit_code=0),
            evidence("implementation", True, kind="tool"),
        ],
    }

    update = verifier_node(state, model=ScriptedModel(['{"status":"passed","reason":"claimed"}']))

    assert update["graph_state"] == "failed"
    assert "latest implementation" in update["verification"].lower()


def test_verifier_requires_green_when_task_requests_tests(base_state: GraphState) -> None:
    state: GraphState = {
        **base_state,
        "task": "implement a feature and run tests",
        "todo": ["implement", "run tests"],
        "attempt": 1,
        "execution": [evidence("implementation", True, kind="tool")],
    }

    update = verifier_node(state, model=ScriptedModel(['{"status":"passed","reason":"claimed"}']))

    assert update["graph_state"] == "failed"
    assert "test" in update["verification"].lower()


def test_verifier_accepts_requested_successful_tests_without_tdd(
    base_state: GraphState,
) -> None:
    state: GraphState = {
        **base_state,
        "task": "implement a feature and run tests",
        "todo": ["implement", "run tests"],
        "attempt": 1,
        "execution": [
            evidence("implementation", True, kind="tool"),
            evidence("test_green", True, exit_code=0),
        ],
    }

    update = verifier_node(state, model=ScriptedModel(['{"status":"passed","reason":"verified"}']))

    assert update == {
        "graph_state": "passed",
        "verification": "verified",
        "attempt": 1,
    }


def test_verifier_rejects_non_tdd_implementation_after_green(
    base_state: GraphState,
) -> None:
    state: GraphState = {
        **base_state,
        "task": "implement a feature and run tests",
        "todo": ["implement", "run tests"],
        "attempt": 1,
        "execution": [
            evidence("implementation", True, kind="tool"),
            evidence("test_green", True, exit_code=0),
            evidence("implementation", True, kind="tool"),
        ],
    }

    update = verifier_node(state, model=ScriptedModel(['{"status":"passed","reason":"claimed"}']))

    assert update["graph_state"] == "failed"
    assert "latest implementation" in update["verification"].lower()


def test_verifier_allows_retry_to_replace_historical_failure(
    base_state: GraphState,
) -> None:
    state: GraphState = {
        **base_state,
        "task": "write a summary",
        "todo": ["write it"],
        "attempt": 2,
        "execution": [
            evidence("other", False, kind="tool", attempt=1),
            evidence("other", True, kind="tool", attempt=2),
        ],
    }

    update = verifier_node(state, model=ScriptedModel(['{"status":"passed","reason":"fixed"}']))

    assert update == {"graph_state": "passed", "verification": "fixed", "attempt": 2}


def test_verifier_invalid_json_fails_and_final_formats(base_state: GraphState) -> None:
    verifying_state: GraphState = {**base_state, "attempt": 1}
    verified = verifier_node(verifying_state, model=ScriptedModel(["bad"]))
    assert verified["graph_state"] == "failed"
    final_state: GraphState = {
        **base_state,
        "attempt": 3,
        "graph_state": "failed",
        "result": "tests failed",
        "verification": "missing demo",
    }
    final = final_node(final_state)
    assert "状态：failed" in final["final_answer"]
    assert "3/3" in final["final_answer"]
