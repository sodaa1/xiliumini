from pathlib import Path
from typing import Literal

from typing_extensions import TypedDict

ActionKind = Literal["tool", "command"]
ActionPhase = Literal["test_red", "implementation", "test_green", "demo", "other"]
GraphStatus = Literal["planning", "acting", "passed", "failed"]


class ActionResult(TypedDict):
    attempt: int
    kind: ActionKind
    phase: ActionPhase
    label: str
    ok: bool
    output: str
    exit_code: int | None
    timed_out: bool
    truncated: bool


class GraphState(TypedDict):
    task: str
    todo: list[str]
    result: str
    execution: list[ActionResult]
    graph_state: GraphStatus
    verification: str
    attempt: int
    max_attempts: int
    final_answer: str
    session_id: str
    workspace: Path
