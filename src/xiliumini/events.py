from __future__ import annotations

from dataclasses import dataclass

from xiliumini.graph.state import TodoItem


@dataclass(frozen=True, slots=True)
class PlannerEvent:
    todos: list[TodoItem]
    summary: str
    attempt: int


@dataclass(frozen=True, slots=True)
class VerifierEvent:
    passed: bool
    reason: str
    attempt: int


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class FinalEvent:
    text: str
    session_id: str


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    code: str
    message: str


RuntimeEvent = PlannerEvent | VerifierEvent | ProgressEvent | FinalEvent | ErrorEvent
