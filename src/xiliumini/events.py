from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlannerEvent:
    todo: list[str]


@dataclass(frozen=True, slots=True)
class ActorEvent:
    result: str
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


RuntimeEvent = PlannerEvent | ActorEvent | VerifierEvent | ProgressEvent | FinalEvent | ErrorEvent
