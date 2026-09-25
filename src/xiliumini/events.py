from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenEvent:
    text: str


@dataclass(frozen=True, slots=True)
class ToolStartedEvent:
    name: str
    call_id: str


@dataclass(frozen=True, slots=True)
class ToolFinishedEvent:
    name: str
    call_id: str
    duration_ms: float
    ok: bool


@dataclass(frozen=True, slots=True)
class FinalEvent:
    text: str
    session_id: str


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    code: str
    message: str


RuntimeEvent = TokenEvent | ToolStartedEvent | ToolFinishedEvent | FinalEvent | ErrorEvent
