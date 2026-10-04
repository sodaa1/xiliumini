from __future__ import annotations

import json
import os
import stat
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from xiliumini.errors import SessionError
from xiliumini.tools.workspace import atomic_write_utf8

SESSION_ROOT = ".xiliumini/session"
SESSION_FILE = "session.json"
SESSION_SUMMARY_FILE = "SESSION_SUMMARY.md"
MAX_SESSION_CONTEXT = 7000
MAX_TURN_CONTENT = 4000


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _session_dir(workspace: Path) -> Path:
    return workspace / Path(SESSION_ROOT).name


def _parse_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise SessionError("session data is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise SessionError("session data is invalid") from None
    if parsed.tzinfo is None:
        raise SessionError("session data is invalid")


def _validate_turn(turn: Any) -> None:
    if not isinstance(turn, dict):
        raise SessionError("session data is invalid")
    role = turn.get("role")
    expected = (
        {"turn", "role", "content", "timestamp"}
        if role == "user"
        else {"turn", "role", "route", "content", "summary"}
        if role == "assistant"
        else set()
    )
    if not expected or set(turn) != expected:
        raise SessionError("session data is invalid")
    number = turn["turn"]
    content = turn["content"]
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise SessionError("session data is invalid")
    if not isinstance(content, str) or len(content) > MAX_TURN_CONTENT:
        raise SessionError("session data is invalid")
    if role == "user":
        _parse_timestamp(turn["timestamp"])
    elif (
        not isinstance(turn["route"], str)
        or turn["route"] not in {"chat", "workflow"}
        or not isinstance(turn["summary"], str)
        or len(turn["summary"]) > MAX_TURN_CONTENT
    ):
        raise SessionError("session data is invalid")


def _validate_session(session: Any) -> dict[str, Any]:
    if not isinstance(session, dict):
        raise SessionError("session data is invalid")
    required = {"session_id", "turn_index", "recent_turns", "created_at", "updated_at"}
    if set(session) != required:
        raise SessionError("session data is invalid")
    try:
        canonical_id = str(UUID(session["session_id"]))
    except (ValueError, AttributeError, TypeError):
        raise SessionError("session data is invalid") from None
    if canonical_id != session["session_id"]:
        raise SessionError("session data is invalid")
    turn_index = session["turn_index"]
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise SessionError("session data is invalid")
    turns = session["recent_turns"]
    if not isinstance(turns, list) or len(turns) > 10:
        raise SessionError("session data is invalid")
    for turn in turns:
        _validate_turn(turn)
    numbers = [turn["turn"] for turn in turns]
    if any(
        current != previous + 1 for previous, current in zip(numbers, numbers[1:], strict=False)
    ):
        raise SessionError("session data is invalid")
    if numbers and turn_index < numbers[-1]:
        raise SessionError("session data is invalid")
    if turn_index and not numbers:
        raise SessionError("session data is invalid")
    _parse_timestamp(session["created_at"])
    _parse_timestamp(session["updated_at"])
    return deepcopy(session)


def _render_summary(session: dict[str, Any]) -> str:
    lines = [
        "# Session Summary",
        "",
        f"- Session ID: `{session['session_id']}`",
        f"- Turn index: {session['turn_index']}",
        f"- Updated at: {session['updated_at']}",
        "",
        "## Recent turns",
        "",
    ]
    if not session["recent_turns"]:
        lines.append("_No turns recorded._")
    for turn in session["recent_turns"]:
        route = f" [{turn['route']}]" if turn["role"] == "assistant" else ""
        summary = turn.get("summary") or turn["content"]
        lines.append(f"- {turn['turn']} {turn['role']}{route}: {summary}")
    return "\n".join(lines) + "\n"


def _append_turn(session: dict[str, Any], turn: dict[str, Any]) -> None:
    _validate_session(session)
    expected = session["turn_index"] + 1
    if turn["turn"] != expected:
        raise SessionError("turn must be the next turn")
    session["turn_index"] = expected
    session["recent_turns"] = [*session["recent_turns"], turn][-10:]


def append_user_turn(session: dict[str, Any], content: str) -> int:
    """Append a bounded user message and return its sequence number."""
    if not isinstance(content, str):
        raise SessionError("turn content must be text")
    turn = session.get("turn_index", 0) + 1
    _append_turn(
        session,
        {
            "turn": turn,
            "role": "user",
            "content": content[:MAX_TURN_CONTENT],
            "timestamp": _now(),
        },
    )
    return turn


def append_assistant_turn(
    session: dict[str, Any],
    *,
    turn: int,
    route: Literal["chat", "workflow"],
    content: str,
    summary: str = "",
) -> None:
    """Append a bounded assistant message for the next sequence number."""
    if not isinstance(route, str) or route not in {"chat", "workflow"}:
        raise SessionError("assistant route must be chat or workflow")
    if not isinstance(content, str) or not isinstance(summary, str):
        raise SessionError("turn content must be text")
    _append_turn(
        session,
        {
            "turn": turn,
            "role": "assistant",
            "route": route,
            "content": content[:MAX_TURN_CONTENT],
            "summary": summary[:MAX_TURN_CONTENT],
        },
    )


def save_session(workspace: Path, session: dict[str, Any]) -> dict[str, Any]:
    """Persist validated session JSON and its human-readable summary."""
    saved = _validate_session(session)
    saved["updated_at"] = _now()
    directory = _session_dir(workspace)
    try:
        atomic_write_utf8(
            directory / SESSION_FILE,
            json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write_utf8(directory / SESSION_SUMMARY_FILE, _render_summary(saved))
    except OSError:
        raise SessionError("could not save session") from None
    session.clear()
    session.update(deepcopy(saved))
    return saved


def load_or_create_session(workspace: Path) -> dict[str, Any]:
    """Load the workspace session or create and persist a new one."""
    path = _session_dir(workspace) / SESSION_FILE
    if not path.exists():
        now = _now()
        return save_session(
            workspace,
            {
                "session_id": str(uuid4()),
                "turn_index": 0,
                "recent_turns": [],
                "created_at": now,
                "updated_at": now,
            },
        )
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SessionError("could not load session") from None
    return _validate_session(loaded)


def _workspace_files(workspace: Path, session_id: str) -> list[str]:
    root = workspace / "workspaces" / session_id
    if not root.exists():
        return []
    candidates: list[tuple[float, str]] = []
    try:
        for current, directories, filenames in os.walk(root):
            directories[:] = [name for name in directories if name != ".xiliumini"]
            for filename in filenames:
                path = Path(current) / filename
                try:
                    info = path.stat()
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    candidates.append((info.st_mtime, path.relative_to(root).as_posix()))
    except OSError:
        return []
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [relative for _, relative in candidates[:30]]


def _turn_line(turn: dict[str, Any]) -> str:
    text = turn.get("summary") or turn["content"]
    text = " ".join(text.splitlines())
    route = f" [{turn['route']}]" if turn["role"] == "assistant" else ""
    return f"- Turn {turn['turn']} {turn['role']}{route}: {text}"


def _take_lines_newest_first(lines: list[str], budget: int) -> tuple[list[str], int]:
    selected: list[str] = []
    for line in reversed(lines):
        cost = len(line) + 1
        if cost <= budget:
            selected.append(line)
            budget -= cost
            continue
        if budget > 1:
            selected.append(line[: budget - 1])
            budget = 0
        break
    selected.reverse()
    return selected, budget


def build_session_context(workspace: Path, session: dict[str, Any] | None = None) -> str:
    """Build bounded routing context from recent turns and isolated workspace files."""
    current = load_or_create_session(workspace) if session is None else _validate_session(session)
    header = [
        "Session context",
        f"Session ID: {current['session_id']}",
        f"Turn index: {current['turn_index']}",
        "",
        "Workspace files (newest first):",
        "",
        "Recent conversation:",
    ]
    base = "\n".join(header)
    budget = MAX_SESSION_CONTEXT - len(base) - 1
    turn_lines = [_turn_line(turn) for turn in current["recent_turns"][-10:]]
    selected_turns, budget = _take_lines_newest_first(turn_lines, budget)

    selected_files: list[str] = []
    for relative in _workspace_files(workspace, current["session_id"]):
        line = f"- {relative}"
        cost = len(line) + 1
        if cost > budget:
            break
        selected_files.append(line)
        budget -= cost

    lines = [
        *header[:5],
        *selected_files,
        "",
        header[-1],
        *selected_turns,
    ]
    return "\n".join(lines)[:MAX_SESSION_CONTEXT]
