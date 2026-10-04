import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from xiliumini.core.session import (
    MAX_SESSION_CONTEXT,
    MAX_TURN_CONTENT,
    SESSION_FILE,
    SESSION_SUMMARY_FILE,
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
)
from xiliumini.errors import SessionError


def test_load_or_create_session_persists_and_reloads_identity(tmp_path):
    workspace = tmp_path / ".xiliumini"

    created = load_or_create_session(workspace)

    assert UUID(created["session_id"]).version == 4
    assert created["turn_index"] == 0
    assert created["recent_turns"] == []
    for field in ("created_at", "updated_at"):
        parsed = datetime.fromisoformat(created[field])
        assert parsed.tzinfo == UTC
    assert (workspace / "session" / SESSION_FILE).is_file()
    assert (workspace / "session" / SESSION_SUMMARY_FILE).is_file()

    loaded = load_or_create_session(workspace)

    assert loaded == created


def test_turns_are_monotonic_truncated_and_retain_the_latest_ten(tmp_path):
    session = load_or_create_session(tmp_path / ".xiliumini")

    user_turn = append_user_turn(session, "u" * (MAX_TURN_CONTENT + 5))
    append_assistant_turn(
        session,
        turn=user_turn + 1,
        route="chat",
        content="answer",
        summary="short answer",
    )
    for index in range(5):
        turn = append_user_turn(session, f"question {index}")
        append_assistant_turn(
            session,
            turn=turn + 1,
            route="workflow",
            content=f"result {index}",
        )

    assert session["turn_index"] == 12
    assert [item["turn"] for item in session["recent_turns"]] == list(range(3, 13))
    assert session["recent_turns"][-1] == {
        "turn": 12,
        "role": "assistant",
        "route": "workflow",
        "content": "result 4",
        "summary": "",
    }

    fresh = load_or_create_session(tmp_path / "other")
    append_user_turn(fresh, "x" * (MAX_TURN_CONTENT + 5))
    assert fresh["recent_turns"][0]["content"] == "x" * MAX_TURN_CONTENT


def test_append_assistant_rejects_invalid_route_and_non_next_turn(tmp_path):
    session = load_or_create_session(tmp_path / ".xiliumini")
    append_user_turn(session, "hello")

    with pytest.raises(SessionError, match="route"):
        append_assistant_turn(session, turn=2, route=cast(Any, "other"), content="bad")
    with pytest.raises(SessionError, match="next turn"):
        append_assistant_turn(session, turn=3, route="chat", content="bad")

    assert session["turn_index"] == 1


@pytest.mark.parametrize(
    "turns",
    [
        [
            {"turn": 1, "role": "user", "content": "one", "timestamp": "2026-10-03T00:00:00+00:00"},
            {
                "turn": 1,
                "role": "user",
                "content": "duplicate",
                "timestamp": "2026-10-03T00:00:01+00:00",
            },
        ],
        [
            {"turn": 1, "role": "user", "content": "one", "timestamp": "2026-10-03T00:00:00+00:00"},
            {"turn": 3, "role": "user", "content": "gap", "timestamp": "2026-10-03T00:00:01+00:00"},
        ],
        [
            {"turn": 2, "role": "user", "content": "two", "timestamp": "2026-10-03T00:00:00+00:00"},
            {
                "turn": 1,
                "role": "user",
                "content": "backward",
                "timestamp": "2026-10-03T00:00:01+00:00",
            },
        ],
    ],
)
def test_save_session_rejects_invalid_turn_order(tmp_path, turns):
    session = load_or_create_session(tmp_path / ".xiliumini")
    session["turn_index"] = max(item["turn"] for item in turns)
    session["recent_turns"] = turns

    with pytest.raises(SessionError, match="invalid"):
        save_session(tmp_path / ".xiliumini", session)


def test_corrupt_session_is_not_overwritten(tmp_path):
    workspace = tmp_path / ".xiliumini"
    path = workspace / "session" / SESSION_FILE
    path.parent.mkdir(parents=True)
    original = b'{"session_id": "broken"}'
    path.write_bytes(original)

    with pytest.raises(SessionError, match="load session|invalid"):
        load_or_create_session(workspace)

    assert path.read_bytes() == original


def test_session_context_contains_recent_files_and_prefers_assistant_summary(tmp_path):
    workspace = tmp_path / ".xiliumini"
    session = load_or_create_session(workspace)
    files = workspace / "workspaces" / session["session_id"]
    files.mkdir(parents=True)
    old = files / "old.txt"
    new = files / "nested" / "new.txt"
    new.parent.mkdir()
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    user_turn = append_user_turn(session, "What changed?")
    append_assistant_turn(
        session,
        turn=user_turn + 1,
        route="workflow",
        content="A long final answer that should not be used here.",
        summary="Changed the newest file.",
    )

    context = build_session_context(workspace, session)

    assert f"Session ID: {session['session_id']}" in context
    assert "Turn index: 2" in context
    assert context.index("nested/new.txt") < context.index("old.txt")
    assert "Turn 1 user: What changed?" in context
    assert "Turn 2 assistant [workflow]: Changed the newest file." in context
    assert "long final answer" not in context


def test_session_context_lists_only_thirty_newest_files(tmp_path):
    workspace = tmp_path / ".xiliumini"
    session = load_or_create_session(workspace)
    files = workspace / "workspaces" / session["session_id"]
    files.mkdir(parents=True)
    for index in range(32):
        path = files / f"file-{index:02}.txt"
        path.write_text(str(index), encoding="utf-8")
        os.utime(path, (index + 1, index + 1))

    context = build_session_context(workspace, session)
    file_lines = [line for line in context.splitlines() if line.startswith("- file-")]

    assert len(file_lines) == 30
    assert file_lines[0] == "- file-31.txt"
    assert file_lines[-1] == "- file-02.txt"
    assert "file-01.txt" not in context


def test_session_context_excludes_internal_checkpoint_and_trace_files(tmp_path):
    workspace = tmp_path / ".xiliumini"
    session = load_or_create_session(workspace)
    files = workspace / "workspaces" / session["session_id"]
    user_file = files / "user.py"
    user_file.parent.mkdir(parents=True)
    user_file.write_text("print('safe')", encoding="utf-8")
    for index in range(35):
        internal = files / ".xiliumini" / "checkpoints" / f"object-{index}"
        internal.parent.mkdir(parents=True, exist_ok=True)
        internal.write_text("internal", encoding="utf-8")
        os.utime(internal, (index + 100, index + 100))
    trace = files / ".xiliumini" / "traces" / "trace.json"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}", encoding="utf-8")

    context = build_session_context(workspace, session)

    assert "user.py" in context
    assert ".xiliumini" not in context


def test_session_context_is_bounded_and_keeps_newest_dialogue(tmp_path):
    workspace = tmp_path / ".xiliumini"
    session = load_or_create_session(workspace)
    for index in range(6):
        turn = append_user_turn(session, f"question-{index}-" + "q" * MAX_TURN_CONTENT)
        append_assistant_turn(
            session,
            turn=turn + 1,
            route="chat",
            content=f"answer-{index}-" + "a" * MAX_TURN_CONTENT,
        )

    context = build_session_context(workspace, session)

    assert len(context) <= MAX_SESSION_CONTEXT
    assert "answer-5-" in context
    assert "question-0-" not in context


def test_session_context_skips_file_removed_during_inventory(monkeypatch, tmp_path):
    workspace = tmp_path / ".xiliumini"
    session = load_or_create_session(workspace)
    files = workspace / "workspaces" / session["session_id"]
    files.mkdir(parents=True)
    vanishing = files / "vanishing.txt"
    stable = files / "stable.txt"
    vanishing.write_text("gone", encoding="utf-8")
    stable.write_text("here", encoding="utf-8")
    real_stat = Path.stat

    def sometimes_missing(path, *args, **kwargs):
        if path == vanishing:
            raise FileNotFoundError
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", sometimes_missing)

    context = build_session_context(workspace, session)

    assert "stable.txt" in context
    assert "vanishing.txt" not in context


def test_load_rejects_non_string_assistant_route_without_overwriting(tmp_path):
    workspace = tmp_path / ".xiliumini"
    session = load_or_create_session(workspace)
    session["turn_index"] = 1
    session["recent_turns"] = [
        {
            "turn": 1,
            "role": "assistant",
            "route": [],
            "content": "bad",
            "summary": "",
        }
    ]
    path = workspace / "session" / SESSION_FILE
    original = json.dumps(session).encode()
    path.write_bytes(original)

    with pytest.raises(SessionError, match="invalid"):
        load_or_create_session(workspace)

    assert path.read_bytes() == original
