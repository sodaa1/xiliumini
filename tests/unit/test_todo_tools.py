import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from xiliumini.errors import WorkspaceError


def drafts(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"id": todo_id, "content": content} for todo_id, content in pairs]


def todo_api():
    return importlib.import_module("xiliumini.tools.todo")


def test_todo_module_exists() -> None:
    assert importlib.util.find_spec("xiliumini.tools.todo") is not None


def test_todo_store_persists_stable_ids_and_preserves_existing_status(tmp_path: Path) -> None:
    store = todo_api().TodoStore(tmp_path)
    assert store.write(drafts(("tests", "write tests"))) == [
        {"id": "tests", "content": "write tests", "status": "pending", "note": ""}
    ]

    store.update("tests", "in_progress")
    store.update("tests", "completed", "2 tests pass")

    assert store.write(drafts(("tests", "write tests"), ("impl", "implement"))) == [
        {
            "id": "tests",
            "content": "write tests",
            "status": "completed",
            "note": "2 tests pass",
        },
        {"id": "impl", "content": "implement", "status": "pending", "note": ""},
    ]
    assert todo_api().TodoStore(tmp_path).read()[0]["status"] == "completed"


def test_todo_store_enforces_transitions_and_blocked_reason(tmp_path: Path) -> None:
    store = todo_api().TodoStore(tmp_path)
    store.write(drafts(("impl", "implement")))

    with pytest.raises(ValueError, match="in_progress before completed"):
        store.update("impl", "completed")
    with pytest.raises(ValueError, match="blocked status requires a note"):
        store.update("impl", "blocked")
    with pytest.raises(ValueError, match="unknown todo id"):
        store.update("missing", "in_progress")

    blocked = store.update("impl", "blocked", "dependency unavailable")
    assert blocked[0]["status"] == "blocked"
    assert blocked[0]["note"] == "dependency unavailable"


def test_todo_write_refuses_to_drop_in_progress_item(tmp_path: Path) -> None:
    store = todo_api().TodoStore(tmp_path)
    store.write(drafts(("active", "active work"), ("later", "later work")))
    store.update("active", "in_progress")

    with pytest.raises(ValueError, match="cannot remove in-progress todo: active"):
        store.write(drafts(("later", "later work")))


def test_todo_tools_return_json_snapshots(tmp_path: Path) -> None:
    created = json.loads(
        todo_api()
        .TodoWriteTool(workspace=tmp_path)
        .invoke({"todos": [{"id": "impl", "content": "implement"}]})
    )
    assert created["ok"] is True
    assert created["todos"][0]["status"] == "pending"

    updated = json.loads(
        todo_api()
        .TodoUpdateTool(workspace=tmp_path)
        .invoke({"todo_id": "impl", "status": "in_progress"})
    )
    assert updated["ok"] is True
    assert updated["todos"][0]["status"] == "in_progress"


def test_corrupt_todo_json_is_not_overwritten(tmp_path: Path) -> None:
    target = tmp_path / ".xiliumini" / "todos.json"
    target.parent.mkdir()
    target.write_text("not-json", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="todo data is invalid"):
        todo_api().TodoStore(tmp_path).read()
    result = json.loads(
        todo_api()
        .TodoWriteTool(workspace=tmp_path)
        .invoke({"todos": [{"id": "new", "content": "new work"}]})
    )
    assert result == {"ok": False, "error": "todo data is invalid"}
    assert target.read_text(encoding="utf-8") == "not-json"


def test_completed_todo_can_be_reopened_for_verifier_feedback(tmp_path):
    store = todo_api().TodoStore(tmp_path)
    store.write(drafts(("impl", "implement")))
    store.update("impl", "in_progress")
    store.update("impl", "completed")
    assert (
        store.update("impl", "in_progress", "recheck verifier feedback")[0]["status"]
        == "in_progress"
    )
    assert store.update("impl", "in_progress")[0]["status"] == "in_progress"
