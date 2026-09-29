import importlib
import importlib.util
import json
from pathlib import Path

from xiliumini.tools.workspace import MAX_FILE_BYTES


def notepad_api():
    return importlib.import_module("xiliumini.tools.notepad")


def memory_api():
    return importlib.import_module("xiliumini.memory")


def test_notepad_and_memory_modules_exist() -> None:
    assert importlib.util.find_spec("xiliumini.tools.notepad") is not None
    assert importlib.util.find_spec("xiliumini.memory") is not None


def test_notepad_appends_without_overwriting_and_reads_empty_workspace(tmp_path: Path) -> None:
    read = notepad_api().NotepadReadTool(workspace=tmp_path)
    append = notepad_api().NotepadAppendTool(workspace=tmp_path)

    assert json.loads(read.invoke({})) == {"ok": True, "content": ""}
    assert json.loads(append.invoke({"entry": "first decision"}))["ok"] is True
    assert json.loads(append.invoke({"entry": "second decision"}))["ok"] is True
    assert json.loads(read.invoke({}))["content"] == "first decision\nsecond decision\n"


def test_notepad_rejects_content_over_size_limit_without_changing_file(tmp_path: Path) -> None:
    append = notepad_api().NotepadAppendTool(workspace=tmp_path)
    assert json.loads(append.invoke({"entry": "keep"}))["ok"] is True

    result = json.loads(append.invoke({"entry": "x" * MAX_FILE_BYTES}))

    assert result["ok"] is False
    assert "limit" in result["error"]
    content = json.loads(notepad_api().NotepadReadTool(workspace=tmp_path).invoke({}))["content"]
    assert content == "keep\n"


def test_memory_snapshot_contains_only_serializable_session_context(tmp_path: Path) -> None:
    notepad_api().NotepadAppendTool(workspace=tmp_path).invoke({"entry": "durable note"})
    state = {
        "task": "research and build",
        "todos": [{"id": "impl", "content": "implement", "status": "pending", "note": ""}],
        "research_notes": [
            {
                "summary": "official docs",
                "queries": ["official docs"],
                "sources": ["https://example.test/docs"],
                "attempt": 1,
            }
        ],
        "session_id": "11111111-1111-4111-8111-111111111111",
        "workspace": tmp_path,
        "api_key": "must-not-leak",
        "model": object(),
    }

    snapshot = memory_api().build_memory_snapshot(state)
    serialized = json.dumps(snapshot)

    assert snapshot["session_id"] == state["session_id"]
    assert snapshot["todos"] == state["todos"]
    assert snapshot["research_notes"] == state["research_notes"]
    assert snapshot["notepad"] == "durable note\n"
    assert "must-not-leak" not in serialized
