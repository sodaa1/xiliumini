import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from xiliumini.errors import WorkspaceError
from xiliumini.graph.memory import HistorySummaryStore
from xiliumini.tools.workspace import MAX_FILE_BYTES


def notepad_api():
    return importlib.import_module("xiliumini.tools.notepad")


def test_notepad_module_exists_and_obsolete_memory_module_is_removed() -> None:
    assert importlib.util.find_spec("xiliumini.tools.notepad") is not None
    assert importlib.util.find_spec("xiliumini.memory") is None


def test_notepad_appends_without_overwriting_and_reads_empty_workspace(tmp_path: Path) -> None:
    read = notepad_api().NotepadReadTool(workspace=tmp_path)
    append = notepad_api().NotepadAppendTool(workspace=tmp_path)

    assert json.loads(read.invoke({})) == {"ok": True, "content": ""}
    assert json.loads(append.invoke({"entry": "first decision"}))["ok"] is True
    assert json.loads(append.invoke({"entry": "second decision"}))["ok"] is True
    assert json.loads(read.invoke({}))["content"] == "first decision\nsecond decision\n"
    assert (tmp_path / "NOTEPAD.md").read_text(encoding="utf-8") == (
        "first decision\nsecond decision\n"
    )


def test_notepad_rejects_content_over_size_limit_without_changing_file(tmp_path: Path) -> None:
    append = notepad_api().NotepadAppendTool(workspace=tmp_path)
    assert json.loads(append.invoke({"entry": "keep"}))["ok"] is True

    result = json.loads(append.invoke({"entry": "x" * MAX_FILE_BYTES}))

    assert result["ok"] is False
    assert "limit" in result["error"]
    content = json.loads(notepad_api().NotepadReadTool(workspace=tmp_path).invoke({}))["content"]
    assert content == "keep\n"


def test_notepad_migrates_legacy_and_canonical_wins(tmp_path: Path) -> None:
    legacy = tmp_path / ".xiliumini" / "notepad.md"
    legacy.parent.mkdir()
    legacy.write_text("legacy note\n", encoding="utf-8")

    assert notepad_api().read_notepad(tmp_path) == "legacy note\n"
    assert (tmp_path / "NOTEPAD.md").read_text(encoding="utf-8") == "legacy note\n"
    assert legacy.read_text(encoding="utf-8") == "legacy note\n"

    (tmp_path / "NOTEPAD.md").write_text("canonical note\n", encoding="utf-8")
    legacy.write_bytes(b"\xff")
    assert notepad_api().read_notepad(tmp_path) == "canonical note\n"


def test_invalid_legacy_notepad_is_preserved(tmp_path: Path) -> None:
    legacy = tmp_path / ".xiliumini" / "notepad.md"
    legacy.parent.mkdir()
    legacy.write_bytes(b"\xff")

    with pytest.raises(WorkspaceError, match="UTF-8"):
        notepad_api().read_notepad(tmp_path)

    assert legacy.read_bytes() == b"\xff"
    assert not (tmp_path / "NOTEPAD.md").exists()


def test_history_summary_store_round_trips_canonical_file(tmp_path: Path) -> None:
    history = HistorySummaryStore(tmp_path)

    assert history.read() == ""
    history.write("# History Summary\n\nImplemented parser.\n")

    assert history.read() == "# History Summary\n\nImplemented parser.\n"
    assert (tmp_path / "HISTORY_SUMMARY.md").read_text(encoding="utf-8") == history.read()


def test_history_summary_rejects_invalid_utf8_without_overwriting(tmp_path: Path) -> None:
    target = tmp_path / "HISTORY_SUMMARY.md"
    target.write_bytes(b"\xff")

    with pytest.raises(WorkspaceError, match="UTF-8"):
        HistorySummaryStore(tmp_path).read()

    assert target.read_bytes() == b"\xff"
