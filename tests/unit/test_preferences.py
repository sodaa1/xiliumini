from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from xiliumini.tools.file_read import FileReadTool
from xiliumini.tools.preferences import (
    PreferenceWriteInput,
    PreferenceWriteTool,
    UserPreferenceStore,
)

NOW = datetime(2026, 9, 29, 1, 2, 3, tzinfo=UTC)


def store(data_dir: Path) -> UserPreferenceStore:
    return UserPreferenceStore(data_dir, clock=lambda: NOW)


def test_preference_store_round_trips_updates_and_removes_by_key(tmp_path: Path) -> None:
    preferences = store(tmp_path)

    assert preferences.read() == []
    assert preferences.upsert("test_style", "Prefer pytest.")[0]["updated_at"] == NOW.isoformat()
    updated = preferences.upsert("test_style", "Prefer focused pytest first.")

    assert updated == [
        {
            "key": "test_style",
            "content": "Prefer focused pytest first.",
            "updated_at": NOW.isoformat(),
        }
    ]
    assert UserPreferenceStore(tmp_path).read() == updated
    assert preferences.remove("test_style") == []


def test_preference_file_has_versioned_markdown_json_format(tmp_path: Path) -> None:
    store(tmp_path).upsert("language", "Reply in Chinese.")

    text = (tmp_path / "USER_PREFERENCES.md").read_text(encoding="utf-8")
    assert text.startswith("# User Preferences\n\n```json\n")
    assert text.endswith("\n```\n")
    payload = json.loads(text.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
    assert payload == {
        "schema_version": 1,
        "preferences": [
            {
                "key": "language",
                "content": "Reply in Chinese.",
                "updated_at": NOW.isoformat(),
            }
        ],
    }


def test_preference_store_preserves_corrupt_file(tmp_path: Path) -> None:
    target = tmp_path / "USER_PREFERENCES.md"
    original = "# User Preferences\n\n```json\nnot-json\n```\n"
    target.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        store(tmp_path).upsert("language", "Chinese")

    assert target.read_text(encoding="utf-8") == original


def test_preference_store_enforces_entry_and_content_bounds(tmp_path: Path) -> None:
    preferences = store(tmp_path)
    for index in range(100):
        preferences.upsert(f"pref_{index}", f"value {index}")

    with pytest.raises(ValueError, match="100"):
        preferences.upsert("one_too_many", "value")
    with pytest.raises(ValueError, match="2,000"):
        preferences.upsert("long", "x" * 2001)


@pytest.mark.parametrize("key", ["", "with space", "../escape", "UPPER"])
def test_preference_store_rejects_invalid_keys(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match="key"):
        store(tmp_path).upsert(key, "value")


@pytest.mark.parametrize("content", ["", "   "])
def test_preference_store_rejects_blank_content(tmp_path: Path, content: str) -> None:
    with pytest.raises(ValueError, match="content"):
        store(tmp_path).upsert("style", content)


def test_preference_store_rejects_removing_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        store(tmp_path).remove("missing")


@pytest.mark.parametrize(
    "secret",
    ["sk-private-value", "api_key=private", "token=private", "password=private"],
)
def test_preference_tool_rejects_secrets_without_echoing_them(tmp_path: Path, secret: str) -> None:
    tool = PreferenceWriteTool(store=store(tmp_path))

    result = json.loads(tool.invoke({"action": "upsert", "key": "credentials", "content": secret}))

    assert result["ok"] is False
    assert secret not in result["error"]
    assert str(tmp_path) not in result["error"]
    assert not (tmp_path / "USER_PREFERENCES.md").exists()


def test_preference_tool_upserts_and_removes_without_path_argument(tmp_path: Path) -> None:
    tool = PreferenceWriteTool(store=store(tmp_path))

    created = json.loads(
        tool.invoke(
            {
                "action": "upsert",
                "key": "language",
                "content": "Reply in Chinese.",
            }
        )
    )
    removed = json.loads(tool.invoke({"action": "remove", "key": "language"}))

    assert created["ok"] is True
    assert created["preferences"][0]["key"] == "language"
    assert removed == {"ok": True, "preferences": []}
    assert tool.args_schema is PreferenceWriteInput
    assert "path" not in PreferenceWriteInput.model_fields


def test_workspace_file_tool_cannot_read_project_preferences(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    workspace = data_dir / "workspaces" / "session"
    workspace.mkdir(parents=True)
    store(data_dir).upsert("language", "Reply in Chinese.")

    result = FileReadTool(workspace=workspace).invoke({"path": "../../USER_PREFERENCES.md"})

    assert result.startswith("Error:")
    assert "Reply in Chinese" not in result
