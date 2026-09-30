from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field, ValidationError

from xiliumini.errors import WorkspaceError
from xiliumini.graph.memory import UserPreference
from xiliumini.tools.workspace import MAX_FILE_BYTES, atomic_write_utf8, read_utf8_text

PREFERENCES_PATH = "USER_PREFERENCES.md"
MAX_PREFERENCES = 100
MAX_PREFERENCE_CHARS = 2_000
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_SECRET = re.compile(r"(?:sk-|api[_-]?key\s*=|token\s*=|password\s*=)", re.IGNORECASE)
_DOCUMENT = re.compile(
    r"\A# User Preferences\n\n```json\n(?P<payload>.*)\n```\n\Z",
    re.DOTALL,
)


class _PreferenceRecord(BaseModel):
    key: str
    content: str
    updated_at: str


class _PreferenceDocument(BaseModel):
    schema_version: Literal[1]
    preferences: list[_PreferenceRecord]


def _validate_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        raise ValueError("preference key must be a lowercase slug")
    return key


def _validate_content(content: str) -> str:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("preference content must not be blank")
    cleaned = content.strip()
    if len(cleaned) > MAX_PREFERENCE_CHARS:
        raise ValueError("preference content exceeds 2,000 characters")
    if _SECRET.search(cleaned):
        raise ValueError("preference content may contain sensitive data")
    return cleaned


class UserPreferenceStore:
    """Persist project-wide preferences outside individual session workspaces."""

    def __init__(
        self,
        data_dir: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def path(self) -> Path:
        return self.data_dir / PREFERENCES_PATH

    def read(self) -> list[UserPreference]:
        if not self.path.exists():
            return []
        try:
            text = read_utf8_text(self.path, MAX_FILE_BYTES)
            match = _DOCUMENT.fullmatch(text)
            if match is None:
                raise ValueError
            parsed = _PreferenceDocument.model_validate_json(match.group("payload"))
            if len(parsed.preferences) > MAX_PREFERENCES:
                raise ValueError
            result: list[UserPreference] = []
            keys: set[str] = set()
            for item in parsed.preferences:
                key = _validate_key(item.key)
                content = _validate_content(item.content)
                datetime.fromisoformat(item.updated_at)
                if key in keys:
                    raise ValueError
                keys.add(key)
                result.append({"key": key, "content": content, "updated_at": item.updated_at})
            return result
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError):
            raise ValueError("preference data is invalid") from None

    def _write(self, preferences: list[UserPreference]) -> list[UserPreference]:
        payload = {"schema_version": 1, "preferences": preferences}
        content = (
            "# User Preferences\n\n```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n```\n"
        )
        atomic_write_utf8(self.path, content)
        return preferences

    def upsert(self, key: str, content: str) -> list[UserPreference]:
        normalized_key = _validate_key(key)
        normalized_content = _validate_content(content)
        preferences = self.read()
        selected = next((item for item in preferences if item["key"] == normalized_key), None)
        record: UserPreference = {
            "key": normalized_key,
            "content": normalized_content,
            "updated_at": self._clock().astimezone(UTC).isoformat(),
        }
        if selected is None:
            if len(preferences) >= MAX_PREFERENCES:
                raise ValueError("preference limit is 100 entries")
            preferences.append(record)
        else:
            selected.update(record)
        return self._write(preferences)

    def remove(self, key: str) -> list[UserPreference]:
        normalized_key = _validate_key(key)
        preferences = self.read()
        if not any(item["key"] == normalized_key for item in preferences):
            raise ValueError("unknown preference key")
        return self._write([item for item in preferences if item["key"] != normalized_key])


class PreferenceWriteInput(BaseModel):
    action: Literal["upsert", "remove"]
    key: str = Field(min_length=1, max_length=100)
    content: str | None = Field(default=None, max_length=MAX_PREFERENCE_CHARS)


class PreferenceWriteTool(BaseTool):
    name: str = "preference_write"
    description: str = "Record or forget one explicit project-wide user preference."
    args_schema: ArgsSchema | None = PreferenceWriteInput
    store: Any = Field(exclude=True)

    def _run(self, action: str, key: str, content: str | None = None) -> str:
        try:
            if action == "upsert":
                if content is None:
                    raise ValueError("preference content is required for upsert")
                preferences = self.store.upsert(key, content)
            else:
                preferences = self.store.remove(key)
            return json.dumps({"ok": True, "preferences": preferences}, ensure_ascii=False)
        except (OSError, ValueError, WorkspaceError):
            return json.dumps(
                {"ok": False, "error": "preference update rejected"},
                ensure_ascii=False,
            )
