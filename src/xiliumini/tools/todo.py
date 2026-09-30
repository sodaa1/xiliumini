from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field, ValidationError, field_validator

from xiliumini.errors import WorkspaceError
from xiliumini.graph.memory import decode_markdown_json, encode_markdown_json
from xiliumini.graph.state import TodoDraft, TodoItem, TodoStatus
from xiliumini.tools.workspace import (
    MAX_FILE_BYTES,
    atomic_write_utf8,
    read_utf8_text,
    resolve_workspace_path,
)

TODO_PATH = "TODO.md"
LEGACY_TODO_PATH = ".xiliumini/todos.json"


class TodoDraftInput(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=2_000)

    @field_validator("id", "content")
    @classmethod
    def strip_nonblank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class TodoWriteInput(BaseModel):
    todos: list[TodoDraftInput] = Field(min_length=1)


class TodoUpdateInput(BaseModel):
    todo_id: str = Field(min_length=1, max_length=100)
    status: TodoStatus
    note: str | None = Field(default=None, max_length=2_000)


class _TodoRecord(TodoDraftInput):
    status: TodoStatus
    note: str = ""


class TodoStore:
    """Persist structured todo progress inside one session workspace."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @property
    def path(self) -> Path:
        return resolve_workspace_path(self.workspace, TODO_PATH)

    def read(self) -> list[TodoItem]:
        target = self.path
        try:
            legacy = resolve_workspace_path(self.workspace, LEGACY_TODO_PATH)
            if target.exists():
                document = decode_markdown_json("TODO", read_utf8_text(target, MAX_FILE_BYTES))
                if document.get("schema_version") != 1:
                    raise ValueError
                raw = document.get("todos")
            elif legacy.exists():
                raw = json.loads(read_utf8_text(legacy, MAX_FILE_BYTES))
            else:
                return []
            if not isinstance(raw, list):
                raise ValueError
            records = [_TodoRecord.model_validate(item) for item in raw]
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
            raise WorkspaceError("todo data is invalid") from None
        result = [cast(TodoItem, record.model_dump()) for record in records]
        if not target.exists():
            self._persist(result)
        return result

    def _persist(self, items: list[TodoItem]) -> list[TodoItem]:
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target = self.path
        atomic_write_utf8(
            target,
            encode_markdown_json("TODO", {"schema_version": 1, "todos": items}),
        )
        return items

    def start_task(self) -> None:
        """Reset progress for a new user task, never overwrite corrupt data."""
        self.read()
        self._persist([])

    def write(self, items: list[TodoDraft]) -> list[TodoItem]:
        normalized = [TodoDraftInput.model_validate(item) for item in items]
        ids = [item.id for item in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError("todo ids must be unique")

        existing = {item["id"]: item for item in self.read()}
        requested_ids = set(ids)
        active = [
            item["id"]
            for item in existing.values()
            if item["status"] == "in_progress" and item["id"] not in requested_ids
        ]
        if active:
            raise ValueError(f"cannot remove in-progress todo: {active[0]}")

        result: list[TodoItem] = []
        for draft in normalized:
            prior = existing.get(draft.id)
            if prior is not None and prior["content"] == draft.content:
                result.append(prior)
            else:
                result.append(
                    {
                        "id": draft.id,
                        "content": draft.content,
                        "status": "pending",
                        "note": "",
                    }
                )
        return self._persist(result)

    def update(
        self,
        todo_id: str,
        status: TodoStatus,
        note: str | None = None,
    ) -> list[TodoItem]:
        items = self.read()
        selected = next((item for item in items if item["id"] == todo_id), None)
        if selected is None:
            raise ValueError(f"unknown todo id: {todo_id}")
        cleaned_note = (note or "").strip()
        if status == "blocked" and not cleaned_note:
            raise ValueError("blocked status requires a note")

        current = selected["status"]
        allowed: dict[TodoStatus, set[TodoStatus]] = {
            "pending": {"in_progress", "blocked"},
            "in_progress": {"in_progress", "completed", "blocked"},
            "completed": {"in_progress", "completed"},
            "blocked": {"in_progress", "blocked"},
        }
        if status not in allowed[current]:
            if current == "pending" and status == "completed":
                raise ValueError("todo must be in_progress before completed")
            raise ValueError(f"invalid todo transition: {current} -> {status}")
        selected["status"] = status
        selected["note"] = cleaned_note
        return self._persist(items)


def _error_payload(exc: Exception) -> str:
    message = str(exc)
    if isinstance(exc, OSError):
        message = "todo persistence failed"
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


class TodoWriteTool(BaseTool):
    name: str = "todo_write"
    description: str = "Create or replace the persistent todo plan for this session."
    args_schema: ArgsSchema | None = TodoWriteInput
    workspace: Path

    def _run(self, todos: list[dict[str, str]]) -> str:
        try:
            result = TodoStore(self.workspace).write(cast(list[TodoDraft], todos))
            return json.dumps({"ok": True, "todos": result}, ensure_ascii=False)
        except (OSError, ValueError, WorkspaceError) as exc:
            return _error_payload(exc)


class TodoUpdateTool(BaseTool):
    name: str = "todo_update"
    description: str = "Update one persistent todo status and optional progress note."
    args_schema: ArgsSchema | None = TodoUpdateInput
    workspace: Path

    def _run(self, todo_id: str, status: TodoStatus, note: str | None = None) -> str:
        try:
            result = TodoStore(self.workspace).update(todo_id, status, note)
            return json.dumps({"ok": True, "todos": result}, ensure_ascii=False)
        except (OSError, ValueError, WorkspaceError) as exc:
            return _error_payload(exc)
