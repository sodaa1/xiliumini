from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field, field_validator

from xiliumini.errors import WorkspaceError
from xiliumini.tools.workspace import (
    MAX_FILE_BYTES,
    atomic_write_utf8,
    read_utf8_text,
    resolve_workspace_path,
    utf8_size,
)

NOTEPAD_PATH = ".xiliumini/notepad.md"


class NotepadReadInput(BaseModel):
    pass


class NotepadAppendInput(BaseModel):
    entry: str = Field(min_length=1, max_length=MAX_FILE_BYTES)

    @field_validator("entry")
    @classmethod
    def strip_nonblank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("entry must not be blank")
        return cleaned


def read_notepad(workspace: Path) -> str:
    target = resolve_workspace_path(workspace, NOTEPAD_PATH)
    if not target.exists():
        return ""
    return read_utf8_text(target, MAX_FILE_BYTES)


def _error(exc: Exception) -> str:
    message = str(exc)
    if isinstance(exc, OSError):
        message = "notepad persistence failed"
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


class NotepadReadTool(BaseTool):
    name: str = "notepad_read"
    description: str = "Read durable notes from the current session workspace."
    args_schema: ArgsSchema | None = NotepadReadInput
    workspace: Path

    def _run(self) -> str:
        try:
            return json.dumps(
                {"ok": True, "content": read_notepad(self.workspace)},
                ensure_ascii=False,
            )
        except (OSError, WorkspaceError) as exc:
            return _error(exc)


class NotepadAppendTool(BaseTool):
    name: str = "notepad_append"
    description: str = "Append one durable finding or decision to the session notepad."
    args_schema: ArgsSchema | None = NotepadAppendInput
    workspace: Path

    def _run(self, entry: str) -> str:
        try:
            existing = read_notepad(self.workspace)
            updated = f"{existing}{entry.strip()}\n"
            if utf8_size(updated) > MAX_FILE_BYTES:
                raise WorkspaceError(f"notepad exceeds the {MAX_FILE_BYTES}-byte limit")
            target = resolve_workspace_path(self.workspace, NOTEPAD_PATH)
            target.parent.mkdir(parents=True, exist_ok=True)
            target = resolve_workspace_path(self.workspace, NOTEPAD_PATH)
            atomic_write_utf8(target, updated)
            return json.dumps(
                {"ok": True, "characters": len(updated)},
                ensure_ascii=False,
            )
        except (OSError, WorkspaceError) as exc:
            return _error(exc)
