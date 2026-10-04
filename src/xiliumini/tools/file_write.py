from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from xiliumini.errors import WorkspaceError
from xiliumini.tools.workspace import (
    MAX_FILE_BYTES,
    atomic_write_utf8,
    ensure_workspace_file_mutable,
    resolve_workspace_path,
    utf8_size,
)


class FileWriteInput(BaseModel):
    path: str = Field(min_length=1)
    content: str = Field(max_length=1_000_000)


class FileWriteTool(BaseTool):
    name: str = "file_write"
    description: str = "Create or replace a UTF-8 file inside the session workspace."
    args_schema: ArgsSchema | None = FileWriteInput
    workspace: Path

    def _run(self, path: str, content: str) -> str:
        try:
            if utf8_size(content) > MAX_FILE_BYTES:
                raise WorkspaceError(f"content exceeds the {MAX_FILE_BYTES}-byte limit")
            target = resolve_workspace_path(self.workspace, path)
            ensure_workspace_file_mutable(self.workspace, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target = resolve_workspace_path(self.workspace, path)
            atomic_write_utf8(target, content)
            return f"Wrote {len(content)} characters to {path}"
        except (OSError, WorkspaceError) as exc:
            return f"Error: {exc}"
