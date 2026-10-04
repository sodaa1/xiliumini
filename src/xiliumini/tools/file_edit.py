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
    read_utf8_text,
    resolve_workspace_path,
    utf8_size,
)


class FileEditInput(BaseModel):
    path: str = Field(min_length=1)
    old_string: str = Field(min_length=1)
    new_string: str = Field(default="", max_length=1_000_000)
    replace_all: bool = False


class FileEditTool(BaseTool):
    name: str = "file_edit"
    description: str = "Replace exact text in a UTF-8 file inside the session workspace."
    args_schema: ArgsSchema | None = FileEditInput
    workspace: Path

    def _run(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        try:
            target = resolve_workspace_path(self.workspace, path)
            ensure_workspace_file_mutable(self.workspace, target)
            original = read_utf8_text(target, MAX_FILE_BYTES)
            matches = original.count(old_string)
            if matches == 0:
                return "Error: old_string was not found"
            if matches > 1 and not replace_all:
                return f"Error: found {matches} matches; add context or set replace_all"

            replacements = matches if replace_all else 1
            updated_size = (
                utf8_size(original)
                - replacements * utf8_size(old_string)
                + replacements * utf8_size(new_string)
            )
            if updated_size > MAX_FILE_BYTES:
                raise WorkspaceError(f"result exceeds the {MAX_FILE_BYTES}-byte limit")
            updated = original.replace(old_string, new_string, -1 if replace_all else 1)
            atomic_write_utf8(target, updated)
            return f"Updated {path} with {replacements} replacements"
        except (OSError, WorkspaceError) as exc:
            return f"Error: {exc}"
