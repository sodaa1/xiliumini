from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from xiliumini.errors import WorkspaceError
from xiliumini.tools.workspace import MAX_FILE_BYTES, read_utf8_text, resolve_workspace_path


class FileReadInput(BaseModel):
    path: str = Field(min_length=1)
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=200, ge=1, le=2_000)


class FileReadTool(BaseTool):
    name: str = "file_read"
    description: str = "Read a bounded range of lines from a UTF-8 file in the session workspace."
    args_schema: ArgsSchema | None = FileReadInput
    workspace: Path

    def _run(self, path: str, offset: int = 1, limit: int = 200) -> str:
        try:
            target = resolve_workspace_path(self.workspace, path)
            text = read_utf8_text(target, MAX_FILE_BYTES)
            lines = text.splitlines()
            selected = lines[offset - 1 : offset - 1 + limit]
            numbered = "\n".join(
                f"{number}: {line}" for number, line in enumerate(selected, start=offset)
            )
            if not selected:
                return f"File: {path}\nLines: none/{len(lines)}"
            end = offset + len(selected) - 1
            return f"File: {path}\nLines: {offset}-{end}/{len(lines)}\n{numbered}"
        except (OSError, WorkspaceError) as exc:
            return f"Error: {exc}"
