from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path, PureWindowsPath

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from xiliumini.errors import WorkspaceError
from xiliumini.tools.workspace import MAX_FILE_BYTES, read_utf8_text, resolve_workspace_path

IGNORED_PARTS = {".git", ".xiliumini", ".venv", "__pycache__"}
MAX_LINE_CHARS = 500


class GrepInput(BaseModel):
    pattern: str = Field(min_length=1)
    path: str = "."
    glob: str | None = None
    max_results: int = Field(default=200, ge=1, le=1_000)


def _validate_glob(pattern: str | None) -> str:
    if pattern is None:
        return "*"
    windows_pattern = PureWindowsPath(pattern)
    is_unc = pattern.startswith(("//", "\\\\"))
    if Path(pattern).is_absolute() or windows_pattern.drive or is_unc:
        raise WorkspaceError("glob must stay inside the workspace")
    if ".." in windows_pattern.parts:
        raise WorkspaceError("glob must stay inside the workspace")
    return pattern


def _candidate_files(root: Path, glob: str | None) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    pattern = _validate_glob(glob)
    for candidate in root.rglob(pattern):
        if candidate.is_file() and not candidate.is_symlink():
            yield candidate


class GrepTool(BaseTool):
    name: str = "grep"
    description: str = "Search UTF-8 files inside the session workspace with a regular expression."
    args_schema: ArgsSchema | None = GrepInput
    workspace: Path

    def _run(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        max_results: int = 200,
    ) -> str:
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regular expression: {exc.msg}"

        try:
            workspace = self.workspace.resolve(strict=True)
            root = resolve_workspace_path(workspace, path, allow_root=True)
            matches: list[str] = []
            truncated = False
            for candidate in _candidate_files(root, glob):
                relative = candidate.relative_to(workspace)
                if any(part in IGNORED_PARTS for part in relative.parts):
                    continue
                try:
                    text = read_utf8_text(candidate, MAX_FILE_BYTES)
                except WorkspaceError:
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if not expression.search(line):
                        continue
                    if len(matches) == max_results:
                        truncated = True
                        break
                    matches.append(f"{relative.as_posix()}:{line_number}:{line[:MAX_LINE_CHARS]}")
                if truncated:
                    break
            if not matches:
                return "No matches found"
            suffix = "\n[results truncated]" if truncated else ""
            return "\n".join(matches) + suffix
        except (OSError, WorkspaceError, ValueError) as exc:
            return f"Error: {exc}"
