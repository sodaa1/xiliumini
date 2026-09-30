from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from langchain_core.tools.base import ArgsSchema

from xiliumini.errors import CommandExecutionError
from xiliumini.tools.command import SHELL_TOKENS, CommandInput, CommandTool
from xiliumini.tools.workspace import resolve_workspace_path


class BashInput(CommandInput):
    phase: Literal["test_red", "test_green", "demo", "check", "other"] = "other"
    cwd: str = "."


class BashTool(CommandTool):
    name: str = "bash"
    description: str = (
        "Restricted argv runner without a shell. Supports python script.py, python -m pytest, "
        "python -m ruff check, python -m ruff format --check, python -m pyright, "
        "python -m compileall, and python -m pip check. cwd must be a workspace-relative "
        "directory. Tag test runs test_red/test_green and demos demo."
    )
    args_schema: ArgsSchema | None = BashInput

    def _resolve_command(self, argv: list[str], cwd: Path) -> list[str]:
        if len(argv) < 3 or argv[:2] != ["python", "-m"]:
            return super()._resolve_command(argv, cwd)
        if any(token in SHELL_TOKENS for token in argv):
            raise CommandExecutionError("command rejected: shell operators are not allowed")
        module = argv[2]
        rest = argv[3:]
        if module == "pip":
            if rest != ["check"]:
                raise CommandExecutionError("command rejected: only python -m pip check is allowed")
            return [sys.executable, "-m", "pip", "check"]
        if module == "compileall":
            prefix = [sys.executable, "-m", "compileall"]
            if rest[:1] == ["-q"]:
                prefix.append("-q")
                rest = rest[1:]
            return [*prefix, *self._resolve_targets(rest, cwd)]
        if module not in {"ruff", "pyright"}:
            return super()._resolve_command(argv, cwd)
        prefix = [sys.executable, "-m", module]
        if module == "ruff":
            if rest[:2] == ["format", "--check"]:
                prefix += rest[:2]
                rest = rest[2:]
            elif rest[:1] == ["check"]:
                prefix += ["check"]
                rest = rest[1:]
            else:
                raise CommandExecutionError("command rejected: only read-only Ruff checks allowed")
        return [*prefix, *self._resolve_targets(rest, cwd)]

    def _resolve_targets(self, targets: list[str], cwd: Path) -> list[str]:
        resolved: list[str] = []
        for target in targets or ["."]:
            if target == ".":
                resolved.append(".")
                continue
            if target.startswith("-"):
                raise CommandExecutionError("command rejected: unsupported checker option")
            path = resolve_workspace_path(cwd, target)
            if not path.exists():
                raise CommandExecutionError("command rejected: checker target does not exist")
            resolved.append(path.relative_to(cwd.resolve()).as_posix())
        return resolved

    def _run(self, argv: list[str], phase: str = "other", cwd: str = ".") -> str:
        return json.dumps({**self.execute(argv, cwd=cwd).model_dump(), "phase": phase})
