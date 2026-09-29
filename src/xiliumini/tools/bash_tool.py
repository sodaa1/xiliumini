from __future__ import annotations

import json
import sys
from typing import Literal

from langchain_core.tools.base import ArgsSchema

from xiliumini.errors import CommandExecutionError
from xiliumini.tools.command import SHELL_TOKENS, CommandInput, CommandTool
from xiliumini.tools.workspace import resolve_workspace_path


class BashInput(CommandInput):
    phase: Literal["test_red", "test_green", "demo", "check", "other"] = "other"


class BashTool(CommandTool):
    name: str = "bash"
    description: str = (
        "BashTool: run argv without a shell in the workspace. Supports python script.py, "
        "python -m pytest, python -m ruff check, python -m ruff format --check, "
        "python -m pyright. Tag test runs test_red/test_green and demos demo."
    )
    args_schema: ArgsSchema | None = BashInput

    def _resolve_command(self, argv: list[str]) -> list[str]:
        if len(argv) < 3 or argv[:2] != ["python", "-m"] or argv[2] not in {"ruff", "pyright"}:
            return super()._resolve_command(argv)
        if any(token in SHELL_TOKENS for token in argv):
            raise CommandExecutionError("command rejected: shell operators are not allowed")
        module = argv[2]
        rest = argv[3:]
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
        for target in rest or ["."]:
            if target == ".":
                prefix.append(".")
                continue
            if target.startswith("-"):
                raise CommandExecutionError("command rejected: unsupported checker option")
            path = resolve_workspace_path(self.workspace, target)
            if not path.exists():
                raise CommandExecutionError("command rejected: checker target does not exist")
            prefix.append(path.relative_to(self.workspace.resolve()).as_posix())
        return prefix

    def _run(self, argv: list[str], phase: str = "other") -> str:
        return json.dumps({**self.execute(argv).model_dump(), "phase": phase})
