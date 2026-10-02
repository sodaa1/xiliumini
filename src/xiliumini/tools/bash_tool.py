from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langchain_core.tools.base import ArgsSchema
from pydantic import Field

from xiliumini.core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    classify_command_risk,
    normalize_approval_mode,
)
from xiliumini.errors import CommandExecutionError
from xiliumini.tools.command import SHELL_TOKENS, CommandInput, CommandResult, CommandTool
from xiliumini.tools.workspace import resolve_workspace_path

RISKY_ARGV_PREFIXES = (
    ("pip", "install"),
    ("python", "-m", "pip", "install"),
    ("uv", "add"),
    ("uv", "sync"),
    ("uv", "pip", "install"),
    ("npm", "install"),
    ("pnpm", "install"),
    ("yarn", "install"),
    ("yarn", "add"),
    ("curl",),
    ("wget",),
    ("uvicorn",),
    ("python", "-m", "http.server"),
)


class BashInput(CommandInput):
    phase: Literal["test_red", "test_green", "demo", "check", "other"] = "other"
    cwd: str = "."


class BashTool(CommandTool):
    name: str = "bash"
    description: str = (
        "Restricted argv runner without a shell. Supports python script.py, python -m pytest, "
        "python -m ruff check, python -m ruff format --check, python -m pyright, "
        "python -m compileall, and python -m pip check. cwd must be a workspace-relative "
        "directory. Approval-gated commands include pip/uv/npm/pnpm/yarn installation, "
        "curl/wget downloads, and uvicorn/python -m http.server development servers. "
        "inline requires a configured human approval handler; auto permits risky commands; "
        "deny rejects them. Shell operators remain forbidden. "
        "Tag test runs test_red/test_green and demos demo."
    )
    args_schema: ArgsSchema | None = BashInput
    approval_mode: str = "inline"
    approval_handler: Callable[[ApprovalRequest], ApprovalDecision] | None = Field(
        default=None,
        exclude=True,
    )

    @staticmethod
    def _command_text(argv: list[str]) -> str:
        return " ".join(argv)

    @staticmethod
    def _direct_risk_reason(argv: list[str]) -> str | None:
        for prefix in RISKY_ARGV_PREFIXES:
            if tuple(argv[: len(prefix)]) == prefix:
                return classify_command_risk(" ".join(prefix))
        return None

    @staticmethod
    def _approval_rejection(argv: list[str], message: str) -> CommandResult:
        return CommandResult(
            ok=False,
            argv=argv,
            stderr=f"command rejected: {message}",
            requires_approval=True,
        )

    def execute(self, argv: list[str], *, cwd: str = ".") -> CommandResult:
        submitted_argv = list(argv)
        command = self._command_text(submitted_argv)
        risk_reason = self._direct_risk_reason(submitted_argv)
        if risk_reason is None:
            return super().execute(submitted_argv, cwd=cwd)

        mode = normalize_approval_mode(self.approval_mode)
        if mode == "deny":
            return self._approval_rejection(submitted_argv, f"approval denied: {risk_reason}")

        if mode == "inline":
            if self.approval_handler is None:
                return self._approval_rejection(
                    submitted_argv,
                    "approval handler is not configured",
                )
            request = ApprovalRequest(
                id=f"approval-{uuid4().hex[:8]}",
                command=command,
                risk_reason=risk_reason,
            )
            try:
                decision = self.approval_handler(request)
                if not isinstance(decision, ApprovalDecision):
                    raise TypeError("approval handler returned an invalid decision")
            except Exception:
                return self._approval_rejection(submitted_argv, "approval handler failed")
            if decision.approved is not True:
                reason = decision.reason or risk_reason
                return self._approval_rejection(
                    submitted_argv,
                    f"approval denied: {reason}",
                )

        result = super().execute(submitted_argv, cwd=cwd)
        return result.model_copy(update={"requires_approval": True})

    def _resolve_command(self, argv: list[str], cwd: Path) -> list[str]:
        if any(token in SHELL_TOKENS for token in argv):
            raise CommandExecutionError("command rejected: shell operators are not allowed")
        if self._direct_risk_reason(argv) is not None:
            if argv[:1] == ["python"]:
                return [sys.executable, *argv[1:]]
            return argv.copy()
        if len(argv) < 3 or argv[:2] != ["python", "-m"]:
            return super()._resolve_command(argv, cwd)
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
