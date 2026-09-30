from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from xiliumini.errors import CommandExecutionError, WorkspaceError
from xiliumini.tools.workspace import resolve_workspace_path

SHELL_TOKENS = {"|", "||", "&&", ";", "<", ">", ">>"}
PYTEST_FLAGS = {
    "-q",
    "--quiet",
    "-v",
    "--verbose",
    "-x",
    "--exitfirst",
    "--disable-warnings",
    "--strict-markers",
}
PYTEST_VALUE_OPTIONS = {"-k", "-m", "--maxfail", "--tb"}


class CommandInput(BaseModel):
    argv: list[str] = Field(min_length=1)


class CommandResult(BaseModel):
    ok: bool
    argv: list[str]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False


class CommandTool(BaseTool):
    name: str = "command"
    description: str = "Run a workspace Python script or python -m pytest without a shell."
    args_schema: ArgsSchema | None = CommandInput
    workspace: Path
    timeout_seconds: float = 30
    max_output_bytes: int = 20_000

    def _resolve_pytest_args(self, args: list[str], cwd: Path) -> list[str]:
        resolved: list[str] = []
        index = 0
        while index < len(args):
            value = args[index]
            if value in PYTEST_FLAGS:
                resolved.append(value)
            elif value in PYTEST_VALUE_OPTIONS:
                if index + 1 >= len(args):
                    raise CommandExecutionError(
                        f"command rejected: pytest option {value} requires a value"
                    )
                option_value = args[index + 1]
                self._validate_pytest_option(value, option_value)
                resolved.extend([value, option_value])
                index += 1
            elif value.startswith("--maxfail="):
                self._validate_pytest_option("--maxfail", value.partition("=")[2])
                resolved.append(value)
            elif value.startswith("--tb="):
                self._validate_pytest_option("--tb", value.partition("=")[2])
                resolved.append(value)
            elif value.startswith("-"):
                raise CommandExecutionError(
                    f"command rejected: pytest option is not allowed: {value}"
                )
            else:
                path_text, separator, node_id = value.partition("::")
                target = resolve_workspace_path(cwd, path_text)
                if not target.exists():
                    raise CommandExecutionError(
                        "command rejected: pytest target must exist in the workspace"
                    )
                relative = target.relative_to(cwd.resolve()).as_posix()
                resolved.append(f"{relative}{separator}{node_id}")
            index += 1
        return resolved

    @staticmethod
    def _validate_pytest_option(option: str, value: str) -> None:
        if not value:
            raise CommandExecutionError(
                f"command rejected: pytest option {option} requires a value"
            )
        if option == "--maxfail" and not value.isdigit():
            raise CommandExecutionError("command rejected: --maxfail must be an integer")
        if option == "--tb" and value not in {
            "auto",
            "long",
            "short",
            "line",
            "native",
            "no",
        }:
            raise CommandExecutionError("command rejected: unsupported --tb value")

    def _resolve_cwd(self, cwd: str) -> Path:
        directory = resolve_workspace_path(self.workspace, cwd, allow_root=True)
        if not directory.is_dir():
            raise CommandExecutionError(
                "command rejected: working directory must be an existing workspace directory"
            )
        return directory

    def _resolve_command(self, argv: list[str], cwd: Path) -> list[str]:
        if not argv or argv[0] != "python":
            raise CommandExecutionError("command rejected: only python is allowed")
        if any(token in SHELL_TOKENS for token in argv):
            raise CommandExecutionError("command rejected: shell operators are not allowed")
        if len(argv) >= 3 and argv[1:3] == ["-m", "pytest"]:
            return [sys.executable, "-m", "pytest", *self._resolve_pytest_args(argv[3:], cwd)]
        if len(argv) < 2 or argv[1].startswith("-"):
            raise CommandExecutionError(
                "command rejected: expected a Python script or python -m pytest"
            )
        script = resolve_workspace_path(cwd, argv[1])
        if script.suffix.lower() != ".py" or not script.is_file():
            raise CommandExecutionError(
                "command rejected: script must be an existing workspace .py file"
            )
        return [sys.executable, str(script), *argv[2:]]

    @staticmethod
    def _decode(raw: bytes, budget: int) -> str:
        data = raw[:budget]
        text = data.decode("utf-8", errors="ignore").replace("\r\n", "\n")
        return text

    def _bounded_output(self, stdout: bytes, stderr: bytes) -> tuple[str, str, bool]:
        total = len(stdout) + len(stderr)
        if total <= self.max_output_bytes:
            return self._decode(stdout, len(stdout)), self._decode(stderr, len(stderr)), False
        stdout_budget = min(len(stdout), self.max_output_bytes // 2)
        stderr_budget = min(len(stderr), self.max_output_bytes - stdout_budget)
        remaining = self.max_output_bytes - stdout_budget - stderr_budget
        stderr_budget += min(remaining, len(stderr) - stderr_budget)
        remaining = self.max_output_bytes - stdout_budget - stderr_budget
        stdout_budget += min(remaining, len(stdout) - stdout_budget)
        return (
            self._decode(stdout, stdout_budget),
            self._decode(stderr, stderr_budget),
            True,
        )

    def execute(self, argv: list[str], *, cwd: str = ".") -> CommandResult:
        try:
            execution_cwd = self._resolve_cwd(cwd)
            command = self._resolve_command(argv, execution_cwd)
            completed = subprocess.run(
                command,
                cwd=execution_cwd,
                shell=False,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            stdout, stderr, truncated = self._bounded_output(completed.stdout, completed.stderr)
            root = str(self.workspace.resolve())
            return CommandResult(
                ok=completed.returncode == 0,
                argv=argv,
                exit_code=completed.returncode,
                stdout=stdout.replace(root, "<workspace>"),
                stderr=stderr.replace(root, "<workspace>"),
                truncated=truncated,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                ok=False,
                argv=argv,
                stderr="command timed out",
                timed_out=True,
            )
        except WorkspaceError as exc:
            return CommandResult(
                ok=False,
                argv=argv,
                stderr=f"command rejected: {exc}",
            )
        except (CommandExecutionError, OSError) as exc:
            return CommandResult(ok=False, argv=argv, stderr=str(exc))

    def _run(self, argv: list[str]) -> str:
        return self.execute(argv).model_dump_json()
