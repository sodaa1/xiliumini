from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
import time
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from xiliumini.tools.workspace import resolve_workspace_path

_running_hook: ContextVar[bool] = ContextVar("xiliumini_hook_running", default=False)
DEFAULT_HOOKS: list[dict[str, Any]] = [
    {
        "id": "python-ruff-format-check",
        "event": "tool.after",
        "match": {
            "tools": ["file_write", "file_edit"],
            "path_glob": "**/*.py",
            "success_only": True,
        },
        "action": {"type": "builtin.ruff_format_check", "target": "changed_file"},
        "on_failure": "report",
    }
]


class HookEngine:
    def __init__(self, config_path: Path, *, runner: Callable[..., Any] | None = None):
        self.config_path = config_path
        self.runner = runner or subprocess.run
        if config_path.exists():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("hooks"), list):
                raise ValueError("invalid hook configuration")
            self.hooks = raw["hooks"]
        else:
            self.hooks = DEFAULT_HOOKS
        for hook in self.hooks:
            if (
                not isinstance(hook, dict)
                or hook.get("event")
                not in {"tool.before", "tool.after", "run.start", "run.success", "run.failure"}
                or hook.get("action", {}).get("type") != "builtin.ruff_format_check"
                or hook.get("action", {}).get("target")
                != (
                    "workspace" if str(hook.get("event", "")).startswith("run.") else "changed_file"
                )
                or hook.get("on_failure", "report") not in {"report", "block"}
            ):
                raise ValueError("unsupported hook configuration")

    @staticmethod
    def _matches(path: str, pattern: str) -> bool:
        normalized = path.replace("\\", "/")
        return fnmatch.fnmatchcase(normalized, pattern) or (
            pattern.startswith("**/") and fnmatch.fnmatchcase(normalized, pattern[3:])
        )

    def _run_hook(
        self,
        hook: dict[str, Any],
        *,
        tool_name: str,
        path: str,
        context: Any,
        gateway: Any,
    ) -> dict[str, Any]:
        started = time.monotonic()
        event: dict[str, Any] = {
            "type": "hook_result",
            "hook_id": hook["id"],
            "trigger_tool": tool_name,
            "path": path,
        }
        token = _running_hook.set(True)
        try:
            if not gateway.authorize_hook(context, hook["id"]):
                event.update(status="denied", exit_code=None)
            else:
                workspace_target = hook["action"]["target"] == "workspace"
                target = (
                    context.workspace.resolve()
                    if workspace_target
                    else resolve_workspace_path(context.workspace, path)
                )
                if not (target.is_dir() if workspace_target else target.is_file()):
                    event.update(status="failed", exit_code=None)
                else:
                    relative = (
                        "."
                        if workspace_target
                        else target.relative_to(context.workspace.resolve()).as_posix()
                    )
                    completed = self.runner(
                        [sys.executable, "-m", "ruff", "format", "--check", relative],
                        cwd=context.workspace,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    event.update(
                        status="passed" if completed.returncode == 0 else "failed",
                        exit_code=completed.returncode,
                    )
        except (OSError, ValueError, subprocess.TimeoutExpired):
            event.update(status="failed", exit_code=None)
        finally:
            _running_hook.reset(token)
        event["duration_ms"] = int((time.monotonic() - started) * 1000)
        if event["status"] != "passed" and hook.get("on_failure") == "block":
            context.hook_failures.append(hook["id"])
        return event

    def lifecycle(self, event_name: str, *, context: Any, gateway: Any) -> list[dict[str, Any]]:
        if event_name not in {"run.start", "run.success", "run.failure"}:
            raise ValueError("invalid run lifecycle event")
        events = [{"type": "hook_lifecycle", "event": event_name, "run_id": context.run_id}]
        for hook in self.hooks:
            if hook["event"] == event_name:
                events.append(
                    self._run_hook(
                        hook, tool_name=event_name, path=".", context=context, gateway=gateway
                    )
                )
        return events

    def before(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any,
        gateway: Any,
    ) -> tuple[list[dict[str, Any]], bool]:
        if _running_hook.get():
            return [], False
        path = arguments.get("path")
        if not isinstance(path, str) or not path.lower().endswith(".py"):
            return [], False
        results = []
        blocked = False
        for hook in self.hooks:
            if hook["event"] != "tool.before":
                continue
            match = hook.get("match", {})
            if tool_name not in match.get("tools", []) or not self._matches(
                path, match.get("path_glob", "")
            ):
                continue
            event = self._run_hook(
                hook, tool_name=tool_name, path=path, context=context, gateway=gateway
            )
            results.append(event)
            blocked = blocked or (event["status"] != "passed" and hook.get("on_failure") == "block")
        return results, blocked

    def after(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        output: Any,
        context: Any,
        gateway: Any,
    ) -> list[dict[str, Any]]:
        if _running_hook.get() or tool_name not in {"file_write", "file_edit"}:
            return []
        if not isinstance(output, str) or not output.startswith(
            "Wrote " if tool_name == "file_write" else "Updated "
        ):
            return []
        path = arguments.get("path")
        if not isinstance(path, str) or not path.lower().endswith(".py"):
            return []
        results = []
        for hook in self.hooks:
            if hook["event"] != "tool.after":
                continue
            match = hook.get("match", {})
            if tool_name not in match.get("tools", []) or not self._matches(
                path, match.get("path_glob", "")
            ):
                continue
            results.append(
                self._run_hook(
                    hook, tool_name=tool_name, path=path, context=context, gateway=gateway
                )
            )
        return results
