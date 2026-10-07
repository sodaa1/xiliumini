from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from xiliumini.automation.store import AutomationStore
from xiliumini.events import ErrorEvent, FinalEvent, VerifierEvent
from xiliumini.runtime import create_runtime
from xiliumini.tools.workspace import atomic_write_utf8


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str
    trace_id: str | None
    report_path: Path | None


class AutomationRunner:
    def __init__(
        self,
        store: AutomationStore,
        settings: Any,
        *,
        runtime_factory: Callable[..., Any] = create_runtime,
    ) -> None:
        self.store = store
        self.settings = settings
        self.runtime_factory = runtime_factory

    def run_once(self, task_id: str, *, scheduled_for: str | None = None) -> RunResult:
        task = self.store.get(task_id)
        if task is None:
            raise ValueError("automation task does not exist")
        run_id = str(uuid4())
        slot = scheduled_for or f"manual:{run_id}"
        if not self.store.claim_run(task_id, slot, run_id):
            raise ValueError("automation trigger was already claimed")

        trace_id: str | None = None
        answer = ""
        error = ""
        verifier_passed = False
        verifier_reason = "Verifier did not pass"
        runtime = None
        try:
            runtime = self.runtime_factory(
                self.settings,
                approval_mode="deny",
                trace_mode="full",
                actor_type="automation",
                policy_profile="background",
            )
            for event in runtime.stream(task.prompt, self.store.session_id(task_id)):
                if isinstance(event, ErrorEvent):
                    error = f"{event.code}: {event.message}"[:1000]
                elif isinstance(event, VerifierEvent):
                    verifier_passed = event.passed
                    verifier_reason = event.reason[:1000]
                elif isinstance(event, FinalEvent):
                    answer = event.text[:20_000]
            trace_id = getattr(runtime, "last_trace_id", None)
            if not error and not verifier_passed:
                error = verifier_reason
            if not error and not answer:
                error = "FinalEvent was not produced"
        except Exception as exc:
            if runtime is not None:
                trace_id = getattr(runtime, "last_trace_id", None)
            error = f"Automation runtime failed: {type(exc).__name__}"

        status = "failed" if error else "success"
        report = self.store.data_dir / "automation-reports" / task_id / f"{run_id}.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# {task.name}",
            "",
            f"Run ID: {run_id}",
            f"Scheduled for: {slot}",
            f"Completed at: {datetime.now(UTC).isoformat()}",
            f"Status: {status}",
            f"Trace ID: {trace_id or 'unavailable'}",
            "",
            answer if status == "success" else f"Error: {error}",
            "",
        ]
        atomic_write_utf8(report, "\n".join(lines))
        self.store.finish_run(
            run_id,
            status=status,
            trace_id=trace_id,
            result_path=str(report),
            error=error or None,
        )
        return RunResult(run_id, status, trace_id, report)
