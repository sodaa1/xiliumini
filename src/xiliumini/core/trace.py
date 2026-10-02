"""Run-local execution trace persistence."""

from __future__ import annotations

import re
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from xiliumini.core.harness_io import (
    append_jsonl,
    normalize_trace_mode,
    sanitize_for_persistence,
    write_json_atomic,
)
from xiliumini.errors import TraceError
from xiliumini.tools.workspace import atomic_write_utf8

_KEY_EVENTS = {
    "resume",
    "error",
    "handoff",
    "approval",
    "approval_request",
    "approval_decision",
    "checkpoint_saved",
}
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class TraceRecorder:
    def __init__(self, runtime: Any, task: str = ""):
        self.workspace = Path(runtime.workspace)
        self.mode = normalize_trace_mode(runtime.trace_mode)
        configured = getattr(runtime, "trace_id", None)
        if isinstance(configured, str) and not configured.strip():
            configured = None
        self.trace_id = f"trace-{uuid4().hex}" if configured is None else configured
        if (
            not isinstance(self.trace_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.trace_id)
            or self.trace_id.upper() in _RESERVED
        ):
            raise TraceError("Invalid trace ID.")
        self.root = self.workspace / ".xiliumini" / "traces" / self.trace_id
        self.task = task
        self.session_id = getattr(runtime, "session_id", None)
        self._state = "new"
        self._summary: dict[str, Any] | None = None
        self._end_request: dict[str, Any] | None = None
        self._head: list[dict[str, Any]] = []
        self._tail: deque[dict[str, Any]] = deque(maxlen=80)
        self._sequence = 0
        self.node_visits: dict[str, int] = {}
        self.tool_calls = 0
        self.failed_tool_calls = 0
        self.approval_count = 0
        self.checkpoint_count = 0
        self.handoff_count = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def _validate_paths(self) -> None:
        root = self.workspace.resolve()
        for path in (
            self.workspace / ".xiliumini",
            self.workspace / ".xiliumini" / "traces",
            self.root,
            self.root / "events.jsonl",
            self.root / "trace.json",
            self.root / "timeline.md",
        ):
            if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
                raise ValueError("unsafe trace path")

    def _record(self, kind: str, payload: Any) -> None:
        if self._state not in {"starting", "running", "ending"}:
            raise TraceError("Trace is not running.")
        try:
            self._validate_paths()
            item = {
                "sequence": self._sequence + 1,
                "timestamp": datetime.now(UTC).isoformat(),
                "type": sanitize_for_persistence(kind, self.workspace, max_text=200),
                "payload": sanitize_for_persistence(payload, self.workspace, max_text=2000),
            }
            append_jsonl(self.root / "events.jsonl", item)
            self._sequence += 1
            if len(self._head) < 20:
                self._head.append(item)
            else:
                self._tail.append(item)
        except Exception:
            self._state = "failed"
            raise TraceError("Unable to record execution trace.") from None

    def start(self, inputs: Any, *, resumed: bool = False, resume_event: Any = None) -> None:
        if not self.enabled:
            return
        if self._state in {"running", "ended"}:
            return
        if self._state != "new":
            raise TraceError("Trace is not running.")
        try:
            self._validate_paths()
            control = self.workspace / ".xiliumini"
            for path in (control, control / "traces"):
                if path.is_symlink() or not path.resolve().is_relative_to(self.workspace.resolve()):
                    raise ValueError("unsafe trace directory")
                path.mkdir(exist_ok=True)
            self.root.mkdir(exist_ok=False)
            self._started_at = datetime.now(UTC).isoformat()
            self._start_clock = time.monotonic()
            self._state = "starting"
            self._record(
                "run_start",
                {
                    "task": self.task,
                    "session_id": self.session_id,
                    "mode": self.mode,
                    "resumed": resumed,
                    "resume_event": resume_event,
                },
            )
            self._state = "running"
        except Exception:
            self._state = "failed"
            raise TraceError("Unable to start execution trace.") from None

    def record_custom_event(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._state != "running":
            raise TraceError("Trace is not running.")
        try:
            if not isinstance(event, dict):
                raise ValueError("invalid event")
            kind = event.get("type", "custom")
            if not isinstance(kind, str):
                raise ValueError("invalid event type")
            if (
                self.mode == "full"
                or kind in _KEY_EVENTS
                or event.get("requires_approval") is True
                or event.get("ok") is False
            ):
                self._record(kind, event)
            if kind == "tool_call":
                self.tool_calls += 1
            elif kind == "tool_result":
                self.failed_tool_calls += int(event.get("ok") is False)
                self.approval_count += int(event.get("requires_approval") is True)
            elif kind == "handoff":
                self.handoff_count += 1
            elif kind == "checkpoint_saved":
                self.checkpoint_count += 1
        except Exception:
            self._state = "failed"
            raise TraceError("Unable to record execution trace.") from None

    def record_graph_update(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._state != "running":
            raise TraceError("Trace is not running.")
        try:
            self._record("graph_update", event)
            for node in event:
                self.node_visits[node] = self.node_visits.get(node, 0) + 1
        except Exception:
            self._state = "failed"
            raise TraceError("Unable to record execution trace.") from None

    def end(
        self, *, status: str, latest_node: str | None, final_state: Any
    ) -> dict[str, Any] | None:
        if not self.enabled or self._state == "ended":
            return self._summary
        if self._state != "running":
            raise TraceError("Trace is not running.")
        self._state = "ending"
        try:
            ended_at = datetime.now(UTC).isoformat()
            duration_ms = max(0, int((time.monotonic() - self._start_clock) * 1000))
            self._end_request = {
                "status": status,
                "latest_node": latest_node,
                "final_state": sanitize_for_persistence(final_state, self.workspace, max_text=2000),
                "ended_at": ended_at,
            }
            self._record("run_end", {"status": status, "latest_node": latest_node})
            self._validate_paths()
            summary = {
                "trace_id": self.trace_id,
                "task": self.task,
                "status": status,
                "started_at": self._started_at,
                "ended_at": ended_at,
                "duration_ms": duration_ms,
                "node_visits": self.node_visits,
                "tool_calls": self.tool_calls,
                "failed_tool_calls": self.failed_tool_calls,
                "approval_count": self.approval_count,
                "checkpoint_count": self.checkpoint_count,
                "handoff_count": self.handoff_count,
                "timeline_omitted": max(0, self._sequence - 100),
            }
            safe_summary = sanitize_for_persistence(summary, self.workspace, max_text=2000)
            safe_summary.update(timeline_head=self._head, timeline_tail=list(self._tail))
            write_json_atomic(self.root / "trace.json", safe_summary)
            lines = [
                "# Execution timeline",
                "",
                f"Task: {safe_summary['task']}",
                f"Status: {safe_summary['status']}",
                "",
            ]
            for item in self._head + list(self._tail):
                payload = item["payload"]
                detail = ""
                if item["type"] == "graph_update" and isinstance(payload, dict):
                    detail = " nodes=" + ", ".join(payload)
                lines.append(f"{item['sequence']} {item['timestamp']} {item['type']}{detail}")
            if summary["timeline_omitted"]:
                lines.append(f"\nOmitted events: {summary['timeline_omitted']}")
            atomic_write_utf8(self.root / "timeline.md", "\n".join(lines) + "\n")
            self._summary = safe_summary
            self._state = "ended"
            return self._summary
        except Exception:
            self._state = "failed"
            raise TraceError("Unable to finish execution trace.") from None
