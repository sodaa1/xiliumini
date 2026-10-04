from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Collapsible, Header, Input, Static

from xiliumini.cli.tui.approval import ApprovalGate, ApprovalModal
from xiliumini.cli.tui.logo import LOGO_FRAME_COUNT, render_logo
from xiliumini.core.agent import stream_session_events
from xiliumini.core.approval import ApprovalDecision, ApprovalRequest
from xiliumini.core.session import SESSION_ROOT, load_or_create_session

StreamFactory = Callable[..., Iterator[dict[str, Any]]]
TimeSource = Callable[[], float]
_MAX_EVENT_TEXT = 800


class AgentEventMessage(Message):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__()
        self.payload = payload


class ApprovalRequestedMessage(Message):
    def __init__(self, request: ApprovalRequest, workspace: Path, gate: ApprovalGate) -> None:
        super().__init__()
        self.request = request
        self.workspace = workspace
        self.gate = gate


class TurnFinishedMessage(Message):
    def __init__(self, error: str | None = None) -> None:
        super().__init__()
        self.error = error


def _bounded(value: Any, limit: int = _MAX_EVENT_TEXT) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    marker = "… [truncated]"
    return text[: limit - len(marker)] + marker


def _event_target(details: dict[str, Any]) -> str:
    args = details.get("args")
    if not isinstance(args, dict):
        return ""
    for key in ("path", "query", "cwd"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            target = _bounded(" ".join(value.splitlines()), 160)
            return f'"{target}"' if key == "query" else target
    return ""


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}分钟{remaining_seconds}秒"
    return f"{remaining_seconds}秒"


class ConversationTurn(Vertical):
    """One user turn with private execution details and a visible result."""

    def __init__(self, task: str, *, started_at: float) -> None:
        self.started_at = started_at
        self._thought = Text()
        self.thought_content = Static("", classes="turn-reasoning-content", markup=False)
        self.thought_scroll = VerticalScroll(self.thought_content, classes="turn-reasoning-scroll")
        self.reasoning = Collapsible(
            self.thought_scroll,
            title="思考过程 · 运行中…",
            collapsed=True,
            classes="turn-thinking",
        )
        self.answer = Static("", classes="turn-answer empty", markup=False)
        self._result = Text()
        super().__init__(
            Static(Text(f"💬 You: {task}", style="bold #d8e7f3"), classes="turn-user"),
            self.reasoning,
            self.answer,
            classes="conversation-turn",
        )

    def write_thought(self, text: str, *, style: str = "") -> None:
        if self._thought:
            self._thought.append("\n")
        self._thought.append(text, style=style)
        self.thought_content.update(self._thought)

    def write_result(self, text: str, *, style: str = "") -> None:
        if self._result:
            self._result.append("\n")
        self._result.append(text, style=style)
        self.answer.update(self._result)
        self.answer.remove_class("empty")

    def finish(self, finished_at: float) -> None:
        elapsed = _format_elapsed(finished_at - self.started_at)
        self.reasoning.title = f"思考过程 · 用时 {elapsed}"


class MokioClawTuiApp(App[None]):
    """Full-screen multi-turn interface for xiliumini."""

    CSS_PATH = "styles.tcss"
    TITLE = "xiliumini"

    def __init__(
        self,
        *,
        session_id: str,
        session_workspace: Path,
        max_attempts: int = 3,
        approval_mode: str = "inline",
        checkpoint_mode: str = "light",
        trace_mode: str = "on",
        stream_factory: StreamFactory = stream_session_events,
        animate_logo: bool = True,
        time_source: TimeSource = monotonic,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.session_workspace = session_workspace.resolve()
        self.execution_workspace = (
            self.session_workspace / "workspaces" / self.session_id
        ).resolve()
        self.max_attempts = max_attempts
        self.approval_mode = approval_mode
        self.checkpoint_mode = checkpoint_mode
        self.trace_mode = trace_mode
        self.stream_factory = stream_factory
        self.animate_logo = animate_logo
        self.time_source = time_source
        self._logo_frame = 0
        self._logo_timer: Timer | None = None
        self._busy = False
        self._shutting_down = False
        self._approval_lock = Lock()
        self._pending_gates: set[ApprovalGate] = set()
        self._active_turn: ConversationTurn | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="main"):
            with Vertical(id="brand-row"):
                yield Static(
                    render_logo(frame=0 if self.animate_logo else None),
                    id="logo",
                )
                yield Static(
                    f"session: {self.session_id[:8]}  •  idle",
                    id="status-bar",
                    markup=False,
                )
            yield VerticalScroll(id="conversation")
            yield Input(placeholder="💬 Input", id="prompt")

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()
        self.query_one("#logo", Static).update(
            render_logo(
                width=max(0, self.size.width - 4),
                frame=0 if self.animate_logo else None,
            )
        )
        if self.animate_logo:
            self._logo_timer = self.set_interval(0.16, self._advance_logo)

    def _advance_logo(self) -> None:
        self._logo_frame += 1
        logo = self.query_one("#logo", Static)
        if self._logo_frame >= LOGO_FRAME_COUNT:
            if self._logo_timer is not None:
                self._logo_timer.stop()
            logo.update(render_logo(width=max(0, self.size.width - 4), frame=None))
            return
        logo.update(
            render_logo(
                width=max(0, self.size.width - 4),
                frame=self._logo_frame,
            )
        )

    def _set_status(self, status: str) -> None:
        self.query_one("#status-bar", Static).update(
            Text(f"session: {self.session_id[:8]}  •  {status}")
        )

    def _start_turn(self, task: str) -> ConversationTurn:
        turn = ConversationTurn(task, started_at=self.time_source())
        self.query_one("#conversation", VerticalScroll).mount(turn)
        self._active_turn = turn
        return turn

    def _current_turn(self) -> ConversationTurn:
        if self._active_turn is None:
            return self._start_turn("Runtime event")
        return self._active_turn

    def _write_event(self, text: str, *, style: str = "") -> None:
        self._current_turn().write_thought(text, style=style)

    def _write_result(self, text: str, *, style: str = "") -> None:
        self._current_turn().write_result(text, style=style)

    def _scroll_to_latest(self) -> None:
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    def _finish_turn(self) -> None:
        if self._active_turn is not None:
            self._active_turn.finish(self.time_source())
            self._active_turn = None
        self._busy = False
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        self._set_status("idle")

    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        payload = message.payload
        event = payload.get("event")
        if not isinstance(event, dict):
            self._write_event("• Invalid runtime event", style="yellow")
            return
        if payload.get("type") == "custom_event":
            self._render_custom_event(event)
        else:
            self._render_graph_event(event)
        self._scroll_to_latest()

    def _render_graph_event(self, event: dict[str, Any]) -> None:
        todos = event.get("todos")
        summary = event.get("summary")
        attempt = event.get("attempt")
        if isinstance(todos, list) and isinstance(summary, str) and isinstance(attempt, int):
            icons = {
                "completed": "✅",
                "in_progress": "🔄",
                "pending": "⬜",
                "blocked": "⛔",
            }
            lines = [f"[Plan] {_bounded(summary, 240)} · attempt {attempt}"]
            for todo in todos:
                if not isinstance(todo, dict):
                    continue
                content = _bounded(todo.get("content", "Untitled task"), 240)
                status_value = todo.get("status")
                status = status_value if isinstance(status_value, str) else "pending"
                lines.append(f"{icons.get(status, '⬜')} {content}")
            self._write_event("\n".join(lines), style="#a9bed0")
            return
        passed = event.get("passed")
        reason = event.get("reason")
        if isinstance(passed, bool) and isinstance(reason, str):
            icon = "✅" if passed else "❌"
            self._write_event(f"{icon} Verifier: {_bounded(reason)}")
            return
        answer = event.get("text")
        if isinstance(answer, str):
            self._write_result(f"📝 Final: {answer}", style="bold #d8e7f3")
            return
        code = event.get("code")
        error_message = event.get("message")
        if isinstance(code, str) and isinstance(error_message, str):
            self._write_result(
                f"⚠ Error [{_bounded(code, 80)}]: {_bounded(error_message)}",
                style="bold red",
            )
            return
        self._write_event(f"• graph_event: {_bounded(event)}", style="yellow")

    def _render_custom_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event_type")
        kind = event_type if isinstance(event_type, str) else "progress"
        details = event.get("details")
        details = details if isinstance(details, dict) else {}
        message = _bounded(event.get("message", ""))
        tool_value = details.get("tool")
        tool = tool_value if isinstance(tool_value, str) else "tool"
        target = _event_target(details)
        target_suffix = f" → {target}" if target else ""
        if kind == "tool_call":
            self._write_event(f"🔧 {tool}{target_suffix}", style="#20d6d0")
        elif kind == "tool_result":
            icon = "✅" if details.get("ok") is True else "❌"
            self._write_event(f"{icon} {tool}: {message}")
        elif kind == "search_results":
            self._write_event(f"🔍 {tool}{target_suffix}: {message}", style="#2796ff")
        elif kind == "handoff":
            target_value = details.get("agent")
            target = target_value if isinstance(target_value, str) else "unknown"
            stage_value = event.get("stage")
            stage = stage_value if isinstance(stage_value, str) else "planner"
            ok = details.get("ok")
            outcome = "✅" if ok is True else "❌" if ok is False else "•"
            summary_value = details.get("summary")
            summary = (
                f" {_bounded(summary_value, 240)}"
                if isinstance(summary_value, str) and summary_value
                else ""
            )
            self._write_event(
                f"🔄 Handoff: {stage} → {target} · {outcome}{summary}",
                style="#ffb36b",
            )
        elif kind == "checkpoint_saved":
            status_value = details.get("status")
            node_value = details.get("latest_node")
            status = status_value if isinstance(status_value, str) else "saved"
            node = node_value if isinstance(node_value, str) else "unknown"
            self._write_event(f"💾 Checkpoint: {status} · {node}", style="#7890a8")
        else:
            self._write_event(f"• {kind}: {message}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        task = event.value.strip()
        event.input.value = ""
        if not task:
            return
        if self._busy:
            self.notify("A session turn is already running.", severity="warning")
            return
        if self._logo_timer is not None:
            self._logo_timer.stop()
        self.query_one("#logo", Static).add_class("collapsed")
        self.query_one("#brand-row", Vertical).add_class("conversation-started")
        self._busy = True
        event.input.disabled = True
        self._set_status("running")
        self._start_turn(_bounded(task))
        self._scroll_to_latest()
        self.call_after_refresh(self._run_turn, task)

    def _request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Bridge a blocking tool approval request to the Textual UI thread."""
        gate = ApprovalGate()
        with self._approval_lock:
            if self._shutting_down:
                return ApprovalDecision(False, "Denied because the TUI closed.")
            self._pending_gates.add(gate)

        try:
            self.post_message(ApprovalRequestedMessage(request, self.execution_workspace, gate))
            approved = gate.wait()
        finally:
            with self._approval_lock:
                self._pending_gates.discard(gate)

        if approved:
            return ApprovalDecision(True, "Approved by user.")
        reason = "Denied because the TUI closed." if self._shutting_down else "Denied by user."
        return ApprovalDecision(False, reason)

    def on_approval_requested_message(self, message: ApprovalRequestedMessage) -> None:
        if self._shutting_down:
            message.gate.resolve(approved=False)
            return

        self._set_status("approval")

        def resolve(decision: bool | None) -> None:
            message.gate.resolve(approved=decision is True)
            if self._busy and not self._shutting_down:
                self._set_status("running")

        self.push_screen(
            ApprovalModal(request=message.request, workspace=message.workspace),
            resolve,
        )

    @work(thread=True, exclusive=True, group="session-turn", exit_on_error=False)
    def _run_turn(self, task: str) -> None:
        events: Iterator[dict[str, Any]] | None = None
        error: str | None = None
        try:
            events = self.stream_factory(
                task,
                session_workspace=self.session_workspace,
                max_attempts=self.max_attempts,
                approval_mode=self.approval_mode,
                approval_handler=(
                    self._request_approval if self.approval_mode == "inline" else None
                ),
                checkpoint_mode=self.checkpoint_mode,
                trace_mode=self.trace_mode,
            )
            for payload in events:
                if self._shutting_down:
                    break
                self.post_message(AgentEventMessage(payload))
        except Exception:
            error = "Session turn failed."
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    error = "Session turn failed."
            if not self._shutting_down:
                self.post_message(TurnFinishedMessage(error))

    def on_turn_finished_message(self, message: TurnFinishedMessage) -> None:
        if message.error:
            self._write_result(f"⚠ {message.error}", style="bold red")
        self._finish_turn()
        self._scroll_to_latest()

    def on_unmount(self) -> None:
        self._shutting_down = True
        if self._logo_timer is not None:
            self._logo_timer.stop()

        with self._approval_lock:
            pending_gates = tuple(self._pending_gates)
            self._pending_gates.clear()
        for gate in pending_gates:
            gate.resolve(approved=False)


def run_tui(
    *,
    session_workspace: Path | None = None,
    max_attempts: int = 3,
    approval_mode: str = "inline",
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
) -> None:
    """Load the persistent session, then start the full-screen interface."""
    workspace = (session_workspace or Path(SESSION_ROOT).parent).expanduser().resolve()
    session = load_or_create_session(workspace)
    MokioClawTuiApp(
        session_id=session["session_id"],
        session_workspace=workspace,
        max_attempts=max_attempts,
        approval_mode=approval_mode,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
    ).run()


__all__ = [
    "AgentEventMessage",
    "ApprovalRequestedMessage",
    "MokioClawTuiApp",
    "TurnFinishedMessage",
    "run_tui",
]
