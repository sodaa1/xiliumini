from __future__ import annotations

from pathlib import Path
from threading import Event, Lock

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from xiliumini.core.approval import ApprovalRequest


class ApprovalGate:
    """Synchronize one approval decision between a worker and the TUI."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._decision: bool | None = None

    @property
    def resolved(self) -> bool:
        with self._lock:
            return self._decision is not None

    def resolve(self, *, approved: bool) -> bool:
        with self._lock:
            if self._decision is not None:
                return False
            self._decision = bool(approved)
            self._event.set()
            return True

    def wait(self, timeout: float | None = None) -> bool:
        if not self._event.wait(timeout):
            raise TimeoutError("approval decision timed out")
        with self._lock:
            decision = self._decision
        if decision is None:  # pragma: no cover - Event and decision change under one lock.
            raise RuntimeError("approval decision is unavailable")
        return decision


class ApprovalModal(ModalScreen[bool]):
    """Request a human decision for one approval-gated command."""

    CSS_PATH = "styles.tcss"
    BINDINGS = [
        Binding("y", "approve", "Approve", show=False, priority=True),
        Binding("enter", "approve", "Approve", show=False, priority=True),
        Binding("n", "deny", "Deny", show=False, priority=True),
        Binding("escape", "deny", "Deny", show=False, priority=True),
    ]

    def __init__(self, request: ApprovalRequest, workspace: Path) -> None:
        super().__init__()
        self.request = request
        self.workspace = workspace.resolve()

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static("⚠ Approval required", id="approval-title", markup=False)
            yield Static(f"Tool: {self.request.tool_name}", id="approval-tool", markup=False)
            yield Static(f"Risk: {self.request.risk_reason}", id="approval-risk", markup=False)
            yield Static(f"Workspace: {self.workspace}", id="approval-workspace", markup=False)
            with VerticalScroll(id="approval-command"):
                yield Static(self.request.command, id="approval-command-text", markup=False)
            with Horizontal(id="approval-actions"):
                yield Button("[Y] Approve", id="approve", variant="success")
                yield Button("[N] Deny", id="deny", variant="error")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")
