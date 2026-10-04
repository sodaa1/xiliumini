from __future__ import annotations

from pathlib import Path
from threading import Barrier, Event, Thread

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from xiliumini.core.approval import ApprovalRequest


def test_approval_gate_blocks_until_approved() -> None:
    from xiliumini.cli.tui.approval import ApprovalGate

    gate = ApprovalGate()
    waiting = Event()
    results: list[bool] = []

    def wait_for_decision() -> None:
        waiting.set()
        results.append(gate.wait())

    thread = Thread(target=wait_for_decision)
    thread.start()
    assert waiting.wait(1)
    assert thread.is_alive()

    assert gate.resolve(approved=True) is True
    thread.join(1)

    assert results == [True]
    assert gate.resolved is True


def test_approval_gate_returns_denial() -> None:
    from xiliumini.cli.tui.approval import ApprovalGate

    gate = ApprovalGate()

    assert gate.resolve(approved=False) is True
    assert gate.wait() is False


def test_approval_gate_first_concurrent_decision_wins() -> None:
    from xiliumini.cli.tui.approval import ApprovalGate

    gate = ApprovalGate()
    barrier = Barrier(3)
    outcomes: list[tuple[bool, bool]] = []

    def resolve(approved: bool) -> None:
        barrier.wait()
        outcomes.append((approved, gate.resolve(approved=approved)))

    threads = [Thread(target=resolve, args=(approved,)) for approved in (True, False)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(1)

    accepted = [approved for approved, installed in outcomes if installed]
    assert len(accepted) == 1
    assert gate.wait() is accepted[0]
    assert sum(installed for _, installed in outcomes) == 1


def test_approval_gate_timeout_does_not_resolve() -> None:
    from xiliumini.cli.tui.approval import ApprovalGate

    gate = ApprovalGate()

    with pytest.raises(TimeoutError, match="approval decision timed out"):
        gate.wait(timeout=0.01)

    assert gate.resolved is False


class ApprovalHost(App[None]):
    def __init__(self, modal) -> None:
        super().__init__()
        self.modal = modal
        self.result: bool | None = None

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(self.modal, self._capture)

    def _capture(self, result: bool | None) -> None:
        self.result = result


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        id="approval-1234",
        command="python -m pip install package-with-a-long-name",
        risk_reason="Python package installation",
        tool_name="BashTool",
    )


@pytest.mark.asyncio
async def test_approval_modal_displays_complete_request(tmp_path: Path) -> None:
    from xiliumini.cli.tui.approval import ApprovalModal

    app = ApprovalHost(ApprovalModal(_request(), tmp_path))

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        assert "BashTool" in str(app.screen.query_one("#approval-tool", Static).content)
        assert "Python package installation" in str(
            app.screen.query_one("#approval-risk", Static).content
        )
        assert str(tmp_path.resolve()) in str(
            app.screen.query_one("#approval-workspace", Static).content
        )
        assert _request().command in str(
            app.screen.query_one("#approval-command-text", Static).content
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "expected"),
    [("y", True), ("enter", True), ("n", False), ("escape", False)],
)
async def test_approval_modal_keyboard_decisions(tmp_path: Path, key: str, expected: bool) -> None:
    from xiliumini.cli.tui.approval import ApprovalModal

    app = ApprovalHost(ApprovalModal(_request(), tmp_path))

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(key)
        await pilot.pause()

    assert app.result is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("selector", "expected"), [("#approve", True), ("#deny", False)])
async def test_approval_modal_button_decisions(
    tmp_path: Path, selector: str, expected: bool
) -> None:
    from xiliumini.cli.tui.approval import ApprovalModal

    app = ApprovalHost(ApprovalModal(_request(), tmp_path))

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        await pilot.click(selector)
        await pilot.pause()

    assert app.result is expected
