from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Header, Input, Static

from xiliumini.core.approval import ApprovalDecision, ApprovalRequest

SESSION_ID = "11111111-1111-4111-8111-111111111111"


def empty_stream(task: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    del task, kwargs
    return iter(())


@pytest.mark.asyncio
async def test_tui_composes_required_regions_and_initial_status(tmp_path: Path) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=empty_stream,
        animate_logo=False,
    )

    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        assert app.query_one(Header)
        status = app.query_one("#status-bar", Static)
        assert "session: 11111111" in str(status.content)
        assert "idle" in str(status.content)
        logo = app.query_one("#logo", Static)
        assert "你的专属智能 Agent" in str(logo.content)
        assert logo.styles.text_align == "center"
        assert app.query_one("#conversation", VerticalScroll)
        assert len(app.query(".conversation-turn")) == 0
        prompt = app.query_one("#prompt", Input)
        assert prompt.disabled is False
        assert prompt.has_focus


@pytest.mark.asyncio
async def test_tui_static_logo_mode_does_not_schedule_animation(tmp_path: Path) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=empty_stream,
        animate_logo=False,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        before = str(app.query_one("#logo", Static).content)
        await pilot.pause(1)
        after = str(app.query_one("#logo", Static).content)

    assert before == after


@pytest.mark.asyncio
async def test_animated_logo_settles_to_persistent_brand_row(tmp_path: Path) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=empty_stream,
        animate_logo=True,
    )

    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause(1)
        logo = app.query_one("#logo", Static)
        status = app.query_one("#status-bar", Static)

        assert "你的专属智能 Agent" in str(logo.content)
        assert not logo.has_class("collapsed")
        assert logo.parent is status.parent
        assert logo.parent is not None and logo.parent.id == "brand-row"


@pytest.mark.asyncio
async def test_logo_hides_after_first_conversation_starts(tmp_path: Path) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=empty_stream,
        animate_logo=True,
    )

    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause(1)
        logo = app.query_one("#logo", Static)
        brand_row = app.query_one("#brand-row")
        assert not logo.has_class("collapsed")

        prompt = app.query_one("#prompt", Input)
        prompt.value = "你好"
        await pilot.press("enter")
        await pilot.pause()

        assert logo.has_class("collapsed")
        assert brand_row.has_class("conversation-started")


@pytest.mark.asyncio
async def test_turn_reasoning_is_collapsed_by_default_and_user_can_expand_it(
    tmp_path: Path,
) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp

    def stream(task: str, **kwargs: Any):
        del task, kwargs
        yield {
            "type": "custom_event",
            "event": {
                "stage": "code_agent",
                "message": "created app.py",
                "event_type": "tool_result",
                "details": {"tool": "file_write", "ok": True},
            },
        }

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=stream,
        animate_logo=False,
    )

    async with app.run_test(size=(110, 36)) as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "创建页面"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: len(app.query(Collapsible)) == 1)

        reasoning = app.query_one(Collapsible)
        assert reasoning.collapsed is True

        assert await pilot.click("CollapsibleTitle") is True
        await pilot.pause()
        assert reasoning.collapsed is False
        assert "created app.py" in str(
            reasoning.query_one(".turn-reasoning-content", Static).content
        )


@pytest.mark.asyncio
async def test_final_answer_stays_visible_outside_reasoning_and_shows_elapsed_time(
    tmp_path: Path,
) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp

    times = iter((0.0, 76.8))

    def stream(task: str, **kwargs: Any):
        del task, kwargs
        yield {
            "type": "custom_event",
            "event": {
                "stage": "search_agent",
                "message": "search complete",
                "event_type": "search_results",
                "details": {"tool": "web_search", "ok": True},
            },
        }
        yield {
            "type": "graph_event",
            "event": {"text": "页面已经创建完成。", "session_id": SESSION_ID},
        }

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=stream,
        animate_logo=False,
        time_source=lambda: next(times),
    )

    async with app.run_test(size=(110, 36)) as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "创建页面"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: not prompt.disabled)

        reasoning = app.query_one(Collapsible)
        answer = app.query_one(".turn-answer", Static)

        assert reasoning.collapsed is True
        assert reasoning.title == "思考过程 · 用时 1分钟16秒"
        assert "页面已经创建完成。" in str(answer.content)
        assert "search complete" not in str(answer.content)


@pytest.mark.asyncio
async def test_each_conversation_turn_keeps_its_own_reasoning_and_answer(
    tmp_path: Path,
) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp

    def stream(task: str, **kwargs: Any):
        del kwargs
        yield {
            "type": "custom_event",
            "event": {
                "stage": "code_agent",
                "message": f"tool for {task}",
                "event_type": "tool_result",
                "details": {"tool": "file_write", "ok": True},
            },
        }
        yield {
            "type": "graph_event",
            "event": {"text": f"answer for {task}", "session_id": SESSION_ID},
        }

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=stream,
        animate_logo=False,
    )

    async with app.run_test(size=(110, 36)) as pilot:
        prompt = app.query_one("#prompt", Input)
        for task in ("first", "second"):
            prompt.value = task
            await pilot.press("enter")
            await _wait_until(pilot, lambda: not prompt.disabled)

        turns = list(app.query(".conversation-turn"))
        assert len(turns) == 2
        first_thought = str(turns[0].query_one(".turn-reasoning-content", Static).content)
        second_thought = str(turns[1].query_one(".turn-reasoning-content", Static).content)

        assert "tool for first" in first_thought
        assert "tool for second" not in first_thought
        assert "tool for second" in second_thought
        assert "answer for first" in str(turns[0].query_one(".turn-answer", Static).content)
        assert "answer for second" in str(turns[1].query_one(".turn-answer", Static).content)


def _log_text(app) -> str:
    widgets = (
        list(app.query(".turn-user"))
        + list(app.query(".turn-reasoning-content"))
        + list(app.query(".turn-answer"))
    )
    content = [str(widget.content) for widget in widgets if isinstance(widget, Static)]
    return "\n".join(content)


async def _wait_until(pilot, predicate, timeout: float = 3) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError("condition was not reached before timeout")


@pytest.mark.asyncio
async def test_plan_snapshot_is_written_inside_collapsed_reasoning(tmp_path: Path) -> None:
    from xiliumini.cli.tui.app import AgentEventMessage, MokioClawTuiApp

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=empty_stream,
        animate_logo=False,
    )
    payload = {
        "type": "graph_event",
        "event": {
            "todos": [
                {"id": "one", "content": "write code", "status": "completed", "note": ""},
                {"id": "two", "content": "run tests", "status": "in_progress", "note": ""},
                {"id": "three", "content": "update docs", "status": "pending", "note": ""},
                {"id": "four", "content": "publish", "status": "blocked", "note": ""},
            ],
            "summary": "Implement the TUI",
            "attempt": 2,
        },
    }

    async with app.run_test() as pilot:
        app.post_message(AgentEventMessage(payload))
        await pilot.pause()
        reasoning = app.query_one(Collapsible)
        plan = str(reasoning.query_one(".turn-reasoning-content", Static).content)

    assert reasoning.collapsed is True
    assert "[Plan] Implement the TUI · attempt 2" in plan
    assert "✅ write code" in plan
    assert "🔄 run tests" in plan
    assert "⬜ update docs" in plan
    assert "⛔ publish" in plan


@pytest.mark.asyncio
async def test_custom_events_render_distinct_rows_and_truncate_safely(tmp_path: Path) -> None:
    from xiliumini.cli.tui.app import AgentEventMessage, MokioClawTuiApp

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=empty_stream,
        animate_logo=False,
    )
    long_message = "Z" * 2_000
    events = [
        {
            "type": "custom_event",
            "event": {
                "stage": "code_agent",
                "message": "code_agent: file_write",
                "event_type": "tool_call",
                "details": {"tool": "file_write", "args": {"path": "app.py"}},
            },
        },
        {
            "type": "custom_event",
            "event": {
                "stage": "code_agent",
                "message": "created app.py",
                "event_type": "tool_result",
                "details": {"tool": "file_write", "ok": True},
            },
        },
        {
            "type": "custom_event",
            "event": {
                "stage": "search_agent",
                "message": "3 sources",
                "event_type": "search_results",
                "details": {
                    "tool": "web_search",
                    "args": {"query": "Flask tutorial"},
                    "ok": True,
                },
            },
        },
        {
            "type": "custom_event",
            "event": {
                "stage": "planner",
                "message": "planner → code_agent: Delegation completed",
                "event_type": "handoff",
                "details": {"agent": "code_agent", "ok": True},
            },
        },
        {
            "type": "custom_event",
            "event": {
                "stage": "checkpoint",
                "message": "Checkpoint saved: running",
                "event_type": "checkpoint_saved",
                "details": {"status": "running", "latest_node": "code_agent"},
            },
        },
        {
            "type": "custom_event",
            "event": {
                "stage": "other",
                "message": long_message,
                "event_type": "future_event",
                "details": {},
            },
        },
    ]

    async with app.run_test(size=(110, 36)) as pilot:
        for event in events:
            app.post_message(AgentEventMessage(event))
        await pilot.pause()
        rendered = _log_text(app)

    assert "🔧 file_write" in rendered
    assert "🔧 file_write → app.py" in rendered
    assert "✅ file_write: created app.py" in rendered
    assert '🔍 web_search → "Flask tutorial": 3 sources' in rendered
    assert "🔄 Handoff: planner → code_agent · ✅" in rendered
    assert "💾 Checkpoint: running · code_agent" in rendered
    assert "future_event" in rendered
    assert "… [truncated]" in rendered
    assert rendered.count("Z") < 1_000


@pytest.mark.asyncio
async def test_final_answer_is_not_truncated(tmp_path: Path) -> None:
    from xiliumini.cli.tui.app import AgentEventMessage, MokioClawTuiApp

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=empty_stream,
        animate_logo=False,
    )
    answer = "A" * 1_000 + " END-OF-ANSWER"

    async with app.run_test(size=(110, 36)) as pilot:
        app.post_message(
            AgentEventMessage(
                {
                    "type": "graph_event",
                    "event": {"text": answer, "session_id": SESSION_ID},
                }
            )
        )
        await pilot.pause()
        rendered = _log_text(app)

    assert "END-OF-ANSWER" in rendered
    assert "[truncated]" not in rendered


@pytest.mark.asyncio
async def test_graph_events_render_verifier_final_and_error_then_restore_input(
    tmp_path: Path,
) -> None:
    from xiliumini.cli.tui.app import (
        AgentEventMessage,
        MokioClawTuiApp,
        TurnFinishedMessage,
    )

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=empty_stream,
        animate_logo=False,
    )
    events = [
        {
            "type": "graph_event",
            "event": {"passed": False, "reason": "tests failed", "attempt": 1},
        },
        {
            "type": "graph_event",
            "event": {"text": "Done after repair", "session_id": SESSION_ID},
        },
        {
            "type": "graph_event",
            "event": {"code": "provider_error", "message": "Provider failed"},
        },
    ]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.disabled = True
        for event in events:
            app.post_message(AgentEventMessage(event))
        await pilot.pause()
        rendered = _log_text(app)
        assert prompt.disabled is True

        app.post_message(TurnFinishedMessage())
        await pilot.pause()
        status = str(app.query_one("#status-bar", Static).content)

    assert "❌ Verifier: tests failed" in rendered
    assert "📝 Final: Done after repair" in rendered
    assert "⚠ Error [provider_error]: Provider failed" in rendered
    assert prompt.disabled is False
    assert "idle" in status


class BlockingEvents:
    def __init__(self, task: str, started: Event, release: Event, closed: list[str]) -> None:
        self.task = task
        self.started = started
        self.release = release
        self.closed = closed
        self.sent = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.sent:
            raise StopIteration
        self.started.set()
        if not self.release.wait(3):
            raise RuntimeError("test release timed out")
        self.sent = True
        return {
            "type": "graph_event",
            "event": {"text": f"answer: {self.task}", "session_id": SESSION_ID},
        }

    def close(self) -> None:
        self.closed.append(self.task)


class FinalThenBlockingEvents:
    def __init__(self, waiting: Event, release: Event, closed: Event) -> None:
        self.waiting = waiting
        self.release = release
        self.closed = closed
        self.sent = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.sent:
            self.sent = True
            return {
                "type": "graph_event",
                "event": {"text": "answer", "session_id": SESSION_ID},
            }
        self.waiting.set()
        if not self.release.wait(3):
            raise RuntimeError("test release timed out")
        raise StopIteration

    def close(self) -> None:
        self.closed.set()


@pytest.mark.asyncio
async def test_session_worker_serializes_turns_reuses_workspace_and_closes_streams(
    tmp_path: Path,
) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp

    starts = [Event(), Event()]
    releases = [Event(), Event()]
    calls: list[tuple[str, dict[str, Any]]] = []
    closed: list[str] = []

    def stream(task: str, **kwargs: Any):
        index = len(calls)
        calls.append((task, kwargs))
        return BlockingEvents(task, starts[index], releases[index], closed)

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=stream,
        animate_logo=False,
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "first"
        await pilot.press("enter")
        assert await asyncio.to_thread(starts[0].wait, 1)
        await pilot.pause()
        assert prompt.disabled is True
        assert "running" in str(app.query_one("#status-bar", Static).content)

        prompt.value = "should not run"
        await pilot.press("enter")
        await pilot.pause()
        assert len(calls) == 1

        releases[0].set()
        await _wait_until(pilot, lambda: not prompt.disabled)
        assert prompt.has_focus

        prompt.value = "second"
        await pilot.press("enter")
        assert await asyncio.to_thread(starts[1].wait, 1)
        releases[1].set()
        await _wait_until(pilot, lambda: len(closed) == 2)
        await _wait_until(pilot, lambda: not prompt.disabled)
        rendered = _log_text(app)

    assert [task for task, _ in calls] == ["first", "second"]
    assert all(call[1]["session_workspace"] == tmp_path.resolve() for call in calls)
    assert all(call[1]["max_attempts"] == 3 for call in calls)
    assert closed == ["first", "second"]
    assert "💬 You: first" in rendered
    assert "📝 Final: answer: first" in rendered
    assert "📝 Final: answer: second" in rendered


@pytest.mark.asyncio
async def test_worker_exception_restores_input_without_leaking_exception(
    tmp_path: Path,
) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp

    def broken_stream(task: str, **kwargs: Any):
        del task, kwargs
        raise RuntimeError("private provider detail")

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=broken_stream,
        animate_logo=False,
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "fail"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: "Session turn failed" in _log_text(app))
        rendered = _log_text(app)

    assert prompt.disabled is False
    assert "private provider detail" not in rendered
    assert "Session turn failed" in rendered


@pytest.mark.asyncio
async def test_final_event_keeps_input_locked_until_stream_cleanup(tmp_path: Path) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp

    waiting = Event()
    release = Event()
    closed = Event()

    def stream(task: str, **kwargs: Any):
        del task, kwargs
        return FinalThenBlockingEvents(waiting, release, closed)

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=stream,
        animate_logo=False,
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "first"
        await pilot.press("enter")
        assert await asyncio.to_thread(waiting.wait, 1)
        await _wait_until(pilot, lambda: "📝 Final: answer" in _log_text(app))

        assert prompt.disabled is True
        assert app._busy is True

        release.set()
        assert await asyncio.to_thread(closed.wait, 1)
        await _wait_until(pilot, lambda: not prompt.disabled)


@pytest.mark.asyncio
@pytest.mark.parametrize(("key", "approved"), [("y", True), ("n", False)])
async def test_worker_approval_modal_returns_decision_and_resumes(
    tmp_path: Path, key: str, approved: bool
) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp
    from xiliumini.cli.tui.approval import ApprovalModal

    decisions: list[ApprovalDecision] = []

    def stream(task: str, **kwargs: Any):
        del task
        handler = kwargs["approval_handler"]
        decisions.append(
            handler(
                ApprovalRequest(
                    id="approval-1",
                    command="python -m pip install demo",
                    risk_reason="Python package installation",
                )
            )
        )
        yield {
            "type": "graph_event",
            "event": {"text": "continued", "session_id": SESSION_ID},
        }

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=stream,
        animate_logo=False,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "install it"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: isinstance(app.screen, ApprovalModal))
        assert "approval" in str(app.query_one("#status-bar", Static).content)
        expected_workspace = tmp_path.resolve() / "workspaces" / SESSION_ID
        assert str(expected_workspace) in str(
            app.screen.query_one("#approval-workspace", Static).content
        )
        await pilot.press(key)
        await _wait_until(pilot, lambda: len(decisions) == 1)
        await _wait_until(pilot, lambda: not prompt.disabled)

    assert decisions == [
        ApprovalDecision(
            approved=approved,
            reason="Approved by user." if approved else "Denied by user.",
        )
    ]


@pytest.mark.asyncio
async def test_first_approval_resolution_wins_over_late_modal_choice(tmp_path: Path) -> None:
    from xiliumini.cli.tui.app import ApprovalRequestedMessage, MokioClawTuiApp
    from xiliumini.cli.tui.approval import ApprovalModal

    decisions: list[ApprovalDecision] = []
    requests: list[ApprovalRequestedMessage] = []

    def stream(task: str, **kwargs: Any):
        del task
        decisions.append(
            kwargs["approval_handler"](
                ApprovalRequest("approval-2", "uv add textual", "Project dependency change")
            )
        )
        return iter(())

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=stream,
        animate_logo=False,
    )

    def capture(message) -> None:
        if isinstance(message, ApprovalRequestedMessage):
            requests.append(message)

    async with app.run_test(message_hook=capture) as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "install"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: isinstance(app.screen, ApprovalModal))
        assert requests[0].gate.resolve(approved=False) is True
        await pilot.press("y")
        await _wait_until(pilot, lambda: len(decisions) == 1)

    assert decisions == [ApprovalDecision(False, "Denied by user.")]


@pytest.mark.asyncio
async def test_shutdown_denies_pending_approval_and_releases_worker(tmp_path: Path) -> None:
    from xiliumini.cli.tui.app import MokioClawTuiApp
    from xiliumini.cli.tui.approval import ApprovalModal

    completed = Event()
    decisions: list[ApprovalDecision] = []

    def stream(task: str, **kwargs: Any):
        del task
        decisions.append(
            kwargs["approval_handler"](
                ApprovalRequest("approval-3", "uv sync", "Dependency synchronization")
            )
        )
        completed.set()
        return iter(())

    app = MokioClawTuiApp(
        session_id=SESSION_ID,
        session_workspace=tmp_path,
        stream_factory=stream,
        animate_logo=False,
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "sync"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: isinstance(app.screen, ApprovalModal))
        app.exit()
        await pilot.pause()

    assert completed.wait(1)
    assert decisions == [ApprovalDecision(False, "Denied because the TUI closed.")]


def test_run_tui_loads_session_before_starting_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xiliumini.cli.tui import app as app_module

    calls: list[tuple[str, object]] = []

    def load_session(workspace: Path) -> dict[str, object]:
        calls.append(("load", workspace))
        return {"session_id": SESSION_ID}

    def run(self) -> None:
        calls.append(("run", self))

    monkeypatch.setattr(app_module, "load_or_create_session", load_session)
    monkeypatch.setattr(app_module.MokioClawTuiApp, "run", run)

    app_module.run_tui(
        session_workspace=tmp_path,
        max_attempts=7,
        approval_mode="deny",
        checkpoint_mode="off",
        trace_mode="off",
    )

    assert calls[0] == ("load", tmp_path.resolve())
    started = calls[1][1]
    assert isinstance(started, app_module.MokioClawTuiApp)
    assert started.session_id == SESSION_ID
    assert started.session_workspace == tmp_path.resolve()
    assert started.max_attempts == 7
    assert started.approval_mode == "deny"
    assert started.checkpoint_mode == "off"
    assert started.trace_mode == "off"
