from __future__ import annotations

from xiliumini.automation.runner import AutomationRunner
from xiliumini.automation.store import AutomationStore
from xiliumini.events import FinalEvent, VerifierEvent
from tests.unit.test_automation import task


def test_run_once_consumes_events_and_records_real_trace(tmp_path):
    store = AutomationStore(tmp_path)
    store.add(task())
    seen = []

    class FakeRuntime:
        last_trace_id = "trace-from-runtime"

        def stream(self, prompt, session_id):
            seen.append((prompt, session_id))
            yield VerifierEvent(passed=True, reason="passed", attempt=1)
            yield FinalEvent(text="Brief completed", session_id=session_id)

    runner = AutomationRunner(store, settings=object(), runtime_factory=lambda *_args, **_kwargs: FakeRuntime())
    result = runner.run_once("daily-brief")
    assert result.status == "success"
    record = store.get_run(result.run_id)
    assert record is not None
    assert record["trace_id"] == "trace-from-runtime"
    assert seen == [("Read AIHot latest news", store.session_id("daily-brief"))]
    assert (tmp_path / "automation-reports" / "daily-brief" / f"{result.run_id}.md").exists()


def test_run_once_fails_without_verifier_success(tmp_path):
    store = AutomationStore(tmp_path)
    store.add(task())

    class FakeRuntime:
        last_trace_id = "trace-failed"

        def stream(self, prompt, session_id):
            yield FinalEvent(text="Unverified answer", session_id=session_id)

    runner = AutomationRunner(store, settings=object(), runtime_factory=lambda *_args, **_kwargs: FakeRuntime())
    result = runner.run_once("daily-brief")
    assert result.status == "failed"
    record = store.get_run(result.run_id)
    assert record is not None
    assert record["status"] == "failed"


def test_run_once_uses_final_verifier_after_retry(tmp_path):
    store = AutomationStore(tmp_path)
    store.add(task())

    class FakeRuntime:
        last_trace_id = "trace-retry"

        def stream(self, prompt, session_id):
            yield VerifierEvent(passed=False, reason="needs another attempt", attempt=1)
            yield VerifierEvent(passed=True, reason="fixed", attempt=2)
            yield FinalEvent(text="done", session_id=session_id)

    runner = AutomationRunner(store, settings=object(), runtime_factory=lambda *_args, **_kwargs: FakeRuntime())
    assert runner.run_once("daily-brief").status == "success"


def test_run_once_keeps_trace_and_error_type_on_runtime_failure(tmp_path):
    store = AutomationStore(tmp_path)
    store.add(task())

    class FakeRuntime:
        last_trace_id = "trace-before-error"

        def stream(self, prompt, session_id):
            yield VerifierEvent(passed=False, reason="started", attempt=1)
            raise PermissionError("secret local path")

    runner = AutomationRunner(
        store, settings=object(), runtime_factory=lambda *_args, **_kwargs: FakeRuntime()
    )
    result = runner.run_once("daily-brief")
    record = store.get_run(result.run_id)
    assert result.status == "failed"
    assert record is not None
    assert record["trace_id"] == "trace-before-error"
    assert record["error"] == "Automation runtime failed: PermissionError"
    assert result.report_path is not None
    assert "secret local path" not in result.report_path.read_text(encoding="utf-8")
