from __future__ import annotations

from datetime import datetime, timedelta
from threading import Event
from zoneinfo import ZoneInfo

from xiliumini.automation.models import AutomationTask
from xiliumini.automation.scheduler import AutomationScheduler
from xiliumini.automation.store import AutomationStore
from tests.unit.test_automation import task


def test_scheduler_restores_cron_job_from_sqlite(tmp_path):
    store = AutomationStore(tmp_path)
    store.add(task())
    service = AutomationScheduler(AutomationStore(tmp_path), runner=object())
    service.start(paused=True)
    try:
        job = service.get_job("daily-brief")
        assert job is not None
        assert job.next_run_time is not None
        assert job.trigger.fields[5].__str__() == "9"
    finally:
        service.shutdown()


def test_near_future_date_runs_once_in_separate_scheduler(tmp_path):
    store = AutomationStore(tmp_path)
    fired = Event()
    calls = []

    class FakeRunner:
        def run_once(self, task_id, *, scheduled_for=None):
            calls.append((task_id, scheduled_for))
            fired.set()

    at = datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(seconds=2)
    store.add(AutomationTask(
        id="once", name="Once", prompt="Task", trigger_type="date",
        schedule={"at": at.isoformat()}, timezone="Asia/Shanghai",
    ))
    service = AutomationScheduler(store, runner=FakeRunner())
    service.start()
    try:
        assert fired.wait(6)
        assert calls == [("once", at.isoformat())]
    finally:
        service.shutdown()
