from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from xiliumini.automation.models import AutomationTask
from xiliumini.automation.store import AutomationStore


def task() -> AutomationTask:
    return AutomationTask(
        id="daily-brief", name="Daily brief", prompt="Read AIHot latest news",
        trigger_type="cron", schedule={"hour": 9, "minute": 0},
        timezone="Asia/Shanghai",
    )


def test_task_persists_and_run_slot_is_unique(tmp_path):
    store = AutomationStore(tmp_path)
    store.add(task())
    reopened = AutomationStore(tmp_path)
    saved = reopened.get("daily-brief")
    assert saved is not None
    assert saved.schedule == {"hour": 9, "minute": 0}
    slot = datetime(2026, 10, 8, 9, tzinfo=ZoneInfo("Asia/Shanghai")).isoformat()
    assert reopened.claim_run("daily-brief", slot, "run-1")
    assert not reopened.claim_run("daily-brief", slot, "run-2")
    reopened.finish_run("run-1", status="success", trace_id="trace-1", result_path="report.md")
    run = reopened.get_run("run-1")
    assert run is not None
    assert run["status"] == "success"


def test_owner_lease_prevents_second_scheduler_and_marks_interrupted(tmp_path):
    store = AutomationStore(tmp_path)
    assert store.acquire_owner("first")
    assert not AutomationStore(tmp_path).acquire_owner("second")
    store.add(task())
    assert store.claim_run("daily-brief", "2026-10-08T09:00:00+08:00", "run-1")
    store.mark_interrupted()
    run = store.get_run("run-1")
    assert run is not None
    assert run["status"] == "interrupted"
    store.release_owner("first")
    assert store.acquire_owner("second")


def test_schedule_validation_rejects_bad_clock_and_naive_date():
    with pytest.raises(ValueError):
        AutomationTask.model_validate(
            {**task().model_dump(), "schedule": {"hour": 25, "minute": 0}}
        )
    with pytest.raises(ValueError):
        AutomationTask(
            id="once", name="Once", prompt="Task", trigger_type="date",
            schedule={"at": "2026-10-08T09:00:00"}, timezone="Asia/Shanghai",
        )
