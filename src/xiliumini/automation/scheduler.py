from __future__ import annotations

import json
from datetime import UTC, datetime
from threading import Event
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from xiliumini.automation.models import AutomationTask
from xiliumini.automation.store import AutomationStore


class AutomationScheduler:
    """One process owns scheduling; SQLite remains the source of truth."""

    def __init__(self, store: AutomationStore, runner: Any):
        self.store = store
        self.runner = runner
        self.instance_id = str(uuid4())
        self._scheduler = BackgroundScheduler()
        self._fingerprints: dict[str, str] = {}
        self._started = False

    @staticmethod
    def _trigger(task: AutomationTask):
        zone = ZoneInfo(task.timezone)
        if task.trigger_type == "date":
            return DateTrigger(run_date=datetime.fromisoformat(task.schedule["at"]), timezone=zone)
        if task.trigger_type == "cron":
            return CronTrigger(
                hour=task.schedule["hour"], minute=task.schedule["minute"], timezone=zone
            )
        return IntervalTrigger(seconds=task.schedule["seconds"], timezone=zone)

    @staticmethod
    def _slot(task: AutomationTask) -> str:
        if task.trigger_type == "date":
            return task.schedule["at"]
        now = datetime.now(ZoneInfo(task.timezone))
        if task.trigger_type == "cron":
            return now.replace(second=0, microsecond=0).isoformat()
        seconds = task.schedule["seconds"]
        utc = datetime.now(UTC)
        bucket = int(utc.timestamp()) // seconds * seconds
        return datetime.fromtimestamp(bucket, UTC).isoformat()

    def _execute(self, task_id: str) -> None:
        task = self.store.get(task_id)
        if task is None or not task.enabled:
            return
        try:
            self.runner.run_once(task_id, scheduled_for=self._slot(task))
        except ValueError:
            # A duplicate slot is intentionally not retried.
            return

    def sync_jobs(self) -> None:
        tasks = {task.id: task for task in self.store.list(enabled_only=True)}
        for task_id in set(self._fingerprints) - tasks.keys():
            self._scheduler.remove_job(task_id)
            del self._fingerprints[task_id]
        for task_id, task in tasks.items():
            fingerprint = json.dumps(task.model_dump(mode="json"), sort_keys=True)
            if self._fingerprints.get(task_id) == fingerprint:
                continue
            self._scheduler.add_job(
                self._execute,
                trigger=self._trigger(task),
                args=[task_id],
                id=task_id,
                replace_existing=True,
                coalesce=True,
                max_instances=1,
                misfire_grace_time=60,
            )
            self._fingerprints[task_id] = fingerprint

    def get_job(self, task_id: str):
        return self._scheduler.get_job(task_id)

    def start(self, *, paused: bool = False) -> None:
        if not self.store.acquire_owner(self.instance_id):
            raise RuntimeError("another automation scheduler owns this data directory")
        self.store.mark_interrupted()
        try:
            self._scheduler.start(paused=paused)
            self._started = True
            self.sync_jobs()
        except BaseException:
            self.store.release_owner(self.instance_id)
            raise

    def serve_forever(self) -> None:
        self.start()
        stop = Event()
        try:
            while not stop.wait(10):
                if not self.store.acquire_owner(self.instance_id):
                    raise RuntimeError("automation scheduler lease was lost")
                self.sync_jobs()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._started:
            self._scheduler.shutdown(wait=False)
            self._started = False
        self.store.release_owner(self.instance_id)
