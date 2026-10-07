from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from xiliumini.automation.models import AutomationTask


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AutomationStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.path = data_dir / "automation.sqlite3"
        data_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS automations (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, prompt TEXT NOT NULL,
                    trigger_type TEXT NOT NULL, schedule_json TEXT NOT NULL,
                    timezone TEXT NOT NULL, enabled INTEGER NOT NULL,
                    policy_profile TEXT NOT NULL, session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS automation_runs (
                    run_id TEXT PRIMARY KEY, automation_id TEXT NOT NULL,
                    scheduled_for TEXT NOT NULL, status TEXT NOT NULL,
                    started_at TEXT NOT NULL, finished_at TEXT,
                    trace_id TEXT, result_path TEXT, error TEXT,
                    UNIQUE(automation_id, scheduled_for)
                );
                CREATE TABLE IF NOT EXISTS scheduler_owner (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    instance_id TEXT NOT NULL, heartbeat_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def session_id(task_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, "xiliumini:automation:" + task_id))

    def add(self, task: AutomationTask) -> None:
        now = _now()
        try:
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO automations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        task.id,
                        task.name,
                        task.prompt,
                        task.trigger_type,
                        json.dumps(task.schedule, sort_keys=True),
                        task.timezone,
                        int(task.enabled),
                        task.policy_profile,
                        self.session_id(task.id),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError:
            raise ValueError("automation ID already exists") from None

    def list(self, *, enabled_only: bool = False) -> list[AutomationTask]:
        sql = "SELECT * FROM automations"
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY id"
        with self._connect() as db:
            return [self._task(row) for row in db.execute(sql)]

    @staticmethod
    def _task(row: sqlite3.Row) -> AutomationTask:
        return AutomationTask(
            id=row["id"],
            name=row["name"],
            prompt=row["prompt"],
            trigger_type=row["trigger_type"],
            schedule=json.loads(row["schedule_json"]),
            timezone=row["timezone"],
            enabled=bool(row["enabled"]),
            policy_profile=row["policy_profile"],
        )

    def get(self, task_id: str) -> AutomationTask | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM automations WHERE id = ?", (task_id,)).fetchone()
        return self._task(row) if row else None

    def set_enabled(self, task_id: str, enabled: bool) -> bool:
        with self._connect() as db:
            updated = db.execute(
                "UPDATE automations SET enabled = ?, updated_at = ? WHERE id = ?",
                (int(enabled), _now(), task_id),
            )
            return updated.rowcount == 1

    def remove(self, task_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM automations WHERE id = ?", (task_id,)).rowcount == 1

    def claim_run(self, task_id: str, scheduled_for: str, run_id: str) -> bool:
        with self._connect() as db:
            result = db.execute(
                """
                INSERT OR IGNORE INTO automation_runs
                (run_id, automation_id, scheduled_for, status, started_at)
                SELECT ?, id, ?, 'running', ? FROM automations WHERE id = ?
            """,
                (run_id, scheduled_for, _now(), task_id),
            )
            return result.rowcount == 1

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        trace_id: str | None,
        result_path: str | None,
        error: str | None = None,
    ) -> None:
        if status not in {"success", "failed", "interrupted"}:
            raise ValueError("invalid automation run status")
        with self._connect() as db:
            result = db.execute(
                """
                UPDATE automation_runs SET status = ?, finished_at = ?,
                    trace_id = ?, result_path = ?, error = ?
                WHERE run_id = ? AND status = 'running'
            """,
                (status, _now(), trace_id, result_path, error, run_id),
            )
            if result.rowcount != 1:
                raise ValueError("automation run is not active")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM automation_runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def mark_interrupted(self) -> int:
        with self._connect() as db:
            result = db.execute(
                "UPDATE automation_runs SET status='interrupted', finished_at=? "
                "WHERE status='running'",
                (_now(),),
            )
            return result.rowcount

    def acquire_owner(self, instance_id: str) -> bool:
        now = datetime.now(UTC)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM scheduler_owner WHERE singleton=1").fetchone()
            if row is not None and row["instance_id"] != instance_id:
                heartbeat = datetime.fromisoformat(row["heartbeat_at"])
                if now - heartbeat < timedelta(seconds=30):
                    return False
            db.execute(
                """
                INSERT INTO scheduler_owner VALUES (1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                instance_id=excluded.instance_id, heartbeat_at=excluded.heartbeat_at
            """,
                (instance_id, now.isoformat()),
            )
            return True

    def release_owner(self, instance_id: str) -> None:
        with self._connect() as db:
            db.execute(
                "DELETE FROM scheduler_owner WHERE singleton=1 AND instance_id=?",
                (instance_id,),
            )
