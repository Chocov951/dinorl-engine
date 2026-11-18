"""Single-process SQLite/WAL persistence and fair job selection for RL-L9."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Final

__all__ = ["JobPriority", "JobStatus", "OrchestrationError", "OrchestrationStore"]

_HEARTBEAT_SECONDS: Final = 30
_TRAINING_SLICE_UNITS: Final = 5
_MAX_TECHNICAL_ATTEMPTS: Final = 3


class JobPriority(IntEnum):
    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3


class JobStatus(StrEnum):
    QUEUED = "queued"
    COMPILING = "compiling"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OrchestrationError(RuntimeError):
    """An immutable job or budget invariant was violated."""


class OrchestrationStore:
    """The dedicated local authority before a remote budget authority exists."""

    def __init__(self, path: Path, *, now: Callable[[], datetime] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._now = now or (lambda: datetime.now(UTC))
        with self._session() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > 2:
                raise OrchestrationError(
                    "orchestration database migration is newer than this engine"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY, configuration_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, reserved_units INTEGER NOT NULL,
                    validated_units INTEGER NOT NULL DEFAULT 0, released_units INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, run_id TEXT, kind TEXT NOT NULL,
                    priority INTEGER NOT NULL, status TEXT NOT NULL, idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL, heartbeat_at TEXT, phase TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS job_events (
                    event_id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    event TEXT NOT NULL, FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS unit_ledger (
                    job_id TEXT NOT NULL, unit_id TEXT NOT NULL, PRIMARY KEY(job_id, unit_id),
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (artifact_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, size INTEGER NOT NULL, pinned INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS evaluations (evaluation_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, protocol TEXT NOT NULL, result_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluation_cache (cache_key TEXT PRIMARY KEY, result_json TEXT NOT NULL);
                """
            )
            if version < 2:
                columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
                }
                if "cancel_requested" not in columns:
                    connection.execute(
                        "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
                    )
                connection.execute("PRAGMA user_version=2")

    @contextmanager
    def _session(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _timestamp(self) -> str:
        value = self._now()
        if value.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @property
    def journal_mode(self) -> str:
        with self._session() as connection:
            row = connection.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower() if row is not None else ""

    def register_experiment(self, experiment_id: str, configuration: bytes) -> bool:
        """Freeze one exact configuration payload before reserving its first run."""

        if not experiment_id or not configuration:
            raise ValueError("experiment identity and configuration are required")
        digest = hashlib.sha256(configuration).hexdigest()
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT configuration_sha256 FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row is not None:
                if row[0] != digest:
                    raise OrchestrationError("experiment configuration is immutable")
                return False
            connection.execute(
                "INSERT INTO experiments(experiment_id,configuration_sha256) VALUES (?,?)",
                (experiment_id, digest),
            )
            return True

    def submit_job(
        self, *, owner_id: str, run_id: str | None, priority: JobPriority, kind: str, key: str
    ) -> str:
        if not owner_id or not kind or not key:
            raise ValueError("job fields must be non-empty")
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT job_id, owner_id, run_id, priority, kind FROM jobs WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if existing is not None:
                if tuple(existing[1:]) != (owner_id, run_id, int(priority), kind):
                    raise OrchestrationError("job idempotency key conflicts")
                return str(existing[0])
            if kind == "private_test":
                duplicate = connection.execute(
                    "SELECT 1 FROM jobs WHERE owner_id = ? AND kind = 'private_test' AND status IN ('queued','compiling','running','evaluating')",
                    (owner_id,),
                ).fetchone()
                if duplicate is not None:
                    raise OrchestrationError("owner already has an active private test")
            job_id = f"job-{uuid.uuid4().hex}"
            connection.execute(
                "INSERT INTO jobs(job_id,owner_id,run_id,kind,priority,status,idempotency_key,created_at) VALUES (?,?,?,?,?,'queued',?,?)",
                (job_id, owner_id, run_id, kind, int(priority), key, self._timestamp()),
            )
            self._event(connection, job_id, "submitted")
        return job_id

    def submit_training(self, *, owner_id: str, run_id: str, budget_units: int, key: str) -> str:
        if budget_units <= 0:
            raise ValueError("budget units must be positive")
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT owner_id,reserved_units FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO runs(run_id,owner_id,reserved_units) VALUES (?,?,?)",
                    (run_id, owner_id, budget_units),
                )
            elif tuple(existing) != (owner_id, budget_units):
                raise OrchestrationError("run reservation differs from existing budget")
        return self.submit_job(
            owner_id=owner_id, run_id=run_id, priority=JobPriority.P3, kind="training", key=key
        )

    def _event(self, connection: sqlite3.Connection, job_id: str, event: str) -> None:
        connection.execute(
            "INSERT INTO job_events(job_id,created_at,event) VALUES (?,?,?)",
            (job_id, self._timestamp(), event),
        )

    def claim_next(self) -> str | None:
        """Claim P0–P2 first; choose the least recently served P3 owner thereafter."""
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT job_id FROM jobs WHERE status='queued' AND priority < 3 ORDER BY priority, created_at LIMIT 1"
            ).fetchone()
            if row is None:
                row = connection.execute(
                    """
                    SELECT j.job_id FROM jobs j
                    WHERE j.status='queued' AND j.priority=3
                    ORDER BY COALESCE((SELECT MAX(e.event_id) FROM job_events e JOIN jobs served ON served.job_id=e.job_id WHERE e.event='unit_validated' AND served.owner_id=j.owner_id), 0), j.created_at
                    LIMIT 1
                    """
                ).fetchone()
            if row is None:
                return None
            job_id = str(row[0])
            connection.execute("UPDATE jobs SET status='running' WHERE job_id=?", (job_id,))
            self._event(connection, job_id, "claimed")
            return job_id

    def transition(self, job_id: str, target: JobStatus, event: str) -> None:
        allowed = {
            JobStatus.QUEUED: {JobStatus.COMPILING, JobStatus.RUNNING, JobStatus.CANCELLED},
            JobStatus.COMPILING: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED},
            JobStatus.RUNNING: {
                JobStatus.EVALUATING,
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            },
            JobStatus.EVALUATING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED},
        }
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or target not in allowed.get(JobStatus(str(row[0])), set()):
                raise OrchestrationError("invalid job transition")
            connection.execute("UPDATE jobs SET status=? WHERE job_id=?", (target.value, job_id))
            self._event(connection, job_id, event)

    def complete_unit(self, job_id: str, unit_id: str) -> bool:
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                "SELECT run_id,status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None or job[0] is None or job[1] != JobStatus.RUNNING.value:
                raise OrchestrationError("only a running training job can validate a unit")
            inserted = connection.execute(
                "INSERT OR IGNORE INTO unit_ledger(job_id,unit_id) VALUES (?,?)", (job_id, unit_id)
            ).rowcount
            if inserted:
                connection.execute(
                    "UPDATE runs SET validated_units=validated_units+1 WHERE run_id=?", (job[0],)
                )
                self._event(connection, job_id, "unit_validated")
                units = connection.execute(
                    "SELECT reserved_units,validated_units FROM runs WHERE run_id=?", (job[0],)
                ).fetchone()
                if units is None:
                    raise OrchestrationError("training run disappeared during unit validation")
                cancellation = connection.execute(
                    "SELECT cancel_requested FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if cancellation is None:
                    raise OrchestrationError("training job disappeared during unit validation")
                target = (
                    "cancelled"
                    if cancellation[0]
                    else "evaluating"
                    if units[0] == units[1]
                    else "queued"
                )
                connection.execute("UPDATE jobs SET status=? WHERE job_id=?", (target, job_id))
                if target == "cancelled":
                    connection.execute(
                        "UPDATE runs SET released_units=reserved_units-validated_units WHERE run_id=?",
                        (job[0],),
                    )
                    self._event(connection, job_id, "cancelled_at_unit_boundary")
            return bool(inserted)

    def technical_failure(self, job_id: str, reason: str) -> JobStatus:
        """Retry a transient failure at most three times from the last validated unit."""

        if not reason:
            raise ValueError("technical failure reason must be non-empty")
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT run_id,attempts,status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None or row[2] not in {
                JobStatus.COMPILING.value,
                JobStatus.RUNNING.value,
                JobStatus.EVALUATING.value,
            }:
                raise OrchestrationError("job cannot receive a technical retry")
            attempts = int(row[1]) + 1
            terminal = attempts >= _MAX_TECHNICAL_ATTEMPTS
            status = JobStatus.FAILED if terminal else JobStatus.QUEUED
            connection.execute(
                "UPDATE jobs SET attempts=?,status=? WHERE job_id=?",
                (attempts, status.value, job_id),
            )
            if terminal and row[0] is not None:
                connection.execute(
                    "UPDATE runs SET released_units=reserved_units-validated_units WHERE run_id=?",
                    (row[0],),
                )
            self._event(connection, job_id, f"technical_failure:{reason}")
            return status

    def cache_evaluation(self, cache_key: str, result_json: str) -> bool:
        """Persist a deterministic suite result once; callers reuse its exact JSON."""

        if not cache_key or not result_json:
            raise ValueError("evaluation cache fields must be non-empty")
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT result_json FROM evaluation_cache WHERE cache_key=?", (cache_key,)
            ).fetchone()
            if row is not None:
                if row[0] != result_json:
                    raise OrchestrationError("evaluation cache key conflicts")
                return False
            connection.execute(
                "INSERT INTO evaluation_cache(cache_key,result_json) VALUES (?,?)",
                (cache_key, result_json),
            )
            return True

    def cached_evaluation(self, cache_key: str) -> str | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT result_json FROM evaluation_cache WHERE cache_key=?", (cache_key,)
            ).fetchone()
        return None if row is None else str(row[0])

    def register_artifact(
        self, *, job_id: str, path: str, sha256: str, size: int, pinned: bool = False
    ) -> str:
        if not path or len(sha256) != 64 or size < 0:
            raise ValueError("artifact metadata is invalid")
        artifact_id = f"artifact-{uuid.uuid4().hex}"
        with self._session() as connection:
            connection.execute(
                "INSERT INTO artifacts(artifact_id,job_id,path,sha256,size,pinned) VALUES (?,?,?,?,?,?)",
                (artifact_id, job_id, path, sha256, size, int(pinned)),
            )
        return artifact_id

    def purge_unpinned_artifacts(self, *, job_id: str) -> list[dict[str, object]]:
        """Return removable metadata; filesystem deletion remains an explicit worker action."""

        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT artifact_id,path,sha256,size FROM artifacts WHERE job_id=? AND pinned=0",
                (job_id,),
            ).fetchall()
            connection.execute("DELETE FROM artifacts WHERE job_id=? AND pinned=0", (job_id,))
        return [dict(row) for row in rows]

    def heartbeat(self, job_id: str, *, phase: str) -> bool:
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT heartbeat_at FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise OrchestrationError("unknown job")
            now = self._timestamp()
            if (
                row[0] is not None
                and datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")).timestamp()
                + _HEARTBEAT_SECONDS
                > datetime.fromisoformat(now.replace("Z", "+00:00")).timestamp()
            ):
                return False
            connection.execute(
                "UPDATE jobs SET heartbeat_at=?,phase=? WHERE job_id=?", (now, phase, job_id)
            )
            self._event(connection, job_id, "heartbeat")
            return True

    def cancel(self, job_id: str) -> None:
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT run_id,status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None or row[1] not in {
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
                JobStatus.EVALUATING.value,
            }:
                raise OrchestrationError("job cannot be cancelled")
            if row[1] == JobStatus.RUNNING.value:
                connection.execute("UPDATE jobs SET cancel_requested=1 WHERE job_id=?", (job_id,))
                self._event(connection, job_id, "cancel_requested")
                return
            connection.execute("UPDATE jobs SET status='cancelled' WHERE job_id=?", (job_id,))
            if row[0] is not None:
                connection.execute(
                    "UPDATE runs SET released_units=reserved_units-validated_units WHERE run_id=?",
                    (row[0],),
                )
            self._event(connection, job_id, "cancelled")

    def status(self, job_id: str) -> dict[str, object]:
        with self._session() as connection:
            row = connection.execute(
                "SELECT j.job_id,j.run_id,j.status,j.phase,j.heartbeat_at,r.reserved_units,r.validated_units,r.released_units FROM jobs j LEFT JOIN runs r ON r.run_id=j.run_id WHERE j.job_id=?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise OrchestrationError("unknown job")
        result = dict(row)
        if result["run_id"] is not None and result["reserved_units"] is not None:
            result["remaining_units"] = (
                int(result["reserved_units"])
                - int(result["validated_units"])
                - int(result["released_units"])
            )
            result["eta_confidence"] = "provisional"
        else:
            result["remaining_units"] = 0
            result["eta_confidence"] = "not_applicable"
        return result

    def status_for_run(self, run_id: str) -> dict[str, object]:
        with self._session() as connection:
            row = connection.execute(
                "SELECT job_id FROM jobs WHERE run_id=? ORDER BY created_at DESC LIMIT 1", (run_id,)
            ).fetchone()
        if row is None:
            raise OrchestrationError("unknown run")
        return self.status(str(row[0]))
