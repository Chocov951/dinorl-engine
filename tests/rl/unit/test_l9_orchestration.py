"""Red-first contracts for the RL-L9 SQLite scheduler."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from dinorl_engine.rl.orchestration.database import JobPriority, JobStatus, OrchestrationStore


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 25, tzinfo=UTC)

    def now(self) -> datetime:
        return self.value


def test_l9_enables_wal_and_idempotently_reserves_a_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = OrchestrationStore(tmp_path / "l9.sqlite3")

    first = store.submit_training(owner_id="alice", run_id="run-a", budget_units=7, key="key-a")
    second = store.submit_training(owner_id="alice", run_id="run-a", budget_units=7, key="key-a")

    assert store.journal_mode == "wal"
    assert (tmp_path / "l9.sqlite3").is_file()
    assert first == second
    assert store.status(first)["reserved_units"] == 7


def test_l9_serves_interactive_work_then_rotates_training_owners(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = OrchestrationStore(tmp_path / "l9.sqlite3")
    first = store.submit_training(owner_id="alice", run_id="run-a", budget_units=10, key="a")
    second = store.submit_training(owner_id="bob", run_id="run-b", budget_units=10, key="b")
    interactive = store.submit_job(
        owner_id="alice", run_id=None, priority=JobPriority.P0, kind="interactive", key="p0"
    )

    assert store.claim_next() == interactive
    store.transition(interactive, JobStatus.COMPLETED, "done")
    assert store.claim_next() == first
    store.complete_unit(first, "unit-1")
    assert store.claim_next() == second


def test_l9_cancellation_releases_unspent_budget_and_heartbeat_is_rate_limited(tmp_path) -> None:  # type: ignore[no-untyped-def]
    clock = Clock()
    store = OrchestrationStore(tmp_path / "l9.sqlite3", now=clock.now)
    job = store.submit_training(owner_id="alice", run_id="run-a", budget_units=3, key="a")
    assert store.claim_next() == job
    assert store.heartbeat(job, phase="collecting") is True
    assert store.heartbeat(job, phase="collecting") is False
    clock.value += timedelta(seconds=30)
    assert store.heartbeat(job, phase="optimizing") is True

    store.complete_unit(job, "unit-1")
    store.cancel(job)

    status = store.status(job)
    assert status["status"] == "cancelled"
    assert status["validated_units"] == 1
    assert status["released_units"] == 2


def test_l9_running_cancellation_waits_for_the_current_unit_boundary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = OrchestrationStore(tmp_path / "l9.sqlite3")
    job = store.submit_training(owner_id="alice", run_id="run-a", budget_units=2, key="a")
    assert store.claim_next() == job
    store.cancel(job)
    assert store.status(job)["status"] == "running"
    store.complete_unit(job, "unit-1")
    assert store.status(job)["status"] == "cancelled"
    assert store.status(job)["released_units"] == 1


def test_l9_retries_only_transient_jobs_three_times_and_caches_evaluations(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = OrchestrationStore(tmp_path / "l9.sqlite3")
    job = store.submit_training(owner_id="alice", run_id="run-a", budget_units=3, key="a")
    assert store.claim_next() == job

    assert store.technical_failure(job, "temporary") is JobStatus.QUEUED
    assert store.claim_next() == job
    assert store.technical_failure(job, "temporary") is JobStatus.QUEUED
    assert store.claim_next() == job
    assert store.technical_failure(job, "temporary") is JobStatus.FAILED
    assert store.status(job)["released_units"] == 3
    assert store.cache_evaluation("suite-key", '{"result":"pass"}') is True
    assert store.cache_evaluation("suite-key", '{"result":"pass"}') is False
    assert store.cached_evaluation("suite-key") == '{"result":"pass"}'


def test_l9_retention_never_selects_pinned_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = OrchestrationStore(tmp_path / "l9.sqlite3")
    job = store.submit_job(
        owner_id="alice", run_id=None, priority=JobPriority.P1, kind="evaluation", key="e"
    )
    digest = "0" * 64
    store.register_artifact(job_id=job, path="old.json", sha256=digest, size=1)
    store.register_artifact(job_id=job, path="pinned.json", sha256=digest, size=1, pinned=True)

    assert [item["path"] for item in store.purge_unpinned_artifacts(job_id=job)] == ["old.json"]


def test_l9_concurrent_reservations_share_one_idempotent_job(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "l9.sqlite3"

    def submit() -> str:
        return OrchestrationStore(path).submit_training(
            owner_id="alice", run_id="run-a", budget_units=4, key="same-key"
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = list(executor.map(lambda _: submit(), range(4)))

    assert len(set(jobs)) == 1
