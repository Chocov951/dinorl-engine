"""Single-process, unit-boundary worker for the initial RL-L9 scheduler."""

from __future__ import annotations

from collections.abc import Callable

from dinorl_engine.rl.orchestration.database import JobStatus, OrchestrationStore

__all__ = ["SingleWorker"]


class SingleWorker:
    """Execute one claimed job atomically; training work never crosses a unit boundary."""

    def __init__(self, store: OrchestrationStore) -> None:
        self._store = store

    def run_once(self, execute: Callable[[str], str | None]) -> str | None:
        """Run one unit/task; ``None`` means successful non-training work."""

        job_id = self._store.claim_next()
        if job_id is None:
            return None
        try:
            unit_id = execute(job_id)
        except OSError as error:
            self._store.technical_failure(job_id, str(error) or "os_error")
            return job_id
        if unit_id is None:
            self._store.transition(job_id, JobStatus.COMPLETED, "completed")
        else:
            self._store.complete_unit(job_id, unit_id)
        return job_id
