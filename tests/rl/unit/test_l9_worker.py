"""RL-L9 worker boundary and restart contracts."""

from __future__ import annotations

from dinorl_engine.rl.orchestration.database import OrchestrationStore
from dinorl_engine.rl.orchestration.worker import SingleWorker


def test_worker_debits_only_a_completed_unit_and_requeues_after_os_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = OrchestrationStore(tmp_path / "l9.sqlite3")
    job = store.submit_training(owner_id="a", run_id="r", budget_units=2, key="k")
    worker = SingleWorker(store)

    assert worker.run_once(lambda _: "unit-1") == job
    assert store.status(job)["validated_units"] == 1
    assert worker.run_once(lambda _: (_ for _ in ()).throw(OSError("temporary"))) == job
    assert store.status(job)["validated_units"] == 1
