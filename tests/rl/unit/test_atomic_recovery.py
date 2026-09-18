"""Contracts for RL-L5 durable unit recovery and idempotent billing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.training.runner import (
    AtomicRecoveryStore,
    CrashPoint,
    SimulatedCrash,
)


def _state(unit_number: int) -> dict[str, object]:
    return {"unit_number": unit_number, "payload": {"counter": unit_number * 10}}


def test_crashing_before_the_atomic_rename_keeps_the_previous_valid_state(
    tmp_path: Path,
) -> None:
    store = AtomicRecoveryStore(tmp_path / "recovery")
    store.commit_json_state(unit_id="unit-1", sequence=1, state=_state(1))

    for crash_point in (
        CrashPoint.SERIALIZED,
        CrashPoint.FSYNCED,
        CrashPoint.HASHED,
        CrashPoint.MANIFEST_WRITTEN,
    ):
        with pytest.raises(SimulatedCrash):
            store.commit_json_state(
                unit_id=f"failed-{crash_point.value}",
                sequence=2,
                state=_state(2),
                crash_at=crash_point,
            )

        recovered = store.load_latest()
        assert recovered is not None
        assert recovered.unit_id == "unit-1"
        assert recovered.state == _state(1)


def test_renamed_unit_is_reconciled_without_a_duplicate_debit(tmp_path: Path) -> None:
    store = AtomicRecoveryStore(tmp_path / "recovery")
    ledger = UnitLedger(tmp_path / "ledger.sqlite3")
    ledger.reserve(run_id="run-1", units=2, idempotency_key="reserve-run-1")

    with pytest.raises(SimulatedCrash):
        store.commit_json_state(
            unit_id="unit-1",
            sequence=1,
            state=_state(1),
            crash_at=CrashPoint.RENAMED,
        )

    recovered = store.load_latest()
    assert recovered is not None
    assert recovered.unit_id == "unit-1"
    assert ledger.debit_validated_unit(
        run_id="run-1", unit_id=recovered.unit_id, idempotency_key="debit-unit-1"
    )
    assert not ledger.debit_validated_unit(
        run_id="run-1", unit_id=recovered.unit_id, idempotency_key="debit-unit-1"
    )
    assert ledger.debited_units("run-1") == 1
    assert ledger.remaining_units("run-1") == 1


def test_recovery_rejects_a_tampered_state_file(tmp_path: Path) -> None:
    store = AtomicRecoveryStore(tmp_path / "recovery")
    record = store.commit_json_state(unit_id="unit-1", sequence=1, state=_state(1))
    (record.directory / "state.json").write_text('{"tampered":true}\n', encoding="utf-8")

    assert store.load_latest() is None


def test_discard_before_does_not_fail_when_a_sync_client_locks_stale_recovery(
    tmp_path: Path,
) -> None:
    store = AtomicRecoveryStore(tmp_path / "recovery")
    stale = store.commit_json_state(unit_id="unit-1", sequence=1, state=_state(1))
    current = store.commit_json_state(unit_id="unit-2", sequence=2, state=_state(2))

    with patch("dinorl_engine.rl.training.runner.shutil.rmtree", side_effect=PermissionError):
        store.discard_before(2)

    assert stale.directory.is_dir()
    latest = store.load_latest()
    assert latest is not None
    assert latest.directory == current.directory
