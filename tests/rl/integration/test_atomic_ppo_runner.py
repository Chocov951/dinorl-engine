"""End-to-end persistence contract for one selected-vector PPO unit."""

from __future__ import annotations

from pathlib import Path

import torch

from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore
from dinorl_engine.rl.training.unit import create_maskable_ppo


def test_atomic_ppo_unit_can_be_restored_and_reconciled_once(tmp_path: Path) -> None:
    configuration = VectorEnvironmentConfig(backend=VectorBackend.DUMMY, n_envs=8, seed=19)
    environment = create_vector_environment(configuration)
    ledger = UnitLedger(tmp_path / "ledger.sqlite3")
    ledger.reserve(run_id="run-1", units=2, idempotency_key="reserve-run-1")
    runner = AtomicPPOUnitRunner(
        run_id="run-1",
        configuration=configuration,
        model=create_maskable_ppo(environment, seed=19, n_steps=configuration.n_steps),
        environment=environment,
        recovery_store=AtomicRecoveryStore(tmp_path / "recovery"),
        ledger=ledger,
    )
    try:
        completed = runner.run_unit(unit_id="unit-1", sequence=1)
        expected_snapshots = [item.snapshot_recovery_state() for item in environment.envs]
    finally:
        environment.close()

    restored, metrics, record = AtomicPPOUnitRunner.restore_latest(
        run_id="run-1",
        configuration=configuration,
        recovery_store=AtomicRecoveryStore(tmp_path / "recovery"),
        ledger=ledger,
    )
    try:
        assert completed.debited
        assert record.unit_id == "unit-1"
        assert metrics == completed.metrics
        assert restored.model.num_timesteps == 2048
        actual_snapshots = [item.snapshot_recovery_state() for item in restored.environment.envs]
        assert actual_snapshots == expected_snapshots
        assert ledger.debited_units("run-1") == 1
    finally:
        restored.environment.close()


def test_continuous_and_resumed_units_produce_identical_model_weights(tmp_path: Path) -> None:
    configuration = VectorEnvironmentConfig(backend=VectorBackend.DUMMY, n_envs=8, seed=19)

    continuous_environment = create_vector_environment(configuration)
    continuous_ledger = UnitLedger(tmp_path / "continuous-ledger.sqlite3")
    continuous_ledger.reserve(run_id="run-1", units=2, idempotency_key="reserve-continuous")
    continuous_runner = AtomicPPOUnitRunner(
        run_id="run-1",
        configuration=configuration,
        model=create_maskable_ppo(continuous_environment, seed=19, n_steps=configuration.n_steps),
        environment=continuous_environment,
        recovery_store=AtomicRecoveryStore(tmp_path / "continuous-recovery"),
        ledger=continuous_ledger,
    )
    try:
        continuous_runner.run_unit(unit_id="unit-1", sequence=1)
        continuous_second = continuous_runner.run_unit(unit_id="unit-2", sequence=2)
        continuous_weights = {
            name: tensor.detach().clone()
            for name, tensor in continuous_runner.model.policy.state_dict().items()
        }
    finally:
        continuous_environment.close()

    resumed_environment = create_vector_environment(configuration)
    resumed_ledger = UnitLedger(tmp_path / "resumed-ledger.sqlite3")
    resumed_ledger.reserve(run_id="run-1", units=2, idempotency_key="reserve-resumed")
    first_runner = AtomicPPOUnitRunner(
        run_id="run-1",
        configuration=configuration,
        model=create_maskable_ppo(resumed_environment, seed=19, n_steps=configuration.n_steps),
        environment=resumed_environment,
        recovery_store=AtomicRecoveryStore(tmp_path / "resumed-recovery"),
        ledger=resumed_ledger,
    )
    try:
        first_runner.run_unit(unit_id="unit-1", sequence=1)
    finally:
        resumed_environment.close()
    resumed_runner, _metrics, _record = AtomicPPOUnitRunner.restore_latest(
        run_id="run-1",
        configuration=configuration,
        recovery_store=AtomicRecoveryStore(tmp_path / "resumed-recovery"),
        ledger=resumed_ledger,
    )
    try:
        resumed_second = resumed_runner.run_unit(unit_id="unit-2", sequence=2)
        resumed_weights = resumed_runner.model.policy.state_dict()
    finally:
        resumed_runner.environment.close()

    assert resumed_second.metrics == continuous_second.metrics
    assert resumed_ledger.debited_units("run-1") == 2
    assert continuous_ledger.debited_units("run-1") == 2
    assert resumed_weights.keys() == continuous_weights.keys()
    for name, continuous_weight in continuous_weights.items():
        assert torch.equal(resumed_weights[name], continuous_weight), name
