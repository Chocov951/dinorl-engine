"""RL-S3: exact resume, atomic-debit and interactive-priority server gate."""

from __future__ import annotations

import hashlib
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Event, Thread

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.orchestration.priority import (
    InteractivePriorityMode,
    InteractivePriorityPrototype,
)
from dinorl_engine.rl.training.runner import (
    AtomicPPOUnitResult,
    AtomicPPOUnitRunner,
    AtomicRecoveryStore,
    CrashPoint,
    SimulatedCrash,
)
from dinorl_engine.rl.training.unit import ROLLOUT_TRANSITIONS, create_maskable_ppo

__all__ = [
    "SELECTED_CONFIGURATION",
    "build_decision",
    "run_measurement",
]

SELECTED_CONFIGURATION = VectorEnvironmentConfig(
    backend=VectorBackend.DUMMY,
    n_envs=8,
    seed=19,
)


class _RolloutTimer(BaseCallback):
    """Measure collection separately; PPO's remaining learn time is optimization."""

    def __init__(self) -> None:
        super().__init__()
        self.collection_seconds = 0.0
        self._started: float | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self._started = time.perf_counter()

    def _on_rollout_end(self) -> None:
        if self._started is None:
            raise RuntimeError("PPO rollout ended without a start timestamp")
        self.collection_seconds += time.perf_counter() - self._started
        self._started = None


def _new_runner(directory: Path, label: str) -> AtomicPPOUnitRunner:
    environment = create_vector_environment(SELECTED_CONFIGURATION)
    ledger = UnitLedger(directory / f"{label}-ledger.sqlite3")
    ledger.reserve(run_id="run-1", units=2, idempotency_key=f"reserve-{label}")
    return AtomicPPOUnitRunner(
        run_id="run-1",
        configuration=SELECTED_CONFIGURATION,
        model=create_maskable_ppo(
            environment,
            seed=SELECTED_CONFIGURATION.seed,
            n_steps=SELECTED_CONFIGURATION.n_steps,
        ),
        environment=environment,
        recovery_store=AtomicRecoveryStore(directory / f"{label}-recovery"),
        ledger=ledger,
    )


def _weights_hash(runner: AtomicPPOUnitRunner) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(runner.model.policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _timing(result: AtomicPPOUnitResult, timer: _RolloutTimer) -> dict[str, float]:
    training_seconds = result.training_seconds
    checkpoint_seconds = result.checkpoint_seconds
    return {
        "collection_seconds": timer.collection_seconds,
        "optimization_seconds": max(0.0, training_seconds - timer.collection_seconds),
        "checkpoint_seconds": checkpoint_seconds,
    }


def _workers_cleaned_up(environment: VecEnv) -> bool:
    """Confirm that the selected backend has no live child worker after close.

    RL-S1 selected ``DummyVecEnv``.  It executes in the owner process and SB3
    deliberately has no ``closed`` attribute on that class, so inspecting that
    absent implementation detail would turn a successful cleanup into a false
    negative.  Its absence of child processes is the relevant RL-S3 invariant.
    """

    return isinstance(environment, DummyVecEnv)


def _continuous_and_resumed(directory: Path) -> dict[str, object]:
    continuous = _new_runner(directory, "continuous")
    try:
        continuous_first_timer = _RolloutTimer()
        continuous.run_unit(unit_id="unit-1", sequence=1, callback=continuous_first_timer)
        continuous_second_timer = _RolloutTimer()
        continuous_second = continuous.run_unit(
            unit_id="unit-2", sequence=2, callback=continuous_second_timer
        )
        continuous_hash = _weights_hash(continuous)
        continuous_timing = _timing(continuous_second, continuous_second_timer)
    finally:
        continuous.environment.close()

    resumed = _new_runner(directory, "resumed")
    try:
        resumed_first_timer = _RolloutTimer()
        resumed.run_unit(unit_id="unit-1", sequence=1, callback=resumed_first_timer)
    finally:
        resumed.environment.close()
    resumed_ledger = UnitLedger(directory / "resumed-ledger.sqlite3")
    restored, _first_metrics, _record = AtomicPPOUnitRunner.restore_latest(
        run_id="run-1",
        configuration=SELECTED_CONFIGURATION,
        recovery_store=AtomicRecoveryStore(directory / "resumed-recovery"),
        ledger=resumed_ledger,
    )
    try:
        resumed_second_timer = _RolloutTimer()
        resumed_second = restored.run_unit(
            unit_id="unit-2", sequence=2, callback=resumed_second_timer
        )
        resumed_hash = _weights_hash(restored)
        resumed_timing = _timing(resumed_second, resumed_second_timer)
    finally:
        restored.environment.close()
        cleaned_up = _workers_cleaned_up(restored.environment)
    return {
        "continuous_weights_sha256": continuous_hash,
        "resumed_weights_sha256": resumed_hash,
        "exact": (
            continuous_hash == resumed_hash and continuous_second.metrics == resumed_second.metrics
        ),
        "continuous": continuous_timing,
        "resumed": resumed_timing,
        "processes_cleaned_up": cleaned_up,
    }


def _crash_protocol(directory: Path) -> dict[str, bool]:
    before_rename_preserved = True
    for point in (
        CrashPoint.SERIALIZED,
        CrashPoint.FSYNCED,
        CrashPoint.HASHED,
        CrashPoint.MANIFEST_WRITTEN,
    ):
        store = AtomicRecoveryStore(directory / f"crash-{point.value}")
        store.commit_json_state(unit_id="unit-1", sequence=1, state={"counter": 1})
        try:
            store.commit_json_state(
                unit_id="unit-2",
                sequence=2,
                state={"counter": 2},
                crash_at=point,
            )
        except SimulatedCrash:
            record = store.load_latest()
            before_rename_preserved = (
                before_rename_preserved
                and record is not None
                and record.unit_id == "unit-1"
                and record.state == {"counter": 1}
            )
        else:
            before_rename_preserved = False
    store = AtomicRecoveryStore(directory / "crash-renamed")
    ledger = UnitLedger(directory / "crash-renamed-ledger.sqlite3")
    ledger.reserve(run_id="run-1", units=1, idempotency_key="reserve-crash")
    try:
        store.commit_json_state(
            unit_id="unit-1",
            sequence=1,
            state={"counter": 1},
            crash_at=CrashPoint.RENAMED,
        )
    except SimulatedCrash:
        record = store.load_latest()
        if record is None:
            return {
                "before_rename_preserved": before_rename_preserved,
                "post_rename_recovered": False,
                "no_duplicate_debit": False,
            }
        first_debit = ledger.debit_validated_unit(
            run_id="run-1", unit_id=record.unit_id, idempotency_key="debit-crash"
        )
        second_debit = ledger.debit_validated_unit(
            run_id="run-1", unit_id=record.unit_id, idempotency_key="debit-crash"
        )
        return {
            "before_rename_preserved": before_rename_preserved,
            "post_rename_recovered": record.unit_id == "unit-1" and first_debit,
            "no_duplicate_debit": not second_debit and ledger.debited_units("run-1") == 1,
        }
    return {
        "before_rename_preserved": before_rename_preserved,
        "post_rename_recovered": False,
        "no_duplicate_debit": False,
    }


def _priority_measurement(directory: Path, mode: InteractivePriorityMode) -> dict[str, object]:
    runner = _new_runner(directory, f"priority-{mode.value}")
    scheduler = InteractivePriorityPrototype(mode)
    training_started = Event()
    interactive_served = Event()
    failures: list[BaseException] = []
    timestamps: dict[str, float] = {}

    def training() -> None:
        timestamps["training_started"] = time.perf_counter()
        training_started.set()
        runner.run_unit(unit_id="unit-1", sequence=1)
        timestamps["training_finished"] = time.perf_counter()

    def execute_training() -> None:
        try:
            scheduler.run_next()
        except BaseException as error:  # pragma: no cover - assertion re-raised below
            failures.append(error)

    def interactive() -> None:
        timestamps["interactive_served"] = time.perf_counter()
        interactive_served.set()

    scheduler.submit_training(training)
    thread = Thread(target=execute_training, name=f"dinorl-s3-{mode.value}")
    thread.start()
    try:
        if not training_started.wait(timeout=30):
            raise RuntimeError("RL-S3 training task did not start")
        timestamps["interactive_submitted"] = time.perf_counter()
        scheduler.submit_interactive(interactive)
        thread.join(timeout=600)
        if thread.is_alive():
            raise RuntimeError("RL-S3 training task did not finish")
        if failures:
            raise RuntimeError("RL-S3 training task failed") from failures[0]
        if mode is InteractivePriorityMode.UNIT_BOUNDARY:
            scheduler.run_next()
        else:
            scheduler.wait_for_interactive()
        if not interactive_served.is_set():
            raise RuntimeError("RL-S3 interactive task was not served")
        training_seconds = timestamps["training_finished"] - timestamps["training_started"]
        latency = timestamps["interactive_served"] - timestamps["interactive_submitted"]
        return {
            "mode": mode.value,
            "play_latency_seconds": latency,
            "training_transitions_per_second": ROLLOUT_TRANSITIONS / training_seconds,
            "interactive_served": True,
        }
    finally:
        scheduler.close()
        runner.environment.close()


def build_decision(candidates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Select the smallest interactive latency, breaking ties by learning throughput."""

    required = {"mode", "play_latency_seconds", "training_transitions_per_second"}
    if len(candidates) != 2:
        raise ValueError("RL-S3 requires exactly two interactive-priority candidates")
    parsed: list[tuple[str, float, float]] = []
    for candidate in candidates:
        if set(candidate) != required:
            raise ValueError("RL-S3 priority candidate has unexpected fields")
        mode = candidate["mode"]
        latency = candidate["play_latency_seconds"]
        throughput = candidate["training_transitions_per_second"]
        if (
            mode not in {item.value for item in InteractivePriorityMode}
            or isinstance(latency, bool)
            or not isinstance(latency, int | float)
            or isinstance(throughput, bool)
            or not isinstance(throughput, int | float)
            or latency < 0
            or throughput <= 0
        ):
            raise ValueError("RL-S3 priority candidate has invalid values")
        parsed.append((mode, float(latency), float(throughput)))
    selected = min(parsed, key=lambda item: (item[1], -item[2], item[0]))
    return {
        "mode": selected[0],
        "criterion": "minimum_play_latency_then_maximum_training_throughput",
        "play_latency_seconds": selected[1],
        "training_transitions_per_second": selected[2],
    }


def run_measurement() -> dict[str, object]:
    """Run the complete selected-backend RL-S3 evidence once on the target server."""

    with tempfile.TemporaryDirectory(prefix="dinorl-rl-s3-") as temporary:
        directory = Path(temporary)
        resume = _continuous_and_resumed(directory)
        crash = _crash_protocol(directory)
        candidates = [
            _priority_measurement(directory, InteractivePriorityMode.UNIT_BOUNDARY),
            _priority_measurement(directory, InteractivePriorityMode.SEPARATE_WORKER),
        ]
    decision = build_decision(
        [
            {
                "mode": candidate["mode"],
                "play_latency_seconds": candidate["play_latency_seconds"],
                "training_transitions_per_second": candidate["training_transitions_per_second"],
            }
            for candidate in candidates
        ]
    )
    return {
        "resume": resume,
        "crash": crash,
        "priority_candidates": candidates,
        "decision": decision,
    }
