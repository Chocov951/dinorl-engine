"""Small, deterministic state machine for the RL-S5c-A first-stable gate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dinorl_engine.controllers.protocol import Controller
from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID, RandomLegalController
from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.evaluation.gates import deterministic_gate
from dinorl_engine.rl.evaluation.suites import evaluate_deterministic
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.policies.local_mlp_v2 import LocalMLPV2Architecture
from dinorl_engine.rl.s5c.config import S5cConfig
from dinorl_engine.rl.s5c.rewards import ARCHETYPES, specialist_rewards
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore
from dinorl_engine.rl.training.unit import create_maskable_ppo

__all__ = ["CalibrationError", "CalibrationSeedState", "calibrate"]

_FORMAT = "s5c-calibration-v1"
_EVALUATION_INTERVAL = 5


class CalibrationError(RuntimeError):
    """Raised when calibration would continue beyond the first stable gate."""


@dataclass(slots=True)
class CalibrationSeedState:
    """Persistable gate state for one zero-initialized specialist seed."""

    archetype: str
    seed: int
    max_units: int
    consecutive_passes: int = 0
    first_stable_gate: int | None = None

    def __post_init__(self) -> None:
        if self.archetype not in ARCHETYPES:
            raise CalibrationError("unknown specialist archetype")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise CalibrationError("seed must be an unsigned 32-bit integer")
        if self.max_units != 147:
            raise CalibrationError("max_units must remain 147")

    @property
    def status(self) -> str:
        return "first_stable_gate" if self.first_stable_gate is not None else "training"

    def record_evaluation(self, *, unit: int, thresholds_passed: bool) -> None:
        """Record one five-unit evaluation and stop exactly on pass number two."""

        if self.first_stable_gate is not None:
            raise CalibrationError("run already stopped at first_stable_gate")
        if type(unit) is not int or unit <= 0 or unit > self.max_units or unit % 5:
            raise CalibrationError("evaluation unit must be a five-unit point within the budget")
        self.consecutive_passes = self.consecutive_passes + 1 if thresholds_passed else 0
        if self.consecutive_passes == 2:
            self.first_stable_gate = unit


class _RandomLegalTrainingPool:
    """The only allowed S5c training opponent, with recoverable private RNG."""

    @staticmethod
    def _controller(seed: int) -> RandomLegalController:
        digest = hashlib.sha256(f"dinorl/s5c/random-legal/{seed}".encode()).digest()
        return RandomLegalController(seed=int.from_bytes(digest[:8], "big"))

    def select(
        self, *, selection_seed: int, learner_actor: Actor, first_actor: Actor
    ) -> tuple[str, Controller]:
        del learner_actor, first_actor
        return RANDOM_LEGAL_CONTROLLER_ID, self._controller(selection_seed)

    def restore(
        self,
        *,
        opponent_id: str,
        selection_seed: int,
        learner_actor: Actor,
        first_actor: Actor,
        controller_state: object,
    ) -> Controller:
        del learner_actor, first_actor
        if opponent_id != RANDOM_LEGAL_CONTROLLER_ID:
            raise CalibrationError("S5c recovery opponent is not random-legal")
        controller = self._controller(selection_seed)
        if controller_state is not None:
            controller.restore_recovery_state(controller_state)
        return controller


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_state(path: Path, *, archetype: str, seed: int, max_units: int) -> CalibrationSeedState:
    if not path.is_file():
        return CalibrationSeedState(archetype=archetype, seed=seed, max_units=max_units)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError("calibration state cannot be read") from error
    if (
        not isinstance(value, dict)
        or value.get("format") != _FORMAT
        or value.get("archetype") != archetype
        or value.get("seed") != seed
        or value.get("max_units") != max_units
        or type(value.get("consecutive_passes")) is not int
        or value.get("first_stable_gate") is not None
        and type(value.get("first_stable_gate")) is not int
    ):
        raise CalibrationError("calibration state is incompatible")
    return CalibrationSeedState(
        archetype=archetype,
        seed=seed,
        max_units=max_units,
        consecutive_passes=value["consecutive_passes"],
        first_stable_gate=value["first_stable_gate"],
    )


def _role_total(cycles: list[dict[str, object]]) -> dict[str, int]:
    names = ("learner_first", "learner_second", "learner_a", "learner_b")
    totals = {name: 0 for name in names}
    for cycle in cycles:
        counts = cycle.get("role_counts")
        if not isinstance(counts, dict) or set(counts) != set(names):
            raise CalibrationError("calibration cycle has invalid realised role counters")
        for name in names:
            value = counts[name]
            if type(value) is not int or value < 0:
                raise CalibrationError("calibration role counter is invalid")
            totals[name] += value
    return totals


def _prepare_campaign(config: S5cConfig) -> None:
    """Persist the frozen campaign inputs before its first seed starts."""

    path = config.output_directory / "manifest.json"
    rewards = specialist_rewards()
    expected = {
        "config": config.to_mapping(),
        "format": "s5c-run-manifest-v1",
        "git_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True, cwd=Path.cwd()
        ).strip(),
        "reward_sha256": {name: rewards[name].cache_key for name in ARCHETYPES},
        "run_id": config.run_id,
    }
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError("S5c campaign manifest cannot be read") from error
        if current != expected:
            raise CalibrationError("S5c campaign manifest is not immutable")
        return
    _atomic_json(path, expected)


def _run_seed(
    config: S5cConfig,
    *,
    archetype: str,
    seed: int,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    branch = config.output_directory / "calibration" / archetype / f"seed-{seed}"
    state_path = branch / "calibration-state.json"
    state = _read_state(state_path, archetype=archetype, seed=seed, max_units=config.max_units)
    if state.first_stable_gate is not None:
        raise CalibrationError("cannot resume a run after first_stable_gate")
    reward = specialist_rewards()[archetype]
    opponents = _RandomLegalTrainingPool()
    vector = VectorEnvironmentConfig(backend=VectorBackend.DUMMY, n_envs=2, seed=seed)
    metadata = {
        "architecture": config.architecture,
        "archetype": archetype,
        "parent_checkpoint": None,
        "reward_sha256": reward.cache_key,
        "training_opponent": config.training_opponent,
    }
    recovery = AtomicRecoveryStore(branch / "recovery")
    ledger = UnitLedger(branch / "ledger.sqlite3")
    run_id = f"s5c-a-{archetype}-{seed}"
    ledger.reserve(run_id=run_id, units=config.max_units, idempotency_key=f"{run_id}-reserve")
    latest = recovery.load_latest()
    if latest is None:
        environment = create_vector_environment(
            vector, training_opponent_pool=opponents, reward_program=reward
        )
        model = create_maskable_ppo(
            environment,
            seed=seed,
            n_steps=vector.n_steps,
            architecture=LocalMLPV2Architecture.COMPACT,
        )
        runner = AtomicPPOUnitRunner(
            run_id=run_id,
            configuration=vector,
            model=model,
            environment=environment,
            recovery_store=recovery,
            ledger=ledger,
            recovery_metadata=metadata,
        )
        completed = 0
    else:
        runner, _metrics, latest = AtomicPPOUnitRunner.restore_latest(
            run_id=run_id,
            configuration=vector,
            recovery_store=recovery,
            ledger=ledger,
            environment_factory=lambda vector: create_vector_environment(
                vector, training_opponent_pool=opponents, reward_program=reward
            ),
            recovery_metadata=metadata,
        )
        completed = latest.sequence
    cycles_path = branch / "cycles.json"
    cycles: list[dict[str, object]] = []
    if cycles_path.is_file():
        try:
            value = json.loads(cycles_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError("calibration cycles cannot be read") from error
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise CalibrationError("calibration cycles are invalid")
        cycles = list(value)
    try:
        for unit in range(completed + 1, config.max_units + 1):
            result = runner.run_unit(unit_id=f"unit-{unit}", sequence=unit)
            cycles.append(
                {
                    "engine_actions": result.metrics.engine_actions,
                    "learner_transitions": result.metrics.learner_transitions,
                    "role_counts": result.metrics.role_counts,
                    "unit": unit,
                }
            )
            _atomic_json(cycles_path, cycles)
            if progress is not None:
                progress(f"S5c-A {archetype} seed {seed}: {unit}/{config.max_units} units")
            if unit % _EVALUATION_INTERVAL:
                continue
            evaluation = evaluate_deterministic(runner.model, seed=config.evaluation_seeds[0])
            scores = evaluation.get("scores")
            if not isinstance(scores, dict):
                raise CalibrationError("deterministic calibration evaluation has no scores")
            gate = deterministic_gate(scores, previous_passed=state.consecutive_passes == 1)
            state.record_evaluation(unit=unit, thresholds_passed=gate["thresholds_passed"] is True)
            _atomic_json(
                branch / "evaluations" / f"unit-{unit}.json",
                {"deterministic": evaluation, "gate": gate, "unit": unit},
            )
            _atomic_json(
                state_path,
                {
                    "archetype": archetype,
                    "consecutive_passes": state.consecutive_passes,
                    "first_stable_gate": state.first_stable_gate,
                    "format": _FORMAT,
                    "max_units": config.max_units,
                    "role_counts": _role_total(cycles),
                    "seed": seed,
                    "status": state.status,
                },
            )
            if state.first_stable_gate is not None:
                break
        return {
            "archetype": archetype,
            "first_stable_gate": state.first_stable_gate,
            "reward_sha256": reward.cache_key,
            "role_counts": _role_total(cycles),
            "seed": seed,
            "status": state.status,
        }
    finally:
        runner.environment.close()


def calibrate(
    config: S5cConfig, *, archetype: str, progress: Callable[[str], None] | None = None
) -> dict[str, object]:
    """Run the three zero-start calibration seeds for one archetype only."""

    if archetype not in ARCHETYPES:
        raise CalibrationError("unknown specialist archetype")
    _prepare_campaign(config)
    reward = specialist_rewards()[archetype]
    result = {
        "architecture": config.architecture,
        "archetype": archetype,
        "format": _FORMAT,
        "max_rounds": config.max_rounds,
        "reward_sha256": reward.cache_key,
        "seeds": [
            _run_seed(config, archetype=archetype, seed=seed, progress=progress)
            for seed in config.calibration_seeds
        ],
        "training_opponent": config.training_opponent,
    }
    _atomic_json(config.output_directory / "calibration" / archetype / "result.json", result)
    return result
