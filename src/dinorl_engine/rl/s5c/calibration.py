"""Resumable first-stable-gate calibration for RL-S5c-A and RL-S5c-A2."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
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
from dinorl_engine.rl.s5c.evaluation import (
    evaluate_gate_checkpoint,
    read_evaluation_evidence,
    write_evaluation_evidence,
)
from dinorl_engine.rl.s5c.evidence import controller_zero_shove_failure, validate_absolute_style
from dinorl_engine.rl.s5c.provenance import initialization_record, repository_content_sha256
from dinorl_engine.rl.s5c.rewards import ARCHETYPES, specialist_rewards
from dinorl_engine.rl.s5c.strength import evaluate_against_strong_pool
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore
from dinorl_engine.rl.training.unit import create_maskable_ppo

__all__ = ["CalibrationError", "CalibrationSeedState", "calibrate", "calibrate_campaign"]

_FORMAT = "s5c-calibration-v1"
_A2_FORMAT = "s5c-a2-calibration-v1"
_EVALUATION_INTERVAL = 5
_TERMINAL = {"first_stable_gate", "failed_budget", "failed_style", "failed_integrity"}
_WINDOWS_TRANSIENT_RENAME_ERRORS = frozenset({5, 32, 33})
_ATOMIC_RENAME_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4) + (0.5,) * 60


class CalibrationError(RuntimeError):
    """Raised when a calibration violates its frozen state-machine contract."""


@dataclass(slots=True)
class CalibrationSeedState:
    """Persistable gate state for one zero-initialized specialist seed."""

    archetype: str
    seed: int
    max_units: int
    consecutive_passes: int = 0
    first_stable_gate: int | None = None
    current_status: str = "training"
    last_evaluation_unit: int = 0

    def __post_init__(self) -> None:
        if self.archetype not in ARCHETYPES:
            raise CalibrationError("unknown specialist archetype")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise CalibrationError("seed must be an unsigned 32-bit integer")
        if self.max_units != 147:
            raise CalibrationError("max_units must remain 147")
        if self.current_status not in {
            "training",
            "first_gate_pass",
            "first_stable_gate",
            "failed_budget",
            "failed_style",
            "failed_integrity",
            "interrupted",
        }:
            raise CalibrationError("unknown calibration status")
        if (
            type(self.last_evaluation_unit) is not int
            or self.last_evaluation_unit < 0
            or self.last_evaluation_unit > self.max_units
            or self.last_evaluation_unit % _EVALUATION_INTERVAL
        ):
            raise CalibrationError("last_evaluation_unit is invalid")

    @property
    def status(self) -> str:
        return self.current_status

    @property
    def can_resume(self) -> bool:
        return self.current_status in {"training", "interrupted"}

    def record_evaluation(self, *, unit: int, thresholds_passed: bool) -> None:
        if self.current_status in _TERMINAL:
            raise CalibrationError(f"run already stopped at {self.current_status}")
        if type(unit) is not int or unit <= 0 or unit > self.max_units or unit % 5:
            raise CalibrationError("evaluation unit must be a five-unit point within the budget")
        if unit <= self.last_evaluation_unit:
            raise CalibrationError("evaluation unit was already recorded")
        self.last_evaluation_unit = unit
        self.consecutive_passes = self.consecutive_passes + 1 if thresholds_passed else 0
        if self.consecutive_passes == 2:
            self.first_stable_gate = unit
            self.current_status = "first_stable_gate"
        elif self.consecutive_passes == 1:
            self.current_status = "first_gate_pass"
        else:
            self.current_status = "training"

    def mark_failed_budget(self, *, unit: int) -> None:
        if unit != self.max_units or self.first_stable_gate is not None:
            raise CalibrationError("failed_budget requires the exhausted pre-gate budget")
        self.current_status = "failed_budget"

    def mark_failed_integrity(self) -> None:
        if self.current_status in {"failed_budget", "interrupted"}:
            raise CalibrationError("cannot fail integrity from this calibration state")
        self.current_status = "failed_integrity"

    def mark_failed_style(self) -> None:
        if self.current_status not in {"training", "first_gate_pass", "first_stable_gate"}:
            raise CalibrationError("cannot fail style from this calibration state")
        self.current_status = "failed_style"

    def mark_interrupted(self) -> None:
        if self.current_status not in {"training", "first_gate_pass", "interrupted"}:
            raise CalibrationError("cannot interrupt a terminal calibration")
        self.current_status = "interrupted"


class _RandomLegalTrainingPool:
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
        for delay in (*_ATOMIC_RENAME_DELAYS_SECONDS, None):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                if (
                    os.name != "nt"
                    or getattr(error, "winerror", None) not in _WINDOWS_TRANSIENT_RENAME_ERRORS
                    or delay is None
                ):
                    raise
                time.sleep(delay)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _state_mapping(
    state: CalibrationSeedState, cycles: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "archetype": state.archetype,
        "consecutive_passes": state.consecutive_passes,
        "first_stable_gate": state.first_stable_gate,
        "format": _A2_FORMAT,
        "max_units": state.max_units,
        "last_evaluation_unit": state.last_evaluation_unit,
        "role_counts": _role_total(cycles),
        "seed": state.seed,
        "status": state.status,
    }


def _read_state(path: Path, *, archetype: str, seed: int, max_units: int) -> CalibrationSeedState:
    if not path.is_file():
        return CalibrationSeedState(archetype=archetype, seed=seed, max_units=max_units)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError("calibration state cannot be read") from error
    if (
        not isinstance(value, dict)
        or value.get("format") not in {_FORMAT, _A2_FORMAT}
        or value.get("archetype") != archetype
        or value.get("seed") != seed
        or value.get("max_units") != max_units
        or type(value.get("consecutive_passes")) is not int
        or (
            value.get("first_stable_gate") is not None
            and type(value.get("first_stable_gate")) is not int
        )
    ):
        raise CalibrationError("calibration state is incompatible")
    status = value.get("status", "training")
    if not isinstance(status, str):
        raise CalibrationError("calibration status is invalid")
    return CalibrationSeedState(
        archetype=archetype,
        seed=seed,
        max_units=max_units,
        consecutive_passes=value["consecutive_passes"],
        first_stable_gate=value["first_stable_gate"],
        current_status=status,
        last_evaluation_unit=(
            value.get("last_evaluation_unit", value.get("first_stable_gate", 0))
            if type(value.get("last_evaluation_unit", value.get("first_stable_gate", 0))) is int
            else 0
        ),
    )


def _read_cycles(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError("calibration cycles cannot be read") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise CalibrationError("calibration cycles are invalid")
    return list(value)


def _verify_manifest(config: S5cConfig) -> Mapping[str, object]:
    path = config.output_directory / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError("run preflight before calibration") from error
    provenance = value.get("provenance") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("format") != "s5c-a2-run-manifest-v1"
        or value.get("run_id") != config.run_id
        or value.get("config_sha256") != config.resolved_sha256
        or value.get("status") != "READY"
        or not isinstance(provenance, Mapping)
        or provenance.get("content_sha256") != repository_content_sha256(Path.cwd())
    ):
        raise CalibrationError("A2 manifest does not match the resolved configuration")
    return value


def _write_initialization(
    *, config: S5cConfig, branch: Path, archetype: str, seed: int, model: object
) -> None:
    record = initialization_record(model, seed=seed)  # type: ignore[arg-type]
    per_seed = config.output_directory / "initialization" / f"seed-{seed}.json"
    if per_seed.is_file():
        reference = json.loads(per_seed.read_text(encoding="utf-8"))
        if (
            not isinstance(reference, dict)
            or reference.get("weights_sha256") != record["weights_sha256"]
        ):
            raise CalibrationError("initial weights differ between archetypes for a common seed")
    else:
        _atomic_json(per_seed, {**record, "reference_archetype": archetype})
    _atomic_json(branch / "initialization.json", {**record, "archetype": archetype})


def _seed_result(
    state: CalibrationSeedState, cycles: list[dict[str, object]], reward_hash: str
) -> dict[str, object]:
    return {
        "archetype": state.archetype,
        "first_stable_gate": state.first_stable_gate,
        "reward_sha256": reward_hash,
        "role_counts": _role_total(cycles),
        "seed": state.seed,
        "status": state.status,
    }


def _accept_a2_evaluation(
    *,
    config: S5cConfig,
    branch: Path,
    archetype: str,
    unit: int,
    state: CalibrationSeedState,
    evaluation: Mapping[str, object],
) -> None:
    gate_value = evaluation.get("gate")
    records = evaluation.get("records")
    if not isinstance(gate_value, Mapping) or not isinstance(records, list):
        raise CalibrationError("A2 evaluation evidence is incomplete")
    state.record_evaluation(
        unit=unit, thresholds_passed=gate_value.get("thresholds_passed") is True
    )
    style_validation = validate_absolute_style(archetype, records)
    if state.first_stable_gate == unit and style_validation["passed"] is not True:
        state.mark_failed_style()
    completed = {
        **evaluation,
        "gate": {**gate_value, "stable": state.first_stable_gate == unit},
        "style_validation": style_validation,
    }
    write_evaluation_evidence(
        branch / "evaluations" / f"unit-{unit}",
        completed,
        diagnostic_replay_sample=config.diagnostic_replay_sample,
    )


def _run_seed(
    config: S5cConfig,
    *,
    archetype: str,
    seed: int,
    progress: Callable[[str], None] | None,
    resume: bool,
    stop_after_unit: int | None = None,
    internal_continue: bool = False,
    defer_strength: bool = False,
) -> dict[str, object]:
    branch = config.output_directory / "calibration" / archetype / f"seed-{seed}"
    state_path = branch / "calibration-state.json"
    state = _read_state(state_path, archetype=archetype, seed=seed, max_units=config.max_units)
    cycles_path = branch / "cycles.json"
    cycles = _read_cycles(cycles_path)
    recovery = AtomicRecoveryStore(branch / "recovery")
    latest = recovery.load_latest()
    reward = specialist_rewards()[archetype]
    if state.status in _TERMINAL:
        if not resume:
            raise CalibrationError(f"run already stopped at {state.status}")
        strength_complete = (branch / "strength.json").is_file()
        if (
            config.phase != "RL-S5c-A2"
            or state.status not in {"first_stable_gate", "failed_budget", "failed_style"}
            or strength_complete
        ):
            return _seed_result(state, cycles, reward.cache_key)
    if latest is not None and not resume:
        raise CalibrationError("existing seed requires explicit --resume")
    initially_terminal = state.status in {
        "first_stable_gate",
        "failed_budget",
        "failed_style",
    }
    if (
        latest is not None
        and not state.can_resume
        and not internal_continue
        and not initially_terminal
    ):
        raise CalibrationError(f"cannot resume seed in state {state.status}")
    opponents = _RandomLegalTrainingPool()
    vector = VectorEnvironmentConfig(backend=VectorBackend.DUMMY, n_envs=2, seed=seed)
    metadata = {
        "architecture": config.architecture,
        "archetype": archetype,
        "parent_checkpoint": None,
        "reward_sha256": reward.cache_key,
        "training_opponent": config.training_opponent,
    }
    ledger = UnitLedger(branch / "ledger.sqlite3")
    run_id = f"s5c-a2-{archetype}-{seed}"
    ledger.reserve(run_id=run_id, units=config.max_units, idempotency_key=f"{run_id}-reserve")
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
        _write_initialization(
            config=config, branch=branch, archetype=archetype, seed=seed, model=model
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
            environment_factory=lambda value: create_vector_environment(
                value, training_opponent_pool=opponents, reward_program=reward
            ),
            recovery_metadata=metadata,
        )
        completed = latest.sequence
        if state.status == "interrupted":
            state.current_status = "training"
    if (
        config.phase == "RL-S5c-A2"
        and completed > state.last_evaluation_unit
        and completed % _EVALUATION_INTERVAL == 0
    ):
        evaluation_directory = branch / "evaluations" / f"unit-{completed}"
        pending = read_evaluation_evidence(evaluation_directory)
        if pending is None:
            if latest is None:
                raise CalibrationError("pending evaluation has no recovery checkpoint")
            pending = evaluate_gate_checkpoint(
                runner.model,
                config=config,
                archetype=archetype,
                training_seed=seed,
                unit=completed,
                previous_passed=state.consecutive_passes == 1,
                checkpoint_sha256=latest.files.get("model.zip"),
                progress=progress,
            )
        _accept_a2_evaluation(
            config=config,
            branch=branch,
            archetype=archetype,
            unit=completed,
            state=state,
            evaluation=pending,
        )
        _atomic_json(state_path, _state_mapping(state, cycles))
    final_evaluation_only = state.status in {
        "first_stable_gate",
        "failed_budget",
        "failed_style",
    }
    limit = config.max_units if stop_after_unit is None else min(stop_after_unit, config.max_units)
    try:
        for unit in range(completed + 1, limit + 1) if not final_evaluation_only else ():
            result = runner.run_unit(unit_id=f"unit-{unit}", sequence=unit)
            cycles.append(
                {
                    "engine_actions": result.metrics.engine_actions,
                    "learner_transitions": result.metrics.learner_transitions,
                    "role_counts": result.metrics.role_counts,
                    "training_seconds": result.training_seconds,
                    "checkpoint_seconds": result.checkpoint_seconds,
                    "unit": unit,
                }
            )
            _atomic_json(cycles_path, cycles)
            if progress is not None:
                elapsed = sum(
                    float(cycle.get("training_seconds", 0.0))
                    + float(cycle.get("checkpoint_seconds", 0.0))
                    for cycle in cycles
                )
                rate = unit / elapsed if elapsed > 0.0 else 0.0
                eta = (config.max_units - unit) / rate if rate > 0.0 else 0.0
                progress(
                    f"S5c-A2 {archetype} seed {seed}: {unit}/{config.max_units} units "
                    f"({100.0 * unit / config.max_units:.1f}%) "
                    f"rate={rate:.3f} unit/s ETA={eta:.0f}s"
                )
            if unit % _EVALUATION_INTERVAL:
                continue
            if config.phase == "RL-S5c-A2":
                evaluation = evaluate_gate_checkpoint(
                    runner.model,
                    config=config,
                    archetype=archetype,
                    training_seed=seed,
                    unit=unit,
                    previous_passed=state.consecutive_passes == 1,
                    checkpoint_sha256=result.recovery.files.get("model.zip"),
                    progress=progress,
                )
                _accept_a2_evaluation(
                    config=config,
                    branch=branch,
                    archetype=archetype,
                    unit=unit,
                    state=state,
                    evaluation=evaluation,
                )
            else:
                evaluation = evaluate_deterministic(runner.model, seed=config.evaluation_seeds[0])
                scores = evaluation.get("scores")
                if not isinstance(scores, dict):
                    raise CalibrationError("deterministic calibration evaluation has no scores")
                gate = deterministic_gate(scores, previous_passed=state.consecutive_passes == 1)
                state.record_evaluation(
                    unit=unit, thresholds_passed=gate["thresholds_passed"] is True
                )
                _atomic_json(
                    branch / "evaluations" / f"unit-{unit}.json",
                    {"deterministic": evaluation, "gate": gate, "unit": unit},
                )
            _atomic_json(state_path, _state_mapping(state, cycles))
            if state.first_stable_gate is not None:
                break
        completed_now = int(cycles[-1]["unit"]) if cycles else completed
        if completed_now == config.max_units and state.first_stable_gate is None:
            state.mark_failed_budget(unit=config.max_units)
        _atomic_json(state_path, _state_mapping(state, cycles))
        if (
            state.status in {"first_stable_gate", "failed_budget", "failed_style"}
            and config.phase == "RL-S5c-A2"
            and not defer_strength
        ):
            if config.rl_s5b_pool is None or config.rl_s5b_pool_sha256 is None:
                raise CalibrationError("A2 strong-pool inputs are missing")
            strength_path = branch / "strength.json"
            if not strength_path.is_file():
                strength = evaluate_against_strong_pool(
                    runner.model,
                    pool_path=config.rl_s5b_pool,
                    expected_sha256=config.rl_s5b_pool_sha256,
                    evaluation_seeds=config.evaluation_seeds,
                    confrontations=config.strong_pool_confrontations,
                    learner_policy_id=f"{archetype}-s{seed}-u{completed_now}",
                )
                _atomic_json(strength_path, strength)
        return _seed_result(state, cycles, reward.cache_key)
    except KeyboardInterrupt:
        state.mark_interrupted()
        _atomic_json(state_path, _state_mapping(state, cycles))
        raise
    finally:
        runner.environment.close()


def _controller_observations(config: S5cConfig) -> dict[int, dict[int, tuple[int, int]]]:
    result: dict[int, dict[int, tuple[int, int]]] = {}
    for seed in config.calibration_seeds:
        checkpoints: dict[int, tuple[int, int]] = {}
        for unit in (5, 10):
            path = (
                config.output_directory
                / "calibration"
                / "controller"
                / f"seed-{seed}"
                / "evaluations"
                / f"unit-{unit}"
                / "summary.json"
            )
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                style = summary["aggregates"]["global"]["style_events"]
                checkpoints[unit] = (
                    int(style["shove_attempted"]),
                    int(style["shove_legal_opportunity"]),
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        result[seed] = checkpoints
    return result


def _mark_controller_integrity_failure(
    config: S5cConfig, observations: Mapping[int, Mapping[int, tuple[int, int]]]
) -> None:
    diagnostic = {
        "format": "s5c-a2-controller-integrity-v1",
        "observations": {
            str(seed): {str(unit): list(values) for unit, values in units.items()}
            for seed, units in observations.items()
        },
        "reason": (
            "zero shove attempts at units 5 and 10 across all seeds despite legal opportunities"
        ),
        "status": "failed_integrity",
    }
    _atomic_json(
        config.output_directory / "calibration" / "controller" / "integrity-diagnostic.json",
        diagnostic,
    )
    for seed in config.calibration_seeds:
        branch = config.output_directory / "calibration" / "controller" / f"seed-{seed}"
        state = _read_state(
            branch / "calibration-state.json",
            archetype="controller",
            seed=seed,
            max_units=config.max_units,
        )
        if state.status != "failed_integrity":
            state.mark_failed_integrity()
            _atomic_json(
                branch / "calibration-state.json",
                _state_mapping(state, _read_cycles(branch / "cycles.json")),
            )


def calibrate(
    config: S5cConfig,
    *,
    archetype: str,
    progress: Callable[[str], None] | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Run all three common-seed calibrations for one archetype."""

    if archetype not in ARCHETYPES:
        raise CalibrationError("unknown specialist archetype")
    if config.phase == "RL-S5c-A2":
        _verify_manifest(config)
    if archetype == "controller" and config.phase == "RL-S5c-A2":
        initial = [
            _run_seed(
                config,
                archetype=archetype,
                seed=seed,
                progress=progress,
                resume=resume,
                stop_after_unit=10,
                internal_continue=True,
                defer_strength=True,
            )
            for seed in config.calibration_seeds
        ]
        observations = _controller_observations(config)
        if controller_zero_shove_failure(observations):
            _mark_controller_integrity_failure(config, observations)
            seeds = [
                _seed_result(
                    _read_state(
                        config.output_directory
                        / "calibration"
                        / archetype
                        / f"seed-{seed}"
                        / "calibration-state.json",
                        archetype=archetype,
                        seed=seed,
                        max_units=config.max_units,
                    ),
                    _read_cycles(
                        config.output_directory
                        / "calibration"
                        / archetype
                        / f"seed-{seed}"
                        / "cycles.json"
                    ),
                    specialist_rewards()[archetype].cache_key,
                )
                for seed in config.calibration_seeds
            ]
        else:
            seeds = [
                _run_seed(
                    config,
                    archetype=archetype,
                    seed=int(item["seed"]),
                    progress=progress,
                    resume=True,
                    internal_continue=True,
                )
                for item in initial
            ]
    else:
        seeds = [
            _run_seed(config, archetype=archetype, seed=seed, progress=progress, resume=resume)
            for seed in config.calibration_seeds
        ]
    reward = specialist_rewards()[archetype]
    result = {
        "architecture": config.architecture,
        "archetype": archetype,
        "format": _A2_FORMAT if config.phase == "RL-S5c-A2" else _FORMAT,
        "max_rounds": config.max_rounds,
        "reward_sha256": reward.cache_key,
        "seeds": seeds,
        "training_opponent": config.training_opponent,
    }
    _atomic_json(config.output_directory / "calibration" / archetype / "result.json", result)
    return result


def calibrate_campaign(
    config: S5cConfig, *, progress: Callable[[str], None] | None = None, resume: bool = False
) -> dict[str, object]:
    """Run or idempotently resume all three A2 archetypes; never start phase B."""

    _verify_manifest(config)
    results = [
        calibrate(config, archetype=name, progress=progress, resume=resume) for name in ARCHETYPES
    ]
    campaign = {
        "format": "s5c-a2-campaign-result-v1",
        "phase": config.phase,
        "run_id": config.run_id,
        "archetypes": results,
        "status": "completed",
    }
    _atomic_json(config.output_directory / "result.json", campaign)
    return campaign
