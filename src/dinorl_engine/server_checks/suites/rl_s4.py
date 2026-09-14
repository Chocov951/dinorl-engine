"""RL-S4: ten-unit architecture smoke on the server-selected PPO configuration."""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks

from dinorl_engine.rl.env.vectorization import create_vector_environment
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.policies.factory import (
    PolicyArchitecture,
    architecture_specification,
    count_trainable_parameters,
)
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore
from dinorl_engine.rl.training.unit import (
    PPO_BATCH_SIZE,
    PPO_EPOCHS,
    ROLLOUT_TRANSITIONS,
    create_maskable_ppo,
)
from dinorl_engine.server_checks.suites.rl_s3 import SELECTED_CONFIGURATION

__all__ = [
    "ARCHITECTURES",
    "UNITS_PER_ARCHITECTURE",
    "run_measurement",
    "smoke_passed",
]

ARCHITECTURES = (PolicyArchitecture.MLP, PolicyArchitecture.SMALL_CNN)
UNITS_PER_ARCHITECTURE = 10
_INFERENCE_REPETITIONS = 64


def _finite_diagnostics(diagnostics: Mapping[str, float]) -> bool:
    return bool(diagnostics) and all(math.isfinite(value) for value in diagnostics.values())


def _legal_rollout_actions(runner: AtomicPPOUnitRunner) -> bool:
    buffer = runner.model.rollout_buffer
    masks = np.asarray(buffer.action_masks).reshape(-1, 9)
    actions = np.asarray(buffer.actions).reshape(-1)
    return len(masks) == ROLLOUT_TRANSITIONS and all(
        bool(mask[int(action)]) for mask, action in zip(masks, actions, strict=True)
    )


def _inference_seconds_per_observation(runner: AtomicPPOUnitRunner) -> float:
    observation = runner.model._last_obs
    if not isinstance(observation, dict):
        raise RuntimeError("RL-S4 model has no last observation for inference measurement")
    masks = get_action_masks(runner.environment)
    started = time.perf_counter()
    for _ in range(_INFERENCE_REPETITIONS):
        actions, _state = runner.model.predict(observation, action_masks=masks, deterministic=True)
        if not all(bool(mask[int(action)]) for mask, action in zip(masks, actions, strict=True)):
            raise RuntimeError("RL-S4 inference selected an action forbidden by its mask")
    elapsed = time.perf_counter() - started
    return elapsed / (_INFERENCE_REPETITIONS * runner.environment.num_envs)


def _run_candidate(directory: Path, architecture: PolicyArchitecture) -> dict[str, object]:
    environment = create_vector_environment(SELECTED_CONFIGURATION)
    identifier = architecture.value
    ledger = UnitLedger(directory / f"{identifier}-ledger.sqlite3")
    ledger.reserve(
        run_id=identifier,
        units=UNITS_PER_ARCHITECTURE,
        idempotency_key=f"reserve-{identifier}",
    )
    runner = AtomicPPOUnitRunner(
        run_id=identifier,
        configuration=SELECTED_CONFIGURATION,
        model=create_maskable_ppo(
            environment,
            seed=SELECTED_CONFIGURATION.seed,
            n_steps=SELECTED_CONFIGURATION.n_steps,
            architecture=architecture,
        ),
        environment=environment,
        recovery_store=AtomicRecoveryStore(directory / f"{identifier}-recovery"),
        ledger=ledger,
    )
    try:
        units: list[dict[str, object]] = []
        for sequence in range(1, UNITS_PER_ARCHITECTURE + 1):
            result = runner.run_unit(unit_id=f"unit-{sequence}", sequence=sequence)
            metrics = result.metrics
            units.append(
                {
                    "learner_transitions": metrics.learner_transitions,
                    "engine_actions": metrics.engine_actions,
                    "epochs": metrics.epochs,
                    "optimizer_steps": metrics.optimizer_steps,
                    "diagnostics_finite": _finite_diagnostics(metrics.diagnostics),
                    "legal_actions_only": _legal_rollout_actions(runner),
                    "training_seconds": result.training_seconds,
                    "checkpoint_seconds": result.checkpoint_seconds,
                }
            )
        inference_seconds = _inference_seconds_per_observation(runner)
        specification = architecture_specification(architecture)
        policy_head_parameters = count_trainable_parameters(runner.model.policy.action_net)
        value_head_parameters = count_trainable_parameters(runner.model.policy.value_net)
        return {
            "architecture": architecture.value,
            "encoder_parameters": count_trainable_parameters(
                runner.model.policy.features_extractor
            ),
            "policy_head_parameters": policy_head_parameters,
            "value_head_parameters": value_head_parameters,
            "total_parameters": count_trainable_parameters(runner.model.policy),
            "approximate_inference_multiply_accumulates": (
                specification.encoder_multiply_accumulates + 64 * 9 + 64
            ),
            "inference_seconds_per_observation": inference_seconds,
            "units": units,
        }
    finally:
        environment.close()


def _finite_non_negative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and value >= 0
    )


def _candidate_passed(candidate: object, architecture: PolicyArchitecture) -> bool:
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "architecture",
        "encoder_parameters",
        "policy_head_parameters",
        "value_head_parameters",
        "total_parameters",
        "approximate_inference_multiply_accumulates",
        "inference_seconds_per_observation",
        "units",
    }:
        return False
    specification = architecture_specification(architecture)
    if (
        candidate.get("architecture") != architecture.value
        or candidate.get("encoder_parameters") != specification.encoder_parameters
        or candidate.get("policy_head_parameters") != 585
        or candidate.get("value_head_parameters") != 65
        or candidate.get("total_parameters") != specification.encoder_parameters + 650
        or candidate.get("approximate_inference_multiply_accumulates")
        != specification.encoder_multiply_accumulates + 640
        or not _finite_non_negative(candidate.get("inference_seconds_per_observation"))
    ):
        return False
    units = candidate.get("units")
    if not isinstance(units, list) or len(units) != UNITS_PER_ARCHITECTURE:
        return False
    expected_unit_keys = {
        "learner_transitions",
        "engine_actions",
        "epochs",
        "optimizer_steps",
        "diagnostics_finite",
        "legal_actions_only",
        "training_seconds",
        "checkpoint_seconds",
    }
    for unit in units:
        if (
            not isinstance(unit, Mapping)
            or set(unit) != expected_unit_keys
            or unit.get("learner_transitions") != ROLLOUT_TRANSITIONS
            or not isinstance(unit.get("engine_actions"), int)
            or unit["engine_actions"] < ROLLOUT_TRANSITIONS
            or unit.get("epochs") != PPO_EPOCHS
            or unit.get("optimizer_steps") != PPO_EPOCHS * (ROLLOUT_TRANSITIONS // PPO_BATCH_SIZE)
            or unit.get("diagnostics_finite") is not True
            or unit.get("legal_actions_only") is not True
            or not _finite_non_negative(unit.get("training_seconds"))
            or not _finite_non_negative(unit.get("checkpoint_seconds"))
        ):
            return False
    return True


def smoke_passed(measurement: object) -> bool:
    """Validate the fixed RL-S4 smoke evidence without selecting a winner."""

    if not isinstance(measurement, Mapping) or set(measurement) != {"candidates"}:
        return False
    candidates = measurement.get("candidates")
    return (
        isinstance(candidates, list)
        and len(candidates) == len(ARCHITECTURES)
        and all(
            _candidate_passed(candidate, architecture)
            for candidate, architecture in zip(candidates, ARCHITECTURES, strict=True)
        )
    )


def run_measurement() -> dict[str, object]:
    """Run ten identical-budget PPO units for each frozen architecture candidate."""

    with tempfile.TemporaryDirectory(prefix="dinorl-rl-s4-") as temporary:
        directory = Path(temporary)
        candidates = [_run_candidate(directory, architecture) for architecture in ARCHITECTURES]
    return {"candidates": candidates}
