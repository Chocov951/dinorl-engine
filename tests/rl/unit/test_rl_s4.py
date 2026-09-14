"""RL-S4 smoke-result structure and no-divergence decision contracts."""

from __future__ import annotations

import copy

from dinorl_engine.rl.policies.factory import PolicyArchitecture, architecture_specification
from dinorl_engine.server_checks.suites.rl_s4 import (
    ARCHITECTURES,
    UNITS_PER_ARCHITECTURE,
    smoke_passed,
)


def _candidate(architecture: PolicyArchitecture) -> dict[str, object]:
    specification = architecture_specification(architecture)
    return {
        "architecture": architecture.value,
        "encoder_parameters": specification.encoder_parameters,
        "policy_head_parameters": 585,
        "value_head_parameters": 65,
        "total_parameters": specification.encoder_parameters + 650,
        "approximate_inference_multiply_accumulates": (
            specification.encoder_multiply_accumulates + 640
        ),
        "inference_seconds_per_observation": 0.001,
        "units": [
            {
                "learner_transitions": 2048,
                "engine_actions": 3000,
                "epochs": 4,
                "optimizer_steps": 32,
                "diagnostics_finite": True,
                "legal_actions_only": True,
                "training_seconds": 1.0,
                "checkpoint_seconds": 0.1,
            }
            for _ in range(UNITS_PER_ARCHITECTURE)
        ],
    }


def test_rl_s4_requires_ten_finite_mask_respecting_units_for_each_candidate() -> None:
    measurement = {"candidates": [_candidate(architecture) for architecture in ARCHITECTURES]}

    assert smoke_passed(measurement)

    broken = copy.deepcopy(measurement)
    candidates = broken["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert isinstance(candidate, dict)
    units = candidate["units"]
    assert isinstance(units, list)
    unit = units[4]
    assert isinstance(unit, dict)
    unit["diagnostics_finite"] = False

    assert not smoke_passed(broken)
