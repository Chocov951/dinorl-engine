"""RL-S5 fixed-budget comparison evidence contract."""

from __future__ import annotations

from dinorl_engine.rl.policies.factory import PolicyArchitecture
from dinorl_engine.server_checks.suites.rl_s5 import (
    DEVELOPMENT_SEEDS,
    UNITS_PER_SEED,
    comparison_passed,
)


def _candidate(
    architecture: PolicyArchitecture,
    *,
    transitions_to_gate: int,
    wall_seconds: float,
    final_score: float,
) -> dict[str, object]:
    return {
        "architecture": architecture.value,
        "seeds": [
            {
                "seed": seed,
                "units": UNITS_PER_SEED,
                "learner_transitions": UNITS_PER_SEED * 2048,
                "engine_actions": UNITS_PER_SEED * 3000,
                "epochs": UNITS_PER_SEED * 4,
                "optimizer_steps": UNITS_PER_SEED * 32,
                "wall_seconds": wall_seconds,
                "cpu_seconds": 9.0,
                "inference_seconds": 0.01,
                "final_score": final_score,
                "transitions_to_gate": transitions_to_gate,
                "deterministic_gate": True,
                "publication_gate": True,
            }
            for seed in DEVELOPMENT_SEEDS
        ],
        "aggregate": {
            "transitions_to_gate": transitions_to_gate,
            "wall_seconds": wall_seconds,
            "final_score": final_score,
            "inference_seconds": 0.01,
        },
    }


def test_rl_s5_requires_each_architecture_and_each_common_seed_to_finish_147_units() -> None:
    measurement = {
        "candidates": [
            _candidate(
                PolicyArchitecture.MLP, transitions_to_gate=100, wall_seconds=10.0, final_score=0.8
            ),
            _candidate(
                PolicyArchitecture.SMALL_CNN,
                transitions_to_gate=200,
                wall_seconds=20.0,
                final_score=0.7,
            ),
        ],
        "decision": {"status": "selected", "architecture": "mlp-v1"},
    }

    assert comparison_passed(measurement)
    measurement["candidates"][0]["seeds"][0]["units"] = UNITS_PER_SEED - 1  # type: ignore[index]
    assert not comparison_passed(measurement)
