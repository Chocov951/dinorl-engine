"""RL-S1 result construction and decision contracts."""

from __future__ import annotations

from dinorl_engine.rl.env.vectorization import VectorBackend, VectorEnvironmentConfig
from dinorl_engine.server_checks.suites.rl_s1 import (
    REPEATED_MEASUREMENTS,
    build_decision,
    candidate_configurations,
)


def test_rl_s1_declares_all_six_server_owned_candidates() -> None:
    candidates = candidate_configurations(seed=19)

    assert len(candidates) == 6
    assert {(candidate.backend, candidate.n_envs) for candidate in candidates} == {
        (backend, n_envs) for backend in VectorBackend for n_envs in (2, 4, 8)
    }
    assert all(candidate.n_steps * candidate.n_envs == 2048 for candidate in candidates)


def test_rl_s1_decision_selects_highest_median_end_to_end_throughput() -> None:
    dummy = VectorEnvironmentConfig(VectorBackend.DUMMY, n_envs=2, seed=19)
    subprocess = VectorEnvironmentConfig(VectorBackend.SUBPROCESS, n_envs=4, seed=19)
    candidates = [
        {
            "backend": dummy.backend.value,
            "n_envs": dummy.n_envs,
            "n_steps": dummy.n_steps,
            "repetitions": [
                {"duration_seconds": 4.0, "learner_transitions": 2048}
                for _ in range(REPEATED_MEASUREMENTS)
            ],
        },
        {
            "backend": subprocess.backend.value,
            "n_envs": subprocess.n_envs,
            "n_steps": subprocess.n_steps,
            "repetitions": [
                {"duration_seconds": 2.0, "learner_transitions": 2048}
                for _ in range(REPEATED_MEASUREMENTS)
            ],
        },
    ]

    decision = build_decision(candidates)

    assert decision["backend"] == "subprocess"
    assert decision["n_envs"] == 4
    assert decision["median_transitions_per_second"] == 1024.0
