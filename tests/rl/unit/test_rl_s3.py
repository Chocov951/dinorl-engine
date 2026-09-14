"""Decision contract for the RL-S3 interactive-priority server suite."""

from __future__ import annotations

from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.orchestration.priority import InteractivePriorityMode
from dinorl_engine.server_checks.suites.rl_s3 import _workers_cleaned_up, build_decision


def test_rl_s3_prioritizes_play_latency_before_training_throughput() -> None:
    decision = build_decision(
        [
            {
                "mode": InteractivePriorityMode.UNIT_BOUNDARY.value,
                "play_latency_seconds": 3.0,
                "training_transitions_per_second": 100.0,
            },
            {
                "mode": InteractivePriorityMode.SEPARATE_WORKER.value,
                "play_latency_seconds": 0.1,
                "training_transitions_per_second": 10.0,
            },
        ]
    )

    assert decision["mode"] == InteractivePriorityMode.SEPARATE_WORKER.value
    assert decision["criterion"] == "minimum_play_latency_then_maximum_training_throughput"


def test_selected_dummy_backend_has_no_child_worker_left_to_clean_up() -> None:
    environment = create_vector_environment(
        VectorEnvironmentConfig(backend=VectorBackend.DUMMY, n_envs=2, seed=19)
    )
    environment.close()

    assert _workers_cleaned_up(environment)
