"""Decision contract for the RL-S3 interactive-priority server suite."""

from __future__ import annotations

from dinorl_engine.rl.orchestration.priority import InteractivePriorityMode
from dinorl_engine.server_checks.suites.rl_s3 import build_decision


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
