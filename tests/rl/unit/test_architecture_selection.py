"""RL-S5 architecture comparison decision contract."""

from __future__ import annotations

from dinorl_engine.rl.evaluation.comparison import select_architecture


def test_architecture_selection_requires_speed_wall_time_and_score_to_converge() -> None:
    decision = select_architecture(
        {
            "mlp-v1": {"transitions_to_gate": 100, "wall_seconds": 10.0, "final_score": 0.8},
            "small-cnn-v1": {
                "transitions_to_gate": 200,
                "wall_seconds": 20.0,
                "final_score": 0.7,
            },
        }
    )

    assert decision == {"status": "selected", "architecture": "mlp-v1"}


def test_architecture_selection_stops_for_a_collective_decision_when_criteria_diverge() -> None:
    decision = select_architecture(
        {
            "mlp-v1": {"transitions_to_gate": 100, "wall_seconds": 20.0, "final_score": 0.7},
            "small-cnn-v1": {
                "transitions_to_gate": 200,
                "wall_seconds": 10.0,
                "final_score": 0.8,
            },
        }
    )

    assert decision["status"] == "collective_decision_required"
    assert decision["winners"] == {
        "speed": "mlp-v1",
        "wall_time": "small-cnn-v1",
        "final_score": "small-cnn-v1",
    }
