"""RL-L7 deterministic and stochastic publication gate boundaries."""

from __future__ import annotations

from dinorl_engine.rl.evaluation.gates import deterministic_gate, publication_gate


def test_deterministic_gate_requires_every_documented_threshold_and_two_consecutive_passes() -> (
    None
):
    scores = {
        "random-legal-v1": 0.9,
        "aggressive-v1": 0.6,
        "prudent-v1": 0.6,
        "opportunist-v1": 0.6,
    }

    first = deterministic_gate(scores, previous_passed=False)
    second = deterministic_gate(scores, previous_passed=True)

    assert first["thresholds_passed"] is True
    assert first["passed"] is False
    assert second["passed"] is True
    assert (
        deterministic_gate({**scores, "prudent-v1": 0.4}, previous_passed=True)["passed"] is False
    )


def test_publication_gate_detects_stochastic_collapse_and_near_boundaries() -> None:
    deterministic = {
        "random-legal-v1": 0.95,
        "aggressive-v1": 0.7,
        "prudent-v1": 0.7,
        "opportunist-v1": 0.7,
    }
    stochastic = {
        "random-legal-v1": 0.85,
        "aggressive-v1": 0.6,
        "prudent-v1": 0.6,
        "opportunist-v1": 0.6,
    }

    result = publication_gate(deterministic, stochastic)

    assert result["passed"] is True
    assert result["near_boundary"] is True
    assert publication_gate(deterministic, {**stochastic, "aggressive-v1": 0.54})["passed"] is False
