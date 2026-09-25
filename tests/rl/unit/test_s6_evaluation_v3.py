"""TDD contracts for the no-new-games RL-S6 v3 decision."""

from __future__ import annotations

from dinorl_engine.rl.s6_evaluation_v3 import publication_gate_v3


def _v2_seed(*, prudent_drop: float, prudent_ci: list[float]) -> dict[str, object]:
    opponents: dict[str, object] = {
        "random-legal-v1": {"stochastic_ci95": [0.90, 0.95], "drop": 0.05, "drop_ci95": [0.0, 0.1]},
        "aggressive-v1": {"stochastic_ci95": [0.70, 0.80], "drop": 0.05, "drop_ci95": [0.0, 0.1]},
        "opportunist-v1": {"stochastic_ci95": [0.70, 0.80], "drop": 0.05, "drop_ci95": [0.0, 0.1]},
        "prudent-v1": {
            "stochastic_ci95": [0.70, 0.80],
            "drop": prudent_drop,
            "drop_ci95": prudent_ci,
        },
    }
    return {
        "main": {"stochastic_ci95": [0.70, 0.80], "drop_ci95": [0.0, 0.1]},
        "opponents": opponents,
    }


def test_v3_warns_about_an_individual_drop_without_turning_it_into_a_failure() -> None:
    result = publication_gate_v3(_v2_seed(prudent_drop=0.26, prudent_ci=[0.23, 0.29]))

    assert result["decision"] == "pass"
    assert result["failed_criteria"] == []
    assert result["warnings"] == ["individual_drop_exceeds_0_15:prudent-v1"]


def test_v3_keeps_the_main_drop_as_a_blocking_confidence_bound() -> None:
    source = _v2_seed(prudent_drop=0.14, prudent_ci=[0.12, 0.16])
    source["main"] = {"stochastic_ci95": [0.70, 0.80], "drop_ci95": [0.01, 0.11]}

    result = publication_gate_v3(source)

    assert result["decision"] == "fail"
    assert result["failed_criteria"] == ["main_drop_ci95"]
    assert result["warnings"] == ["individual_drop_ci95_crosses_0_15:prudent-v1"]
