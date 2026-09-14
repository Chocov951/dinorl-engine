"""RL-L7 metric contracts independent of policy inference and the server."""

from __future__ import annotations

import pytest

from dinorl_engine.rl.evaluation.metrics import (
    EvaluationAggregate,
    OutcomeCounts,
    aggregate_seed_reports,
)


def test_chess_score_counts_wins_draws_and_losses() -> None:
    outcomes = OutcomeCounts(wins=3, draws=2, losses=5)

    assert outcomes.games == 10
    assert outcomes.score == pytest.approx(0.4)


def test_seed_aggregation_uses_median_and_interquartile_range() -> None:
    aggregate = aggregate_seed_reports(
        [
            EvaluationAggregate(
                seed=19, final_score=0.30, wall_seconds=30.0, inference_seconds=0.01
            ),
            EvaluationAggregate(
                seed=20, final_score=0.80, wall_seconds=20.0, inference_seconds=0.03
            ),
            EvaluationAggregate(
                seed=21, final_score=0.50, wall_seconds=10.0, inference_seconds=0.02
            ),
        ]
    )

    assert aggregate["final_score"] == {
        "median": 0.5,
        "iqr": [0.3, 0.8],
        "mean": pytest.approx((0.3 + 0.8 + 0.5) / 3),
    }
    assert aggregate["wall_seconds"]["median"] == 20.0
    assert aggregate["inference_seconds"]["iqr"] == [0.01, 0.03]


def test_seed_aggregation_rejects_an_empty_or_non_finite_fixture() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_seed_reports([])
    with pytest.raises(ValueError, match="finite"):
        aggregate_seed_reports(
            [
                EvaluationAggregate(
                    seed=19, final_score=float("nan"), wall_seconds=1.0, inference_seconds=0.01
                )
            ]
        )
