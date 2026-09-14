"""Finite, deterministic aggregation primitives for RL-L7 evaluation reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, median

__all__ = [
    "EvaluationAggregate",
    "OutcomeCounts",
    "aggregate_seed_reports",
]


@dataclass(frozen=True, slots=True)
class OutcomeCounts:
    """Wins, draws and losses from the learner's perspective."""

    wins: int
    draws: int
    losses: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0 for value in (self.wins, self.draws, self.losses)
        ):
            raise ValueError("outcome counts must be non-negative integers")

    @property
    def games(self) -> int:
        """Return the number of completed games."""

        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Return the chess-style score, with draws worth one half."""

        if self.games == 0:
            raise ValueError("score requires at least one game")
        return (self.wins + 0.5 * self.draws) / self.games


@dataclass(frozen=True, slots=True)
class EvaluationAggregate:
    """One architecture/seed terminal point used by the RL-S5 comparison."""

    seed: int
    final_score: float
    wall_seconds: float
    inference_seconds: float

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an unsigned 32-bit integer")
        for value in (self.final_score, self.wall_seconds, self.inference_seconds):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError("aggregate values must be finite numbers")
        if not 0.0 <= self.final_score <= 1.0:
            raise ValueError("final_score must be in [0, 1]")
        if self.wall_seconds < 0.0 or self.inference_seconds < 0.0:
            raise ValueError("durations must be non-negative")


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    if not values:
        raise ValueError("at least one seed report is required")
    ordered = sorted(values)
    lower_index = (len(ordered) - 1) // 4
    upper_index = (3 * (len(ordered) - 1) + 3) // 4
    return {
        "median": float(median(ordered)),
        "iqr": [ordered[lower_index], ordered[upper_index]],
        "mean": float(fmean(ordered)),
    }


def aggregate_seed_reports(
    reports: list[EvaluationAggregate],
) -> dict[str, dict[str, float | list[float]]]:
    """Aggregate development seeds with median, IQR endpoints and mean."""

    if not reports:
        raise ValueError("at least one seed report is required")
    return {
        "final_score": _summary([report.final_score for report in reports]),
        "wall_seconds": _summary([report.wall_seconds for report in reports]),
        "inference_seconds": _summary([report.inference_seconds for report in reports]),
    }
