"""Configurable RL-S7 curve extraction; this module never trains a policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["CalibrationError", "extract_curve_times", "propose_daily_budget"]


class CalibrationError(ValueError):
    """A server curve cannot support a reproducible product calibration."""


def extract_curve_times(
    points: Sequence[Mapping[str, object]],
    *,
    visible_score: float,
    gate_score: float,
    plateau_delta: float,
) -> dict[str, int | None]:
    """Extract confirmed threshold crossings and a noise-resistant plateau.

    ``T_visible`` is reported at the *first* of two consecutive qualifying
    observations.  The second observation is its confirmation.  A plateau
    likewise requires three consecutive marginal improvements at or below
    ``plateau_delta`` (four observations), never one noisy interval.
    """

    if not 0 <= visible_score <= 1 or not 0 <= gate_score <= 1 or plateau_delta < 0:
        raise CalibrationError("calibration thresholds are invalid")
    values: list[tuple[int, float]] = []
    for point in points:
        unit, score = point.get("unit"), point.get("score")
        if type(unit) is not int or isinstance(score, bool) or not isinstance(score, int | float):
            raise CalibrationError("curve point is invalid")
        if not 0 <= float(score) <= 1:
            raise CalibrationError("curve score is outside [0, 1]")
        values.append((unit, float(score)))
    if (
        len(values) < 3
        or values != sorted(values)
        or len({unit for unit, _ in values}) != len(values)
    ):
        raise CalibrationError(
            "curve points must be strictly increasing and contain at least three values"
        )
    visible = next(
        (
            values[index][0]
            for index in range(len(values) - 1)
            if values[index][1] >= visible_score and values[index + 1][1] >= visible_score
        ),
        None,
    )
    gate = next((unit for unit, score in values if score >= gate_score), None)
    plateau = next(
        (
            values[index + 1][0]
            for index in range(len(values) - 3)
            if all(
                values[offset + 1][1] - values[offset][1] <= plateau_delta
                for offset in range(index, index + 3)
            )
        ),
        None,
    )
    return {"T_visible": visible, "T_gate": gate, "T_plateau": plateau}


def propose_daily_budget(*, visible: int, plateau: int) -> int:
    """Choose the largest integer satisfying ``visible <= D < plateau / 2``."""

    if visible <= 0 or plateau <= 0 or visible >= plateau / 2:
        raise CalibrationError("no daily budget satisfies the RL-S7 inequality")
    return max(visible, (plateau - 1) // 2)
