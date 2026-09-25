"""RL-S6 v3 gate derived solely from pre-existing paired V2 evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS

__all__ = ["S6EvaluationV3Error", "publication_gate_v3"]


class S6EvaluationV3Error(ValueError):
    """V2 evidence lacks the fixed intervals required by the V3 rule."""


def _interval(value: object, field: str, *, minimum: float = 0.0) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        raise S6EvaluationV3Error(f"{field} must be a two-value confidence interval")
    lower, upper = value
    if (
        isinstance(lower, bool)
        or not isinstance(lower, int | float)
        or isinstance(upper, bool)
        or not isinstance(upper, int | float)
        or not minimum <= float(lower) <= float(upper) <= 1.0
    ):
        raise S6EvaluationV3Error(f"{field} confidence interval is invalid")
    return float(lower), float(upper)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise S6EvaluationV3Error(f"{field} is invalid")
    return value


def publication_gate_v3(v2_gate: Mapping[str, object]) -> dict[str, object]:
    """Apply V3’s blocking criteria and non-blocking individual-drop warnings."""

    main = _mapping(v2_gate.get("main"), "V2 main evidence")
    opponents = _mapping(v2_gate.get("opponents"), "V2 opponent evidence")
    expected = {*SCRIPTED_CONTROLLER_IDS, RANDOM_LEGAL_CONTROLLER_ID}
    if set(opponents) != expected:
        raise S6EvaluationV3Error("V2 opponent catalog is incomplete")
    random = _mapping(opponents[RANDOM_LEGAL_CONTROLLER_ID], "V2 random evidence")
    random_ci = _interval(random.get("stochastic_ci95"), "V2 random score")
    main_score_ci = _interval(main.get("stochastic_ci95"), "V2 main score")
    main_drop_ci = _interval(main.get("drop_ci95"), "V2 main drop", minimum=-1.0)
    script_score_cis = {
        opponent: _interval(
            _mapping(opponents[opponent], f"V2 {opponent} evidence").get("stochastic_ci95"),
            f"V2 {opponent} score",
        )
        for opponent in SCRIPTED_CONTROLLER_IDS
    }
    criteria = {
        "random_score_ci95": random_ci[0] >= 0.85,
        "main_score_ci95": main_score_ci[0] > 0.55,
        "individual_scores_ci95": all(interval[0] > 0.40 for interval in script_score_cis.values()),
        "main_drop_ci95": main_drop_ci[1] <= 0.10,
    }
    warnings: list[str] = []
    individual_drops: dict[str, dict[str, object]] = {}
    for opponent in SCRIPTED_CONTROLLER_IDS:
        evidence = _mapping(opponents[opponent], f"V2 {opponent} evidence")
        point = evidence.get("drop")
        if isinstance(point, bool) or not isinstance(point, int | float):
            raise S6EvaluationV3Error(f"V2 {opponent} drop is invalid")
        interval = _interval(evidence.get("drop_ci95"), f"V2 {opponent} drop", minimum=-1.0)
        warning: str | None = None
        if float(point) > 0.15:
            warning = f"individual_drop_exceeds_0_15:{opponent}"
        elif interval[0] <= 0.15 <= interval[1]:
            warning = f"individual_drop_ci95_crosses_0_15:{opponent}"
        if warning is not None:
            warnings.append(warning)
        individual_drops[opponent] = {
            "drop": float(point),
            "drop_ci95": list(interval),
            "warning": warning,
        }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "format": "rl-s6-publication-evaluation-v3",
        "blocking_criteria": criteria,
        "failed_criteria": failed,
        "warnings": warnings,
        "individual_drops": individual_drops,
        "decision": "pass" if not failed else "fail",
    }
