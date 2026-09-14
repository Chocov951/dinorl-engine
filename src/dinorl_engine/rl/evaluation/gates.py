"""Frozen RL-L7 score gates; only server measurements may change their inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping

from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS

__all__ = ["deterministic_gate", "publication_gate"]

_RANDOM_DETERMINISTIC_MINIMUM = 0.90
_MAIN_AVERAGE_EXCLUSIVE_MINIMUM = 0.55
_MAIN_INDIVIDUAL_EXCLUSIVE_MINIMUM = 0.40
_STOCHASTIC_AVERAGE_MAXIMUM_DROP = 0.10
_STOCHASTIC_INDIVIDUAL_MAXIMUM_DROP = 0.15
_STOCHASTIC_RANDOM_MINIMUM = 0.85
_BOUNDARY_DISTANCE = 0.02


def _scores(scores: Mapping[str, object]) -> dict[str, float]:
    expected = {*SCRIPTED_CONTROLLER_IDS, RANDOM_LEGAL_CONTROLLER_ID}
    if set(scores) != expected:
        raise ValueError("scores must cover the complete versioned opponent catalog")
    normalized: dict[str, float] = {}
    for opponent, score in scores.items():
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            raise ValueError("scores must be finite numbers")
        if not 0.0 <= score <= 1.0:
            raise ValueError("scores must be in [0, 1]")
        normalized[opponent] = float(score)
    return normalized


def deterministic_gate(scores: Mapping[str, object], *, previous_passed: bool) -> dict[str, object]:
    """Evaluate the section-5 deterministic gate, including the consecutive-pass rule."""

    normalized = _scores(scores)
    main_average = sum(normalized[opponent] for opponent in SCRIPTED_CONTROLLER_IDS) / len(
        SCRIPTED_CONTROLLER_IDS
    )
    thresholds_passed = (
        normalized[RANDOM_LEGAL_CONTROLLER_ID] >= _RANDOM_DETERMINISTIC_MINIMUM
        and main_average > _MAIN_AVERAGE_EXCLUSIVE_MINIMUM
        and all(
            normalized[opponent] > _MAIN_INDIVIDUAL_EXCLUSIVE_MINIMUM
            for opponent in SCRIPTED_CONTROLLER_IDS
        )
    )
    return {
        "random_score": normalized[RANDOM_LEGAL_CONTROLLER_ID],
        "main_average_score": main_average,
        "thresholds_passed": thresholds_passed,
        "previous_passed": previous_passed,
        "passed": thresholds_passed and previous_passed,
    }


def publication_gate(
    deterministic_scores: Mapping[str, object], stochastic_scores: Mapping[str, object]
) -> dict[str, object]:
    """Reject a stochastic policy which collapses relative to deterministic evidence."""

    deterministic = _scores(deterministic_scores)
    stochastic = _scores(stochastic_scores)
    deterministic_average = sum(
        deterministic[opponent] for opponent in SCRIPTED_CONTROLLER_IDS
    ) / len(SCRIPTED_CONTROLLER_IDS)
    stochastic_average = sum(stochastic[opponent] for opponent in SCRIPTED_CONTROLLER_IDS) / len(
        SCRIPTED_CONTROLLER_IDS
    )
    average_drop = deterministic_average - stochastic_average
    individual_drops = {
        opponent: deterministic[opponent] - stochastic[opponent]
        for opponent in SCRIPTED_CONTROLLER_IDS
    }
    boundaries = [
        abs(stochastic[RANDOM_LEGAL_CONTROLLER_ID] - _STOCHASTIC_RANDOM_MINIMUM),
        abs(average_drop - _STOCHASTIC_AVERAGE_MAXIMUM_DROP),
        *(abs(drop - _STOCHASTIC_INDIVIDUAL_MAXIMUM_DROP) for drop in individual_drops.values()),
    ]
    passed = (
        average_drop <= _STOCHASTIC_AVERAGE_MAXIMUM_DROP
        and all(drop <= _STOCHASTIC_INDIVIDUAL_MAXIMUM_DROP for drop in individual_drops.values())
        and stochastic[RANDOM_LEGAL_CONTROLLER_ID] >= _STOCHASTIC_RANDOM_MINIMUM
    )
    return {
        "deterministic_main_average": deterministic_average,
        "stochastic_main_average": stochastic_average,
        "main_average_drop": average_drop,
        "individual_drops": individual_drops,
        "stochastic_random_score": stochastic[RANDOM_LEGAL_CONTROLLER_ID],
        "near_boundary": any(distance <= _BOUNDARY_DISTANCE for distance in boundaries),
        "passed": passed,
    }
