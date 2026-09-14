"""Traceable RL-S5 architecture decision from server-measured aggregates."""

from __future__ import annotations

import math
from collections.abc import Mapping

from dinorl_engine.rl.policies.factory import PolicyArchitecture

__all__ = ["select_architecture"]


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _unique_best(values: Mapping[str, float], *, highest: bool) -> str | None:
    target = (max if highest else min)(values.values())
    winners = [name for name, value in values.items() if value == target]
    return winners[0] if len(winners) == 1 else None


def select_architecture(candidates: Mapping[str, object]) -> dict[str, object]:
    """Select only when speed, wall time and final score name the same unique candidate."""

    expected = {architecture.value for architecture in PolicyArchitecture}
    if set(candidates) != expected:
        raise ValueError("comparison must contain exactly the two frozen architectures")
    transitions: dict[str, float] = {}
    wall_times: dict[str, float] = {}
    scores: dict[str, float] = {}
    for architecture, value in candidates.items():
        if not isinstance(value, Mapping):
            raise ValueError("candidate aggregate must be a mapping")
        if set(value) not in (
            {"transitions_to_gate", "wall_seconds", "final_score"},
            {"transitions_to_gate", "wall_seconds", "final_score", "inference_seconds"},
        ):
            raise ValueError("candidate aggregate has unexpected fields")
        transition_value = value["transitions_to_gate"]
        if transition_value is None:
            transitions[architecture] = math.inf
        elif type(transition_value) is int and transition_value > 0:
            transitions[architecture] = float(transition_value)
        else:
            raise ValueError("transitions_to_gate must be a positive integer or null")
        wall_times[architecture] = _number(value["wall_seconds"], "wall_seconds")
        scores[architecture] = _number(value["final_score"], "final_score")
        if wall_times[architecture] < 0.0 or not 0.0 <= scores[architecture] <= 1.0:
            raise ValueError("candidate aggregate is outside its valid range")
    winners = {
        "speed": _unique_best(transitions, highest=False),
        "wall_time": _unique_best(wall_times, highest=False),
        "final_score": _unique_best(scores, highest=True),
    }
    unique_winners = set(winners.values())
    if len(unique_winners) == 1 and None not in unique_winners:
        return {"status": "selected", "architecture": unique_winners.pop()}
    return {"status": "collective_decision_required", "winners": winners}
