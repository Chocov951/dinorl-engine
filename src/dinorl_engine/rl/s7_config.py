"""Strict, user-supplied RL-S7 calibration protocol configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["S7Config", "S7ConfigError"]


class S7ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class S7Config:
    seeds: tuple[int, ...]
    max_units: int
    visible_score: float
    gate_score: float
    plateau_delta: float

    @classmethod
    def load(cls, path: Path) -> S7Config:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise S7ConfigError("RL-S7 config is unreadable") from error
        required = {"format", "seeds", "max_units", "visible_score", "gate_score", "plateau_delta"}
        if (
            not isinstance(value, dict)
            or set(value) != required
            or value["format"] != "dinorl-s7-config-v1"
        ):
            raise S7ConfigError("RL-S7 config is invalid")
        result = cls(
            tuple(value["seeds"]),
            value["max_units"],
            value["visible_score"],
            value["gate_score"],
            value["plateau_delta"],
        )
        if (
            len(result.seeds) < 3
            or len(set(result.seeds)) != len(result.seeds)
            or any(type(seed) is not int for seed in result.seeds)
        ):
            raise S7ConfigError("RL-S7 requires at least three distinct integer seeds")
        if type(result.max_units) is not int or result.max_units <= 0:
            raise S7ConfigError("RL-S7 max_units is invalid")
        if (
            not all(
                type(score) in {int, float} and 0 <= score <= 1
                for score in (result.visible_score, result.gate_score)
            )
            or type(result.plateau_delta) not in {int, float}
            or result.plateau_delta < 0
        ):
            raise S7ConfigError("RL-S7 thresholds are invalid")
        return result
