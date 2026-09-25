"""Frozen, server-owned parameters for the corrective RL-S6 v2 measurement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["S6EvaluationV2Config", "S6EvaluationV2ConfigError"]


class S6EvaluationV2ConfigError(ValueError):
    """The corrective measurement protocol was incomplete or altered."""


@dataclass(frozen=True, slots=True)
class S6EvaluationV2Config:
    confrontations_per_opponent: int
    bootstrap_replicates: int
    bootstrap_seed: int

    @classmethod
    def load(cls, path: Path) -> S6EvaluationV2Config:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise S6EvaluationV2ConfigError("RL-S6 v2 protocol cannot be read") from error
        required = {
            "format",
            "confrontations_per_opponent",
            "bootstrap_replicates",
            "bootstrap_seed",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise S6EvaluationV2ConfigError("RL-S6 v2 protocol has missing or unexpected fields")
        if value["format"] != "dinorl-s6-evaluation-v2-config-v1":
            raise S6EvaluationV2ConfigError("RL-S6 v2 protocol format is invalid")
        confrontations = value["confrontations_per_opponent"]
        replicates = value["bootstrap_replicates"]
        seed = value["bootstrap_seed"]
        if (
            type(confrontations) is not int
            or confrontations < 200
            or type(replicates) is not int
            or replicates < 10_000
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise S6EvaluationV2ConfigError("RL-S6 v2 protocol volume or seed is invalid")
        return cls(
            confrontations_per_opponent=confrontations,
            bootstrap_replicates=replicates,
            bootstrap_seed=seed,
        )
