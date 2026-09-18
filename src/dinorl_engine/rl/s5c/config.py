"""Frozen, dependency-free configuration for an RL-S5c campaign."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID

__all__ = ["S5cConfig", "S5cConfigError", "load_config"]

_FORMAT: Final = "dinorl-s5c-config-v1"
_ARCHITECTURE: Final = "mlp-compact-v2"
_MAX_ROUNDS: Final = 30


class S5cConfigError(ValueError):
    """Raised when a specialist campaign would drift from its frozen protocol."""


def _seeds(value: object, field: str, expected: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != expected
        or len(set(value)) != expected
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in value)
    ):
        raise S5cConfigError(f"{field} must contain {expected} distinct unsigned 32-bit seeds")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class S5cConfig:
    """All settings that must be identical across the three specialist profiles."""

    run_id: str
    output_directory: Path
    architecture: str
    max_rounds: int
    training_opponent: str
    calibration_seeds: tuple[int, ...]
    production_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    max_units: int

    def __post_init__(self) -> None:
        if not self.run_id or any(character.isspace() for character in self.run_id):
            raise S5cConfigError("run_id must be a non-empty identifier without whitespace")
        if self.architecture != _ARCHITECTURE:
            raise S5cConfigError("RL-S5c architecture must be mlp-compact-v2")
        if self.max_rounds != _MAX_ROUNDS:
            raise S5cConfigError("RL-S5c max_rounds must remain 30")
        if self.training_opponent != RANDOM_LEGAL_CONTROLLER_ID:
            raise S5cConfigError("RL-S5c training is exclusively against random-legal-v1")
        if len(self.calibration_seeds) != 3 or len(self.production_seeds) != 5:
            raise S5cConfigError("RL-S5c requires three calibration and five production seeds")
        if not self.evaluation_seeds or any(
            type(seed) is not int or not 0 <= seed < 2**32 for seed in self.evaluation_seeds
        ):
            raise S5cConfigError("evaluation_seeds must be unsigned 32-bit integers")
        if type(self.max_units) is not int or self.max_units != 147:
            raise S5cConfigError("RL-S5c max_units must remain 147")

    @classmethod
    def from_mapping(cls, value: object, *, base_directory: Path | None = None) -> S5cConfig:
        expected = {
            "format",
            "run_id",
            "output_directory",
            "architecture",
            "max_rounds",
            "training_opponent",
            "calibration_seeds",
            "production_seeds",
            "evaluation_seeds",
            "max_units",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise S5cConfigError("S5c config has missing or unexpected fields")
        if value["format"] != _FORMAT:
            raise S5cConfigError(f"format must be {_FORMAT!r}")
        output = value["output_directory"]
        if not isinstance(output, str) or not output:
            raise S5cConfigError("output_directory must be a non-empty path string")
        path = Path(output)
        if base_directory is not None and not path.is_absolute():
            path = base_directory / path
        return cls(
            run_id=value["run_id"] if isinstance(value["run_id"], str) else "",
            output_directory=path,
            architecture=value["architecture"] if isinstance(value["architecture"], str) else "",
            max_rounds=value["max_rounds"] if type(value["max_rounds"]) is int else -1,
            training_opponent=(
                value["training_opponent"] if isinstance(value["training_opponent"], str) else ""
            ),
            calibration_seeds=_seeds(value["calibration_seeds"], "calibration_seeds", 3),
            production_seeds=_seeds(value["production_seeds"], "production_seeds", 5),
            evaluation_seeds=_seeds(value["evaluation_seeds"], "evaluation_seeds", 3),
            max_units=value["max_units"] if type(value["max_units"]) is int else -1,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "format": _FORMAT,
            "run_id": self.run_id,
            "output_directory": str(self.output_directory),
            "architecture": self.architecture,
            "max_rounds": self.max_rounds,
            "training_opponent": self.training_opponent,
            "calibration_seeds": list(self.calibration_seeds),
            "production_seeds": list(self.production_seeds),
            "evaluation_seeds": list(self.evaluation_seeds),
            "max_units": self.max_units,
        }


def load_config(path: Path) -> S5cConfig:
    """Load the JSON-compatible YAML configuration without a YAML dependency."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S5cConfigError(f"cannot read JSON-compatible YAML config: {path}") from error
    return S5cConfig.from_mapping(value, base_directory=path.parent.resolve())
