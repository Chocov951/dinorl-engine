"""Strict, dependency-free configuration for an RL-S5b campaign."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = ["S5bCheckpoint", "S5bConfig", "S5bConfigError", "load_config"]

_FORMAT: Final = "dinorl-s5b-config-v1"
_ARCHITECTURES: Final = frozenset({"mlp-v1", "mlp-compact-v2", "mlp-balanced-v2", "mlp-deep-v2"})


class S5bConfigError(ValueError):
    """Raised when a campaign configuration is not a strict S5b document."""


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise S5bConfigError(f"{field} must be a positive integer")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or any(character.isspace() for character in value):
        raise S5bConfigError(f"{field} must be a non-empty identifier without whitespace")
    return value


@dataclass(frozen=True, slots=True)
class S5bCheckpoint:
    """A checkpoint directory supplied to the immutable benchmark pool."""

    checkpoint_id: str
    architecture: str
    seed: int
    directory: Path

    def __post_init__(self) -> None:
        _identifier(self.checkpoint_id, "checkpoint_id")
        if self.architecture not in _ARCHITECTURES:
            raise S5bConfigError("checkpoint architecture is not an RL-S5 architecture")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise S5bConfigError("checkpoint seed must be an unsigned 32-bit integer")


@dataclass(frozen=True, slots=True)
class S5bConfig:
    """All inputs which determine a comparable S5b phase-A campaign."""

    run_id: str
    output_directory: Path
    checkpoints: tuple[S5bCheckpoint, ...]
    final_architectures: tuple[str, ...]
    confrontations: int
    evaluation_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier(self.run_id, "run_id")
        if not self.checkpoints:
            raise S5bConfigError("checkpoints must not be empty")
        identifiers = [checkpoint.checkpoint_id for checkpoint in self.checkpoints]
        if len(set(identifiers)) != len(identifiers):
            raise S5bConfigError("checkpoint identifiers must be unique")
        final_architectures = set(self.final_architectures)
        if not final_architectures or not final_architectures <= _ARCHITECTURES:
            raise S5bConfigError("final_architectures must name RL-S5 architectures")
        if not final_architectures <= {checkpoint.architecture for checkpoint in self.checkpoints}:
            raise S5bConfigError("every finalist must have a supplied checkpoint")
        _positive_int(self.confrontations, "confrontations")
        if not self.evaluation_seeds or any(
            type(seed) is not int or not 0 <= seed < 2**32 for seed in self.evaluation_seeds
        ):
            raise S5bConfigError("evaluation_seeds must contain unsigned 32-bit integers")

    @classmethod
    def from_mapping(cls, value: object, *, base_directory: Path) -> S5bConfig:
        """Parse a JSON-compatible YAML document with no implicit defaults."""

        if not isinstance(value, dict) or set(value) != {
            "format",
            "run_id",
            "output_directory",
            "checkpoints",
            "final_architectures",
            "confrontations",
            "evaluation_seeds",
        }:
            raise S5bConfigError("S5b config has missing or unexpected fields")
        if value["format"] != _FORMAT:
            raise S5bConfigError(f"format must be {_FORMAT!r}")
        raw_checkpoints = value["checkpoints"]
        if not isinstance(raw_checkpoints, list):
            raise S5bConfigError("checkpoints must be an array")
        checkpoints: list[S5bCheckpoint] = []
        for index, raw_checkpoint in enumerate(raw_checkpoints):
            if not isinstance(raw_checkpoint, dict) or set(raw_checkpoint) != {
                "id",
                "architecture",
                "seed",
                "directory",
            }:
                raise S5bConfigError(f"checkpoints[{index}] has missing or unexpected fields")
            raw_directory = raw_checkpoint["directory"]
            if not isinstance(raw_directory, str) or not raw_directory:
                raise S5bConfigError(f"checkpoints[{index}].directory must be a path string")
            directory = Path(raw_directory)
            checkpoints.append(
                S5bCheckpoint(
                    checkpoint_id=_identifier(raw_checkpoint["id"], f"checkpoints[{index}].id"),
                    architecture=(
                        raw_checkpoint["architecture"]
                        if isinstance(raw_checkpoint["architecture"], str)
                        else ""
                    ),
                    seed=(raw_checkpoint["seed"] if type(raw_checkpoint["seed"]) is int else -1),
                    directory=directory if directory.is_absolute() else base_directory / directory,
                )
            )
        final_architectures = value["final_architectures"]
        evaluation_seeds = value["evaluation_seeds"]
        if not isinstance(final_architectures, list) or not all(
            isinstance(item, str) for item in final_architectures
        ):
            raise S5bConfigError("final_architectures must be an array of strings")
        if not isinstance(evaluation_seeds, list):
            raise S5bConfigError("evaluation_seeds must be an array")
        raw_output = value["output_directory"]
        if not isinstance(raw_output, str) or not raw_output:
            raise S5bConfigError("output_directory must be a path string")
        output = Path(raw_output)
        return cls(
            run_id=_identifier(value["run_id"], "run_id"),
            output_directory=output if output.is_absolute() else base_directory / output,
            checkpoints=tuple(checkpoints),
            final_architectures=tuple(final_architectures),
            confrontations=_positive_int(value["confrontations"], "confrontations"),
            evaluation_seeds=tuple(seed if type(seed) is int else -1 for seed in evaluation_seeds),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return every behavior-affecting setting in canonical path form."""

        return {
            "format": _FORMAT,
            "run_id": self.run_id,
            "output_directory": str(self.output_directory.resolve()),
            "checkpoints": [
                {
                    "id": checkpoint.checkpoint_id,
                    "architecture": checkpoint.architecture,
                    "seed": checkpoint.seed,
                    "directory": str(checkpoint.directory.resolve()),
                }
                for checkpoint in self.checkpoints
            ],
            "final_architectures": list(self.final_architectures),
            "confrontations": self.confrontations,
            "evaluation_seeds": list(self.evaluation_seeds),
        }


def load_config(path: Path) -> S5bConfig:
    """Load JSON-compatible YAML without adding a YAML parser dependency.

    JSON is a valid YAML subset.  Keeping this first S5b configuration in that
    subset preserves the requested ``.yaml`` interface while keeping the core
    deployable on the quota-constrained reference server.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S5bConfigError(f"cannot read JSON-compatible YAML config: {path}") from error
    return S5bConfig.from_mapping(raw, base_directory=path.parent.resolve())
