"""Frozen RL-S6 configuration; no specialist or curriculum knobs are accepted."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.rl.contracts import canonical_json_bytes
from dinorl_engine.rl.rewards.reference import REFERENCE_REWARD_SOURCE

__all__ = ["S6Config", "S6ConfigError"]


class S6ConfigError(ValueError):
    """The frozen RL-S6 contract was substituted or incompletely specified."""


@dataclass(frozen=True, slots=True)
class S6Config:
    run_id: str
    output_directory: Path
    architecture: str
    architecture_dimensions: tuple[int, int, int]
    starter_directory: Path
    starter_weights_sha256: str
    beta_validation_pool: Path
    beta_validation_pool_sha256: str
    training_seeds: tuple[int, ...]
    evaluation_seed: int
    stochastic_evaluation_seed: int
    stochastic_extension_seed: int
    max_units: int
    evaluation_interval_units: int
    training_opponents: tuple[str, ...]
    reward_dsl: str

    @classmethod
    def load(cls, path: Path) -> S6Config:
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise S6ConfigError("RL-S6 config cannot be read as canonical JSON") from error
        required = {
            "format",
            "phase",
            "run_id",
            "output_directory",
            "architecture",
            "architecture_dimensions",
            "starter_directory",
            "starter_weights_sha256",
            "beta_validation_pool",
            "beta_validation_pool_sha256",
            "training_seeds",
            "evaluation_seed",
            "stochastic_evaluation_seed",
            "stochastic_extension_seed",
            "max_units",
            "evaluation_interval_units",
            "training_opponents",
            "reward_dsl",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise S6ConfigError("RL-S6 config has missing or unexpected fields")
        if value["format"] != "dinorl-s6-config-v1" or value["phase"] != "RL-S6":
            raise S6ConfigError("RL-S6 config format or phase is invalid")
        base = path.parent.resolve()
        try:
            result = cls(
                run_id=str(value["run_id"]),
                output_directory=(base / Path(str(value["output_directory"]))).resolve(),
                architecture=str(value["architecture"]),
                architecture_dimensions=tuple(value["architecture_dimensions"]),
                starter_directory=(base / Path(str(value["starter_directory"]))).resolve(),
                starter_weights_sha256=str(value["starter_weights_sha256"]),
                beta_validation_pool=(base / Path(str(value["beta_validation_pool"]))).resolve(),
                beta_validation_pool_sha256=str(value["beta_validation_pool_sha256"]),
                training_seeds=tuple(value["training_seeds"]),
                evaluation_seed=int(value["evaluation_seed"]),
                stochastic_evaluation_seed=int(value["stochastic_evaluation_seed"]),
                stochastic_extension_seed=int(value["stochastic_extension_seed"]),
                max_units=int(value["max_units"]),
                evaluation_interval_units=int(value["evaluation_interval_units"]),
                training_opponents=tuple(value["training_opponents"]),
                reward_dsl=str(value["reward_dsl"]),
            )
        except (TypeError, ValueError) as error:
            raise S6ConfigError("RL-S6 config has invalid value types") from error
        result.validate()
        return result

    def validate(self) -> None:
        if self.architecture != "mlp-compact-v2" or self.architecture_dimensions != (663, 64, 64):
            raise S6ConfigError("RL-S6 requires mlp-compact-v2 dimensions 663->64->64")
        if self.training_seeds != (19, 20, 21, 22, 23) or len(set(self.training_seeds)) != 5:
            raise S6ConfigError("RL-S6 requires exactly the five frozen training seeds")
        if self.max_units != 147 or self.evaluation_interval_units != 5:
            raise S6ConfigError("RL-S6 budget and evaluation interval are frozen")
        if self.training_opponents != tuple(SCRIPTED_CONTROLLER_IDS):
            raise S6ConfigError("RL-S6 training opponents must be the three scripted bots")
        if self.reward_dsl != REFERENCE_REWARD_SOURCE:
            raise S6ConfigError("RL-S6 must use the canonical generalist reference Reward DSL")
        if any(
            len(value) != 64 or value.lower() != value
            for value in (self.starter_weights_sha256, self.beta_validation_pool_sha256)
        ):
            raise S6ConfigError("RL-S6 input hashes are invalid")

    @property
    def mapping(self) -> dict[str, object]:
        return {
            "format": "dinorl-s6-config-v1",
            "phase": "RL-S6",
            "run_id": self.run_id,
            "output_directory": str(self.output_directory),
            "architecture": self.architecture,
            "architecture_dimensions": list(self.architecture_dimensions),
            "starter_directory": str(self.starter_directory),
            "starter_weights_sha256": self.starter_weights_sha256,
            "beta_validation_pool": str(self.beta_validation_pool),
            "beta_validation_pool_sha256": self.beta_validation_pool_sha256,
            "training_seeds": list(self.training_seeds),
            "evaluation_seed": self.evaluation_seed,
            "stochastic_evaluation_seed": self.stochastic_evaluation_seed,
            "stochastic_extension_seed": self.stochastic_extension_seed,
            "max_units": self.max_units,
            "evaluation_interval_units": self.evaluation_interval_units,
            "training_opponents": list(self.training_opponents),
            "reward_dsl": self.reward_dsl,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.mapping)).hexdigest()
