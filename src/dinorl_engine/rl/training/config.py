"""Strict, immutable models for the canonical experiment configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from dinorl_engine.rl.versions import CONFIG_SCHEMA_VERSION

__all__ = ["ConfigError", "ExperimentConfig", "PPOConfig"]


_BATCH_SIZES: Final = frozenset((64, 128, 256, 512, 1024, 2048))
_OPAQUE_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


class ConfigError(ValueError):
    """Raised when a canonical RL configuration is structurally invalid."""


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{field_name} must be a finite number")
    return number


def _unsigned_32_bit_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ConfigError(f"{field_name} must be an unsigned 32-bit integer")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return value


def _opaque_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID.fullmatch(value) is None:
        raise ConfigError(f"{field_name} must be an opaque identifier")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ConfigError(f"{context} has missing={missing!r} unexpected={unexpected!r}")


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Player-selectable PPO settings within the immutable V1 domains."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    batch_size: int = 256

    def __post_init__(self) -> None:
        learning_rate = _finite_number(self.learning_rate, "learning_rate")
        gamma = _finite_number(self.gamma, "gamma")
        entropy_coef = _finite_number(self.entropy_coef, "entropy_coef")
        gae_lambda = _finite_number(self.gae_lambda, "gae_lambda")
        clip_range = _finite_number(self.clip_range, "clip_range")
        if not 1e-8 <= learning_rate <= 0.1:
            raise ConfigError("learning_rate must be in [1e-8, 0.1]")
        if not 0 <= gamma <= 1:
            raise ConfigError("gamma must be in [0, 1]")
        if not 0 <= entropy_coef <= 1:
            raise ConfigError("entropy_coef must be in [0, 1]")
        if not 0 <= gae_lambda <= 1:
            raise ConfigError("gae_lambda must be in [0, 1]")
        if not 0 < clip_range <= 1:
            raise ConfigError("clip_range must be in (0, 1]")
        if isinstance(self.batch_size, bool) or self.batch_size not in _BATCH_SIZES:
            raise ConfigError("batch_size must divide 2048 and be an allowed V1 batch size")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "entropy_coef", entropy_coef)
        object.__setattr__(self, "gae_lambda", gae_lambda)
        object.__setattr__(self, "clip_range", clip_range)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PPOConfig:
        """Build a configuration only if it has exactly the V1 fields."""

        _require_exact_keys(
            value,
            frozenset(
                {
                    "learning_rate",
                    "gamma",
                    "entropy_coef",
                    "gae_lambda",
                    "clip_range",
                    "batch_size",
                }
            ),
            "ppo",
        )
        return cls(
            learning_rate=_finite_number(value["learning_rate"], "learning_rate"),
            gamma=_finite_number(value["gamma"], "gamma"),
            entropy_coef=_finite_number(value["entropy_coef"], "entropy_coef"),
            gae_lambda=_finite_number(value["gae_lambda"], "gae_lambda"),
            clip_range=_finite_number(value["clip_range"], "clip_range"),
            batch_size=value["batch_size"] if isinstance(value["batch_size"], int) else -1,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the exact JSON-compatible PPO object."""

        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Canonical configuration submitted to an RL training command."""

    experiment_id: str
    parent_checkpoint_id: str | None
    seed: int
    budget_units: int
    reward_source: str
    ppo: PPOConfig
    schema_version: str = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
        _opaque_id(self.experiment_id, "experiment_id")
        if self.parent_checkpoint_id is not None:
            _opaque_id(self.parent_checkpoint_id, "parent_checkpoint_id")
        _unsigned_32_bit_integer(self.seed, "seed")
        _positive_integer(self.budget_units, "budget_units")
        if not isinstance(self.reward_source, str) or not self.reward_source:
            raise ConfigError("reward_source must be a non-empty string")
        if len(self.reward_source.encode("utf-8")) > 32 * 1024:
            raise ConfigError("reward_source must be at most 32 KiB UTF-8")
        if not isinstance(self.ppo, PPOConfig):
            raise ConfigError("ppo must be a PPOConfig")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExperimentConfig:
        """Parse a strict JSON-compatible object without accepting extras."""

        _require_exact_keys(
            value,
            frozenset(
                {
                    "schema_version",
                    "experiment_id",
                    "parent_checkpoint_id",
                    "seed",
                    "budget_units",
                    "reward_source",
                    "ppo",
                }
            ),
            "experiment",
        )
        ppo_value = value["ppo"]
        if not isinstance(ppo_value, Mapping):
            raise ConfigError("ppo must be an object")
        return cls(
            schema_version=value["schema_version"]
            if isinstance(value["schema_version"], str)
            else "",
            experiment_id=_opaque_id(value["experiment_id"], "experiment_id"),
            parent_checkpoint_id=(
                None
                if value["parent_checkpoint_id"] is None
                else _opaque_id(value["parent_checkpoint_id"], "parent_checkpoint_id")
            ),
            seed=_unsigned_32_bit_integer(value["seed"], "seed"),
            budget_units=_positive_integer(value["budget_units"], "budget_units"),
            reward_source=value["reward_source"] if isinstance(value["reward_source"], str) else "",
            ppo=PPOConfig.from_mapping(ppo_value),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the exact JSON-compatible V1 document."""

        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "seed": self.seed,
            "budget_units": self.budget_units,
            "reward_source": self.reward_source,
            "ppo": self.ppo.to_mapping(),
        }
