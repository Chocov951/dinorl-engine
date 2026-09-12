"""Deterministic factories for the server-owned vector environment choices."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import Any, Final, cast

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv

__all__ = [
    "ROLLOUT_TRANSITIONS",
    "VECTOR_ENV_COUNTS",
    "VectorBackend",
    "VectorEnvironmentConfig",
    "create_vector_environment",
    "derive_environment_seed",
    "make_environment_factory",
]

ROLLOUT_TRANSITIONS: Final = 2048
VECTOR_ENV_COUNTS: Final = (2, 4, 8)
_MAX_SEED: Final = 2**32 - 1


class VectorBackend(StrEnum):
    """The two server-measured vectorization backends."""

    DUMMY = "dummy"
    SUBPROCESS = "subprocess"


@dataclass(frozen=True, slots=True)
class VectorEnvironmentConfig:
    """Immutable server configuration for one vectorization candidate."""

    backend: VectorBackend
    n_envs: int
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.backend, VectorBackend):
            raise ValueError("backend must be a VectorBackend")
        if self.n_envs not in VECTOR_ENV_COUNTS:
            raise ValueError(f"n_envs must be one of {VECTOR_ENV_COUNTS!r}")
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_SEED:
            raise ValueError("seed must be an unsigned 32-bit integer")

    @property
    def n_steps(self) -> int:
        """Return rollout steps per environment for exactly one global unit."""

        return ROLLOUT_TRANSITIONS // self.n_envs


def derive_environment_seed(seed: int, environment_index: int) -> int:
    """Derive an independent, reproducible unsigned seed for one worker."""

    if type(seed) is not int or not 0 <= seed <= _MAX_SEED:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if type(environment_index) is not int or environment_index < 0:
        raise ValueError("environment_index must be a non-negative integer")
    payload = f"dinorl/v1/vector-environment/{seed}/{environment_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def make_environment_factory(
    seed: int, environment_index: int
) -> Callable[[], DinoRLSingleAgentEnv]:
    """Create a picklable factory with a unique deterministic RNG stream."""

    return partial(
        DinoRLSingleAgentEnv,
        seed=derive_environment_seed(seed, environment_index),
        environment_index=environment_index,
    )


def create_vector_environment(configuration: VectorEnvironmentConfig) -> VecEnv:
    """Create one configured backend without exposing process choices to players."""

    factories = [
        make_environment_factory(configuration.seed, environment_index)
        for environment_index in range(configuration.n_envs)
    ]
    vec_factories = cast(list[Callable[[], gym.Env[Any, Any]]], factories)
    if configuration.backend is VectorBackend.DUMMY:
        return DummyVecEnv(vec_factories)
    # The reference server is Linux/fork; Windows development uses spawn safely.
    start_method = "spawn" if sys.platform == "win32" else "fork"
    return SubprocVecEnv(vec_factories, start_method=start_method)
