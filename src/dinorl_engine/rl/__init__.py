"""Versioned reinforcement-learning contracts for DinoRL.

The package deliberately contains no game-rule implementation at RL-L0.  It is
allowed to depend on the engine's public API in later tickets; the core package
must never depend on this package.
"""

from dinorl_engine.rl.versions import (
    CHECKPOINT_VERSION,
    EXPERIMENT_VERSION,
    OBSERVATION_VERSION,
    REWARD_CATALOG_VERSION,
    REWARD_DSL_VERSION,
    SNAPSHOT_VERSION,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "EXPERIMENT_VERSION",
    "OBSERVATION_VERSION",
    "REWARD_CATALOG_VERSION",
    "REWARD_DSL_VERSION",
    "SNAPSHOT_VERSION",
]
