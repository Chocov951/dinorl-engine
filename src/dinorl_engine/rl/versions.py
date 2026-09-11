"""Immutable identifiers for the versioned RL contracts."""

from typing import Final

__all__ = [
    "CHECKPOINT_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "EXPERIMENT_VERSION",
    "MLP_ARCHITECTURE_VERSION",
    "OBSERVATION_VERSION",
    "REWARD_CATALOG_VERSION",
    "REWARD_DSL_VERSION",
    "SMALL_CNN_ARCHITECTURE_VERSION",
    "SNAPSHOT_VERSION",
]

CONFIG_SCHEMA_VERSION: Final = "1.0.0"
OBSERVATION_VERSION: Final = "rl-observation-v1"
EXPERIMENT_VERSION: Final = "rl-experiment-v1"
REWARD_DSL_VERSION: Final = "reward-dsl-v1"
REWARD_CATALOG_VERSION: Final = "reward-catalog-v1"
CHECKPOINT_VERSION: Final = "training-checkpoint-v1"
SNAPSHOT_VERSION: Final = "policy-snapshot-v1"
MLP_ARCHITECTURE_VERSION: Final = "mlp-v1"
SMALL_CNN_ARCHITECTURE_VERSION: Final = "small-cnn-v1"
