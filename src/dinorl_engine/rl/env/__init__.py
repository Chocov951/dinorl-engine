"""RL environment adapters."""

from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv, IllegalActionEscapeError
from dinorl_engine.rl.env.vectorization import (
    VECTOR_ENV_COUNTS,
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)

__all__ = [
    "VECTOR_ENV_COUNTS",
    "DinoRLSingleAgentEnv",
    "IllegalActionEscapeError",
    "VectorBackend",
    "VectorEnvironmentConfig",
    "create_vector_environment",
]
