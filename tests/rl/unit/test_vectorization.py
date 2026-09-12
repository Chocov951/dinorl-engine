"""Contracts for the RL-L3 seeded vector-environment factories."""

import sys

import numpy as np
import pytest

from dinorl_engine.rl.env.vectorization import (
    VECTOR_ENV_COUNTS,
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
    derive_environment_seed,
)


def test_environment_seeds_are_stable_and_unique_per_vector_index() -> None:
    first = tuple(derive_environment_seed(19, index) for index in range(8))

    assert first == tuple(derive_environment_seed(19, index) for index in range(8))
    assert len(set(first)) == 8
    assert first != tuple(derive_environment_seed(20, index) for index in range(8))


@pytest.mark.parametrize("n_envs", VECTOR_ENV_COUNTS)
def test_each_supported_configuration_keeps_one_global_2048_transition_rollout(
    n_envs: int,
) -> None:
    configuration = VectorEnvironmentConfig(
        backend=VectorBackend.DUMMY,
        n_envs=n_envs,
        seed=19,
    )

    assert configuration.n_steps == 2048 // n_envs
    assert configuration.n_envs * configuration.n_steps == 2048


@pytest.mark.skipif(sys.platform == "win32", reason="SubprocVecEnv is measured on Linux target")
def test_dummy_and_subprocess_backends_start_from_the_same_seeded_observations() -> None:
    dummy = create_vector_environment(
        VectorEnvironmentConfig(backend=VectorBackend.DUMMY, n_envs=2, seed=19)
    )
    subprocess = create_vector_environment(
        VectorEnvironmentConfig(backend=VectorBackend.SUBPROCESS, n_envs=2, seed=19)
    )
    try:
        dummy_observation = dummy.reset()
        subprocess_observation = subprocess.reset()

        np.testing.assert_array_equal(dummy_observation["grid"], subprocess_observation["grid"])
        np.testing.assert_array_equal(
            dummy_observation["features"], subprocess_observation["features"]
        )
    finally:
        dummy.close()
        subprocess.close()


@pytest.mark.skipif(sys.platform == "win32", reason="SubprocVecEnv is measured on Linux target")
def test_subprocess_backend_closes_all_workers() -> None:
    environment = create_vector_environment(
        VectorEnvironmentConfig(backend=VectorBackend.SUBPROCESS, n_envs=2, seed=19)
    )

    environment.reset()
    environment.close()

    assert environment.closed
    assert all(not process.is_alive() for process in environment.processes)
