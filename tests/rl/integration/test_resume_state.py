"""RL-L5 recovery state keeps a partially played game bit-for-bit reproducible."""

from __future__ import annotations

import random

import numpy as np
import torch

from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.training.resume import (
    capture_process_rng_state,
    export_vector_recovery_state,
    restore_process_rng_state,
    restore_vector_recovery_state,
)


def _first_legal_action(environment: DinoRLSingleAgentEnv) -> int:
    return next(index for index, allowed in enumerate(environment.action_masks()) if allowed)


def test_single_agent_recovery_restores_a_game_mid_episode_without_resetting() -> None:
    original = DinoRLSingleAgentEnv(seed=19, environment_index=3)
    original.reset()
    for _ in range(7):
        _observation, _reward, terminated, truncated, _info = original.step(
            _first_legal_action(original)
        )
        assert not truncated
        if terminated:
            original.reset()

    recovery_state = original.export_recovery_state()
    recovered = DinoRLSingleAgentEnv(seed=0, environment_index=0)
    recovered.restore_recovery_state(recovery_state)

    assert recovered.snapshot_recovery_state() == original.snapshot_recovery_state()
    np.testing.assert_array_equal(recovered.action_masks(), original.action_masks())
    action = _first_legal_action(original)
    original_result = original.step(action)
    recovered_result = recovered.step(action)
    np.testing.assert_equal(original_result[0], recovered_result[0])
    assert original_result[1:] == recovered_result[1:]


def test_process_rng_recovery_restores_python_numpy_and_torch_streams() -> None:
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    state = capture_process_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(4))

    random.random()
    np.random.random()
    torch.rand(4)
    restore_process_rng_state(state)
    actual = (random.random(), float(np.random.random()), torch.rand(4))

    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_vector_recovery_preserves_distinct_partial_games_without_a_reset() -> None:
    configuration = VectorEnvironmentConfig(backend=VectorBackend.DUMMY, n_envs=2, seed=19)
    original = create_vector_environment(configuration)
    recovered = create_vector_environment(configuration)
    try:
        original.reset()
        for index, environment in enumerate(original.envs):
            for _ in range(index + 3):
                environment.step(_first_legal_action(environment))
        state = export_vector_recovery_state(original)

        restore_vector_recovery_state(recovered, state)

        assert [item.snapshot_recovery_state() for item in recovered.envs] == [
            item.snapshot_recovery_state() for item in original.envs
        ]
    finally:
        original.close()
        recovered.close()
