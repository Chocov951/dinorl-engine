"""Minimal masked PPO unit against the real single-agent wrapper."""

import math
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.training.unit import (
    PPO_EPOCHS,
    ROLLOUT_TRANSITIONS,
    create_maskable_ppo,
    load_maskable_ppo,
    save_maskable_ppo,
    train_one_unit,
)


def _assert_prediction_is_legal(model: MaskablePPO, env: DinoRLSingleAgentEnv) -> None:
    observation, _ = env.reset(seed=19)
    action, _ = model.predict(observation, action_masks=env.action_masks(), deterministic=True)

    assert env.action_masks()[int(action)]


def test_maskable_ppo_trains_one_exact_unit_and_round_trips_a_saved_model(tmp_path: Path) -> None:
    env = DinoRLSingleAgentEnv(seed=19)
    model = create_maskable_ppo(env, seed=19)
    _assert_prediction_is_legal(model, env)

    result = train_one_unit(model, env)

    assert result.metrics.learner_transitions == ROLLOUT_TRANSITIONS
    assert result.metrics.epochs == PPO_EPOCHS
    assert result.metrics.optimizer_steps == 32
    assert result.metrics.engine_actions >= ROLLOUT_TRANSITIONS
    assert all(math.isfinite(value) for value in result.metrics.diagnostics.values())
    assert model.rollout_buffer.action_masks.shape == (ROLLOUT_TRANSITIONS, 9)
    assert np.all(model.rollout_buffer.action_masks.any(axis=-1))
    assert all(
        bool(mask[int(action)])
        for mask, action in zip(
            model.rollout_buffer.action_masks,
            model.rollout_buffer.actions.reshape(-1),
            strict=True,
        )
    )

    path = tmp_path / "minimal-ppo.zip"
    save_maskable_ppo(model, path)
    loaded = load_maskable_ppo(path, env)
    _assert_prediction_is_legal(loaded, env)
