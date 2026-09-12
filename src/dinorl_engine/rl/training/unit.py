"""Minimal reproducible masked-PPO training unit for RL-L2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv

from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.policies.mlp import DinoRLMLPFeaturesExtractor

__all__ = [
    "PPO_BATCH_SIZE",
    "PPO_EPOCHS",
    "ROLLOUT_TRANSITIONS",
    "PPOUnitMetrics",
    "PPOUnitResult",
    "ROLLOUT_TRANSITIONS",
    "create_maskable_ppo",
    "load_maskable_ppo",
    "save_maskable_ppo",
    "train_one_unit",
]

ROLLOUT_TRANSITIONS: Final = 2048
PPO_EPOCHS: Final = 4
PPO_BATCH_SIZE: Final = 256
_TORCH_THREADS: Final = 1


@dataclass(frozen=True, slots=True)
class PPOUnitMetrics:
    """Correction metrics from one PPO update, not a performance benchmark."""

    learner_transitions: int
    engine_actions: int
    epochs: int
    optimizer_steps: int
    diagnostics: dict[str, float]


@dataclass(frozen=True, slots=True)
class PPOUnitResult:
    """Result of a successfully completed minimal PPO unit."""

    metrics: PPOUnitMetrics


def create_maskable_ppo(
    env: DinoRLSingleAgentEnv | VecEnv, *, seed: int, n_steps: int = ROLLOUT_TRANSITIONS
) -> MaskablePPO:
    """Build the fixed CPU-only V1 provisional MLP policy."""

    torch.set_num_threads(_TORCH_THREADS)
    return MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.01,
        vf_coef=0.50,
        max_grad_norm=0.50,
        normalize_advantage=True,
        device="cpu",
        seed=seed,
        verbose=0,
        policy_kwargs={
            "features_extractor_class": DinoRLMLPFeaturesExtractor,
            "net_arch": [],
            "normalize_images": False,
        },
    )


def _finite_diagnostics(model: MaskablePPO) -> dict[str, float]:
    """Extract the numerical diagnostics emitted by the completed PPO update."""

    diagnostics: dict[str, float] = {}
    for key, value in model.logger.name_to_value.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError(f"non-finite PPO diagnostic: {key}")
        diagnostics[key] = number
    if not diagnostics:
        raise RuntimeError("PPO did not emit numerical diagnostics")
    return diagnostics


def _total_counter(env: DinoRLSingleAgentEnv | VecEnv, attribute: str) -> int:
    """Read an aggregate monotonic counter from one or many environments."""

    if isinstance(env, VecEnv):
        values = env.get_attr(attribute)
        if not all(type(value) is int for value in values):
            raise RuntimeError(f"vector environment has an invalid {attribute} counter")
        return sum(values)
    value = getattr(env, attribute)
    if type(value) is not int:
        raise RuntimeError(f"environment has an invalid {attribute} counter")
    return value


def train_one_unit(model: MaskablePPO, env: DinoRLSingleAgentEnv | VecEnv) -> PPOUnitResult:
    """Collect exactly 2,048 learner transitions and optimize for four epochs."""

    learner_before = _total_counter(env, "total_learner_transitions")
    engine_before = _total_counter(env, "total_engine_actions")
    model.learn(total_timesteps=ROLLOUT_TRANSITIONS, reset_num_timesteps=False)
    learner_transitions = _total_counter(env, "total_learner_transitions") - learner_before
    engine_actions = _total_counter(env, "total_engine_actions") - engine_before
    if learner_transitions != ROLLOUT_TRANSITIONS:
        raise RuntimeError(
            f"PPO unit collected an unexpected number of learner transitions: {learner_transitions}"
        )
    optimizer_steps = PPO_EPOCHS * (ROLLOUT_TRANSITIONS // PPO_BATCH_SIZE)
    return PPOUnitResult(
        metrics=PPOUnitMetrics(
            learner_transitions=learner_transitions,
            engine_actions=engine_actions,
            epochs=PPO_EPOCHS,
            optimizer_steps=optimizer_steps,
            diagnostics=_finite_diagnostics(model),
        )
    )


def save_maskable_ppo(model: MaskablePPO, path: Path) -> None:
    """Persist an internal minimal SB3 model for the RL-S0 round-trip check."""

    model.save(str(path))


def load_maskable_ppo(path: Path, env: DinoRLSingleAgentEnv | VecEnv) -> MaskablePPO:
    """Reload a locally-created internal SB3 model on the CPU."""

    return MaskablePPO.load(str(path), env=env, device="cpu")
