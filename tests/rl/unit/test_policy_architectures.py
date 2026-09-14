"""Frozen RL-L6 contracts for the two server-owned policy encoders."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.policies.cnn import DinoRLSmallCNNFeaturesExtractor
from dinorl_engine.rl.policies.factory import (
    PolicyArchitecture,
    architecture_specification,
    count_trainable_parameters,
)
from dinorl_engine.rl.policies.mlp import DinoRLMLPFeaturesExtractor
from dinorl_engine.rl.training.unit import (
    create_maskable_ppo,
    load_maskable_ppo,
    save_maskable_ppo,
)


@pytest.mark.parametrize(
    ("architecture", "extractor_type", "expected_parameters"),
    [
        (PolicyArchitecture.MLP, DinoRLMLPFeaturesExtractor, 93_248),
        (PolicyArchitecture.SMALL_CNN, DinoRLSmallCNNFeaturesExtractor, 91_936),
    ],
)
def test_candidate_encoders_have_frozen_shapes_finite_gradients_and_parameter_counts(
    architecture: PolicyArchitecture,
    extractor_type: type[torch.nn.Module],
    expected_parameters: int,
) -> None:
    environment = DinoRLSingleAgentEnv(seed=19)
    extractor = extractor_type(environment.observation_space)
    observations = {
        "grid": torch.rand((3, 8, 9, 9), dtype=torch.float32),
        "features": torch.rand((3, 15), dtype=torch.float32),
    }

    encoded = extractor(observations)
    encoded.square().mean().backward()

    specification = architecture_specification(architecture)
    assert encoded.shape == (3, 64)
    assert count_trainable_parameters(extractor) == expected_parameters
    assert specification.encoder_parameters == expected_parameters
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in extractor.parameters()
    )


@pytest.mark.parametrize("architecture", list(PolicyArchitecture))
def test_candidate_architectures_keep_the_same_masked_ppo_heads_and_round_trip(
    architecture: PolicyArchitecture, tmp_path: Path
) -> None:
    environment = DinoRLSingleAgentEnv(seed=19)
    model = create_maskable_ppo(environment, seed=19, architecture=architecture)
    observation, _ = environment.reset(seed=19)
    action, _ = model.predict(
        observation, action_masks=environment.action_masks(), deterministic=True
    )

    path = tmp_path / f"{architecture.value}.zip"
    save_maskable_ppo(model, path)
    loaded = load_maskable_ppo(path, environment)

    assert environment.action_masks()[int(action)]
    assert model.policy.action_net.in_features == 64
    assert model.policy.action_net.out_features == 9
    assert model.policy.value_net.in_features == 64
    assert model.policy.value_net.out_features == 1
    assert (
        type(loaded.policy.features_extractor)
        is architecture_specification(architecture).features_extractor_class
    )


def test_policy_architecture_is_a_closed_server_owned_choice() -> None:
    with pytest.raises(ValueError, match="PolicyArchitecture"):
        architecture_specification("player-choice")  # type: ignore[arg-type]
