"""Local-only MLP V2 architecture and tournament contracts."""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.policies.factory import count_trainable_parameters
from dinorl_engine.rl.policies.local_mlp_v2 import (
    LOCAL_MLP_V2_ARCHITECTURES,
    LocalMLPV2Architecture,
    local_mlp_v2_specification,
)
from dinorl_engine.rl.training.unit import create_maskable_ppo
from dinorl_engine.server_checks.suites.rl_s5_v2 import (
    aggregate_candidate_seeds,
    tournament_pairs,
)


def test_three_local_mlp_v2_candidates_have_distinct_frozen_capacity() -> None:
    expected = {
        LocalMLPV2Architecture.COMPACT: (46_656, 46_528),
        LocalMLPV2Architecture.BALANCED: (69_952, 69_792),
        LocalMLPV2Architecture.DEEP: (109_760, 109_440),
    }
    environment = DinoRLSingleAgentEnv(seed=19)
    observations = {
        "grid": torch.rand((3, 8, 9, 9), dtype=torch.float32),
        "features": torch.rand((3, 15), dtype=torch.float32),
    }

    assert tuple(LOCAL_MLP_V2_ARCHITECTURES) == tuple(expected)
    for architecture, (parameters, multiply_accumulates) in expected.items():
        specification = local_mlp_v2_specification(architecture)
        extractor = specification.features_extractor_class(environment.observation_space)
        encoded = extractor(observations)
        encoded.square().mean().backward()

        assert encoded.shape == (3, 64)
        assert count_trainable_parameters(extractor) == parameters
        assert specification.encoder_parameters == parameters
        assert specification.encoder_multiply_accumulates == multiply_accumulates
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in extractor.parameters()
        )


def test_each_local_mlp_v2_candidate_builds_the_same_masked_ppo_heads() -> None:
    for architecture in LOCAL_MLP_V2_ARCHITECTURES:
        environment = DinoRLSingleAgentEnv(seed=19)
        try:
            model = create_maskable_ppo(environment, seed=19, architecture=architecture)
            observation, _info = environment.reset(seed=19)
            action, _state = model.predict(
                observation,
                action_masks=environment.action_masks(),
                deterministic=True,
            )

            assert isinstance(model.policy.features_extractor, BaseFeaturesExtractor)
            assert model.policy.action_net.in_features == 64
            assert model.policy.value_net.in_features == 64
            assert environment.action_masks()[int(action)]
        finally:
            environment.close()


def test_tournament_contains_every_distinct_pair_and_no_self_play() -> None:
    pairs = tournament_pairs()

    assert len(pairs) == 6
    assert all(left != right for left, right in pairs)
    assert set(pairs) == {
        ("mlp-v1", "mlp-compact-v2"),
        ("mlp-v1", "mlp-balanced-v2"),
        ("mlp-v1", "mlp-deep-v2"),
        ("mlp-compact-v2", "mlp-balanced-v2"),
        ("mlp-compact-v2", "mlp-deep-v2"),
        ("mlp-balanced-v2", "mlp-deep-v2"),
    }


def test_v2_seed_aggregation_keeps_median_iqr_and_learning_speed() -> None:
    seeds = [
        {
            "transitions_to_gate": transitions,
            "wall_seconds": wall,
            "final_score": score,
            "inference_seconds": inference,
        }
        for transitions, wall, score, inference in (
            (225_280, 12.0, 0.8, 0.003),
            (163_840, 10.0, 1.0, 0.001),
            (204_800, 11.0, 0.9, 0.002),
        )
    ]

    aggregate = aggregate_candidate_seeds(seeds)

    assert aggregate["transitions_to_gate"] == {
        "median": 204_800,
        "iqr": [163_840, 225_280],
    }
    assert aggregate["wall_seconds"] == {"mean": 11.0, "median": 11.0, "iqr": [10.0, 12.0]}
    assert aggregate["final_score"] == {"mean": 0.9, "median": 0.9, "iqr": [0.8, 1.0]}
    assert aggregate["inference_seconds"] == {
        "mean": 0.002,
        "median": 0.002,
        "iqr": [0.001, 0.003],
    }
