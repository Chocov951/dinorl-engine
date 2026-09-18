"""Local-only MLP candidates used by the RL-S5 V2 diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import ClassVar, cast

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from dinorl_engine.rl.env.observation import FEATURE_COUNT, GRID_CHANNELS, GRID_COLUMNS, GRID_ROWS

__all__ = [
    "LOCAL_MLP_V2_ARCHITECTURES",
    "DinoRLBalancedMLPV2FeaturesExtractor",
    "DinoRLCompactMLPV2FeaturesExtractor",
    "DinoRLDeepMLPV2FeaturesExtractor",
    "LocalMLPV2Architecture",
    "LocalMLPV2Specification",
    "local_mlp_v2_specification",
]

_FLAT_FEATURES = GRID_CHANNELS * GRID_ROWS * GRID_COLUMNS + FEATURE_COUNT
_OUTPUT_FEATURES = 64


class LocalMLPV2Architecture(StrEnum):
    """Experimental identifiers which can never enter a V1 server archive."""

    COMPACT = "mlp-compact-v2"
    BALANCED = "mlp-balanced-v2"
    DEEP = "mlp-deep-v2"


class _DinoRLMLPV2FeaturesExtractor(BaseFeaturesExtractor):
    hidden_features: ClassVar[tuple[int, ...]]

    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=_OUTPUT_FEATURES)
        dimensions = (_FLAT_FEATURES, *self.hidden_features, _OUTPUT_FEATURES)
        layers: list[torch.nn.Module] = []
        for input_features, output_features in pairwise(dimensions):
            layers.extend((torch.nn.Linear(input_features, output_features), torch.nn.ReLU()))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        flattened_grid = observations["grid"].flatten(start_dim=1)
        combined = torch.cat((flattened_grid, observations["features"]), dim=1)
        return cast(torch.Tensor, self.network(combined))


class DinoRLCompactMLPV2FeaturesExtractor(_DinoRLMLPV2FeaturesExtractor):
    """Half-size encoder: ``663 -> 64 -> 64``."""

    hidden_features = (64,)


class DinoRLBalancedMLPV2FeaturesExtractor(_DinoRLMLPV2FeaturesExtractor):
    """Balanced encoder: ``663 -> 96 -> 64``."""

    hidden_features = (96,)


class DinoRLDeepMLPV2FeaturesExtractor(_DinoRLMLPV2FeaturesExtractor):
    """Depth candidate: ``663 -> 128 -> 128 -> 64``."""

    hidden_features = (128, 128)


@dataclass(frozen=True, slots=True)
class LocalMLPV2Specification:
    architecture: LocalMLPV2Architecture
    features_extractor_class: type[BaseFeaturesExtractor]
    encoder_parameters: int
    encoder_multiply_accumulates: int


_SPECIFICATIONS = {
    LocalMLPV2Architecture.COMPACT: LocalMLPV2Specification(
        architecture=LocalMLPV2Architecture.COMPACT,
        features_extractor_class=DinoRLCompactMLPV2FeaturesExtractor,
        encoder_parameters=46_656,
        encoder_multiply_accumulates=46_528,
    ),
    LocalMLPV2Architecture.BALANCED: LocalMLPV2Specification(
        architecture=LocalMLPV2Architecture.BALANCED,
        features_extractor_class=DinoRLBalancedMLPV2FeaturesExtractor,
        encoder_parameters=69_952,
        encoder_multiply_accumulates=69_792,
    ),
    LocalMLPV2Architecture.DEEP: LocalMLPV2Specification(
        architecture=LocalMLPV2Architecture.DEEP,
        features_extractor_class=DinoRLDeepMLPV2FeaturesExtractor,
        encoder_parameters=109_760,
        encoder_multiply_accumulates=109_440,
    ),
}

LOCAL_MLP_V2_ARCHITECTURES = tuple(LocalMLPV2Architecture)


def local_mlp_v2_specification(
    architecture: LocalMLPV2Architecture,
) -> LocalMLPV2Specification:
    if not isinstance(architecture, LocalMLPV2Architecture):
        raise ValueError("architecture must be a LocalMLPV2Architecture")
    return _SPECIFICATIONS[architecture]
