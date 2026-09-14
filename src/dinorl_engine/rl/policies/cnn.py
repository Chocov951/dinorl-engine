"""Small frozen CNN feature extractor for the RL-L6 architecture smoke."""

from __future__ import annotations

from typing import cast

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from dinorl_engine.rl.env.observation import FEATURE_COUNT, GRID_CHANNELS, GRID_COLUMNS, GRID_ROWS

__all__ = ["DinoRLSmallCNNFeaturesExtractor"]

_GRID_CONV_CHANNELS = 16
_GRID_FEATURES = 64
_SCALAR_FEATURES = 16
_HIDDEN_FEATURES = 64
_FLATTENED_GRID_FEATURES = _GRID_CONV_CHANNELS * GRID_ROWS * GRID_COLUMNS


class DinoRLSmallCNNFeaturesExtractor(BaseFeaturesExtractor):
    """Encode grid and scalars as ``CNN -> 64``, ``MLP -> 16``, then ``80 -> 64``."""

    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=_HIDDEN_FEATURES)
        self.grid_network = torch.nn.Sequential(
            torch.nn.Conv2d(GRID_CHANNELS, _GRID_CONV_CHANNELS, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                _GRID_CONV_CHANNELS,
                _GRID_CONV_CHANNELS,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(_FLATTENED_GRID_FEATURES, _GRID_FEATURES),
            torch.nn.ReLU(),
        )
        self.scalar_network = torch.nn.Sequential(
            torch.nn.Linear(FEATURE_COUNT, _SCALAR_FEATURES),
            torch.nn.ReLU(),
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(_GRID_FEATURES + _SCALAR_FEATURES, _HIDDEN_FEATURES),
            torch.nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode one batch of canonical float32 observations without image normalization."""

        grid_features = self.grid_network(observations["grid"])
        scalar_features = self.scalar_network(observations["features"])
        return cast(torch.Tensor, self.fusion(torch.cat((grid_features, scalar_features), dim=1)))
