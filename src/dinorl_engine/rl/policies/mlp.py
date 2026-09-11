"""Provisional MLP feature extractor for the V1 masked PPO policy."""

from __future__ import annotations

from typing import cast

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from dinorl_engine.rl.env.observation import FEATURE_COUNT, GRID_CHANNELS, GRID_COLUMNS, GRID_ROWS

__all__ = ["DinoRLMLPFeaturesExtractor"]

_FLAT_FEATURES = GRID_CHANNELS * GRID_ROWS * GRID_COLUMNS + FEATURE_COUNT
_HIDDEN_FEATURES = 64


class DinoRLMLPFeaturesExtractor(BaseFeaturesExtractor):
    """Flatten the canonical observation then project ``663 -> 128 -> 64``."""

    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=_HIDDEN_FEATURES)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(_FLAT_FEATURES, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, _HIDDEN_FEATURES),
            torch.nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode one batch of already-normalized float32 observations."""

        flattened_grid = observations["grid"].flatten(start_dim=1)
        return cast(
            torch.Tensor,
            self.network(torch.cat((flattened_grid, observations["features"]), dim=1)),
        )
