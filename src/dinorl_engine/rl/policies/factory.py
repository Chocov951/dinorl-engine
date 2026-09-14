"""Closed server-owned policy architecture catalog for the RL-L6 smoke."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from dinorl_engine.rl.policies.cnn import DinoRLSmallCNNFeaturesExtractor
from dinorl_engine.rl.policies.mlp import DinoRLMLPFeaturesExtractor
from dinorl_engine.rl.versions import MLP_ARCHITECTURE_VERSION, SMALL_CNN_ARCHITECTURE_VERSION

__all__ = [
    "ArchitectureSpecification",
    "PolicyArchitecture",
    "architecture_specification",
    "count_trainable_parameters",
]


class PolicyArchitecture(StrEnum):
    """The only architecture identifiers selectable by server-owned benchmark code."""

    MLP = MLP_ARCHITECTURE_VERSION
    SMALL_CNN = SMALL_CNN_ARCHITECTURE_VERSION


@dataclass(frozen=True, slots=True)
class ArchitectureSpecification:
    """Frozen encoder dimensions and approximate forward multiply-accumulates."""

    architecture: PolicyArchitecture
    features_extractor_class: type[BaseFeaturesExtractor]
    encoder_parameters: int
    encoder_multiply_accumulates: int


_SPECIFICATIONS = {
    PolicyArchitecture.MLP: ArchitectureSpecification(
        architecture=PolicyArchitecture.MLP,
        features_extractor_class=DinoRLMLPFeaturesExtractor,
        encoder_parameters=93_248,
        encoder_multiply_accumulates=93_056,
    ),
    PolicyArchitecture.SMALL_CNN: ArchitectureSpecification(
        architecture=PolicyArchitecture.SMALL_CNN,
        features_extractor_class=DinoRLSmallCNNFeaturesExtractor,
        encoder_parameters=91_936,
        encoder_multiply_accumulates=368_240,
    ),
}


def architecture_specification(architecture: PolicyArchitecture) -> ArchitectureSpecification:
    """Resolve one frozen architecture without accepting a player-provided string."""

    if not isinstance(architecture, PolicyArchitecture):
        raise ValueError("architecture must be a PolicyArchitecture")
    return _SPECIFICATIONS[architecture]


def count_trainable_parameters(module: torch.nn.Module) -> int:
    """Return the exact number of trainable scalar parameters in ``module``."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
