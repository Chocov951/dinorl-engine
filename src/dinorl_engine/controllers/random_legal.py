"""Uniform diagnostic controller over a supplied legal-action mask."""

from __future__ import annotations

from random import Random
from typing import Final

from dinorl_engine.core.actions import Action
from dinorl_engine.core.state import PublicSnapshot

__all__ = ["RANDOM_LEGAL_CONTROLLER_ID", "RandomLegalController"]

RANDOM_LEGAL_CONTROLLER_ID: Final = "random-legal-v1"


class RandomLegalController:
    """Choose uniformly from legal actions using an isolated random stream."""

    controller_id: Final = RANDOM_LEGAL_CONTROLLER_ID

    def __init__(self, *, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        self._random = Random(seed)

    def choose_action(self, state: PublicSnapshot, legal_actions: tuple[bool, ...]) -> Action:
        """Return a uniformly sampled legal action from a complete V1 mask."""

        del state
        if len(legal_actions) != len(Action) or any(
            type(legal) is not bool for legal in legal_actions
        ):
            raise ValueError("legal_actions must be a boolean mask of length 9")
        choices = [Action(index) for index, legal in enumerate(legal_actions) if legal]
        if not choices:
            raise ValueError("legal_actions must contain at least one action")
        return self._random.choice(choices)
