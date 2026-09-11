"""Native V1 reward used before the programmable Reward DSL."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import GameResult
from dinorl_engine.core.events import (
    ActionTransition,
    CarcassPointAwardedEffect,
    DamageDealtEffect,
)

__all__ = ["reference_reward"]

_CARCASS_POINT_REWARD = 0.10
_DAMAGE_POINT_REWARD = 0.05


def reference_reward(
    transition: ActionTransition,
    *,
    learner_actor: Actor,
    result: GameResult | None,
) -> float:
    """Return the native reward for one atomic transition in learner orientation.

    Effects which occur while ending a turn are deliberately included: they are
    public game events and must affect the same learner transition as the action
    that caused them.
    """

    reward = 0.0
    for effect in (*transition.effects, *transition.automatic_effects):
        if isinstance(effect, CarcassPointAwardedEffect):
            multiplier = 1.0 if effect.actor is learner_actor else -1.0
            reward += multiplier * _CARCASS_POINT_REWARD * effect.amount
        elif isinstance(effect, DamageDealtEffect):
            multiplier = 1.0 if effect.actor is learner_actor else -1.0
            reward += multiplier * _DAMAGE_POINT_REWARD * effect.amount

    if result is not None:
        if result.winner is learner_actor:
            reward += 1.0
        elif isinstance(result.winner, Actor):
            reward -= 1.0
    return reward
