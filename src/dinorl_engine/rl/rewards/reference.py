"""Native V1 reward used before the programmable Reward DSL."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import GameResult
from dinorl_engine.core.events import (
    ActionTransition,
    CarcassPointAwardedEffect,
    DamageDealtEffect,
)

__all__ = ["REFERENCE_REWARD_SOURCE", "reference_reward", "reference_reward_public"]

REFERENCE_REWARD_SOURCE = """\
fn reward(t: Transition) -> Number {
    return terminal_score(t, 1.0, -1.0, 0.0)
         + 0.10 * carcass_points_gained(t, SELF)
         - 0.10 * carcass_points_gained(t, OPPONENT)
         + 0.05 * damage_dealt(t, SELF)
         - 0.05 * damage_dealt(t, OPPONENT);
}
"""

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


def reference_reward_public(transition: dict[str, object]) -> float:
    """Evaluate the fixed V1 reward from its already-public transition values.

    This is the native, precalculated control used by RL-S2.  It intentionally
    mirrors the source term order; the specialized VM bytecode performs the
    same reads without interpreting generic expression instructions.
    """

    events = transition.get("events")
    if not isinstance(events, dict):
        events = {}

    def metric(name: str, actor: str) -> float:
        values = events.get(name)
        if not isinstance(values, dict):
            return 0.0
        value = values.get(actor, 0.0)
        return (
            float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0
        )

    outcome = transition.get("outcome")
    terminal = 1.0 if outcome == "WIN" else -1.0 if outcome == "LOSS" else 0.0
    return (
        terminal
        + 0.10 * metric("carcass_points_gained", "SELF")
        - 0.10 * metric("carcass_points_gained", "OPPONENT")
        + 0.05 * metric("damage_dealt", "SELF")
        - 0.05 * metric("damage_dealt", "OPPONENT")
    )
