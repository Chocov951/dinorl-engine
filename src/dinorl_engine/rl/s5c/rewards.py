"""Versioned event-driven reward programs for the three RL-S5c specialists."""

from __future__ import annotations

from typing import Final

from dinorl_engine.rl.rewards.runtime import CompiledReward, compile_reward

__all__ = ["ARCHETYPES", "specialist_rewards"]

ARCHETYPES: Final = ("scavenger", "predator", "controller")

_SOURCES: Final = {
    "scavenger": """\
fn reward(t: Transition) -> Number {
    return terminal_score(t, 1.0, -1.0, 0.0)
         + 0.10 * damage_dealt(t, SELF) - 0.05 * damage_dealt(t, OPPONENT)
         + 0.15 * carcass_points_gained(t, SELF) - 0.10 * carcass_points_gained(t, OPPONENT)
         + safe_feed(t);
}
fn safe_feed(t: Transition) -> Number {
    if feed_started(t, SELF) &&
       manhattan(t.before.self.position, t.before.opponent.position) > 3.0 {
        return 0.03;
    }
    return 0.0;
}
""",
    "predator": """\
fn reward(t: Transition) -> Number {
    return terminal_score(t, 1.0, -1.0, 0.0)
         + 0.10 * damage_dealt(t, SELF) - 0.03 * damage_dealt(t, OPPONENT);
}
""",
    "controller": """\
fn reward(t: Transition) -> Number {
    return terminal_score(t, 1.0, -1.0, 0.0)
         + 0.05 * damage_dealt(t, SELF) - 0.05 * damage_dealt(t, OPPONENT)
         + useful_shove(t);
}
fn useful_shove(t: Transition) -> Number {
    if shove_moved_target(t, SELF) &&
       (shove_wall_damage(t, SELF) > 0.0 || entered_mud(t, OPPONENT) ||
        feed_interrupted(t, OPPONENT)) {
        return 0.08;
    }
    return 0.0;
}
""",
}


def specialist_rewards() -> dict[str, CompiledReward]:
    """Compile immutable calibration programs; coefficients live only in the DSL."""

    return {archetype: compile_reward(_SOURCES[archetype]) for archetype in ARCHETYPES}
