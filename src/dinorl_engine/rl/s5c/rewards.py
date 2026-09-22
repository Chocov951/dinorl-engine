"""Versioned event-driven reward programs for the three RL-S5c specialists."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Final

from dinorl_engine.rl.rewards.runtime import CompiledReward, compile_reward

__all__ = ["ARCHETYPES", "a3_rewards", "specialist_reward_terms", "specialist_rewards"]

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
        return 0.12;
    }
    return 0.0;
}
""",
}


@lru_cache(maxsize=1)
def specialist_rewards() -> dict[str, CompiledReward]:
    """Compile immutable calibration programs; coefficients live only in the DSL."""

    return {archetype: compile_reward(_SOURCES[archetype]) for archetype in ARCHETYPES}


@lru_cache(maxsize=1)
def a3_rewards() -> dict[str, CompiledReward]:
    """Compile the four frozen A3 pilot variants; episode caps live in orchestration."""

    predator_balanced = """\
fn reward(t: Transition) -> Number {
    return terminal_score(t, 1.0, -1.0, 0.0)
         + 0.06 * damage_dealt(t, SELF) - 0.04 * damage_dealt(t, OPPONENT)
         + effective_interrupt(t);
}
fn effective_interrupt(t: Transition) -> Number {
    if feed_interrupted(t, OPPONENT) { return 0.04; }
    return 0.0;
}
"""
    return {
        "scavenger-a2-control": compile_reward(_SOURCES["scavenger"]),
        "scavenger-a3-curriculum": compile_reward(_SOURCES["scavenger"]),
        "predator-a2-control": compile_reward(_SOURCES["predator"]),
        "predator-a3-balanced": compile_reward(predator_balanced),
    }


def _owner(events: Mapping[str, object], name: str, owner: str) -> float:
    value = events.get(name)
    if not isinstance(value, Mapping):
        return 0.0
    number = value.get(owner, 0.0)
    return float(number) if isinstance(number, int | float | bool) else 0.0


def specialist_reward_terms(
    reward: CompiledReward, transition: Mapping[str, object]
) -> dict[str, float]:
    """Decompose every specialist auxiliary term without changing its DSL source."""

    archetypes = specialist_rewards()
    archetype = next(
        (name for name, candidate in archetypes.items() if candidate.cache_key == reward.cache_key),
        None,
    )
    if archetype is None:
        return {}
    events_value = transition.get("events")
    events = events_value if isinstance(events_value, Mapping) else {}
    terms = {
        "damage_dealt": (0.05 if archetype == "controller" else 0.10)
        * _owner(events, "damage_dealt", "SELF"),
        "damage_received": (-0.05 if archetype != "predator" else -0.03)
        * _owner(events, "damage_dealt", "OPPONENT"),
    }
    if archetype == "scavenger":
        terms["carcass_points_gained"] = 0.15 * _owner(events, "carcass_points_gained", "SELF")
        terms["opponent_carcass_points"] = -0.10 * _owner(
            events, "carcass_points_gained", "OPPONENT"
        )
        before = transition.get("before")
        self_view = before.get("self") if isinstance(before, Mapping) else None
        opponent_view = before.get("opponent") if isinstance(before, Mapping) else None
        self_position = self_view.get("position") if isinstance(self_view, Mapping) else None
        opponent_position = (
            opponent_view.get("position") if isinstance(opponent_view, Mapping) else None
        )
        distance = 0
        if (
            isinstance(self_position, tuple)
            and isinstance(opponent_position, tuple)
            and len(self_position) == len(opponent_position) == 2
        ):
            distance = abs(int(self_position[0]) - int(opponent_position[0])) + abs(
                int(self_position[1]) - int(opponent_position[1])
            )
        terms["safe_feed"] = 0.03 * float(
            bool(_owner(events, "feed_started", "SELF")) and distance > 3
        )
    if archetype == "controller":
        useful = bool(_owner(events, "shove_moved_target", "SELF")) and bool(
            _owner(events, "shove_wall_damage", "SELF")
            or _owner(events, "entered_mud", "OPPONENT")
            or _owner(events, "feed_interrupted", "OPPONENT")
        )
        terms["useful_shove"] = 0.12 * float(useful)
    return terms
