"""Canonical public transition values consumed by the Reward DSL only."""

from __future__ import annotations

from collections.abc import Mapping

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor, EndReason
from dinorl_engine.core.engine import GameResult
from dinorl_engine.core.events import (
    ActionTransition,
    CarcassPointAwardedEffect,
    ConsumptionInterruptedEffect,
    ConsumptionStartedEffect,
    DamageDealtEffect,
    EnduranceRestoredEffect,
    RestStartedEffect,
    TargetShovedEffect,
)
from dinorl_engine.core.maps import load_map
from dinorl_engine.core.state import PublicSnapshot

__all__ = ["public_reward_transition"]


def _raptor_view(values: Mapping[str, object], *, own: bool) -> dict[str, object]:
    fields = (
        (
            "position",
            "hp",
            "endurance",
            "carcass_score",
            "main_action_available",
            "voluntary_move_done",
        )
        if own
        else (
            "position",
            "hp",
            "endurance",
            "carcass_score",
            "consumption_pending",
            "rest_pending",
        )
    )
    return {field: values[field] for field in fields}


def _carcass_view(snapshot: PublicSnapshot) -> dict[str, object]:
    arena = load_map(MAP_ID)
    snapshots = snapshot["carcasses"]
    turn = snapshot["turn"]
    if not isinstance(snapshots, Mapping) or type(turn) is not int:
        raise RuntimeError("public snapshot carcass view is invalid")
    values: dict[str, object] = {}
    for carcass in arena.carcasses:
        state = snapshots.get(carcass.carcass_id)
        if not isinstance(state, Mapping):
            raise RuntimeError("public snapshot carcass state is invalid")
        status = state.get("status")
        reactivate_on_turn = state.get("reactivate_on_turn")
        available = status in {"available", "active"}
        delay = max(0, reactivate_on_turn - turn) if type(reactivate_on_turn) is int else 0
        values[carcass.carcass_id] = {
            "position": carcass.position,
            "available": available,
            "reactivation_delay": delay,
        }
    return values


def _view(snapshot: PublicSnapshot, learner: Actor, first_actor: Actor) -> dict[str, object]:
    raptors = snapshot["raptors"]
    if not isinstance(raptors, Mapping):
        raise RuntimeError("public snapshot raptors are invalid")
    self_key = learner.name
    opponent_key = (Actor.B if learner is Actor.A else Actor.A).name
    own = raptors.get(self_key)
    opponent = raptors.get(opponent_key)
    if not isinstance(own, Mapping) or not isinstance(opponent, Mapping):
        raise RuntimeError("public snapshot raptor view is invalid")
    active_actor = snapshot["active_actor"]
    if not isinstance(active_actor, str) or active_actor not in Actor.__members__:
        raise RuntimeError("public snapshot active actor is invalid")
    return {
        "self": _raptor_view(own, own=True),
        "opponent": _raptor_view(opponent, own=False),
        "round": snapshot["round"],
        "turn": snapshot["turn"],
        "active_actor": _actor(Actor[active_actor], learner),
        "initiative": _actor(first_actor, learner),
        "carcasses": _carcass_view(snapshot),
    }


def _actor(actor: Actor, learner: Actor) -> str:
    return "SELF" if actor is learner else "OPPONENT"


def public_reward_transition(
    *,
    before: PublicSnapshot,
    after: PublicSnapshot,
    transition: ActionTransition,
    learner_actor: Actor,
    first_actor: Actor,
    result: GameResult | None,
    legal_actions: tuple[bool, ...] | None = None,
) -> dict[str, object]:
    """Build the closed V1 DSL input from public engine data only."""

    events: dict[str, dict[str, float | bool]] = {
        name: {
            "SELF": 0.0
            if name in {"damage_dealt", "shove_wall_damage", "carcass_points_gained"}
            else False,
            "OPPONENT": 0.0
            if name in {"damage_dealt", "shove_wall_damage", "carcass_points_gained"}
            else False,
        }
        for name in (
            "damage_dealt",
            "bite_attempted",
            "bite_hit",
            "shove_attempted",
            "shove_moved_target",
            "shove_wall_damage",
            "entered_mud",
            "exited_mud",
            "feed_started",
            "feed_interrupted",
            "feed_completed",
            "rest_started",
            "rest_completed",
            "carcass_points_gained",
        )
    }
    actor = _actor(transition.actor, learner_actor)
    if transition.action.name == "BITE":
        events["bite_attempted"][actor] = True
    if transition.action.name == "SHOVE":
        events["shove_attempted"][actor] = True
    for effect in (*transition.effects, *transition.automatic_effects):
        if isinstance(effect, DamageDealtEffect):
            owner = _actor(effect.actor, learner_actor)
            events["damage_dealt"][owner] = float(effect.amount)
            if transition.action.name == "BITE":
                events["bite_hit"][owner] = True
            if transition.action.name == "SHOVE":
                events["shove_wall_damage"][owner] = float(effect.amount)
        elif isinstance(effect, TargetShovedEffect):
            events["shove_moved_target"][_actor(effect.actor, learner_actor)] = (
                effect.from_position != effect.to_position
            )
        elif isinstance(effect, ConsumptionStartedEffect):
            events["feed_started"][_actor(effect.actor, learner_actor)] = True
        elif isinstance(effect, ConsumptionInterruptedEffect):
            events["feed_interrupted"][_actor(effect.target, learner_actor)] = True
        elif isinstance(effect, CarcassPointAwardedEffect):
            events["carcass_points_gained"][_actor(effect.actor, learner_actor)] = float(
                effect.amount
            )
            events["feed_completed"][_actor(effect.actor, learner_actor)] = True
        elif isinstance(effect, RestStartedEffect):
            events["rest_started"][_actor(effect.actor, learner_actor)] = True
        elif isinstance(effect, EnduranceRestoredEffect):
            events["rest_completed"][_actor(effect.actor, learner_actor)] = True
    arena = load_map(MAP_ID)
    before_raptors = before["raptors"]
    after_raptors = after["raptors"]
    if not isinstance(before_raptors, Mapping) or not isinstance(after_raptors, Mapping):
        raise RuntimeError("public snapshot raptors are invalid")
    for engine_actor in Actor:
        before_raptor = before_raptors.get(engine_actor.name)
        after_raptor = after_raptors.get(engine_actor.name)
        if not isinstance(before_raptor, Mapping) or not isinstance(after_raptor, Mapping):
            raise RuntimeError("public snapshot raptor state is invalid")
        before_position = before_raptor.get("position")
        after_position = after_raptor.get("position")
        if not isinstance(before_position, tuple) or not isinstance(after_position, tuple):
            raise RuntimeError("public snapshot raptor position is invalid")
        perspective = _actor(engine_actor, learner_actor)
        events["entered_mud"][perspective] = (
            before_position not in arena.mud and after_position in arena.mud
        )
        events["exited_mud"][perspective] = (
            before_position in arena.mud and after_position not in arena.mud
        )
    outcome = "ONGOING"
    end_reason: str | None = None
    if result is not None:
        outcome = (
            "DRAW"
            if result.winner == "draw"
            else "WIN"
            if result.winner is learner_actor
            else "LOSS"
        )
        end_reason = {
            EndReason.KO: "ko",
            EndReason.CARCASS_SCORE: "carcass_score",
            EndReason.ROUND_LIMIT: "round_limit",
        }[result.reason]
    tiles = {
        f"{row},{column}": arena.tile_at((row, column)).name
        for row in range(arena.rows)
        for column in range(arena.columns)
    }
    return {
        "before": _view(before, learner_actor, first_actor),
        "after": _view(after, learner_actor, first_actor),
        "action": {
            "actor": actor,
            "kind": transition.action.name,
            "direction": transition.action.name.removeprefix("MOVE_")
            if transition.action.name.startswith("MOVE_")
            else "NO_DIRECTION",
            "movement_cost": float(transition.cost.movement),
            "endurance_cost": float(transition.cost.endurance),
            "success": True,
        },
        "events": events,
        "legal_actions": {
            action.name: legal_actions[int(action)]
            for action in Action
            if legal_actions is not None
        },
        "tiles": tiles,
        "map": {
            "rows": arena.rows,
            "columns": arena.columns,
            "walls": tuple(sorted(arena.walls)),
            "mud": tuple(sorted(arena.mud)),
        },
        "outcome": outcome,
        "end_reason": end_reason,
    }
