"""Replay v1 recording and canonical JSON serialization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from dinorl_engine.controllers.protocol import is_opaque_controller_id
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import GameResult
from dinorl_engine.core.events import (
    ActionTransition,
    ActorMovedEffect,
    CarcassConsumedEffect,
    CarcassPointAwardedEffect,
    CentralReactivatedEffect,
    CentralRechargeStartedEffect,
    ConsumptionInterruptedEffect,
    ConsumptionStartedEffect,
    DamageDealtEffect,
    Effect,
    EnduranceRestoredEffect,
    RestStartedEffect,
    TargetShovedEffect,
    TurnEndedEffect,
)
from dinorl_engine.core.state import GameState, snapshot_public

__all__ = [
    "ReplayControllerDescriptor",
    "ReplayRecorder",
    "canonical_replay_json",
    "replay_sha256",
]

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

_CONTROLLER_IDS: Final = frozenset({"aggressive-v1", "prudent-v1", "opportunist-v1"})
_ZERO_COST: Final[JsonObject] = {"movement": 0, "endurance": 0}


@dataclass(frozen=True, slots=True)
class ReplayControllerDescriptor:
    """Closed, path-free controller provenance stored in replay v1."""

    kind: Literal["scripted", "manual", "sequence"]
    id: str

    def __post_init__(self) -> None:
        if self.kind not in {"scripted", "manual", "sequence"}:
            raise ValueError("unknown replay controller kind")
        if self.kind == "scripted" and self.id not in _CONTROLLER_IDS:
            raise ValueError("unknown scripted controller id")
        if self.kind == "manual" and self.id != "manual":
            raise ValueError("manual replay controller id must be 'manual'")
        if self.kind == "sequence" and not is_opaque_controller_id(self.id):
            raise ValueError("sequence controller id must be opaque and path-free")


def _controller_json(controller: str | ReplayControllerDescriptor) -> JsonObject:
    descriptor = (
        ReplayControllerDescriptor("scripted", controller)
        if isinstance(controller, str)
        else controller
    )
    return {"kind": descriptor.kind, "id": descriptor.id}


def _json_value(value: object) -> JsonValue:
    """Copy a public immutable value into JSON-compatible mutable containers."""

    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Mapping):
        converted: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            converted[key] = _json_value(item)
        return converted
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported replay value: {type(value).__name__}")


def _state_json(state: GameState) -> JsonObject:
    value = _json_value(snapshot_public(state))
    if not isinstance(value, dict):
        raise AssertionError("public snapshots must be JSON objects")
    return value


def _effect_json(effect: Effect) -> JsonObject:
    value: JsonObject = {"type": effect.type, "actor": effect.actor.name}
    if isinstance(effect, ActorMovedEffect):
        value["from_position"] = list(effect.from_position)
        value["to_position"] = list(effect.to_position)
    elif isinstance(effect, TargetShovedEffect):
        value["target"] = effect.target.name
        value["from_position"] = list(effect.from_position)
        value["to_position"] = list(effect.to_position)
    elif isinstance(effect, DamageDealtEffect):
        value["target"] = effect.target.name
        value["amount"] = effect.amount
    elif isinstance(effect, ConsumptionInterruptedEffect):
        value["target"] = effect.target.name
        value["carcass_id"] = effect.carcass_id
    elif isinstance(
        effect,
        ConsumptionStartedEffect
        | CarcassConsumedEffect
        | CentralRechargeStartedEffect
        | CentralReactivatedEffect,
    ):
        value["carcass_id"] = effect.carcass_id
    elif isinstance(effect, CarcassPointAwardedEffect):
        value["carcass_id"] = effect.carcass_id
        value["amount"] = effect.amount
    elif isinstance(effect, EnduranceRestoredEffect):
        value["amount"] = effect.amount
    elif not isinstance(effect, RestStartedEffect | TurnEndedEffect):
        raise AssertionError(f"Unknown effect type: {type(effect).__name__}")
    return value


def canonical_replay_json(replay: Mapping[str, object]) -> bytes:
    """Serialize a replay to deterministic, compact UTF-8 JSON."""

    return json.dumps(
        replay,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def replay_sha256(replay: Mapping[str, object]) -> str:
    """Return the lowercase SHA-256 of canonical replay JSON."""

    return hashlib.sha256(canonical_replay_json(replay)).hexdigest()


class ReplayRecorder:
    """Collect replay dictionaries only for matches that requested a replay."""

    def __init__(
        self,
        state: GameState,
        *,
        controller_a: str | ReplayControllerDescriptor,
        controller_b: str | ReplayControllerDescriptor,
    ) -> None:
        if state.terminal:
            raise ValueError("cannot start a replay from a terminal state")
        self._header: JsonObject = {
            "replay_version": state.replay_version,
            "rules_version": state.rules_version,
            "action_version": state.action_version,
            "observation_version": state.observation_version,
            "engine_version": state.engine_version,
            "map_id": state.map_id,
            "seed": state.seed,
            "first_actor": state.first_actor.name,
            "controllers": {
                "A": _controller_json(controller_a),
                "B": _controller_json(controller_b),
            },
        }
        self._events: list[JsonObject] = []
        self._round = state.round
        self._turn = state.turn
        self._actor = state.active_actor
        self._finished = False
        self._append_event("match_started", state)
        self._append_event("turn_started", state, automatic_effects=())

    def _append_event(
        self,
        event_type: str,
        state: GameState,
        *,
        round_number: int | None = None,
        turn: int | None = None,
        actor: Actor | None = None,
        action: str | None = None,
        movement_cost: int = 0,
        endurance_cost: int = 0,
        effects: tuple[Effect, ...] = (),
        automatic_effects: tuple[Effect, ...] | None = None,
    ) -> None:
        event: JsonObject = {
            "seq": len(self._events),
            "type": event_type,
            "round": state.round if round_number is None else round_number,
            "turn": state.turn if turn is None else turn,
            "actor": (state.active_actor if actor is None else actor).name,
            "action": action,
            "cost": (
                dict(_ZERO_COST)
                if movement_cost == 0 and endurance_cost == 0
                else {"movement": movement_cost, "endurance": endurance_cost}
            ),
            "effects": [_effect_json(effect) for effect in effects],
            "state": _state_json(state),
        }
        if automatic_effects is not None:
            event["automatic_effects"] = [_effect_json(effect) for effect in automatic_effects]
        self._events.append(event)

    def record_action(self, transition: ActionTransition, state: GameState) -> None:
        """Record one engine transition and any following visible lifecycle events."""

        if self._finished:
            raise RuntimeError("replay is already finished")
        if transition.actor is not self._actor:
            raise ValueError("action actor does not match the current replay turn")
        self._append_event(
            "action_resolved",
            state,
            round_number=self._round,
            turn=self._turn,
            actor=transition.actor,
            action=transition.action.name,
            movement_cost=transition.cost.movement,
            endurance_cost=transition.cost.endurance,
            effects=transition.effects,
        )

        turn_advanced = state.turn != self._turn or state.active_actor is not self._actor
        if transition.turn_ended and turn_advanced:
            self._round = state.round
            self._turn = state.turn
            self._actor = state.active_actor
            self._append_event(
                "turn_started",
                state,
                automatic_effects=transition.automatic_effects,
            )
        if state.terminal:
            self._append_event("match_ended", state)
            self._finished = True

    def finish(self, result: GameResult) -> JsonObject:
        """Build the completed replay contract object."""

        if not self._finished:
            raise RuntimeError("cannot finish a replay before the match ends")
        replay = dict(self._header)
        replay["events"] = _json_value(self._events)
        replay["result"] = {
            "winner": result.winner.name if isinstance(result.winner, Actor) else result.winner,
            "reason": result.reason.name.lower(),
            "rounds_completed": result.rounds_completed,
            "individual_turns": result.individual_turns,
        }
        return replay
