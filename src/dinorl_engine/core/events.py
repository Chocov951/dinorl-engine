"""Typed internal action transitions used without replay dictionaries."""

from dataclasses import dataclass, field
from typing import Literal

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.maps import Position
from dinorl_engine.core.state import CarcassId

__all__ = [
    "ActionCost",
    "ActionTransition",
    "ActorMovedEffect",
    "ConsumptionInterruptedEffect",
    "DamageDealtEffect",
    "Effect",
    "TargetShovedEffect",
    "TurnEndedEffect",
]


@dataclass(frozen=True, slots=True)
class ActionCost:
    """Resources paid by an atomic action."""

    movement: int
    endurance: int


@dataclass(frozen=True, slots=True)
class ActorMovedEffect:
    """A voluntary movement resolved for the active actor."""

    actor: Actor
    from_position: Position
    to_position: Position
    type: Literal["actor_moved"] = field(default="actor_moved", init=False)


@dataclass(frozen=True, slots=True)
class DamageDealtEffect:
    """Immediate damage dealt by an actor to the opposing target."""

    actor: Actor
    target: Actor
    amount: int
    type: Literal["damage_dealt"] = field(default="damage_dealt", init=False)


@dataclass(frozen=True, slots=True)
class ConsumptionInterruptedEffect:
    """Cancellation of a target's pending carcass consumption."""

    actor: Actor
    target: Actor
    carcass_id: CarcassId
    type: Literal["consumption_interrupted"] = field(default="consumption_interrupted", init=False)


@dataclass(frozen=True, slots=True)
class TargetShovedEffect:
    """Forced displacement of the opposing target."""

    actor: Actor
    target: Actor
    from_position: Position
    to_position: Position
    type: Literal["target_shoved"] = field(default="target_shoved", init=False)


@dataclass(frozen=True, slots=True)
class TurnEndedEffect:
    """Explicit closure of an actor's turn."""

    actor: Actor
    type: Literal["turn_ended"] = field(default="turn_ended", init=False)


type Effect = (
    ActorMovedEffect
    | DamageDealtEffect
    | ConsumptionInterruptedEffect
    | TargetShovedEffect
    | TurnEndedEffect
)


@dataclass(frozen=True, slots=True)
class ActionTransition:
    """Immutable result of one resolved atomic action."""

    action: Action
    actor: Actor
    cost: ActionCost
    effects: tuple[Effect, ...]
    turn_ended: bool
