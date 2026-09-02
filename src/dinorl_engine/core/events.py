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
    "CarcassConsumedEffect",
    "CarcassPointAwardedEffect",
    "CentralReactivatedEffect",
    "CentralRechargeStartedEffect",
    "ConsumptionStartedEffect",
    "ConsumptionInterruptedEffect",
    "DamageDealtEffect",
    "EnduranceRestoredEffect",
    "Effect",
    "RestStartedEffect",
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
class ConsumptionStartedEffect:
    """Start of an interruptible carcass consumption."""

    actor: Actor
    carcass_id: CarcassId
    type: Literal["consumption_started"] = field(default="consumption_started", init=False)


@dataclass(frozen=True, slots=True)
class CarcassPointAwardedEffect:
    """Delayed carcass point awarded at turn start."""

    actor: Actor
    carcass_id: CarcassId
    amount: int
    type: Literal["carcass_point_awarded"] = field(default="carcass_point_awarded", init=False)


@dataclass(frozen=True, slots=True)
class CarcassConsumedEffect:
    """Permanent removal of a successfully consumed lateral carcass."""

    actor: Actor
    carcass_id: CarcassId
    type: Literal["carcass_consumed"] = field(default="carcass_consumed", init=False)


@dataclass(frozen=True, slots=True)
class CentralRechargeStartedEffect:
    """Start of the central carcass one-round cooldown."""

    actor: Actor
    carcass_id: CarcassId = "carcass_center"
    type: Literal["central_recharge_started"] = field(
        default="central_recharge_started", init=False
    )


@dataclass(frozen=True, slots=True)
class CentralReactivatedEffect:
    """Reactivation of the central carcass at the scheduled turn."""

    actor: Actor
    carcass_id: CarcassId = "carcass_center"
    type: Literal["central_reactivated"] = field(default="central_reactivated", init=False)


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
class RestStartedEffect:
    """Declaration of a delayed endurance restoration."""

    actor: Actor
    type: Literal["rest_started"] = field(default="rest_started", init=False)


@dataclass(frozen=True, slots=True)
class EnduranceRestoredEffect:
    """Endurance restored at the beginning of an actor's next turn."""

    actor: Actor
    amount: int
    type: Literal["endurance_restored"] = field(default="endurance_restored", init=False)


@dataclass(frozen=True, slots=True)
class TurnEndedEffect:
    """Explicit closure of an actor's turn."""

    actor: Actor
    type: Literal["turn_ended"] = field(default="turn_ended", init=False)


type Effect = (
    ActorMovedEffect
    | ConsumptionStartedEffect
    | CarcassPointAwardedEffect
    | CarcassConsumedEffect
    | CentralRechargeStartedEffect
    | CentralReactivatedEffect
    | DamageDealtEffect
    | ConsumptionInterruptedEffect
    | TargetShovedEffect
    | RestStartedEffect
    | EnduranceRestoredEffect
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
    automatic_effects: tuple[Effect, ...] = ()
