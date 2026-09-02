"""Mutable internal state and immutable public snapshots."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from dinorl_engine.core.constants import Actor, EndReason
from dinorl_engine.core.errors import IllegalActionError
from dinorl_engine.core.maps import Position

__all__ = [
    "CarcassId",
    "CentralCarcassState",
    "GameState",
    "LateralCarcassState",
    "PublicSnapshot",
    "RaptorState",
    "snapshot_public",
]

type CarcassId = Literal["carcass_left_a", "carcass_center", "carcass_left_b"]
type LateralCarcassStatus = Literal["available", "pending", "consumed"]
type CentralCarcassStatus = Literal["active", "pending", "recharging"]
type Winner = Actor | Literal["draw"] | None
type PublicSnapshot = Mapping[str, object]


@dataclass(slots=True)
class RaptorState:
    """The mutable state owned by one actor."""

    position: Position
    hp: int = 6
    endurance: int = 5
    movement_points: int = 0
    carcass_score: int = 0
    main_action_available: bool = True
    voluntary_move_done: bool = False
    rest_pending: bool = False
    consumption_pending: CarcassId | None = None


@dataclass(slots=True)
class LateralCarcassState:
    """Runtime status of one consumable lateral carcass."""

    status: LateralCarcassStatus = "available"
    pending_actor: Actor | None = None


@dataclass(slots=True)
class CentralCarcassState:
    """Runtime status and recharge deadline of the central carcass."""

    status: CentralCarcassStatus = "active"
    pending_actor: Actor | None = None
    reactivate_on_turn: int | None = None


@dataclass(slots=True)
class GameState:
    """Complete mutable state of one deterministic game."""

    engine_version: str
    rules_version: str
    action_version: str
    observation_version: str
    replay_version: str
    map_id: str
    seed: int
    first_actor: Actor
    active_actor: Actor
    raptors: tuple[RaptorState, RaptorState]
    lateral_carcasses: tuple[LateralCarcassState, LateralCarcassState]
    central_carcass: CentralCarcassState
    turn: int = 0
    round: int = 1
    terminal: bool = False
    winner: Winner = None
    end_reason: EndReason | None = None

    def raptor(self, actor: Actor) -> RaptorState:
        """Return an actor's state using the stable actor integer as index."""

        return self.raptors[actor]

    def claim_main_action(self) -> None:
        """Atomically reserve the active actor's sole main action for this turn."""

        actor = self.raptor(self.active_actor)
        if not actor.main_action_available:
            raise IllegalActionError("Main action already used during this turn")
        actor.main_action_available = False


def _raptor_snapshot(raptor: RaptorState) -> PublicSnapshot:
    return MappingProxyType(
        {
            "position": raptor.position,
            "hp": raptor.hp,
            "endurance": raptor.endurance,
            "movement_points": raptor.movement_points,
            "carcass_score": raptor.carcass_score,
            "main_action_available": raptor.main_action_available,
            "voluntary_move_done": raptor.voluntary_move_done,
            "rest_pending": raptor.rest_pending,
            "consumption_pending": raptor.consumption_pending,
        }
    )


def _lateral_carcass_snapshot(carcass: LateralCarcassState) -> PublicSnapshot:
    return MappingProxyType(
        {
            "status": carcass.status,
            "pending_actor": carcass.pending_actor.name
            if carcass.pending_actor is not None
            else None,
        }
    )


def _central_carcass_snapshot(carcass: CentralCarcassState) -> PublicSnapshot:
    return MappingProxyType(
        {
            "status": carcass.status,
            "pending_actor": carcass.pending_actor.name
            if carcass.pending_actor is not None
            else None,
            "reactivate_on_turn": carcass.reactivate_on_turn,
        }
    )


def snapshot_public(state: GameState) -> PublicSnapshot:
    """Build a deeply immutable, schema-shaped view of an internal state."""

    raptors = MappingProxyType(
        {
            "A": _raptor_snapshot(state.raptor(Actor.A)),
            "B": _raptor_snapshot(state.raptor(Actor.B)),
        }
    )
    carcasses = MappingProxyType(
        {
            "carcass_left_a": _lateral_carcass_snapshot(state.lateral_carcasses[0]),
            "carcass_center": _central_carcass_snapshot(state.central_carcass),
            "carcass_left_b": _lateral_carcass_snapshot(state.lateral_carcasses[1]),
        }
    )
    winner = state.winner.name if isinstance(state.winner, Actor) else state.winner
    return MappingProxyType(
        {
            "active_actor": state.active_actor.name,
            "turn": state.turn,
            "round": state.round,
            "terminal": state.terminal,
            "winner": winner,
            "end_reason": state.end_reason.name.lower() if state.end_reason is not None else None,
            "raptors": raptors,
            "carcasses": carcasses,
        }
    )
