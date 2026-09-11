"""Observation-v1 encoder using only public engine state and static map data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import NDArray

from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.maps import Position, load_map
from dinorl_engine.core.state import PublicSnapshot
from dinorl_engine.rl.env.canonical import (
    GRID_COLUMNS,
    GRID_ROWS,
    engine_to_canonical_position,
)

__all__ = [
    "FEATURE_COUNT",
    "GRID_CHANNELS",
    "GRID_COLUMNS",
    "GRID_ROWS",
    "Observation",
    "build_observation",
]

GRID_CHANNELS = 8
FEATURE_COUNT = 15

type Observation = dict[str, NDArray[np.float32]]

_ARENA = load_map(MAP_ID)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"public state field {field_name} must be an object")
    return cast(Mapping[str, object], value)


def _integer(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"public state field {field_name} must be an integer")
    return value


def _position(value: object, field_name: str) -> Position:
    if (
        not isinstance(value, tuple | list)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
    ):
        raise ValueError(f"public state field {field_name} must be a coordinate")
    return value[0], value[1]


def _raptor(snapshot: PublicSnapshot, actor: Actor) -> Mapping[str, object]:
    raptors = _mapping(snapshot.get("raptors"), "raptors")
    return _mapping(raptors.get(actor.name), f"raptors.{actor.name}")


def _carcass(snapshot: PublicSnapshot, carcass_id: str) -> Mapping[str, object]:
    carcasses = _mapping(snapshot.get("carcasses"), "carcasses")
    return _mapping(carcasses.get(carcass_id), f"carcasses.{carcass_id}")


def _set_position(
    grid: NDArray[np.float32], channel: int, position: Position, learner_actor: Actor
) -> None:
    row, column = engine_to_canonical_position(position, learner_actor)
    grid[channel, row, column] = 1.0


def _build_grid(snapshot: PublicSnapshot, learner_actor: Actor) -> NDArray[np.float32]:
    grid = np.zeros((GRID_CHANNELS, GRID_ROWS, GRID_COLUMNS), dtype=np.float32)
    for position in _ARENA.walls:
        _set_position(grid, 0, position, learner_actor)
    for position in _ARENA.mud:
        _set_position(grid, 1, position, learner_actor)

    lateral_carcasses = tuple(
        carcass for carcass in _ARENA.carcasses if carcass.carcass_id != "carcass_center"
    )
    for carcass in lateral_carcasses:
        _set_position(grid, 2, carcass.position, learner_actor)
        state = _carcass(snapshot, carcass.carcass_id)
        if state.get("status") == "available":
            _set_position(grid, 3, carcass.position, learner_actor)

    central = next(
        carcass for carcass in _ARENA.carcasses if carcass.carcass_id == "carcass_center"
    )
    _set_position(grid, 4, central.position, learner_actor)
    if _carcass(snapshot, "carcass_center").get("status") == "active":
        _set_position(grid, 5, central.position, learner_actor)

    opponent_actor = Actor.B if learner_actor is Actor.A else Actor.A
    _set_position(
        grid,
        6,
        _position(_raptor(snapshot, learner_actor).get("position"), "self.position"),
        learner_actor,
    )
    _set_position(
        grid,
        7,
        _position(_raptor(snapshot, opponent_actor).get("position"), "opponent.position"),
        learner_actor,
    )
    return grid


def _bool_feature(value: object) -> float:
    if not isinstance(value, bool):
        raise ValueError("public state boolean feature must be a bool")
    return float(value)


def _bounded_fraction(value: int, denominator: int, field_name: str) -> float:
    if not 0 <= value <= denominator:
        raise ValueError(f"public state field {field_name} is outside its V1 bounds")
    return value / denominator


def _build_features(
    snapshot: PublicSnapshot, learner_actor: Actor, first_actor: Actor
) -> NDArray[np.float32]:
    opponent_actor = Actor.B if learner_actor is Actor.A else Actor.A
    self_raptor = _raptor(snapshot, learner_actor)
    opponent = _raptor(snapshot, opponent_actor)
    central = _carcass(snapshot, "carcass_center")
    central_status = central.get("status")
    if central_status not in {"active", "pending", "recharging"}:
        raise ValueError("public state has an invalid central carcass status")
    turn = _integer(snapshot.get("turn"), "turn")
    reactivate_on_turn = central.get("reactivate_on_turn")
    if central_status == "recharging":
        reactivate_turn = _integer(
            reactivate_on_turn, "carcasses.carcass_center.reactivate_on_turn"
        )
        central_delay = max(0, min(2, reactivate_turn - turn)) / 2
    else:
        central_delay = 0.0

    features = np.asarray(
        (
            _bounded_fraction(_integer(self_raptor.get("hp"), "self.hp"), 6, "self.hp"),
            _bounded_fraction(
                _integer(self_raptor.get("endurance"), "self.endurance"), 5, "self.endurance"
            ),
            _bounded_fraction(
                _integer(self_raptor.get("movement_points"), "self.movement_points"),
                3,
                "self.movement_points",
            ),
            _bounded_fraction(
                _integer(self_raptor.get("carcass_score"), "self.carcass_score"),
                3,
                "self.carcass_score",
            ),
            _bool_feature(self_raptor.get("main_action_available")),
            _bool_feature(self_raptor.get("voluntary_move_done")),
            _bounded_fraction(_integer(opponent.get("hp"), "opponent.hp"), 6, "opponent.hp"),
            _bounded_fraction(
                _integer(opponent.get("endurance"), "opponent.endurance"), 5, "opponent.endurance"
            ),
            _bounded_fraction(
                _integer(opponent.get("carcass_score"), "opponent.carcass_score"),
                3,
                "opponent.carcass_score",
            ),
            float(opponent.get("consumption_pending") is not None),
            _bool_feature(opponent.get("rest_pending")),
            float(central_status != "active"),
            central_delay,
            _bounded_fraction(_integer(snapshot.get("round"), "round"), 30, "round"),
            float(learner_actor is first_actor),
        ),
        dtype=np.float32,
    )
    if features.shape != (FEATURE_COUNT,) or not np.all((features >= 0.0) & (features <= 1.0)):
        raise AssertionError("observation features must have the V1 shape and bounds")
    return features


def build_observation(
    snapshot: PublicSnapshot, *, learner_actor: Actor, first_actor: Actor
) -> Observation:
    """Encode a public state into the fixed observation-v1 Gymnasium dictionary."""

    return {
        "grid": _build_grid(snapshot, learner_actor),
        "features": _build_features(snapshot, learner_actor, first_actor),
    }
