"""Canonical 180-degree transforms for the learner's point of view."""

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.maps import Position

__all__ = [
    "GRID_COLUMNS",
    "GRID_ROWS",
    "canonical_to_engine_action",
    "canonical_to_engine_position",
    "engine_to_canonical_action",
    "engine_to_canonical_position",
    "rotate_action",
    "rotate_position",
]

GRID_ROWS = 9
GRID_COLUMNS = 9

_ROTATED_ACTIONS = {
    Action.MOVE_NORTH: Action.MOVE_SOUTH,
    Action.MOVE_EAST: Action.MOVE_WEST,
    Action.MOVE_SOUTH: Action.MOVE_NORTH,
    Action.MOVE_WEST: Action.MOVE_EAST,
}


def rotate_position(position: Position) -> Position:
    """Rotate a V1 arena coordinate by 180 degrees."""

    row, column = position
    if not 0 <= row < GRID_ROWS or not 0 <= column < GRID_COLUMNS:
        raise ValueError(f"position outside the {GRID_ROWS}x{GRID_COLUMNS} arena: {position!r}")
    return GRID_ROWS - 1 - row, GRID_COLUMNS - 1 - column


def rotate_action(action: Action) -> Action:
    """Rotate a directional action, preserving non-directional actions."""

    return _ROTATED_ACTIONS.get(action, action)


def engine_to_canonical_position(position: Position, learner_actor: Actor) -> Position:
    """Express an engine coordinate from the learner's canonical perspective."""

    return position if learner_actor is Actor.A else rotate_position(position)


def canonical_to_engine_position(position: Position, learner_actor: Actor) -> Position:
    """Convert a canonical coordinate back to the engine perspective."""

    return engine_to_canonical_position(position, learner_actor)


def engine_to_canonical_action(action: Action, learner_actor: Actor) -> Action:
    """Express an engine action in the learner's canonical orientation."""

    return action if learner_actor is Actor.A else rotate_action(action)


def canonical_to_engine_action(action: Action, learner_actor: Actor) -> Action:
    """Convert a learner action to the engine orientation."""

    return engine_to_canonical_action(action, learner_actor)
