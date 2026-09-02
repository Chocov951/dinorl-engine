"""Versioned actions understood by the DinoRL engine."""

from enum import IntEnum, unique

__all__ = ["Action"]


@unique
class Action(IntEnum):
    """An atomic action, ordered for use in fixed-size legal-action masks."""

    MOVE_NORTH = 0
    MOVE_EAST = 1
    MOVE_SOUTH = 2
    MOVE_WEST = 3
    BITE = 4
    SHOVE = 5
    FEED = 6
    REST = 7
    END_TURN = 8
