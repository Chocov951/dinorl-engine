"""Version identifiers and stable enums shared by the pure engine core."""

from enum import IntEnum, unique
from typing import Final

__all__ = [
    "ACTION_VERSION",
    "ENGINE_VERSION",
    "MAP_ID",
    "OBSERVATION_VERSION",
    "PROTOCOL_VERSION",
    "REPLAY_VERSION",
    "RULES_VERSION",
    "Actor",
    "EndReason",
    "Tile",
]

ENGINE_VERSION: Final = "0.1.0"
PROTOCOL_VERSION: Final = "1.0.0"
RULES_VERSION: Final = "1.0.0"
ACTION_VERSION: Final = "1.0.0"
OBSERVATION_VERSION: Final = "1.0.0"
REPLAY_VERSION: Final = "1.0.0"
MAP_ID: Final = "arena_mvp_v1"


@unique
class Tile(IntEnum):
    """A tile type stored in the flattened arena grid."""

    EMPTY = 0
    WALL = 1
    MUD = 2
    CARCASS = 3


@unique
class Actor(IntEnum):
    """One of the two actors in a match."""

    A = 0
    B = 1


@unique
class EndReason(IntEnum):
    """The reason why a match reached a terminal state."""

    KO = 0
    CARCASS_SCORE = 1
    ROUND_LIMIT = 2
