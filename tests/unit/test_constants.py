"""Version and core enum contract tests."""

from enum import IntEnum

from dinorl_engine.core.constants import (
    ACTION_VERSION,
    ENGINE_VERSION,
    MAP_ID,
    OBSERVATION_VERSION,
    PROTOCOL_VERSION,
    REPLAY_VERSION,
    RULES_VERSION,
    Actor,
    EndReason,
    Tile,
)


def test_initial_versions_and_map_id_are_explicit_strings() -> None:
    assert ENGINE_VERSION == "0.1.0"
    assert PROTOCOL_VERSION == "1.0.0"
    assert RULES_VERSION == "1.0.0"
    assert ACTION_VERSION == "1.0.0"
    assert OBSERVATION_VERSION == "1.0.0"
    assert REPLAY_VERSION == "1.0.0"
    assert MAP_ID == "arena_mvp_v1"


def test_core_enums_have_stable_values_and_order() -> None:
    assert issubclass(Tile, IntEnum)
    assert [(member.name, member.value) for member in Tile] == [
        ("EMPTY", 0),
        ("WALL", 1),
        ("MUD", 2),
        ("CARCASS", 3),
    ]

    assert issubclass(Actor, IntEnum)
    assert [(member.name, member.value) for member in Actor] == [("A", 0), ("B", 1)]

    assert issubclass(EndReason, IntEnum)
    assert [(member.name, member.value) for member in EndReason] == [
        ("KO", 0),
        ("CARCASS_SCORE", 1),
        ("ROUND_LIMIT", 2),
    ]


def test_core_enums_have_no_aliases() -> None:
    for enum_type in (Tile, Actor, EndReason):
        assert len(enum_type.__members__) == len(enum_type)
