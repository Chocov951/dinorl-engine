"""Versioned action enum tests."""

from enum import IntEnum

from dinorl_engine.core.actions import Action


def test_actions_have_stable_values_and_order() -> None:
    assert issubclass(Action, IntEnum)
    assert [(action.name, action.value) for action in Action] == [
        ("MOVE_NORTH", 0),
        ("MOVE_EAST", 1),
        ("MOVE_SOUTH", 2),
        ("MOVE_WEST", 3),
        ("BITE", 4),
        ("SHOVE", 5),
        ("FEED", 6),
        ("REST", 7),
        ("END_TURN", 8),
    ]


def test_action_enum_has_no_alias() -> None:
    assert len(Action.__members__) == len(Action)
