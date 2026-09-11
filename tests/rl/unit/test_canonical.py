"""Canonical A/B coordinate and directional transforms."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.env.canonical import (
    GRID_COLUMNS,
    GRID_ROWS,
    canonical_to_engine_action,
    canonical_to_engine_position,
    engine_to_canonical_action,
    engine_to_canonical_position,
    rotate_action,
    rotate_position,
)


@pytest.mark.parametrize("row", range(GRID_ROWS))
@pytest.mark.parametrize("column", range(GRID_COLUMNS))
def test_rotation_is_an_involution_for_all_board_cells(row: int, column: int) -> None:
    position = row, column

    assert rotate_position(rotate_position(position)) == position
    assert rotate_position(position) == (GRID_ROWS - 1 - row, GRID_COLUMNS - 1 - column)
    assert (
        engine_to_canonical_position(canonical_to_engine_position(position, Actor.B), Actor.B)
        == position
    )


@pytest.mark.parametrize("action", tuple(Action))
def test_directional_action_mapping_is_exhaustive_and_bijective(action: Action) -> None:
    assert rotate_action(rotate_action(action)) is action
    assert (
        engine_to_canonical_action(canonical_to_engine_action(action, Actor.B), Actor.B) is action
    )
    assert (
        canonical_to_engine_action(engine_to_canonical_action(action, Actor.A), Actor.A) is action
    )


def test_rotation_reverses_each_direction_without_changing_non_directional_actions() -> None:
    assert rotate_action(Action.MOVE_NORTH) is Action.MOVE_SOUTH
    assert rotate_action(Action.MOVE_EAST) is Action.MOVE_WEST
    assert rotate_action(Action.MOVE_SOUTH) is Action.MOVE_NORTH
    assert rotate_action(Action.MOVE_WEST) is Action.MOVE_EAST
    for action in (Action.BITE, Action.SHOVE, Action.FEED, Action.REST, Action.END_TURN):
        assert rotate_action(action) is action


@pytest.mark.parametrize("position", [(-1, 0), (0, -1), (GRID_ROWS, 0), (0, GRID_COLUMNS)])
def test_rotation_rejects_positions_outside_the_versioned_arena(position: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        rotate_position(position)
