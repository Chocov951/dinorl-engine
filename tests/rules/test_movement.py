"""Voluntary movement rule tests."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.errors import IllegalActionError


def _environment() -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=1)
    env.reset(first_actor=Actor.A)
    return env


@pytest.mark.parametrize(
    ("action", "expected_position"),
    [
        (Action.MOVE_NORTH, (0, 1)),
        (Action.MOVE_EAST, (1, 2)),
        (Action.MOVE_SOUTH, (2, 1)),
        (Action.MOVE_WEST, (1, 0)),
    ],
)
def test_actor_can_move_in_each_orthogonal_direction(
    action: Action, expected_position: tuple[int, int]
) -> None:
    env = _environment()
    actor = env.state.raptor(Actor.A)
    actor.position = (1, 1)
    initial_endurance = actor.endurance

    assert env.legal_actions()[action] is True
    env.step(action)

    assert actor.position == expected_position
    assert actor.movement_points == 2
    assert actor.endurance == initial_endurance
    assert actor.voluntary_move_done is True


@pytest.mark.parametrize("action", [Action.MOVE_SOUTH, Action.MOVE_WEST])
def test_map_boundaries_block_movement_without_mutation(action: Action) -> None:
    env = _environment()
    before = env.snapshot_public()

    assert env.legal_actions()[action] is False
    with pytest.raises(IllegalActionError):
        env.step(action)

    assert env.snapshot_public() == before


def test_wall_blocks_movement_without_mutation() -> None:
    env = _environment()
    env.state.raptor(Actor.A).position = (1, 4)
    before = env.snapshot_public()

    assert env.legal_actions()[Action.MOVE_EAST] is False
    with pytest.raises(IllegalActionError):
        env.step(Action.MOVE_EAST)

    assert env.snapshot_public() == before


def test_opponent_blocks_movement_without_mutation() -> None:
    env = _environment()
    env.state.raptor(Actor.A).position = (1, 1)
    env.state.raptor(Actor.B).position = (1, 2)
    before = env.snapshot_public()

    assert env.legal_actions()[Action.MOVE_EAST] is False
    with pytest.raises(IllegalActionError):
        env.step(Action.MOVE_EAST)

    assert env.snapshot_public() == before


def test_entering_mud_costs_one_movement_point() -> None:
    env = _environment()
    actor = env.state.raptor(Actor.A)
    actor.position = (5, 0)

    env.step(Action.MOVE_SOUTH)

    assert actor.position == (6, 0)
    assert actor.movement_points == 2


def test_leaving_mud_costs_two_movement_points() -> None:
    env = _environment()
    actor = env.state.raptor(Actor.A)
    actor.position = (6, 0)
    actor.movement_points = 2

    env.step(Action.MOVE_NORTH)

    assert actor.position == (5, 0)
    assert actor.movement_points == 0
    assert actor.endurance == 5


def test_leaving_mud_with_one_movement_point_is_illegal_and_atomic() -> None:
    env = _environment()
    actor = env.state.raptor(Actor.A)
    actor.position = (6, 0)
    actor.movement_points = 1
    before = env.snapshot_public()

    assert env.legal_actions()[Action.MOVE_NORTH] is False
    with pytest.raises(IllegalActionError):
        env.step(Action.MOVE_NORTH)

    assert env.snapshot_public() == before
