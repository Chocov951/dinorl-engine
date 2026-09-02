"""Turn closure, actor order and round counter tests."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv


@pytest.mark.parametrize("first_actor", list(Actor))
def test_end_turn_alternates_in_fixed_order_and_counts_complete_rounds(
    first_actor: Actor,
) -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=10)
    state = env.reset(first_actor=first_actor)
    second_actor = Actor.B if first_actor is Actor.A else Actor.A

    state.raptor(first_actor).movement_points = 1
    env.step(Action.END_TURN)

    assert state.active_actor is second_actor
    assert state.turn == 1
    assert state.round == 1
    assert state.raptor(first_actor).movement_points == 0
    assert state.raptor(second_actor).movement_points == 3

    state.raptor(second_actor).movement_points = 2
    env.step(Action.END_TURN)

    assert state.active_actor is first_actor
    assert state.turn == 2
    assert state.round == 2
    assert state.raptor(second_actor).movement_points == 0
    assert state.raptor(first_actor).movement_points == 3


def test_end_turn_is_legal_with_or_without_remaining_movement_points() -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=10)
    state = env.reset(first_actor=Actor.A)

    assert env.legal_actions()[Action.END_TURN] is True
    state.raptor(Actor.A).movement_points = 0
    assert env.legal_actions()[Action.END_TURN] is True


def test_movement_can_be_followed_by_voluntary_end_turn() -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=10)
    state = env.reset(first_actor=Actor.A)

    env.step(Action.MOVE_EAST)
    env.step(Action.END_TURN)

    assert state.raptor(Actor.A).position == (8, 1)
    assert state.raptor(Actor.A).movement_points == 0
    assert state.active_actor is Actor.B
    assert state.turn == 1
    assert state.round == 1
