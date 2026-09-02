"""Consumable lateral carcass lifecycle tests."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv

type Position = tuple[int, int]


def _environment(actor_position: Position, opponent_position: Position = (0, 8)) -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=10)
    env.reset(first_actor=Actor.A)
    env.state.raptor(Actor.A).position = actor_position
    env.state.raptor(Actor.B).position = opponent_position
    return env


@pytest.mark.parametrize(
    ("position", "consumed_index", "available_index"),
    [((3, 0), 0, 1), ((5, 8), 1, 0)],
)
def test_validated_lateral_carcass_is_consumed_independently(
    position: Position,
    consumed_index: int,
    available_index: int,
) -> None:
    env = _environment(position)
    actor = env.state.raptor(Actor.A)
    env.step(Action.FEED)

    transition = env.step(Action.END_TURN)

    consumed = env.state.lateral_carcasses[consumed_index]
    other = env.state.lateral_carcasses[available_index]
    assert actor.carcass_score == 1
    assert consumed.status == "consumed"
    assert consumed.pending_actor is None
    assert other.status == "available"
    assert [effect.type for effect in transition.automatic_effects] == [
        "carcass_point_awarded",
        "carcass_consumed",
    ]
    assert env.legal_actions()[Action.FEED] is False


def test_interrupted_lateral_carcass_remains_available_for_retry() -> None:
    env = _environment((5, 8), (5, 7))
    actor = env.state.raptor(Actor.A)
    env.step(Action.FEED)
    env.step(Action.BITE)

    assert env.state.lateral_carcasses[1].status == "available"
    assert actor.carcass_score == 0
    assert actor.endurance == 3

    env.step(Action.END_TURN)
    assert env.state.active_actor is Actor.A
    assert env.legal_actions()[Action.FEED] is True
