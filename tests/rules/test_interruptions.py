"""End-to-end pending-consumption interruption tests."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv

type Position = tuple[int, int]


def _environment(consumer_position: Position, opponent_position: Position) -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=9)
    env.reset(first_actor=Actor.A)
    env.state.raptor(Actor.A).position = consumer_position
    env.state.raptor(Actor.B).position = opponent_position
    return env


@pytest.mark.parametrize("attack", [Action.BITE, Action.SHOVE])
@pytest.mark.parametrize(
    ("carcass_id", "consumer_position", "opponent_position"),
    [
        ("carcass_left_a", (3, 0), (3, 1)),
        ("carcass_center", (4, 4), (4, 3)),
    ],
)
def test_bite_and_shove_cancel_point_without_refunding_feed(
    attack: Action,
    carcass_id: str,
    consumer_position: Position,
    opponent_position: Position,
) -> None:
    env = _environment(consumer_position, opponent_position)
    consumer = env.state.raptor(Actor.A)
    env.step(Action.FEED)

    transition = env.step(attack)

    assert consumer.consumption_pending is None
    assert consumer.carcass_score == 0
    assert consumer.endurance == 3
    assert "consumption_interrupted" in [effect.type for effect in transition.effects]
    if carcass_id == "carcass_center":
        assert env.state.central_carcass.status == "active"
        assert env.state.central_carcass.pending_actor is None
    else:
        assert env.state.lateral_carcasses[0].status == "available"
        assert env.state.lateral_carcasses[0].pending_actor is None

    env.step(Action.END_TURN)
    assert env.state.active_actor is Actor.A
    assert consumer.carcass_score == 0
    assert consumer.endurance == 3


@pytest.mark.parametrize("opponent_action", [Action.END_TURN, Action.REST])
def test_non_attack_action_does_not_interrupt_consumption(opponent_action: Action) -> None:
    env = _environment((4, 4), (0, 8))
    consumer = env.state.raptor(Actor.A)
    env.step(Action.FEED)

    transition = env.step(opponent_action)

    assert "consumption_interrupted" not in [effect.type for effect in transition.effects]
    assert env.state.active_actor is Actor.A
    assert consumer.carcass_score == 1
    assert consumer.endurance == 3


def test_opponent_movement_does_not_interrupt_consumption() -> None:
    env = _environment((4, 4), (0, 8))
    consumer = env.state.raptor(Actor.A)
    env.step(Action.FEED)

    env.step(Action.MOVE_WEST)

    assert consumer.consumption_pending == "carcass_center"
    assert consumer.carcass_score == 0
