"""Central carcass recharge chronology tests."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv


def _environment(opponent_position: tuple[int, int] = (0, 8)) -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=11)
    env.reset(first_actor=Actor.A)
    env.state.raptor(Actor.A).position = (4, 4)
    env.state.raptor(Actor.B).position = opponent_position
    return env


def test_central_reactivates_after_validation_turn_and_one_opponent_turn() -> None:
    env = _environment()
    consumer = env.state.raptor(Actor.A)
    env.step(Action.FEED)

    validation = env.step(Action.END_TURN)

    central = env.state.central_carcass
    assert env.state.active_actor is Actor.A
    assert env.state.turn == 2
    assert consumer.carcass_score == 1
    assert central.status == "recharging"
    assert central.pending_actor is None
    assert central.reactivate_on_turn == 4
    assert [effect.type for effect in validation.automatic_effects] == [
        "carcass_point_awarded",
        "central_recharge_started",
    ]
    assert env.legal_actions()[Action.FEED] is False

    env.step(Action.END_TURN)
    assert env.state.active_actor is Actor.B
    assert env.state.turn == 3
    assert central.status == "recharging"
    assert central.reactivate_on_turn == 4

    reactivation = env.step(Action.END_TURN)
    assert env.state.active_actor is Actor.A
    assert env.state.turn == 4
    assert central.status == "active"
    assert central.reactivate_on_turn is None
    assert [effect.type for effect in reactivation.automatic_effects] == ["central_reactivated"]
    assert env.legal_actions()[Action.FEED] is True


@pytest.mark.parametrize("attack", [Action.BITE, Action.SHOVE])
def test_interrupted_central_consumption_never_starts_recharge(attack: Action) -> None:
    env = _environment(opponent_position=(4, 3))
    env.step(Action.FEED)

    env.step(attack)

    central = env.state.central_carcass
    assert central.status == "active"
    assert central.pending_actor is None
    assert central.reactivate_on_turn is None
    transition = env.step(Action.END_TURN)
    assert env.state.raptor(Actor.A).carcass_score == 0
    assert central.status == "active"
    assert not {
        "central_recharge_started",
        "central_reactivated",
    } & {effect.type for effect in transition.automatic_effects}


def test_reactivated_central_carcass_can_score_again() -> None:
    env = _environment()
    consumer = env.state.raptor(Actor.A)
    env.step(Action.FEED)
    env.step(Action.END_TURN)
    env.step(Action.END_TURN)
    env.step(Action.END_TURN)

    env.step(Action.FEED)
    env.step(Action.END_TURN)

    assert consumer.carcass_score == 2
    assert env.state.central_carcass.status == "recharging"
    assert env.state.central_carcass.reactivate_on_turn == 8
