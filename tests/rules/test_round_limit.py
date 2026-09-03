"""Thirty-round limit and deferred-transition cutoff tests."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor, EndReason
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.errors import IllegalActionError


@pytest.mark.parametrize("first_actor", list(Actor))
def test_game_draws_after_exactly_thirty_complete_rounds(first_actor: Actor) -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=13)
    state = env.reset(first_actor=first_actor)
    second_actor = Actor.B if first_actor is Actor.A else Actor.A

    for _ in range(59):
        env.step(Action.END_TURN)

    assert state.terminal is False
    assert state.active_actor is second_actor
    assert state.turn == 59
    assert state.round == 30

    transition = env.step(Action.END_TURN)

    assert state.terminal is True
    assert state.winner == "draw"
    assert state.end_reason is EndReason.ROUND_LIMIT
    assert state.active_actor is second_actor
    assert state.turn == 59
    assert state.round == 30
    assert state.raptor(second_actor).movement_points == 0
    assert transition.automatic_effects == ()
    assert not any(env.legal_actions())
    result = env.result
    assert result.winner == "draw"
    assert result.reason is EndReason.ROUND_LIMIT
    assert result.rounds_completed == 30
    assert result.individual_turns == 60
    assert result.actions == 60

    with pytest.raises(IllegalActionError):
        env.step(Action.END_TURN)


def test_round_limit_prevents_pending_score_and_rest_resolution() -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=13)
    state = env.reset(first_actor=Actor.A)
    state.turn = 58
    state.round = 30
    consumer = state.raptor(Actor.A)
    consumer.position = (3, 0)
    consumer.carcass_score = 2
    env.step(Action.FEED)
    resting_actor = state.raptor(Actor.B)
    resting_actor.endurance = 1

    transition = env.step(Action.REST)

    assert state.terminal is True
    assert state.winner == "draw"
    assert state.end_reason is EndReason.ROUND_LIMIT
    assert state.active_actor is Actor.B
    assert state.turn == 59
    assert state.round == 30
    assert consumer.carcass_score == 2
    assert consumer.consumption_pending == "carcass_left_a"
    assert state.lateral_carcasses[0].status == "pending"
    assert resting_actor.endurance == 1
    assert resting_actor.rest_pending is True
    assert transition.automatic_effects == ()


def test_ko_during_first_turn_of_round_thirty_precedes_round_limit() -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=13)
    state = env.reset(first_actor=Actor.A)
    state.turn = 58
    state.round = 30
    state.raptor(Actor.A).position = (1, 1)
    state.raptor(Actor.B).position = (1, 2)
    state.raptor(Actor.B).hp = 2

    env.step(Action.BITE)

    assert state.terminal is True
    assert state.winner is Actor.A
    assert state.end_reason is EndReason.KO
    assert env.result.individual_turns == 59
    assert env.result.rounds_completed == 29
