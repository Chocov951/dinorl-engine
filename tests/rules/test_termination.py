"""Immediate K.O. and carcass-score termination tests."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor, EndReason
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.errors import IllegalActionError


def _environment() -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=12)
    env.reset(first_actor=Actor.A)
    return env


def test_result_is_unavailable_before_terminal_state() -> None:
    env = _environment()

    with pytest.raises(RuntimeError, match="not terminal"):
        _ = env.result


def test_third_carcass_point_stops_remaining_turn_start_transitions() -> None:
    env = _environment()
    actor = env.state.raptor(Actor.A)
    actor.position = (3, 0)
    actor.carcass_score = 2
    env.step(Action.FEED)
    actor.endurance = 1
    actor.rest_pending = True
    env.state.central_carcass.status = "recharging"
    env.state.central_carcass.reactivate_on_turn = 2

    transition = env.step(Action.END_TURN)

    assert env.state.terminal is True
    assert env.state.winner is Actor.A
    assert env.state.end_reason is EndReason.CARCASS_SCORE
    assert actor.carcass_score == 3
    assert actor.endurance == 1
    assert actor.rest_pending is True
    assert actor.movement_points == 0
    assert actor.main_action_available is False
    assert env.state.central_carcass.status == "recharging"
    assert env.state.central_carcass.reactivate_on_turn == 2
    assert [effect.type for effect in transition.automatic_effects] == [
        "carcass_point_awarded",
        "carcass_consumed",
    ]
    assert not any(env.legal_actions())

    result = env.result
    assert result.winner is Actor.A
    assert result.reason is EndReason.CARCASS_SCORE
    assert result.score_a == 3
    assert result.score_b == 0
    assert result.hp_a == 6
    assert result.hp_b == 6
    assert result.rounds_completed == 1
    assert result.individual_turns == 2
    assert result.actions == 2


def test_ko_result_is_immediate_and_counts_current_turn() -> None:
    env = _environment()
    env.state.raptor(Actor.A).position = (1, 1)
    env.state.raptor(Actor.B).position = (1, 2)
    env.state.raptor(Actor.B).hp = 2

    transition = env.step(Action.BITE)

    assert transition.automatic_effects == ()
    result = env.result
    assert result.winner is Actor.A
    assert result.reason is EndReason.KO
    assert result.rounds_completed == 0
    assert result.individual_turns == 1
    assert result.actions == 1


def test_terminal_state_refuses_action_without_changing_result() -> None:
    env = _environment()
    env.state.raptor(Actor.A).position = (1, 3)
    env.state.raptor(Actor.B).position = (1, 4)
    env.state.raptor(Actor.B).hp = 1
    env.step(Action.SHOVE)
    before = env.result

    with pytest.raises(IllegalActionError):
        env.step(Action.END_TURN)

    assert env.result == before
