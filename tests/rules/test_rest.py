"""Delayed rest and non-interruption rule tests."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.errors import IllegalActionError
from dinorl_engine.core.events import ActionCost


def _environment() -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=7)
    env.reset(first_actor=Actor.A)
    return env


def test_rest_starts_pending_recovery_and_ends_turn_without_immediate_restore() -> None:
    env = _environment()
    actor = env.state.raptor(Actor.A)
    actor.endurance = 2

    assert env.legal_actions()[Action.REST] is True
    transition = env.step(Action.REST)

    assert transition.action is Action.REST
    assert transition.actor is Actor.A
    assert transition.cost == ActionCost(movement=0, endurance=0)
    assert [effect.type for effect in transition.effects] == ["rest_started", "turn_ended"]
    assert transition.automatic_effects == ()
    assert transition.turn_ended is True
    assert actor.endurance == 2
    assert actor.rest_pending is True
    assert actor.main_action_available is False
    assert actor.movement_points == 0
    assert env.state.active_actor is Actor.B
    assert env.state.turn == 1


def test_rest_is_illegal_after_voluntary_movement_and_does_not_mutate_state() -> None:
    env = _environment()
    env.step(Action.MOVE_EAST)
    before = env.snapshot_public()

    assert env.legal_actions()[Action.REST] is False
    with pytest.raises(IllegalActionError):
        env.step(Action.REST)

    assert env.snapshot_public() == before


def test_rest_restores_endurance_only_at_start_of_actors_next_turn() -> None:
    env = _environment()
    actor = env.state.raptor(Actor.A)
    actor.endurance = 1
    env.step(Action.REST)

    transition = env.step(Action.END_TURN)

    assert env.state.active_actor is Actor.A
    assert actor.endurance == 5
    assert actor.rest_pending is False
    assert [effect.type for effect in transition.automatic_effects] == ["endurance_restored"]
    assert transition.automatic_effects[0].actor is Actor.A
    assert transition.automatic_effects[0].amount == 4


@pytest.mark.parametrize("attack", [Action.BITE, Action.SHOVE])
def test_bite_and_shove_do_not_interrupt_pending_rest(attack: Action) -> None:
    env = _environment()
    actor = env.state.raptor(Actor.A)
    actor.position = (1, 2)
    actor.endurance = 1
    env.state.raptor(Actor.B).position = (1, 1)
    env.step(Action.REST)

    env.step(attack)

    assert actor.rest_pending is True
    assert actor.endurance == 1
    env.step(Action.END_TURN)
    assert actor.rest_pending is False
    assert actor.endurance == 5
