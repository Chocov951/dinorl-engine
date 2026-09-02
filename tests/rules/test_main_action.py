"""Main-action allowance and turn reset tests."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.errors import IllegalActionError


def _environment() -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=2)
    env.reset(first_actor=Actor.A)
    return env


def test_only_one_main_action_can_be_claimed_during_a_turn() -> None:
    env = _environment()

    env.state.claim_main_action()
    after_first_action = env.snapshot_public()

    assert env.state.raptor(Actor.A).main_action_available is False
    with pytest.raises(IllegalActionError, match="Main action already used"):
        env.state.claim_main_action()
    assert env.snapshot_public() == after_first_action


def test_voluntary_movement_remains_available_before_and_after_main_action() -> None:
    env = _environment()

    env.step(Action.MOVE_EAST)
    env.state.claim_main_action()

    assert env.legal_actions()[Action.MOVE_EAST] is True
    env.step(Action.MOVE_EAST)

    actor = env.state.raptor(Actor.A)
    assert actor.position == (8, 2)
    assert actor.movement_points == 1
    assert actor.main_action_available is False


def test_turn_start_resets_main_action_and_voluntary_movement_flags() -> None:
    env = _environment()
    actor_a = env.state.raptor(Actor.A)
    env.step(Action.MOVE_EAST)
    env.state.claim_main_action()

    env.step(Action.END_TURN)
    env.step(Action.END_TURN)

    assert env.state.active_actor is Actor.A
    assert actor_a.main_action_available is True
    assert actor_a.voluntary_move_done is False
