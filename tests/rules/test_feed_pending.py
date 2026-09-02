"""Pending carcass consumption and delayed point tests."""

from collections.abc import Callable

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.errors import IllegalActionError
from dinorl_engine.core.events import ActionCost

type StateMutation = Callable[[DinoRLEnv], None]


def _environment() -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=8)
    env.reset(first_actor=Actor.A)
    env.state.raptor(Actor.A).position = (4, 4)
    return env


def test_feed_pays_endurance_starts_pending_consumption_and_ends_turn() -> None:
    env = _environment()
    actor = env.state.raptor(Actor.A)

    assert env.legal_actions()[Action.FEED] is True
    transition = env.step(Action.FEED)

    assert transition.action is Action.FEED
    assert transition.actor is Actor.A
    assert transition.cost == ActionCost(movement=0, endurance=2)
    assert [effect.type for effect in transition.effects] == [
        "consumption_started",
        "turn_ended",
    ]
    assert transition.effects[0].carcass_id == "carcass_center"
    assert transition.turn_ended is True
    assert actor.endurance == 3
    assert actor.movement_points == 0
    assert actor.main_action_available is False
    assert actor.carcass_score == 0
    assert actor.consumption_pending == "carcass_center"
    assert env.state.central_carcass.status == "pending"
    assert env.state.central_carcass.pending_actor is Actor.A
    assert env.state.active_actor is Actor.B


def test_feed_point_is_awarded_only_after_full_opponent_turn() -> None:
    env = _environment()
    actor = env.state.raptor(Actor.A)
    env.step(Action.FEED)

    env.step(Action.MOVE_WEST)
    assert actor.carcass_score == 0
    assert actor.consumption_pending == "carcass_center"

    transition = env.step(Action.END_TURN)

    assert env.state.active_actor is Actor.A
    assert actor.carcass_score == 1
    assert actor.consumption_pending is None
    assert [effect.type for effect in transition.automatic_effects] == [
        "carcass_point_awarded",
        "central_recharge_started",
    ]
    assert transition.automatic_effects[0].actor is Actor.A
    assert transition.automatic_effects[0].carcass_id == "carcass_center"
    assert transition.automatic_effects[0].amount == 1


def _move_off_carcass(env: DinoRLEnv) -> None:
    env.state.raptor(Actor.A).position = (4, 3)


def _remove_endurance(env: DinoRLEnv) -> None:
    env.state.raptor(Actor.A).endurance = 1


def _consume_main_action(env: DinoRLEnv) -> None:
    env.state.claim_main_action()


def _disable_central_carcass(env: DinoRLEnv) -> None:
    env.state.central_carcass.status = "recharging"


@pytest.mark.parametrize(
    "mutation",
    [_move_off_carcass, _remove_endurance, _consume_main_action, _disable_central_carcass],
)
def test_illegal_feed_does_not_mutate_state(mutation: StateMutation) -> None:
    env = _environment()
    mutation(env)
    before = env.snapshot_public()

    assert env.legal_actions()[Action.FEED] is False
    with pytest.raises(IllegalActionError):
        env.step(Action.FEED)

    assert env.snapshot_public() == before
