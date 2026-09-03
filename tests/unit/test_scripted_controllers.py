"""Characteristic decisions of the three deterministic scripted bots."""

import pytest

from dinorl_engine.controllers.protocol import Controller
from dinorl_engine.controllers.scripted import (
    SCRIPTED_CONTROLLER_IDS,
    AggressiveController,
    OpportunistController,
    PrudentController,
    can_interrupt_next_turn,
    create_scripted_controller,
)
from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import DinoRLEnv


def _env() -> DinoRLEnv:
    env = DinoRLEnv("arena_mvp_v1", seed=4)
    env.reset(first_actor=Actor.A)
    return env


def _decision(env: DinoRLEnv, controller: Controller) -> Action:
    return controller.choose_action(env.snapshot_public(), env.legal_actions())


def test_only_the_three_versioned_scripted_controllers_are_available() -> None:
    assert SCRIPTED_CONTROLLER_IDS == (
        "aggressive-v1",
        "prudent-v1",
        "opportunist-v1",
    )
    assert isinstance(create_scripted_controller("aggressive-v1"), AggressiveController)
    assert isinstance(create_scripted_controller("prudent-v1"), PrudentController)
    assert isinstance(create_scripted_controller("opportunist-v1"), OpportunistController)
    with pytest.raises(ValueError, match="unknown scripted controller"):
        create_scripted_controller("random-v1")


def test_aggressive_bites_then_uses_shove_only_when_endurance_prevents_bite() -> None:
    env = _env()
    env.state.raptor(Actor.A).position = (1, 1)
    env.state.raptor(Actor.B).position = (1, 2)
    controller = AggressiveController()

    assert _decision(env, controller) is Action.BITE

    env.state.raptor(Actor.A).endurance = 1
    assert _decision(env, controller) is Action.SHOVE


def test_aggressive_path_tie_breaks_north_then_rests_when_nothing_is_payable() -> None:
    env = _env()
    controller = AggressiveController()

    assert _decision(env, controller) is Action.MOVE_NORTH

    env.state.raptor(Actor.A).movement_points = 0
    assert _decision(env, controller) is Action.REST


def test_prudent_rests_at_low_endurance_before_other_priorities() -> None:
    env = _env()
    env.state.raptor(Actor.A).endurance = 2

    assert _decision(env, PrudentController()) is Action.REST


def test_prudent_feeds_safely_and_heads_for_the_nearest_available_lateral() -> None:
    env = _env()
    env.state.raptor(Actor.A).position = (3, 0)

    assert _decision(env, PrudentController()) is Action.FEED

    env.state.raptor(Actor.A).position = (3, 1)
    assert _decision(env, PrudentController()) is Action.MOVE_WEST


def test_prudent_bites_an_immediate_threat() -> None:
    env = _env()
    env.state.raptor(Actor.A).position = (1, 1)
    env.state.raptor(Actor.B).position = (1, 2)

    assert _decision(env, PrudentController()) is Action.BITE


def test_opportunist_prefers_safe_feed_then_bite_then_center() -> None:
    env = _env()
    env.state.raptor(Actor.A).position = (4, 4)
    assert _decision(env, OpportunistController()) is Action.FEED

    env.state.raptor(Actor.A).position = (1, 1)
    env.state.raptor(Actor.B).position = (1, 2)
    assert _decision(env, OpportunistController()) is Action.BITE

    env.state.raptor(Actor.B).position = (0, 8)
    assert _decision(env, OpportunistController()) is Action.MOVE_EAST


def test_opportunist_shoves_when_the_center_is_blocked_and_mud_is_behind_target() -> None:
    env = _env()
    env.state.raptor(Actor.A).position = (3, 4)
    env.state.raptor(Actor.A).endurance = 1
    env.state.raptor(Actor.B).position = (4, 4)

    assert _decision(env, OpportunistController()) is Action.SHOVE


def test_opportunist_approaches_an_opponent_blocking_the_center() -> None:
    env = _env()
    env.state.raptor(Actor.A).position = (4, 2)
    env.state.raptor(Actor.B).position = (4, 4)

    assert _decision(env, OpportunistController()) is Action.MOVE_EAST


def test_interrupt_prediction_uses_weighted_path_reserve_and_known_restoration() -> None:
    env = _env()
    assert can_interrupt_next_turn(env.snapshot_public()) is False

    env.state.raptor(Actor.A).position = (1, 1)
    env.state.raptor(Actor.B).position = (1, 2)
    env.state.raptor(Actor.B).endurance = 0
    assert can_interrupt_next_turn(env.snapshot_public()) is False

    env.state.raptor(Actor.B).rest_pending = True
    assert can_interrupt_next_turn(env.snapshot_public()) is True


def test_every_scripted_decision_is_repeatable_and_legal() -> None:
    env = _env()
    snapshot = env.snapshot_public()
    mask = env.legal_actions()
    for controller_id in SCRIPTED_CONTROLLER_IDS:
        controller = create_scripted_controller(controller_id)
        first = controller.choose_action(snapshot, mask)
        second = controller.choose_action(snapshot, mask)
        assert first is second
        assert mask[first] is True
