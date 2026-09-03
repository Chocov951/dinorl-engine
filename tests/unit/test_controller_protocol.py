"""Tests for the replaceable controller boundary."""

from typing import cast

import pytest

from dinorl_engine.controllers.protocol import (
    Controller,
    ControllerDecisionError,
    step_with_controller,
)
from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.errors import IllegalActionError
from dinorl_engine.core.state import PublicSnapshot


class EndTurnController:
    def choose_action(self, state: PublicSnapshot, legal_actions: tuple[bool, ...]) -> Action:
        assert state["active_actor"] == "A"
        assert legal_actions[Action.END_TURN] is True
        return Action.END_TURN


class IllegalBiteController:
    def choose_action(self, state: PublicSnapshot, legal_actions: tuple[bool, ...]) -> Action:
        return Action.BITE


class IntegerController:
    def choose_action(self, state: PublicSnapshot, legal_actions: tuple[bool, ...]) -> Action:
        return cast(Action, 8)


def _env() -> DinoRLEnv:
    env = DinoRLEnv("arena_mvp_v1", seed=1)
    env.reset(first_actor=Actor.A)
    return env


def test_runtime_controller_protocol_accepts_a_compatible_controller() -> None:
    controller = EndTurnController()

    assert isinstance(controller, Controller)
    transition = step_with_controller(_env(), controller)

    assert transition.action is Action.END_TURN
    assert transition.actor is Actor.A


def test_controller_action_is_still_verified_by_the_engine() -> None:
    with pytest.raises(IllegalActionError, match="Action is not legal"):
        step_with_controller(_env(), IllegalBiteController())


def test_controller_returning_a_raw_integer_is_rejected() -> None:
    with pytest.raises(ControllerDecisionError, match="Action enum"):
        step_with_controller(_env(), IntegerController())


def test_incompatible_controller_is_rejected_before_execution() -> None:
    with pytest.raises(TypeError, match="Controller protocol"):
        step_with_controller(_env(), cast(Controller, object()))
