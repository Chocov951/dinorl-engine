"""Tests for the injected local manual controller."""

from collections.abc import Iterator

import pytest

from dinorl_engine.controllers.protocol import Controller
from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.debug.manual import ManualCancelledError, ManualController


def _decision(responses: Iterator[str], output: list[str]) -> tuple[DinoRLEnv, ManualController]:
    env = DinoRLEnv("arena_mvp_v1", seed=2)
    env.reset(first_actor=Actor.A)
    controller = ManualController(
        reader=lambda: next(responses),
        writer=output.append,
        renderer=lambda _state: "STATE\n",
    )
    return env, controller


def test_manual_controller_lists_only_legal_actions_and_accepts_their_number() -> None:
    output: list[str] = []
    env, controller = _decision(iter(["2"]), output)

    action = controller.choose_action(env.snapshot_public(), env.legal_actions())

    assert isinstance(controller, Controller)
    assert action is Action.MOVE_EAST
    assert output == [
        "STATE\n",
        ("Legal actions:\n1. MOVE_NORTH\n2. MOVE_EAST\n3. REST\n4. END_TURN\n\nAction > "),
    ]


def test_manual_controller_accepts_an_exact_action_name() -> None:
    output: list[str] = []
    env, controller = _decision(iter(["END_TURN"]), output)

    assert controller.choose_action(env.snapshot_public(), env.legal_actions()) is Action.END_TURN


def test_invalid_input_reprompts_without_mutating_the_engine() -> None:
    output: list[str] = []
    env, controller = _decision(iter(["BITE", "END_TURN"]), output)
    before = env.snapshot_public()

    action = controller.choose_action(before, env.legal_actions())

    assert action is Action.END_TURN
    assert env.snapshot_public() == before
    assert output[-2:] == ["Invalid action: BITE\n", "Action > "]


@pytest.mark.parametrize("interruption", [EOFError(), KeyboardInterrupt()])
def test_eof_and_keyboard_interrupt_cancel_instead_of_ending_the_turn(
    interruption: BaseException,
) -> None:
    def interrupt() -> str:
        raise interruption

    controller = ManualController(
        reader=interrupt,
        writer=lambda _text: None,
        renderer=lambda _state: "STATE\n",
    )
    env = DinoRLEnv("arena_mvp_v1", seed=2)
    env.reset(first_actor=Actor.A)

    with pytest.raises(ManualCancelledError):
        controller.choose_action(env.snapshot_public(), env.legal_actions())
