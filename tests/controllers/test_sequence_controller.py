"""Static validation and strict execution of JSON sequence controllers."""

from pathlib import Path
from typing import Any

import pytest

from dinorl_engine.controllers.sequence import (
    SequenceExhaustedError,
    SequenceIllegalActionError,
    SequenceSchemaError,
    load_sequence_controller,
    parse_sequence_controller,
)
from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.match.replay import ReplayControllerDescriptor
from dinorl_engine.match.runner import run_controller_match

ROOT = Path(__file__).parents[2]


def _payload(actions: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "controller_id": "test-sequence-a",
        "turns": [{"actions": ["END_TURN"] if actions is None else actions}],
    }


def test_canonical_fixture_loads_without_exposing_its_path_as_the_id() -> None:
    controller = load_sequence_controller(ROOT / "fixtures" / "debug" / "bite_then_retreat.json")

    assert controller.controller_id == "bite-then-retreat"


@pytest.mark.parametrize(
    "payload",
    [
        {**_payload(), "schema_version": "2.0.0"},
        {**_payload(), "controller_id": "../plan"},
        {**_payload(), "controller_id": "C:\\plan"},
        {**_payload(), "controller_id": "sequence:plan"},
        {**_payload(), "controller_id": "plan/name"},
        {**_payload(), "turns": []},
        {**_payload(), "turns": [{"actions": []}]},
        {**_payload(), "turns": [{"actions": ["UNKNOWN", "END_TURN"]}]},
        {**_payload(), "turns": [{"actions": ["REST", "END_TURN"]}]},
        {**_payload(), "turns": [{"actions": ["MOVE_NORTH"]}]},
        {**_payload(), "extra": True},
        {**_payload(), "turns": [{"actions": ["END_TURN"], "extra": True}]},
    ],
)
def test_static_sequence_validation_rejects_every_forbidden_shape(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(SequenceSchemaError):
        parse_sequence_controller(payload)


def test_sequence_returns_each_action_of_its_own_scripted_turn() -> None:
    controller = parse_sequence_controller(
        {
            "schema_version": "1.0.0",
            "controller_id": "north-then-end",
            "turns": [{"actions": ["MOVE_NORTH", "END_TURN"]}],
        }
    )
    env = DinoRLEnv("arena_mvp_v1", seed=0)
    env.reset(first_actor=Actor.A)

    first = controller.choose_action(env.snapshot_public(), env.legal_actions())
    env.step(first)
    second = controller.choose_action(env.snapshot_public(), env.legal_actions())

    assert first is Action.MOVE_NORTH
    assert second is Action.END_TURN


def test_illegal_runtime_action_stops_with_a_complete_diagnostic() -> None:
    controller = parse_sequence_controller(_payload(["BITE", "END_TURN"]))
    env = DinoRLEnv("arena_mvp_v1", seed=0)
    env.reset(first_actor=Actor.A)

    with pytest.raises(SequenceIllegalActionError) as captured:
        controller.choose_action(env.snapshot_public(), env.legal_actions())

    diagnostic = str(captured.value)
    assert "Controller: test-sequence-a" in diagnostic
    assert "Actor: A" in diagnostic
    assert "Round: 1" in diagnostic
    assert "Turn: 0" in diagnostic
    assert "Scripted turn index: 0" in diagnostic
    assert "Action index: 0" in diagnostic
    assert "Requested action: BITE" in diagnostic
    assert "Legal actions: MOVE_NORTH, MOVE_EAST, REST, END_TURN" in diagnostic
    assert "Reason: engine legal-action mask rejected the action" in diagnostic
    assert "A: HP=6" in diagnostic


def test_exhausted_sequence_stops_without_a_fallback_action() -> None:
    controller = parse_sequence_controller(_payload())
    env = DinoRLEnv("arena_mvp_v1", seed=0)
    env.reset(first_actor=Actor.A)
    assert controller.choose_action(env.snapshot_public(), env.legal_actions()) is Action.END_TURN

    with pytest.raises(SequenceExhaustedError, match="Sequence exhausted"):
        controller.choose_action(env.snapshot_public(), env.legal_actions())


def test_two_complete_sequences_can_run_to_the_round_limit_with_replay() -> None:
    turns = [{"actions": ["END_TURN"]} for _ in range(30)]
    controller_a = parse_sequence_controller(
        {**_payload(), "controller_id": "end-a", "turns": turns}
    )
    controller_b = parse_sequence_controller(
        {**_payload(), "controller_id": "end-b", "turns": turns}
    )

    outcome = run_controller_match(
        map_id="arena_mvp_v1",
        seed=9,
        controller_a=controller_a,
        controller_b=controller_b,
        controller_a_descriptor=ReplayControllerDescriptor("sequence", "end-a"),
        controller_b_descriptor=ReplayControllerDescriptor("sequence", "end-b"),
        include_replay=True,
    )

    assert outcome.result.winner == "draw"
    assert outcome.result.actions == 60
    assert outcome.replay is not None
    assert outcome.replay["controllers"] == {
        "A": {"kind": "sequence", "id": "end-a"},
        "B": {"kind": "sequence", "id": "end-b"},
    }
