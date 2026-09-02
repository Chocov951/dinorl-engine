"""Free shove direction, costs and legality tests."""

from collections.abc import Callable

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.errors import IllegalActionError
from dinorl_engine.core.events import ActionCost

type StateMutation = Callable[[DinoRLEnv], None]
type Position = tuple[int, int]


def _environment(
    attacker_position: Position = (1, 1), target_position: Position = (1, 2)
) -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=4)
    env.reset(first_actor=Actor.A)
    env.state.raptor(Actor.A).position = attacker_position
    env.state.raptor(Actor.B).position = target_position
    return env


@pytest.mark.parametrize(
    ("attacker_position", "target_position", "expected_position"),
    [
        ((3, 1), (2, 1), (0, 1)),
        ((1, 1), (1, 2), (1, 4)),
        ((0, 1), (1, 1), (3, 1)),
        ((1, 4), (1, 3), (1, 1)),
    ],
)
def test_shove_pushes_target_two_free_tiles_in_a_straight_line(
    attacker_position: Position,
    target_position: Position,
    expected_position: Position,
) -> None:
    env = _environment(attacker_position, target_position)

    assert env.legal_actions()[Action.SHOVE] is True
    transition = env.step(Action.SHOVE)

    attacker = env.state.raptor(Actor.A)
    target = env.state.raptor(Actor.B)
    assert transition.action is Action.SHOVE
    assert transition.actor is Actor.A
    assert transition.cost == ActionCost(movement=1, endurance=1)
    assert [effect.type for effect in transition.effects] == ["target_shoved"]
    shove_effect = transition.effects[0]
    assert shove_effect.actor is Actor.A
    assert shove_effect.target is Actor.B
    assert shove_effect.from_position == target_position
    assert shove_effect.to_position == expected_position
    assert transition.turn_ended is False
    assert attacker.movement_points == 2
    assert attacker.endurance == 4
    assert attacker.main_action_available is False
    assert target.position == expected_position
    assert target.hp == 6
    assert target.voluntary_move_done is False
    assert env.state.active_actor is Actor.A
    assert env.legal_actions()[Action.SHOVE] is False


def _move_target_out_of_range(env: DinoRLEnv) -> None:
    env.state.raptor(Actor.B).position = (1, 3)


def _remove_movement_points(env: DinoRLEnv) -> None:
    env.state.raptor(Actor.A).movement_points = 0


def _remove_endurance(env: DinoRLEnv) -> None:
    env.state.raptor(Actor.A).endurance = 0


def _consume_main_action(env: DinoRLEnv) -> None:
    env.state.claim_main_action()


@pytest.mark.parametrize(
    "mutation",
    [_move_target_out_of_range, _remove_movement_points, _remove_endurance, _consume_main_action],
)
def test_illegal_shove_does_not_mutate_state(mutation: StateMutation) -> None:
    env = _environment()
    mutation(env)
    before = env.snapshot_public()

    assert env.legal_actions()[Action.SHOVE] is False
    with pytest.raises(IllegalActionError):
        env.step(Action.SHOVE)

    assert env.snapshot_public() == before


def test_shove_interrupts_consumption_and_keeps_turn_open() -> None:
    env = _environment()
    target = env.state.raptor(Actor.B)
    target.consumption_pending = "carcass_center"
    env.state.central_carcass.status = "pending"
    env.state.central_carcass.pending_actor = Actor.B

    transition = env.step(Action.SHOVE)

    assert target.consumption_pending is None
    assert env.state.central_carcass.status == "active"
    assert [effect.type for effect in transition.effects] == [
        "target_shoved",
        "consumption_interrupted",
    ]
    assert env.legal_actions()[Action.MOVE_SOUTH] is True
