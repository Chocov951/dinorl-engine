"""Shove collision and mud resolution tests."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor, EndReason
from dinorl_engine.core.engine import DinoRLEnv

type Position = tuple[int, int]


def _environment(attacker_position: Position, target_position: Position) -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=5)
    env.reset(first_actor=Actor.A)
    env.state.raptor(Actor.A).position = attacker_position
    env.state.raptor(Actor.B).position = target_position
    return env


@pytest.mark.parametrize(
    ("attacker_position", "target_position", "expected_position"),
    [
        ((1, 3), (1, 4), (1, 4)),
        ((1, 2), (1, 3), (1, 4)),
        ((1, 1), (0, 1), (0, 1)),
        ((2, 1), (1, 1), (0, 1)),
    ],
)
def test_blocked_shove_stops_on_last_valid_tile_and_deals_one_damage(
    attacker_position: Position,
    target_position: Position,
    expected_position: Position,
) -> None:
    env = _environment(attacker_position, target_position)

    transition = env.step(Action.SHOVE)

    target = env.state.raptor(Actor.B)
    assert target.position == expected_position
    assert target.hp == 5
    assert [effect.type for effect in transition.effects] == ["target_shoved", "damage_dealt"]
    assert transition.effects[0].from_position == target_position
    assert transition.effects[0].to_position == expected_position
    assert transition.effects[1].amount == 1


@pytest.mark.parametrize(
    ("attacker_position", "target_position", "expected_position"),
    [
        ((2, 5), (2, 6), (2, 7)),
        ((2, 4), (2, 5), (2, 7)),
        ((6, 4), (5, 4), (3, 4)),
    ],
)
def test_shove_stops_when_target_reaches_mud_without_damage(
    attacker_position: Position,
    target_position: Position,
    expected_position: Position,
) -> None:
    env = _environment(attacker_position, target_position)

    transition = env.step(Action.SHOVE)

    target = env.state.raptor(Actor.B)
    assert target.position == expected_position
    assert target.hp == 6
    assert target.voluntary_move_done is False
    assert [effect.type for effect in transition.effects] == ["target_shoved"]


def test_blocked_shove_still_interrupts_pending_consumption() -> None:
    env = _environment((1, 3), (1, 4))
    target = env.state.raptor(Actor.B)
    target.consumption_pending = "carcass_center"
    env.state.central_carcass.status = "pending"
    env.state.central_carcass.pending_actor = Actor.B

    transition = env.step(Action.SHOVE)

    assert target.position == (1, 4)
    assert target.consumption_pending is None
    assert env.state.central_carcass.status == "active"
    assert [effect.type for effect in transition.effects] == [
        "target_shoved",
        "damage_dealt",
        "consumption_interrupted",
    ]


def test_collision_ko_is_immediate() -> None:
    env = _environment((1, 3), (1, 4))
    env.state.raptor(Actor.B).hp = 1

    env.step(Action.SHOVE)

    assert env.state.raptor(Actor.B).hp == 0
    assert env.state.terminal is True
    assert env.state.winner is Actor.A
    assert env.state.end_reason is EndReason.KO
    assert not any(env.legal_actions())
