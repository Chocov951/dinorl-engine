"""Initial game state and public snapshot tests."""

from types import MappingProxyType

import pytest

from dinorl_engine.core.constants import (
    ACTION_VERSION,
    ENGINE_VERSION,
    MAP_ID,
    OBSERVATION_VERSION,
    REPLAY_VERSION,
    RULES_VERSION,
    Actor,
)
from dinorl_engine.core.engine import DinoRLEnv


@pytest.mark.parametrize("first_actor", list(Actor))
def test_reset_initializes_raptors_at_their_spawns(first_actor: Actor) -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=12345)

    state = env.reset(first_actor=first_actor)

    assert state.engine_version == ENGINE_VERSION
    assert state.rules_version == RULES_VERSION
    assert state.action_version == ACTION_VERSION
    assert state.observation_version == OBSERVATION_VERSION
    assert state.replay_version == REPLAY_VERSION
    assert state.map_id == MAP_ID
    assert state.seed == 12345
    assert state.first_actor is first_actor
    assert state.active_actor is first_actor
    assert state.turn == 0
    assert state.round == 1
    assert state.terminal is False
    assert state.winner is None
    assert state.end_reason is None

    raptor_a, raptor_b = state.raptors
    assert raptor_a.position == (8, 0)
    assert raptor_b.position == (0, 8)
    for actor, raptor in zip(Actor, state.raptors, strict=True):
        assert raptor.hp == 6
        assert raptor.endurance == 5
        assert raptor.movement_points == (3 if actor is first_actor else 0)
        assert raptor.carcass_score == 0
        assert raptor.main_action_available is True
        assert raptor.voluntary_move_done is False
        assert raptor.rest_pending is False
        assert raptor.consumption_pending is None


def test_seeded_first_actor_is_deterministic() -> None:
    first = DinoRLEnv(map_id=MAP_ID, seed=987654321).reset().first_actor
    second = DinoRLEnv(map_id=MAP_ID, seed=987654321).reset().first_actor

    assert first is second
    assert first in Actor


def test_initial_carcasses_have_their_public_statuses() -> None:
    state = DinoRLEnv(map_id=MAP_ID, seed=1).reset(first_actor=Actor.A)

    assert [(carcass.status, carcass.pending_actor) for carcass in state.lateral_carcasses] == [
        ("available", None),
        ("available", None),
    ]
    assert state.central_carcass.status == "active"
    assert state.central_carcass.pending_actor is None
    assert state.central_carcass.reactivate_on_turn is None


def test_public_snapshot_is_stable_and_deeply_immutable() -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=42)
    env.reset(first_actor=Actor.B)

    snapshot = env.snapshot_public()

    assert snapshot == env.snapshot_public()
    assert isinstance(snapshot, MappingProxyType)
    assert snapshot["active_actor"] == "B"
    assert snapshot["turn"] == 0
    assert snapshot["round"] == 1
    assert snapshot["terminal"] is False
    assert snapshot["winner"] is None
    assert snapshot["end_reason"] is None
    raptors = snapshot["raptors"]
    assert isinstance(raptors, MappingProxyType)
    assert raptors["A"]["position"] == (8, 0)
    assert raptors["B"]["movement_points"] == 3
    carcasses = snapshot["carcasses"]
    assert isinstance(carcasses, MappingProxyType)
    assert carcasses["carcass_center"]["status"] == "active"

    with pytest.raises(TypeError):
        snapshot["round"] = 2
    with pytest.raises(TypeError):
        raptors["A"]["hp"] = 0
