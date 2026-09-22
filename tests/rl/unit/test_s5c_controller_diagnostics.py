"""Deterministic controller-shove telemetry required before A2 calibration."""

from __future__ import annotations

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.rl.env.canonical import canonical_to_engine_action, engine_to_canonical_action
from dinorl_engine.rl.s5c.evidence import analyse_shove_transition


def _shove(
    attacker: tuple[int, int],
    target: tuple[int, int],
    *,
    feeding: bool = False,
) -> dict[str, int]:
    environment = DinoRLEnv(MAP_ID, seed=19)
    environment.reset(first_actor=Actor.A)
    environment.state.raptor(Actor.A).position = attacker
    environment.state.raptor(Actor.B).position = target
    if feeding:
        environment.state.raptor(Actor.B).consumption_pending = "carcass_center"
        environment.state.central_carcass.status = "pending"
        environment.state.central_carcass.pending_actor = Actor.B
    before = environment.snapshot_public()
    legal = environment.legal_actions()
    assert legal[Action.SHOVE] is True
    transition = environment.step(Action.SHOVE)
    return analyse_shove_transition(
        before=before,
        after=environment.snapshot_public(),
        transition=transition,
        learner_actor=Actor.A,
        shove_was_legal=legal[Action.SHOVE],
    )


def test_shove_action_index_is_preserved_across_engine_mask_and_both_policy_orientations() -> None:
    assert int(Action.SHOVE) == 5
    assert engine_to_canonical_action(Action.SHOVE, Actor.A) is Action.SHOVE
    assert engine_to_canonical_action(Action.SHOVE, Actor.B) is Action.SHOVE
    assert canonical_to_engine_action(Action.SHOVE, Actor.A) is Action.SHOVE
    assert canonical_to_engine_action(Action.SHOVE, Actor.B) is Action.SHOVE


def test_normal_shove_moves_two_tiles_and_records_cost_and_distance() -> None:
    metrics = _shove((1, 1), (1, 2))

    assert metrics["attempted"] == 1
    assert metrics["legal"] == 1
    assert metrics["succeeded"] == 1
    assert metrics["distance"] == 2
    assert metrics["movement_cost"] == 1
    assert metrics["endurance_cost"] == 1


def test_shove_stops_in_mud_and_is_useful() -> None:
    metrics = _shove((2, 5), (2, 6))

    assert metrics["pushed_into_mud"] == 1
    assert metrics["mud_interrupted"] == 1
    assert metrics["useful"] == 1


def test_shove_collision_against_wall_is_useful_even_without_displacement() -> None:
    metrics = _shove((1, 3), (1, 4))

    assert metrics["wall_collision"] == 1
    assert metrics["distance"] == 0
    assert metrics["useful"] == 1


def test_shove_interruption_of_nutrition_is_recorded() -> None:
    metrics = _shove((4, 3), (4, 4), feeding=True)

    assert metrics["feed_interrupted"] == 1
    assert metrics["useful"] == 1


def test_shove_moves_target_away_from_active_carcass_and_creates_access() -> None:
    metrics = _shove((4, 3), (4, 4))

    assert metrics["away_from_active_carcass"] == 1
    assert metrics["favourable_carcass_access"] == 1
    assert metrics["useful"] == 1
