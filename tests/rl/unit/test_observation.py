"""Observation-v1 spatial channels and normalized scalar features."""

import numpy as np
import pytest

from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.rl.env.observation import (
    FEATURE_COUNT,
    GRID_CHANNELS,
    GRID_COLUMNS,
    GRID_ROWS,
    build_observation,
)


def _initial_engine(first_actor: Actor) -> DinoRLEnv:
    engine = DinoRLEnv(MAP_ID, seed=17, replay=False)
    engine.reset(first_actor=first_actor)
    return engine


def test_observation_has_the_versioned_gym_shape_dtype_and_bounds() -> None:
    engine = _initial_engine(Actor.A)

    observation = build_observation(
        engine.snapshot_public(), learner_actor=Actor.A, first_actor=Actor.A
    )

    assert set(observation) == {"grid", "features"}
    assert observation["grid"].shape == (GRID_CHANNELS, GRID_ROWS, GRID_COLUMNS)
    assert observation["features"].shape == (FEATURE_COUNT,)
    assert observation["grid"].dtype == np.float32
    assert observation["features"].dtype == np.float32
    assert np.all((observation["grid"] >= 0.0) & (observation["grid"] <= 1.0))
    assert np.all((observation["features"] >= 0.0) & (observation["features"] <= 1.0))


@pytest.mark.parametrize("learner_actor", tuple(Actor))
def test_all_scalar_features_stay_bounded_in_public_initial_states(learner_actor: Actor) -> None:
    first_actor = learner_actor
    engine = _initial_engine(first_actor)

    observation = build_observation(
        engine.snapshot_public(), learner_actor=learner_actor, first_actor=first_actor
    )

    assert np.all((observation["features"] >= 0.0) & (observation["features"] <= 1.0))
    assert observation["features"][14] == np.float32(1.0)


def test_rotated_initial_fixtures_encode_to_the_same_canonical_observation() -> None:
    engine_a = _initial_engine(Actor.A)
    engine_b = _initial_engine(Actor.B)

    observation_a = build_observation(
        engine_a.snapshot_public(), learner_actor=Actor.A, first_actor=Actor.A
    )
    observation_b = build_observation(
        engine_b.snapshot_public(), learner_actor=Actor.B, first_actor=Actor.B
    )

    np.testing.assert_array_equal(observation_a["grid"], observation_b["grid"])
    np.testing.assert_array_equal(observation_a["features"], observation_b["features"])


def test_all_scalar_encodings_stay_in_bounds_at_their_rule_extrema() -> None:
    engine = _initial_engine(Actor.A)
    state = engine.state
    self_raptor = state.raptor(Actor.A)
    opponent = state.raptor(Actor.B)
    self_raptor.hp = 0
    self_raptor.endurance = 0
    self_raptor.movement_points = 0
    self_raptor.carcass_score = 3
    self_raptor.main_action_available = False
    self_raptor.voluntary_move_done = True
    opponent.hp = 6
    opponent.endurance = 5
    opponent.carcass_score = 3
    opponent.consumption_pending = "carcass_center"
    opponent.rest_pending = True
    state.central_carcass.status = "recharging"
    state.central_carcass.reactivate_on_turn = 42
    state.turn = 40
    state.round = 30

    observation = build_observation(
        engine.snapshot_public(), learner_actor=Actor.A, first_actor=Actor.B
    )

    assert np.all((observation["features"] >= 0.0) & (observation["features"] <= 1.0))
    np.testing.assert_array_equal(
        observation["features"],
        np.asarray((0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0), dtype=np.float32),
    )
