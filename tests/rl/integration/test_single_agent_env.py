"""Masked Gymnasium adapter behavior for one DinoRL learner."""

import numpy as np
import pytest
from gymnasium.utils.passive_env_checker import env_reset_passive_checker, env_step_passive_checker

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv, IllegalActionEscapeError


def test_adapted_gymnasium_contract_uses_only_legal_actions() -> None:
    env = DinoRLSingleAgentEnv(seed=4)
    observation, info = env_reset_passive_checker(env, seed=4)
    assert env.observation_space.contains(observation)
    assert info["learner_transitions"] == 0
    assert info["engine_actions"] >= 0
    assert env.action_masks().shape == (len(Action),)
    assert env.action_masks().dtype == np.bool_
    assert env.action_masks().any()

    for _ in range(20):
        legal_action = int(np.flatnonzero(env.action_masks())[0])
        observation, _, terminated, truncated, _ = env_step_passive_checker(env, legal_action)
        assert truncated is False
        assert env.observation_space.contains(observation)
        if terminated:
            observation, _ = env.reset()
            assert env.observation_space.contains(observation)


def test_reset_auto_plays_the_opponent_opening_before_returning_an_observation() -> None:
    env = DinoRLSingleAgentEnv(seed=8)

    observation, info = env.reset(
        options={
            "learner_actor": "A",
            "first_actor": "B",
            "opponent_id": "aggressive-v1",
        }
    )

    assert env.observation_space.contains(observation)
    assert env.active_actor is Actor.A
    assert info["learner_transitions"] == 0
    assert info["engine_actions"] > 0


def test_learner_end_turn_auto_plays_the_complete_opponent_turn() -> None:
    env = DinoRLSingleAgentEnv(seed=8)
    env.reset(
        options={
            "learner_actor": "A",
            "first_actor": "A",
            "opponent_id": "aggressive-v1",
        }
    )

    _, _, terminated, truncated, info = env.step(Action.END_TURN)

    assert terminated is False
    assert truncated is False
    assert env.active_actor is Actor.A
    assert info["learner_transitions"] == 1
    assert info["engine_actions"] > 1


def test_termination_during_the_opponent_turn_is_a_real_gym_termination() -> None:
    env = DinoRLSingleAgentEnv(seed=2)
    env.reset(options={"learner_actor": "A", "first_actor": "A"})
    engine = env.engine
    opponent = engine.state.raptor(Actor.B)
    opponent.carcass_score = 2
    opponent.consumption_pending = "carcass_left_b"
    engine.state.lateral_carcasses[1].status = "pending"
    engine.state.lateral_carcasses[1].pending_actor = Actor.B

    observation, _, terminated, truncated, info = env.step(Action.END_TURN)

    assert env.observation_space.contains(observation)
    assert terminated is True
    assert truncated is False
    assert info["result"] == {"winner": "B", "reason": "carcass_score"}
    assert not env.action_masks().any()


def test_learner_actor_and_first_actor_are_drawn_independently() -> None:
    env = DinoRLSingleAgentEnv(seed=29)
    combinations: set[tuple[str, str]] = set()
    opponent_ids: set[str] = set()
    for _ in range(80):
        _, info = env.reset()
        combinations.add((info["learner_actor"], info["first_actor"]))
        opponent_ids.add(info["opponent_id"])

    assert combinations == {("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")}
    assert opponent_ids == {"aggressive-v1", "prudent-v1", "opportunist-v1"}


def test_seeded_resets_are_reproducible_and_options_are_strict() -> None:
    env = DinoRLSingleAgentEnv(seed=1, environment_index=3)
    first_observation, first_info = env.reset(seed=99)
    first_mask = env.action_masks()
    second_observation, second_info = env.reset(seed=99)

    for field in ("grid", "features"):
        np.testing.assert_array_equal(first_observation[field], second_observation[field])
    assert first_info == second_info
    np.testing.assert_array_equal(first_mask, env.action_masks())

    for options in (
        {"unexpected": "value"},
        {"learner_actor": "C"},
        {"first_actor": -1},
        {"opponent_id": "random-legal-v1"},
    ):
        with pytest.raises(ValueError):
            env.reset(options=options)


@pytest.mark.parametrize(
    ("seed", "environment_index"),
    [(-1, 0), (2**32, 0), (True, 0), (0, -1), (0, True)],
)
def test_environment_constructor_validates_seed_and_environment_index(
    seed: object, environment_index: object
) -> None:
    with pytest.raises(ValueError):
        DinoRLSingleAgentEnv(seed=seed, environment_index=environment_index)  # type: ignore[arg-type]


def test_environment_requires_reset_and_rejects_non_masked_action_values() -> None:
    env = DinoRLSingleAgentEnv(seed=5)
    with pytest.raises(RuntimeError):
        _ = env.engine

    env.reset(options={"learner_actor": "A", "first_actor": "A"})
    for action in (-1, len(Action), True, "north"):
        with pytest.raises(IllegalActionEscapeError):
            env.step(action)  # type: ignore[arg-type]


def test_illegal_action_raises_without_mutating_the_engine() -> None:
    env = DinoRLSingleAgentEnv(seed=7)
    env.reset(options={"learner_actor": "A", "first_actor": "A"})
    illegal_action = next(index for index, allowed in enumerate(env.action_masks()) if not allowed)
    state_before = env.engine.snapshot_public()
    transitions_before = env.learner_transitions
    actions_before = env.engine_actions

    with pytest.raises(IllegalActionEscapeError):
        env.step(illegal_action)

    assert env.engine.snapshot_public() == state_before
    assert env.learner_transitions == transitions_before
    assert env.engine_actions == actions_before
    assert env.engine.replay is False
