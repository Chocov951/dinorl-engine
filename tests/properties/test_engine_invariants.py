"""State, resource, chronology and atomicity properties of the complete rule engine."""

import pickle

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor, EndReason
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.errors import IllegalActionError
from dinorl_engine.core.maps import load_map

_MAIN_ACTIONS = frozenset({Action.BITE, Action.SHOVE, Action.FEED, Action.REST})
_NON_END_ACTIONS = tuple(action for action in Action if action is not Action.END_TURN)


def _assert_state_invariants(env: DinoRLEnv) -> None:
    state = env.state
    arena = load_map(MAP_ID)
    positions = [raptor.position for raptor in state.raptors]
    assert len(set(positions)) == 2
    for raptor in state.raptors:
        assert 0 <= raptor.hp <= 6
        assert 0 <= raptor.endurance <= 5
        assert 0 <= raptor.movement_points <= 3
        assert 0 <= raptor.carcass_score <= 3
        assert 0 <= raptor.position[0] < arena.rows
        assert 0 <= raptor.position[1] < arena.columns
        assert raptor.position not in arena.walls
        if raptor.hp == 0:
            assert state.terminal is True
        if raptor.carcass_score == 3:
            assert state.terminal is True


@settings(max_examples=100, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=(1 << 63) - 1),
    first_actor=st.sampled_from(tuple(Actor)),
    choices=st.lists(st.integers(min_value=0, max_value=255), max_size=120),
)
def test_reachable_traces_preserve_state_resource_score_and_main_action_invariants(
    seed: int,
    first_actor: Actor,
    choices: list[int],
) -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=seed)
    env.reset(first_actor=first_actor)
    previous_scores = tuple(raptor.carcass_score for raptor in env.state.raptors)
    claimed_main_actions: set[tuple[int, Actor]] = set()
    _assert_state_invariants(env)

    for choice in choices:
        if env.is_terminal:
            break
        mask = env.legal_actions()
        legal_actions = [action for action in Action if mask[action]]
        action = legal_actions[choice % len(legal_actions)]
        turn = env.state.turn
        actor = env.state.active_actor
        env.step(action)
        if action in _MAIN_ACTIONS:
            key = (turn, actor)
            assert key not in claimed_main_actions
            claimed_main_actions.add(key)
        scores = tuple(raptor.carcass_score for raptor in env.state.raptors)
        assert scores[0] >= previous_scores[0]
        assert scores[1] >= previous_scores[1]
        previous_scores = scores
        _assert_state_invariants(env)


@given(
    first_actor=st.sampled_from(tuple(Actor)),
    rest_turns=st.lists(st.booleans(), min_size=60, max_size=60),
)
def test_every_game_ends_no_later_than_second_turn_of_round_thirty(
    first_actor: Actor,
    rest_turns: list[bool],
) -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=0)
    env.reset(first_actor=first_actor)

    for turn_index, should_rest in enumerate(rest_turns):
        assert env.is_terminal is False
        assert env.state.turn == turn_index
        env.step(Action.REST if should_rest else Action.END_TURN)

    assert env.is_terminal is True
    assert env.state.turn == 59
    assert env.state.round == 30
    assert env.state.end_reason is EndReason.ROUND_LIMIT


@given(
    seed=st.integers(min_value=0, max_value=(1 << 63) - 1),
    first_actor=st.sampled_from(tuple(Actor)),
    action=st.sampled_from(_NON_END_ACTIONS),
)
def test_illegal_action_leaves_nonterminal_state_byte_identical(
    seed: int,
    first_actor: Actor,
    action: Action,
) -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=seed)
    state = env.reset(first_actor=first_actor)
    actor = state.raptor(first_actor)
    actor.movement_points = 0
    actor.main_action_available = False
    actor.voluntary_move_done = True
    before = pickle.dumps(state, protocol=5)

    assert env.legal_actions()[action] is False
    with pytest.raises(IllegalActionError):
        env.step(action)

    assert pickle.dumps(state, protocol=5) == before


@given(action=st.sampled_from(tuple(Action)))
def test_no_action_is_possible_after_terminal_state(action: Action) -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=0)
    state = env.reset(first_actor=Actor.A)
    state.raptor(Actor.A).position = (1, 1)
    state.raptor(Actor.B).position = (1, 2)
    state.raptor(Actor.B).hp = 2
    env.step(Action.BITE)
    before = pickle.dumps(state, protocol=5)

    with pytest.raises(IllegalActionError):
        env.step(action)

    assert pickle.dumps(state, protocol=5) == before
