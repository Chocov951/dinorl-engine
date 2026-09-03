"""Score, interruption and carcass lifecycle properties."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor, EndReason
from dinorl_engine.core.engine import DinoRLEnv

type Position = tuple[int, int]

_CARCASSES: tuple[tuple[str, Position, Position], ...] = (
    ("carcass_left_a", (3, 0), (3, 1)),
    ("carcass_center", (4, 4), (4, 3)),
    ("carcass_left_b", (5, 8), (5, 7)),
)


def _opponent(actor: Actor) -> Actor:
    return Actor.B if actor is Actor.A else Actor.A


@given(
    first_actor=st.sampled_from(tuple(Actor)),
    lateral_index=st.integers(min_value=0, max_value=1),
    later_turns=st.integers(min_value=0, max_value=40),
)
def test_consumed_lateral_carcass_never_becomes_available_again(
    first_actor: Actor,
    lateral_index: int,
    later_turns: int,
) -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=0)
    state = env.reset(first_actor=first_actor)
    position = (3, 0) if lateral_index == 0 else (5, 8)
    state.raptor(first_actor).position = position
    env.step(Action.FEED)
    env.step(Action.END_TURN)

    for _ in range(later_turns):
        assert state.lateral_carcasses[lateral_index].status == "consumed"
        if env.is_terminal:
            break
        env.step(Action.END_TURN)
    assert state.lateral_carcasses[lateral_index].status == "consumed"


@given(
    first_actor=st.sampled_from(tuple(Actor)),
    carcass_index=st.integers(min_value=0, max_value=2),
    attack=st.sampled_from((Action.BITE, Action.SHOVE)),
    initial_score=st.integers(min_value=0, max_value=2),
)
def test_interruption_never_awards_a_point(
    first_actor: Actor,
    carcass_index: int,
    attack: Action,
    initial_score: int,
) -> None:
    carcass_id, consumer_position, opponent_position = _CARCASSES[carcass_index]
    opponent = _opponent(first_actor)
    env = DinoRLEnv(map_id=MAP_ID, seed=0)
    state = env.reset(first_actor=first_actor)
    consumer = state.raptor(first_actor)
    consumer.position = consumer_position
    consumer.carcass_score = initial_score
    state.raptor(opponent).position = opponent_position
    env.step(Action.FEED)

    env.step(attack)
    env.step(Action.END_TURN)

    assert consumer.carcass_score == initial_score
    assert consumer.consumption_pending is None
    assert consumer.endurance == 3
    if carcass_id == "carcass_center":
        assert state.central_carcass.status == "active"
    else:
        lateral_index = 0 if carcass_id == "carcass_left_a" else 1
        assert state.lateral_carcasses[lateral_index].status == "available"


@given(
    first_actor=st.sampled_from(tuple(Actor)),
    carcass_index=st.integers(min_value=0, max_value=2),
)
def test_third_carcass_point_always_implies_terminal_score_victory(
    first_actor: Actor,
    carcass_index: int,
) -> None:
    _, consumer_position, _ = _CARCASSES[carcass_index]
    env = DinoRLEnv(map_id=MAP_ID, seed=0)
    state = env.reset(first_actor=first_actor)
    consumer = state.raptor(first_actor)
    consumer.position = consumer_position
    consumer.carcass_score = 2
    env.step(Action.FEED)

    env.step(Action.END_TURN)

    assert consumer.carcass_score == 3
    assert state.terminal is True
    assert state.winner is first_actor
    assert state.end_reason is EndReason.CARCASS_SCORE


@pytest.mark.parametrize("actor", list(Actor))
def test_scores_never_decrease_across_successful_consumptions(actor: Actor) -> None:
    env = DinoRLEnv(map_id=MAP_ID, seed=0)
    state = env.reset(first_actor=actor)
    consumer = state.raptor(actor)
    consumer.position = (4, 4)
    observed_scores = [consumer.carcass_score]

    env.step(Action.FEED)
    env.step(Action.END_TURN)
    observed_scores.append(consumer.carcass_score)
    env.step(Action.END_TURN)
    env.step(Action.END_TURN)
    env.step(Action.FEED)
    env.step(Action.END_TURN)
    observed_scores.append(consumer.carcass_score)

    assert observed_scores == sorted(observed_scores)
