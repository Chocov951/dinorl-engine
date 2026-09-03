"""Damage and terminal-state properties for bite and shove."""

from hypothesis import given
from hypothesis import strategies as st

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor, EndReason
from dinorl_engine.core.engine import DinoRLEnv


@given(
    first_actor=st.sampled_from(tuple(Actor)),
    target_hp=st.integers(min_value=1, max_value=6),
)
def test_bite_damage_is_bounded_and_zero_hp_is_terminal(
    first_actor: Actor,
    target_hp: int,
) -> None:
    opponent = Actor.B if first_actor is Actor.A else Actor.A
    env = DinoRLEnv(map_id=MAP_ID, seed=0)
    state = env.reset(first_actor=first_actor)
    state.raptor(first_actor).position = (1, 1)
    target = state.raptor(opponent)
    target.position = (1, 2)
    target.hp = target_hp

    env.step(Action.BITE)

    assert target.hp == max(0, target_hp - 2)
    assert 0 <= target.hp <= 6
    if target.hp == 0:
        assert state.terminal is True
        assert state.winner is first_actor
        assert state.end_reason is EndReason.KO


@given(
    first_actor=st.sampled_from(tuple(Actor)),
    target_hp=st.integers(min_value=1, max_value=6),
    collision_on_first_step=st.booleans(),
)
def test_shove_collision_deals_at_most_one_damage_and_zero_hp_is_terminal(
    first_actor: Actor,
    target_hp: int,
    collision_on_first_step: bool,
) -> None:
    opponent = Actor.B if first_actor is Actor.A else Actor.A
    env = DinoRLEnv(map_id=MAP_ID, seed=0)
    state = env.reset(first_actor=first_actor)
    attacker = state.raptor(first_actor)
    target = state.raptor(opponent)
    if collision_on_first_step:
        attacker.position = (1, 3)
        target.position = (1, 4)
    else:
        attacker.position = (1, 2)
        target.position = (1, 3)
    target.hp = target_hp

    env.step(Action.SHOVE)

    assert target_hp - target.hp in {0, 1}
    assert target.hp == target_hp - 1
    assert 0 <= target.hp <= 6
    if target.hp == 0:
        assert state.terminal is True
        assert state.winner is first_actor
        assert state.end_reason is EndReason.KO
