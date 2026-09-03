"""Position invariants preserved by every legal shove on the official map."""

from hypothesis import assume, given
from hypothesis import strategies as st

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.maps import load_map

type Position = tuple[int, int]

_DIRECTIONS: tuple[Position, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


@given(
    attacker_row=st.integers(min_value=0, max_value=8),
    attacker_column=st.integers(min_value=0, max_value=8),
    direction=st.sampled_from(_DIRECTIONS),
    target_hp=st.integers(min_value=1, max_value=6),
)
def test_legal_shove_preserves_position_and_resource_invariants(
    attacker_row: int,
    attacker_column: int,
    direction: Position,
    target_hp: int,
) -> None:
    arena = load_map(MAP_ID)
    attacker_position = (attacker_row, attacker_column)
    target_position = (
        attacker_row + direction[0],
        attacker_column + direction[1],
    )
    assume(attacker_position not in arena.walls)
    assume(0 <= target_position[0] < arena.rows and 0 <= target_position[1] < arena.columns)
    assume(target_position not in arena.walls)
    env = DinoRLEnv(map_id=MAP_ID, seed=6)
    env.reset(first_actor=Actor.A)
    attacker = env.state.raptor(Actor.A)
    target = env.state.raptor(Actor.B)
    attacker.position = attacker_position
    target.position = target_position
    target.hp = target_hp

    env.step(Action.SHOVE)

    assert 0 <= target.position[0] < arena.rows
    assert 0 <= target.position[1] < arena.columns
    assert target.position not in arena.walls
    assert target.position != attacker.position
    assert target_hp - target.hp in {0, 1}
    assert target.movement_points == 0
    assert target.voluntary_move_done is False
    assert attacker.movement_points == 2
    assert attacker.endurance == 4
