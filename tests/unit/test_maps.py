"""Official MVP arena fixture and compilation tests."""

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from dinorl_engine.core.constants import MAP_ID, Tile
from dinorl_engine.core.maps import compile_map, load_map

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "maps" / "arena_mvp_v1.json"
type MapMutation = Callable[[dict[str, Any]], None]


def _set_grid_cell(payload: dict[str, Any], position: tuple[int, int], symbol: str) -> None:
    row, column = position
    grid_row = payload["grid"][row]
    payload["grid"][row] = f"{grid_row[:column]}{symbol}{grid_row[column + 1 :]}"


def _move_spawn_a(payload: dict[str, Any]) -> None:
    payload["spawns"]["spawn_a"] = [8, 1]
    _set_grid_cell(payload, (8, 0), ".")
    _set_grid_cell(payload, (8, 1), "S")


def _remove_wall_symmetrically_invalid(payload: dict[str, Any]) -> None:
    payload["walls"].remove([7, 3])
    _set_grid_cell(payload, (7, 3), ".")


def _remove_mud_symmetrically_invalid(payload: dict[str, Any]) -> None:
    payload["mud"].remove([6, 1])
    _set_grid_cell(payload, (6, 1), ".")


def _move_lateral_carcass(payload: dict[str, Any]) -> None:
    payload["carcasses"]["carcass_left_b"]["position"] = [5, 7]
    _set_grid_cell(payload, (5, 8), ".")
    _set_grid_cell(payload, (5, 7), "C")


def _move_central_carcass(payload: dict[str, Any]) -> None:
    payload["carcasses"]["carcass_center"]["position"] = [4, 3]
    _set_grid_cell(payload, (4, 4), ".")
    _set_grid_cell(payload, (4, 3), "C")


@pytest.fixture
def map_payload() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        payload: dict[str, Any] = json.load(fixture_file)
    return payload


def test_official_map_is_compiled_to_a_flat_grid() -> None:
    arena = load_map(MAP_ID)

    assert arena.map_id == "arena_mvp_v1"
    assert (arena.rows, arena.columns) == (9, 9)
    assert len(arena.tiles) == 81
    assert arena.spawn_a == (8, 0)
    assert arena.spawn_b == (0, 8)
    assert arena.tile_at(arena.spawn_a) is Tile.EMPTY
    assert arena.tile_at(arena.spawn_b) is Tile.EMPTY


def test_official_map_contains_named_carcasses_and_explicit_terrain() -> None:
    arena = load_map(MAP_ID)

    assert [
        (carcass.carcass_id, carcass.kind, carcass.position) for carcass in arena.carcasses
    ] == [
        ("carcass_left_a", "lateral", (3, 0)),
        ("carcass_center", "central", (4, 4)),
        ("carcass_left_b", "lateral", (5, 8)),
    ]
    assert arena.walls == frozenset(
        {(1, 5), (2, 2), (3, 5), (4, 0), (4, 1), (4, 7), (4, 8), (5, 3), (6, 6), (7, 3)}
    )
    assert arena.mud == frozenset({(2, 7), (2, 8), (3, 2), (3, 4), (5, 4), (5, 6), (6, 0), (6, 1)})
    assert all(arena.tile_at(carcass.position) is Tile.CARCASS for carcass in arena.carcasses)
    assert all(arena.tile_at(position) is Tile.WALL for position in arena.walls)
    assert all(arena.tile_at(position) is Tile.MUD for position in arena.mud)


def test_official_map_is_rotationally_symmetric() -> None:
    arena = load_map(MAP_ID)

    def rotate(position: tuple[int, int]) -> tuple[int, int]:
        return arena.rows - 1 - position[0], arena.columns - 1 - position[1]

    assert rotate(arena.spawn_a) == arena.spawn_b
    assert {rotate(position) for position in arena.walls} == set(arena.walls)
    assert {rotate(position) for position in arena.mud} == set(arena.mud)
    assert rotate(arena.carcasses[0].position) == arena.carcasses[2].position
    assert rotate(arena.carcasses[1].position) == arena.carcasses[1].position


def test_official_map_is_compiled_only_once() -> None:
    assert load_map(MAP_ID) is load_map(MAP_ID)


def test_tile_lookup_rejects_a_position_outside_the_map() -> None:
    with pytest.raises(IndexError, match="Position outside map"):
        load_map(MAP_ID).tile_at((-1, 0))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(unexpected=True),
        lambda payload: payload.update(map_id=1),
        lambda payload: payload.update(map_id="another-map"),
        lambda payload: payload.update(rows=True),
        lambda payload: payload.update(rows=8),
        lambda payload: payload.update(grid=[]),
        lambda payload: payload["grid"].__setitem__(0, 1),
        lambda payload: payload["grid"].__setitem__(0, "X.......S"),
        lambda payload: payload.update(spawns=[]),
        lambda payload: payload["spawns"].update(spawn_a=[8]),
        lambda payload: payload["walls"].append([1, 5]),
        lambda payload: payload["walls"].append([9, 0]),
        lambda payload: payload.update(walls="not-an-array"),
        lambda payload: payload["mud"].append([1, 5]),
        lambda payload: payload["walls"].pop(),
        lambda payload: payload["grid"].__setitem__(0, "M.......S"),
        lambda payload: payload.update(carcasses=[]),
        lambda payload: payload["carcasses"].pop("carcass_center"),
        lambda payload: payload["carcasses"]["carcass_center"].pop("kind"),
        lambda payload: payload["carcasses"]["carcass_center"].update(kind="lateral"),
        lambda payload: payload["spawns"].update(spawn_a=[0, 8]),
        lambda payload: payload["spawns"].update(spawn_a=[3, 0]),
        _move_spawn_a,
        _remove_wall_symmetrically_invalid,
        _remove_mud_symmetrically_invalid,
        _move_lateral_carcass,
        _move_central_carcass,
    ],
)
def test_invalid_map_fixture_is_rejected(
    map_payload: dict[str, Any], mutation: MapMutation
) -> None:
    invalid_payload = copy.deepcopy(map_payload)
    mutation(invalid_payload)

    with pytest.raises(ValueError):
        compile_map(invalid_payload)


def test_map_fixture_must_be_an_object() -> None:
    with pytest.raises(ValueError, match="map must be an object"):
        compile_map([])


def test_unknown_map_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown map_id"):
        load_map("unknown-map")
