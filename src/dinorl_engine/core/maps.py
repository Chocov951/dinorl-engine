"""Loading and compact representation of official DinoRL maps."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal, cast

from dinorl_engine.core.constants import MAP_ID, Tile

__all__ = ["ArenaMap", "Carcass", "Position", "compile_map", "load_map"]

type Position = tuple[int, int]
type CarcassKind = Literal["lateral", "central"]

_ROWS = 9
_COLUMNS = 9
_FIXTURE_DIRECTORY = Path(__file__).resolve().parents[3] / "fixtures" / "maps"
_MAP_FIXTURES = {MAP_ID: "arena_mvp_v1.json"}
_ROOT_FIELDS = {"map_id", "rows", "columns", "grid", "spawns", "carcasses", "walls", "mud"}
_CARCASS_IDS = ("carcass_left_a", "carcass_center", "carcass_left_b")
_GRID_SYMBOLS = frozenset(".SMBC")


@dataclass(frozen=True, slots=True)
class Carcass:
    """A named carcass and its fixed position on an arena map."""

    carcass_id: str
    kind: CarcassKind
    position: Position


@dataclass(frozen=True, slots=True)
class ArenaMap:
    """A validated arena compiled to a flat, immutable tile sequence."""

    map_id: str
    rows: int
    columns: int
    tiles: tuple[Tile, ...]
    spawn_a: Position
    spawn_b: Position
    carcasses: tuple[Carcass, ...]
    walls: frozenset[Position]
    mud: frozenset[Position]

    def tile_at(self, position: Position) -> Tile:
        """Return the tile at a validated row-column position."""

        row, column = position
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            raise IndexError(f"Position outside map: {position!r}")
        return self.tiles[row * self.columns + column]


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _exact_fields(value: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} must contain exactly {sorted(expected)!r}")


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _position(value: object, field: str, rows: int, columns: int) -> Position:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-item coordinate")
    row = _integer(value[0], f"{field}[0]")
    column = _integer(value[1], f"{field}[1]")
    if not (0 <= row < rows and 0 <= column < columns):
        raise ValueError(f"{field} is outside the map")
    return row, column


def _positions(value: object, field: str, rows: int, columns: int) -> frozenset[Position]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    positions = [
        _position(item, f"{field}[{index}]", rows, columns) for index, item in enumerate(value)
    ]
    if len(positions) != len(set(positions)):
        raise ValueError(f"{field} contains duplicate coordinates")
    return frozenset(positions)


def _grid(value: object, rows: int, columns: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError("grid must contain one string per row")
    grid: list[str] = []
    for index, item in enumerate(value):
        row = _string(item, f"grid[{index}]")
        if len(row) != columns or not set(row) <= _GRID_SYMBOLS:
            raise ValueError(f"grid[{index}] has invalid dimensions or symbols")
        grid.append(row)
    return tuple(grid)


def _carcasses(value: object, rows: int, columns: int) -> tuple[Carcass, ...]:
    definitions = _mapping(value, "carcasses")
    _exact_fields(definitions, set(_CARCASS_IDS), "carcasses")
    expected_kinds: tuple[CarcassKind, ...] = ("lateral", "central", "lateral")
    carcasses: list[Carcass] = []
    for carcass_id, expected_kind in zip(_CARCASS_IDS, expected_kinds, strict=True):
        definition = _mapping(definitions[carcass_id], f"carcasses.{carcass_id}")
        _exact_fields(definition, {"kind", "position"}, f"carcasses.{carcass_id}")
        kind = _string(definition["kind"], f"carcasses.{carcass_id}.kind")
        if kind != expected_kind:
            raise ValueError(f"carcasses.{carcass_id}.kind must be {expected_kind!r}")
        position = _position(
            definition["position"], f"carcasses.{carcass_id}.position", rows, columns
        )
        carcasses.append(Carcass(carcass_id, expected_kind, position))
    return tuple(carcasses)


def _positions_for_symbol(grid: tuple[str, ...], symbol: str) -> set[Position]:
    return {
        (row_index, column_index)
        for row_index, row in enumerate(grid)
        for column_index, cell in enumerate(row)
        if cell == symbol
    }


def _validate_symmetry(
    rows: int,
    columns: int,
    spawn_a: Position,
    spawn_b: Position,
    carcasses: tuple[Carcass, ...],
    walls: frozenset[Position],
    mud: frozenset[Position],
) -> None:
    def rotate(position: Position) -> Position:
        return rows - 1 - position[0], columns - 1 - position[1]

    if rotate(spawn_a) != spawn_b:
        raise ValueError("spawns are not rotationally symmetric")
    if {rotate(position) for position in walls} != set(walls):
        raise ValueError("walls are not rotationally symmetric")
    if {rotate(position) for position in mud} != set(mud):
        raise ValueError("mud is not rotationally symmetric")
    if rotate(carcasses[0].position) != carcasses[2].position:
        raise ValueError("lateral carcasses are not rotationally symmetric")
    if rotate(carcasses[1].position) != carcasses[1].position:
        raise ValueError("central carcass is not centered")


def compile_map(payload: object) -> ArenaMap:
    """Validate a map fixture and compile it to its immutable runtime form."""

    root = _mapping(payload, "map")
    _exact_fields(root, _ROOT_FIELDS, "map")

    map_id = _string(root["map_id"], "map_id")
    if map_id != MAP_ID:
        raise ValueError(f"Unsupported map_id: {map_id!r}")
    rows = _integer(root["rows"], "rows")
    columns = _integer(root["columns"], "columns")
    if (rows, columns) != (_ROWS, _COLUMNS):
        raise ValueError(f"Official map dimensions must be {_ROWS}x{_COLUMNS}")

    grid = _grid(root["grid"], rows, columns)
    spawns = _mapping(root["spawns"], "spawns")
    _exact_fields(spawns, {"spawn_a", "spawn_b"}, "spawns")
    spawn_a = _position(spawns["spawn_a"], "spawns.spawn_a", rows, columns)
    spawn_b = _position(spawns["spawn_b"], "spawns.spawn_b", rows, columns)
    carcasses = _carcasses(root["carcasses"], rows, columns)
    walls = _positions(root["walls"], "walls", rows, columns)
    mud = _positions(root["mud"], "mud", rows, columns)

    carcass_positions = {carcass.position for carcass in carcasses}
    spawn_positions = {spawn_a, spawn_b}
    if len(spawn_positions) != 2 or len(carcass_positions) != 3:
        raise ValueError("spawns and carcasses must occupy distinct positions")
    if walls & mud or (walls | mud) & (spawn_positions | carcass_positions):
        raise ValueError("terrain, spawns and carcasses must not overlap")
    if spawn_positions & carcass_positions:
        raise ValueError("spawns and carcasses must not overlap")

    expected_positions = {
        "S": spawn_positions,
        "C": carcass_positions,
        "M": set(walls),
        "B": set(mud),
    }
    for symbol, positions in expected_positions.items():
        if _positions_for_symbol(grid, symbol) != positions:
            raise ValueError(f"grid symbol {symbol!r} does not match explicit coordinates")

    _validate_symmetry(rows, columns, spawn_a, spawn_b, carcasses, walls, mud)
    tile_by_symbol = {"M": Tile.WALL, "B": Tile.MUD, "C": Tile.CARCASS}
    tiles = tuple(tile_by_symbol.get(symbol, Tile.EMPTY) for row in grid for symbol in row)
    return ArenaMap(
        map_id=map_id,
        rows=rows,
        columns=columns,
        tiles=tiles,
        spawn_a=spawn_a,
        spawn_b=spawn_b,
        carcasses=carcasses,
        walls=walls,
        mud=mud,
    )


@cache
def load_map(map_id: str = MAP_ID) -> ArenaMap:
    """Load an official map by identifier and cache its compiled form."""

    fixture_name = _MAP_FIXTURES.get(map_id)
    if fixture_name is None:
        raise ValueError(f"Unknown map_id: {map_id!r}")
    with (_FIXTURE_DIRECTORY / fixture_name).open(encoding="utf-8") as fixture_file:
        return compile_map(json.load(fixture_file))
