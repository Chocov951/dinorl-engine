"""Deterministic scripted controllers shipped with the MVP."""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Mapping
from typing import ClassVar, Final, Literal, cast

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID
from dinorl_engine.core.maps import Position, load_map
from dinorl_engine.core.state import PublicSnapshot

__all__ = [
    "SCRIPTED_CONTROLLER_IDS",
    "AggressiveController",
    "ControllerId",
    "OpportunistController",
    "PrudentController",
    "can_interrupt_next_turn",
    "create_scripted_controller",
]

type ControllerId = Literal["aggressive-v1", "prudent-v1", "opportunist-v1"]
type ScriptedController = AggressiveController | PrudentController | OpportunistController

SCRIPTED_CONTROLLER_IDS: Final[tuple[ControllerId, ...]] = (
    "aggressive-v1",
    "prudent-v1",
    "opportunist-v1",
)

_ARENA = load_map(MAP_ID)
_MOVES: Final[tuple[tuple[Action, Position], ...]] = (
    (Action.MOVE_NORTH, (-1, 0)),
    (Action.MOVE_EAST, (0, 1)),
    (Action.MOVE_SOUTH, (1, 0)),
    (Action.MOVE_WEST, (0, -1)),
)
_CARCASS_POSITIONS: Final = {carcass.carcass_id: carcass.position for carcass in _ARENA.carcasses}


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"public state field {field} must be an object")
    return cast(Mapping[str, object], value)


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"public state field {field} must be an integer")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"public state field {field} must be a boolean")
    return value


def _position(value: object, field: str) -> Position:
    if (
        not isinstance(value, tuple | list)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
    ):
        raise ValueError(f"public state field {field} must be a coordinate")
    return value[0], value[1]


def _active_name(state: PublicSnapshot) -> str:
    value = state.get("active_actor")
    if value not in {"A", "B"}:
        raise ValueError("public state has an invalid active_actor")
    return value


def _opponent_name(actor_name: str) -> str:
    return "B" if actor_name == "A" else "A"


def _raptor(state: PublicSnapshot, actor_name: str) -> Mapping[str, object]:
    raptors = _mapping(state.get("raptors"), "raptors")
    return _mapping(raptors.get(actor_name), f"raptors.{actor_name}")


def _raptor_position(state: PublicSnapshot, actor_name: str) -> Position:
    return _position(_raptor(state, actor_name).get("position"), f"raptors.{actor_name}.position")


def _is_inside(position: Position) -> bool:
    return 0 <= position[0] < _ARENA.rows and 0 <= position[1] < _ARENA.columns


def _add(position: Position, delta: Position) -> Position:
    return position[0] + delta[0], position[1] + delta[1]


def _is_legal(mask: tuple[bool, ...], action: Action) -> bool:
    return len(mask) == len(Action) and mask[action]


def _next_step(
    start: Position,
    goals: set[Position],
    blocked: Position,
    legal_actions: tuple[bool, ...],
) -> Action | None:
    """Return the first step of a BFS with the versioned directional tie-break."""

    if start in goals:
        return None
    queue: deque[tuple[Position, Action | None]] = deque([(start, None)])
    visited = {start}
    while queue:
        position, first_action = queue.popleft()
        for action, delta in _MOVES:
            if first_action is None and not _is_legal(legal_actions, action):
                continue
            destination = _add(position, delta)
            if (
                destination in visited
                or destination == blocked
                or not _is_inside(destination)
                or destination in _ARENA.walls
            ):
                continue
            candidate = action if first_action is None else first_action
            if destination in goals:
                return candidate
            visited.add(destination)
            queue.append((destination, candidate))
    return None


def _adjacent_positions(position: Position) -> set[Position]:
    return {
        candidate
        for _, delta in _MOVES
        if _is_inside(candidate := _add(position, delta)) and candidate not in _ARENA.walls
    }


def _weighted_distance(
    start: Position,
    goals: set[Position],
    blocked: Position,
) -> int | None:
    """Find the minimum movement-point cost, charging mud on departure."""

    pending: list[tuple[int, Position]] = [(0, start)]
    distances = {start: 0}
    while pending:
        cost, position = heapq.heappop(pending)
        if cost != distances[position]:
            continue
        if position in goals:
            return cost
        step_cost = 2 if position in _ARENA.mud else 1
        for _, delta in _MOVES:
            destination = _add(position, delta)
            if destination == blocked or not _is_inside(destination) or destination in _ARENA.walls:
                continue
            candidate_cost = cost + step_cost
            if candidate_cost < distances.get(destination, candidate_cost + 1):
                distances[destination] = candidate_cost
                heapq.heappush(pending, (candidate_cost, destination))
    return None


def can_interrupt_next_turn(state: PublicSnapshot) -> bool:
    """Predict whether the opponent can reach and attack after known start effects."""

    turn = _integer(state.get("turn"), "turn")
    if turn >= 59:
        return False
    actor_name = _active_name(state)
    opponent_name = _opponent_name(actor_name)
    actor_position = _raptor_position(state, actor_name)
    opponent = _raptor(state, opponent_name)
    opponent_position = _position(opponent.get("position"), f"raptors.{opponent_name}.position")
    pending = opponent.get("consumption_pending")
    score = _integer(opponent.get("carcass_score"), f"raptors.{opponent_name}.carcass_score")
    if pending is not None and score >= 2:
        return False
    endurance = _integer(opponent.get("endurance"), f"raptors.{opponent_name}.endurance")
    if _boolean(opponent.get("rest_pending"), f"raptors.{opponent_name}.rest_pending"):
        endurance = 5
    if endurance < 1:
        return False
    movement_cost = _weighted_distance(
        opponent_position,
        _adjacent_positions(actor_position),
        actor_position,
    )
    if movement_cost is None or movement_cost + 1 > 3:
        return False
    return endurance >= 2 or endurance >= 1


def _move_toward_opponent(state: PublicSnapshot, legal_actions: tuple[bool, ...]) -> Action | None:
    actor_name = _active_name(state)
    opponent_position = _raptor_position(state, _opponent_name(actor_name))
    return _next_step(
        _raptor_position(state, actor_name),
        _adjacent_positions(opponent_position),
        opponent_position,
        legal_actions,
    )


def _move_toward(
    state: PublicSnapshot,
    legal_actions: tuple[bool, ...],
    goals: set[Position],
) -> Action | None:
    actor_name = _active_name(state)
    opponent_position = _raptor_position(state, _opponent_name(actor_name))
    reachable_goals = set(goals)
    if opponent_position in reachable_goals:
        reachable_goals.remove(opponent_position)
        reachable_goals.update(_adjacent_positions(opponent_position))
    return _next_step(
        _raptor_position(state, actor_name),
        reachable_goals,
        opponent_position,
        legal_actions,
    )


def _available_lateral_positions(state: PublicSnapshot) -> set[Position]:
    carcasses = _mapping(state.get("carcasses"), "carcasses")
    positions: set[Position] = set()
    for carcass_id in ("carcass_left_a", "carcass_left_b"):
        carcass = _mapping(carcasses.get(carcass_id), f"carcasses.{carcass_id}")
        if carcass.get("status") == "available":
            positions.add(_CARCASS_POSITIONS[carcass_id])
    return positions


def _shove_is_tactical(state: PublicSnapshot) -> bool:
    actor_name = _active_name(state)
    actor = _raptor_position(state, actor_name)
    target = _raptor_position(state, _opponent_name(actor_name))
    delta = target[0] - actor[0], target[1] - actor[1]
    position = target
    collision = False
    for _ in range(2):
        destination = _add(position, delta)
        if not _is_inside(destination) or destination in _ARENA.walls:
            collision = True
            break
        position = destination
        if position in _ARENA.mud:
            break
    return collision or position in _ARENA.mud


class AggressiveController:
    """Seek contact, then bite whenever resources allow it."""

    controller_id: ClassVar[ControllerId] = "aggressive-v1"

    def choose_action(self, state: PublicSnapshot, legal_actions: tuple[bool, ...]) -> Action:
        if _is_legal(legal_actions, Action.BITE):
            return Action.BITE
        actor = _raptor(state, _active_name(state))
        endurance = _integer(actor.get("endurance"), "active endurance")
        if endurance < 2 and _is_legal(legal_actions, Action.SHOVE):
            return Action.SHOVE
        movement = _move_toward_opponent(state, legal_actions)
        if movement is not None:
            return movement
        if _is_legal(legal_actions, Action.REST):
            return Action.REST
        return Action.END_TURN


class PrudentController:
    """Preserve endurance and favor safe lateral carcass points."""

    controller_id: ClassVar[ControllerId] = "prudent-v1"

    def choose_action(self, state: PublicSnapshot, legal_actions: tuple[bool, ...]) -> Action:
        actor = _raptor(state, _active_name(state))
        endurance = _integer(actor.get("endurance"), "active endurance")
        if endurance <= 2 and _is_legal(legal_actions, Action.REST):
            return Action.REST
        if _is_legal(legal_actions, Action.FEED) and not can_interrupt_next_turn(state):
            return Action.FEED
        if _is_legal(legal_actions, Action.BITE):
            return Action.BITE
        movement = _move_toward(state, legal_actions, _available_lateral_positions(state))
        if movement is not None:
            return movement
        return Action.END_TURN


class OpportunistController:
    """Exploit safe food, immediate attacks, and tactical central control."""

    controller_id: ClassVar[ControllerId] = "opportunist-v1"

    def choose_action(self, state: PublicSnapshot, legal_actions: tuple[bool, ...]) -> Action:
        if _is_legal(legal_actions, Action.FEED) and not can_interrupt_next_turn(state):
            return Action.FEED
        if _is_legal(legal_actions, Action.BITE):
            return Action.BITE
        movement = _move_toward(
            state,
            legal_actions,
            {_CARCASS_POSITIONS["carcass_center"]},
        )
        if movement is not None:
            return movement
        if _is_legal(legal_actions, Action.SHOVE) and _shove_is_tactical(state):
            return Action.SHOVE
        return Action.END_TURN


def create_scripted_controller(controller_id: str) -> ScriptedController:
    """Resolve one identifier from the closed MVP controller set."""

    if controller_id == "aggressive-v1":
        return AggressiveController()
    if controller_id == "prudent-v1":
        return PrudentController()
    if controller_id == "opportunist-v1":
        return OpportunistController()
    raise ValueError(f"unknown scripted controller: {controller_id!r}")
