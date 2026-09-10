"""Deterministic plain-text rendering of public DinoRL states."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dinorl_engine.core.constants import MAP_ID
from dinorl_engine.core.maps import Position, load_map
from dinorl_engine.core.state import PublicSnapshot

__all__ = ["render_state"]

type ReplayEvent = Mapping[str, object]

_ARENA = load_map(MAP_ID)
_CARCASS_POSITIONS = {carcass.carcass_id: carcass.position for carcass in _ARENA.carcasses}


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _position(value: object, field: str) -> Position:
    if (
        not isinstance(value, tuple | list)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
    ):
        raise ValueError(f"{field} must be a coordinate")
    return value[0], value[1]


def _raptors(state: PublicSnapshot) -> Mapping[str, object]:
    return _mapping(state.get("raptors"), "raptors")


def _raptor(state: PublicSnapshot, actor: str) -> Mapping[str, object]:
    return _mapping(_raptors(state).get(actor), f"raptors.{actor}")


def _yes_no(value: object) -> str:
    if not isinstance(value, bool):
        raise ValueError("MAIN must be a boolean")
    return "yes" if value else "no"


def _raptor_line(state: PublicSnapshot, actor: str) -> str:
    raptor = _raptor(state, actor)
    return (
        f"{actor}: HP={raptor.get('hp')} END={raptor.get('endurance')} "
        f"PM={raptor.get('movement_points')} SCORE={raptor.get('carcass_score')} "
        f"MAIN={_yes_no(raptor.get('main_action_available'))}"
    )


def _grid(state: PublicSnapshot) -> list[str]:
    symbols = [["." for _ in range(_ARENA.columns)] for _ in range(_ARENA.rows)]
    for row, column in _ARENA.walls:
        symbols[row][column] = "/"
    for row, column in _ARENA.mud:
        symbols[row][column] = "~"

    carcasses = _mapping(state.get("carcasses"), "carcasses")
    for carcass_id in ("carcass_left_a", "carcass_left_b"):
        carcass = _mapping(carcasses.get(carcass_id), f"carcasses.{carcass_id}")
        if carcass.get("status") == "available":
            row, column = _CARCASS_POSITIONS[carcass_id]
            symbols[row][column] = "l"
    central = _mapping(carcasses.get("carcass_center"), "carcasses.carcass_center")
    row, column = _CARCASS_POSITIONS["carcass_center"]
    symbols[row][column] = "C" if central.get("status") == "active" else "c"

    for actor in ("A", "B"):
        row, column = _position(_raptor(state, actor).get("position"), f"raptors.{actor}.position")
        symbols[row][column] = actor
    return [" ".join(row) for row in symbols]


def _pending_feed(state: PublicSnapshot) -> str:
    pending = [
        f"{actor} {carcass_id}"
        for actor in ("A", "B")
        if (carcass_id := _raptor(state, actor).get("consumption_pending")) is not None
    ]
    return ", ".join(pending) if pending else "none"


def _pending_rest(state: PublicSnapshot) -> str:
    pending = [actor for actor in ("A", "B") if _raptor(state, actor).get("rest_pending") is True]
    return ", ".join(pending) if pending else "none"


def _last_action(event: ReplayEvent | None) -> str:
    if event is None or event.get("type") != "action_resolved":
        return "none"
    actor = event.get("actor")
    action = event.get("action")
    if not isinstance(actor, str) or not isinstance(action, str):
        return "none"
    target: str | None = None
    effects = event.get("effects")
    if isinstance(effects, list | tuple):
        for effect_value in effects:
            if isinstance(effect_value, Mapping):
                candidate = effect_value.get("target")
                if isinstance(candidate, str):
                    target = candidate
                    break
    return " ".join(value for value in (actor, action, target) if value is not None)


def render_state(
    state: PublicSnapshot,
    *,
    event: ReplayEvent | None = None,
) -> str:
    """Render a complete public state without side effects or ANSI escapes."""

    active_actor = state.get("active_actor")
    if active_actor not in {"A", "B"}:
        raise ValueError("active_actor must be A or B")
    carcasses = _mapping(state.get("carcasses"), "carcasses")
    central = _mapping(carcasses.get("carcass_center"), "carcasses.carcass_center")
    lines = [
        f"Round {state.get('round')} — Turn {state.get('turn')} — Actor {active_actor}",
        _raptor_line(state, "A"),
        _raptor_line(state, "B"),
        "",
        *_grid(state),
        "",
        f"Central carcass: {central.get('status')}",
        f"Pending feed: {_pending_feed(state)}",
        f"Pending rest: {_pending_rest(state)}",
        f"Last action: {_last_action(event)}",
    ]
    return "\n".join(lines) + "\n"
