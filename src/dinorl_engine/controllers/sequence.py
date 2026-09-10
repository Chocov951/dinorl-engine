"""Strict local controller driven by a statically validated JSON sequence."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dinorl_engine.controllers.protocol import is_opaque_controller_id
from dinorl_engine.core.actions import Action
from dinorl_engine.core.state import PublicSnapshot
from dinorl_engine.debug.text_renderer import render_state

__all__ = [
    "SequenceController",
    "SequenceExhaustedError",
    "SequenceIllegalActionError",
    "SequenceSchemaError",
    "load_sequence_controller",
    "parse_sequence_controller",
]

type Renderer = Callable[[PublicSnapshot], str]

_SCHEMA_VERSION = "1.0.0"
_ROOT_FIELDS = {"schema_version", "controller_id", "turns"}
_TURN_FIELDS = {"actions"}
_CLOSING_ACTIONS = frozenset({Action.FEED, Action.REST, Action.END_TURN})


class SequenceSchemaError(ValueError):
    """A sequence document failed static validation."""


class SequenceIllegalActionError(RuntimeError):
    """A statically valid action is illegal in the actual match state."""


class SequenceExhaustedError(RuntimeError):
    """A match requested an action after all scripted turns were consumed."""


@dataclass(frozen=True, slots=True)
class _SequenceDefinition:
    controller_id: str
    turns: tuple[tuple[Action, ...], ...]


def _object(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SequenceSchemaError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _exact_fields(value: Mapping[str, object], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise SequenceSchemaError(f"{name} must contain exactly {sorted(fields)!r}")


def _parse_definition(payload: object) -> _SequenceDefinition:
    root = _object(payload, "sequence")
    _exact_fields(root, _ROOT_FIELDS, "sequence")
    if root["schema_version"] != _SCHEMA_VERSION:
        raise SequenceSchemaError("unknown sequence schema_version")
    controller_id = root["controller_id"]
    if not is_opaque_controller_id(controller_id):
        raise SequenceSchemaError("controller_id must be opaque and path-free")
    turns_value = root["turns"]
    if not isinstance(turns_value, list) or not turns_value:
        raise SequenceSchemaError("turns must be a non-empty array")

    turns: list[tuple[Action, ...]] = []
    for turn_index, turn_value in enumerate(turns_value):
        turn = _object(turn_value, f"turns[{turn_index}]")
        _exact_fields(turn, _TURN_FIELDS, f"turns[{turn_index}]")
        actions_value = turn["actions"]
        if not isinstance(actions_value, list) or not actions_value:
            raise SequenceSchemaError(f"turns[{turn_index}].actions must be non-empty")
        actions: list[Action] = []
        for action_index, action_name in enumerate(actions_value):
            if not isinstance(action_name, str) or action_name not in Action.__members__:
                raise SequenceSchemaError(f"turns[{turn_index}].actions[{action_index}] is unknown")
            actions.append(Action[action_name])
        if actions[-1] not in _CLOSING_ACTIONS:
            raise SequenceSchemaError(f"turns[{turn_index}] must end with FEED, REST or END_TURN")
        if any(action in _CLOSING_ACTIONS for action in actions[:-1]):
            raise SequenceSchemaError(
                f"turns[{turn_index}] contains an action after a turn-ending action"
            )
        turns.append(tuple(actions))
    return _SequenceDefinition(controller_id=controller_id, turns=tuple(turns))


def _state_field(state: PublicSnapshot, field: str) -> object:
    value = state.get(field)
    return "unknown" if value is None else value


class SequenceController:
    """Return successive validated actions without fallback behavior."""

    def __init__(
        self,
        definition: _SequenceDefinition,
        *,
        renderer: Renderer = render_state,
    ) -> None:
        self.controller_id = definition.controller_id
        self._turns = definition.turns
        self._renderer = renderer
        self._turn_index = 0
        self._action_index = 0

    def _diagnostic(
        self,
        state: PublicSnapshot,
        legal_actions: tuple[bool, ...],
        *,
        requested_action: str,
        reason: str,
    ) -> str:
        legal = ", ".join(
            action.name
            for action in Action
            if len(legal_actions) == len(Action) and legal_actions[action]
        )
        return (
            f"Controller: {self.controller_id}\n"
            f"Actor: {_state_field(state, 'active_actor')}\n"
            f"Round: {_state_field(state, 'round')}\n"
            f"Turn: {_state_field(state, 'turn')}\n"
            f"Scripted turn index: {self._turn_index}\n"
            f"Action index: {self._action_index}\n"
            f"Requested action: {requested_action}\n"
            f"Legal actions: {legal}\n"
            f"Reason: {reason}\n"
            "State:\n"
            f"{self._renderer(state)}"
        )

    def choose_action(
        self,
        state: PublicSnapshot,
        legal_actions: tuple[bool, ...],
    ) -> Action:
        """Return the next action or stop strictly with a contextual error."""

        if self._turn_index >= len(self._turns):
            raise SequenceExhaustedError(
                "Sequence exhausted\n"
                + self._diagnostic(
                    state,
                    legal_actions,
                    requested_action="none",
                    reason="no scripted turn remains",
                )
            )
        scripted_turn = self._turns[self._turn_index]
        action = scripted_turn[self._action_index]
        if len(legal_actions) != len(Action) or not legal_actions[action]:
            raise SequenceIllegalActionError(
                self._diagnostic(
                    state,
                    legal_actions,
                    requested_action=action.name,
                    reason="engine legal-action mask rejected the action",
                )
            )
        self._action_index += 1
        if self._action_index == len(scripted_turn):
            self._turn_index += 1
            self._action_index = 0
        return action


def parse_sequence_controller(
    payload: object,
    *,
    renderer: Renderer = render_state,
) -> SequenceController:
    """Validate an in-memory JSON value and create its strict controller."""

    return SequenceController(_parse_definition(payload), renderer=renderer)


def load_sequence_controller(
    path: Path,
    *,
    renderer: Renderer = render_state,
) -> SequenceController:
    """Load a local UTF-8 JSON sequence without retaining or exposing its path."""

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SequenceSchemaError(f"cannot read sequence: {error}") from error
    return parse_sequence_controller(payload, renderer=renderer)
