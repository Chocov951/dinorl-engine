"""Deterministic local environment for the pure DinoRL engine."""

from random import Random

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import (
    ACTION_VERSION,
    ENGINE_VERSION,
    OBSERVATION_VERSION,
    REPLAY_VERSION,
    RULES_VERSION,
    Actor,
)
from dinorl_engine.core.errors import IllegalActionError
from dinorl_engine.core.maps import ArenaMap, load_map
from dinorl_engine.core.state import (
    CentralCarcassState,
    GameState,
    LateralCarcassState,
    PublicSnapshot,
    RaptorState,
    snapshot_public,
)

__all__ = ["DinoRLEnv"]

_MAX_SEED = (1 << 63) - 1
_MOVEMENT_DELTAS: dict[Action, tuple[int, int]] = {
    Action.MOVE_NORTH: (-1, 0),
    Action.MOVE_EAST: (0, 1),
    Action.MOVE_SOUTH: (1, 0),
    Action.MOVE_WEST: (0, -1),
}


class DinoRLEnv:
    """Own and mutate the state of one deterministic game."""

    def __init__(self, map_id: str, seed: int, replay: bool = False) -> None:
        if type(seed) is not int or not 0 <= seed <= _MAX_SEED:
            raise ValueError("seed must be an integer between 0 and 2**63 - 1")
        self._arena: ArenaMap = load_map(map_id)
        self.seed = seed
        self.replay = replay
        self._state: GameState | None = None

    @property
    def state(self) -> GameState:
        """Return the initialized state."""

        if self._state is None:
            raise RuntimeError("reset() must be called before accessing state")
        return self._state

    @property
    def is_terminal(self) -> bool:
        """Report whether the initialized game has ended."""

        return self.state.terminal

    def reset(self, first_actor: Actor | None = None) -> GameState:
        """Create and return a fresh initial state."""

        if first_actor is not None and not isinstance(first_actor, Actor):
            raise ValueError("first_actor must be Actor.A, Actor.B or None")
        selected_actor = first_actor
        if selected_actor is None:
            selected_actor = Random(self.seed).choice(tuple(Actor))

        raptors = (
            RaptorState(position=self._arena.spawn_a),
            RaptorState(position=self._arena.spawn_b),
        )
        raptors[selected_actor].movement_points = 3
        self._state = GameState(
            engine_version=ENGINE_VERSION,
            rules_version=RULES_VERSION,
            action_version=ACTION_VERSION,
            observation_version=OBSERVATION_VERSION,
            replay_version=REPLAY_VERSION,
            map_id=self._arena.map_id,
            seed=self.seed,
            first_actor=selected_actor,
            active_actor=selected_actor,
            raptors=raptors,
            lateral_carcasses=(LateralCarcassState(), LateralCarcassState()),
            central_carcass=CentralCarcassState(),
        )
        return self._state

    def snapshot_public(self) -> PublicSnapshot:
        """Return an immutable public view of the current state."""

        return snapshot_public(self.state)

    def _movement_destination(self, action: Action) -> tuple[int, int]:
        actor = self.state.raptor(self.state.active_actor)
        row_delta, column_delta = _MOVEMENT_DELTAS[action]
        return actor.position[0] + row_delta, actor.position[1] + column_delta

    def _movement_is_legal(self, action: Action) -> bool:
        state = self.state
        actor = state.raptor(state.active_actor)
        destination = self._movement_destination(action)
        row, column = destination
        if not (0 <= row < self._arena.rows and 0 <= column < self._arena.columns):
            return False
        if destination in self._arena.walls:
            return False
        opponent = state.raptor(Actor.B if state.active_actor is Actor.A else Actor.A)
        if destination == opponent.position:
            return False
        movement_cost = 2 if actor.position in self._arena.mud else 1
        return actor.movement_points >= movement_cost

    def legal_actions(self) -> tuple[bool, ...]:
        """Return a fixed-size mask for the actions implemented at the current gate."""

        legal = [False] * len(Action)
        if self.state.terminal:
            return tuple(legal)
        for action in _MOVEMENT_DELTAS:
            legal[action] = self._movement_is_legal(action)
        legal[Action.END_TURN] = True
        return tuple(legal)

    def _end_turn(self) -> None:
        state = self.state
        outgoing_actor = state.active_actor
        state.raptor(outgoing_actor).movement_points = 0
        if outgoing_actor is not state.first_actor:
            state.round += 1
        state.active_actor = Actor.B if outgoing_actor is Actor.A else Actor.A
        state.turn += 1
        incoming_raptor = state.raptor(state.active_actor)
        incoming_raptor.movement_points = 3
        incoming_raptor.main_action_available = True
        incoming_raptor.voluntary_move_done = False

    def step(self, action: Action) -> GameState:
        """Apply one legal atomic action and return the current internal state."""

        if not isinstance(action, Action) or not self.legal_actions()[action]:
            raise IllegalActionError(f"Action is not legal: {action!r}")

        if action is Action.END_TURN:
            self._end_turn()
            return self.state

        actor = self.state.raptor(self.state.active_actor)
        movement_cost = 2 if actor.position in self._arena.mud else 1
        actor.position = self._movement_destination(action)
        actor.movement_points -= movement_cost
        actor.voluntary_move_done = True
        return self.state
