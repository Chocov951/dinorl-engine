"""Deterministic local environment for the pure DinoRL engine."""

from random import Random
from typing import cast

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import (
    ACTION_VERSION,
    ENGINE_VERSION,
    OBSERVATION_VERSION,
    REPLAY_VERSION,
    RULES_VERSION,
    Actor,
    EndReason,
)
from dinorl_engine.core.errors import IllegalActionError
from dinorl_engine.core.events import (
    ActionCost,
    ActionTransition,
    ActorMovedEffect,
    CarcassConsumedEffect,
    CarcassPointAwardedEffect,
    CentralReactivatedEffect,
    CentralRechargeStartedEffect,
    ConsumptionInterruptedEffect,
    ConsumptionStartedEffect,
    DamageDealtEffect,
    Effect,
    EnduranceRestoredEffect,
    RestStartedEffect,
    TargetShovedEffect,
    TurnEndedEffect,
)
from dinorl_engine.core.maps import ArenaMap, load_map
from dinorl_engine.core.state import (
    CarcassId,
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

    def _opponent(self, actor: Actor) -> Actor:
        return Actor.B if actor is Actor.A else Actor.A

    def _actors_are_adjacent(self) -> bool:
        state = self.state
        attacker = state.raptor(state.active_actor)
        target = state.raptor(self._opponent(state.active_actor))
        row_distance = abs(attacker.position[0] - target.position[0])
        column_distance = abs(attacker.position[1] - target.position[1])
        return row_distance + column_distance == 1

    def _bite_is_legal(self) -> bool:
        attacker = self.state.raptor(self.state.active_actor)
        return (
            attacker.main_action_available
            and self._actors_are_adjacent()
            and attacker.movement_points >= 1
            and attacker.endurance >= 2
        )

    def _shove_is_legal(self) -> bool:
        attacker = self.state.raptor(self.state.active_actor)
        return (
            attacker.main_action_available
            and self._actors_are_adjacent()
            and attacker.movement_points >= 1
            and attacker.endurance >= 1
        )

    def _rest_is_legal(self) -> bool:
        actor = self.state.raptor(self.state.active_actor)
        return actor.main_action_available and not actor.voluntary_move_done

    def _carcass_at_active_actor(self) -> CarcassId | None:
        position = self.state.raptor(self.state.active_actor).position
        for carcass in self._arena.carcasses:
            if carcass.position == position:
                return cast(CarcassId, carcass.carcass_id)
        return None

    def _feed_is_legal(self) -> bool:
        actor = self.state.raptor(self.state.active_actor)
        carcass_id = self._carcass_at_active_actor()
        if not actor.main_action_available or actor.endurance < 2 or carcass_id is None:
            return False
        if carcass_id == "carcass_center":
            return self.state.central_carcass.status == "active"
        lateral_index = 0 if carcass_id == "carcass_left_a" else 1
        return self.state.lateral_carcasses[lateral_index].status == "available"

    def legal_actions(self) -> tuple[bool, ...]:
        """Return a fixed-size mask for the actions implemented at the current gate."""

        legal = [False] * len(Action)
        if self.state.terminal:
            return tuple(legal)
        for action in _MOVEMENT_DELTAS:
            legal[action] = self._movement_is_legal(action)
        legal[Action.BITE] = self._bite_is_legal()
        legal[Action.SHOVE] = self._shove_is_legal()
        legal[Action.FEED] = self._feed_is_legal()
        legal[Action.REST] = self._rest_is_legal()
        legal[Action.END_TURN] = True
        return tuple(legal)

    def _start_turn(self) -> tuple[Effect, ...]:
        actor = self.state.raptor(self.state.active_actor)
        effects: list[Effect] = []
        carcass_id = actor.consumption_pending
        if carcass_id is not None:
            actor.carcass_score += 1
            actor.consumption_pending = None
            if carcass_id == "carcass_center":
                central = self.state.central_carcass
                central.status = "recharging"
                central.pending_actor = None
                central.reactivate_on_turn = self.state.turn + 2
            else:
                lateral_index = 0 if carcass_id == "carcass_left_a" else 1
                lateral = self.state.lateral_carcasses[lateral_index]
                lateral.status = "consumed"
                lateral.pending_actor = None
            effects.append(
                CarcassPointAwardedEffect(
                    actor=self.state.active_actor,
                    carcass_id=carcass_id,
                    amount=1,
                )
            )
            if carcass_id != "carcass_center":
                effects.append(
                    CarcassConsumedEffect(
                        actor=self.state.active_actor,
                        carcass_id=carcass_id,
                    )
                )
            else:
                effects.append(CentralRechargeStartedEffect(actor=self.state.active_actor))
        if actor.rest_pending:
            restored = 5 - actor.endurance
            actor.endurance = 5
            actor.rest_pending = False
            effects.append(EnduranceRestoredEffect(actor=self.state.active_actor, amount=restored))
        central = self.state.central_carcass
        if central.status == "recharging" and central.reactivate_on_turn == self.state.turn:
            central.status = "active"
            central.reactivate_on_turn = None
            effects.append(CentralReactivatedEffect(actor=self.state.active_actor))
        actor.movement_points = 3
        actor.main_action_available = True
        actor.voluntary_move_done = False
        return tuple(effects)

    def _end_turn(self) -> tuple[Effect, ...]:
        state = self.state
        outgoing_actor = state.active_actor
        state.raptor(outgoing_actor).movement_points = 0
        if outgoing_actor is not state.first_actor:
            state.round += 1
        state.active_actor = Actor.B if outgoing_actor is Actor.A else Actor.A
        state.turn += 1
        return self._start_turn()

    def _interrupt_consumption(
        self, actor: Actor, target_actor: Actor
    ) -> ConsumptionInterruptedEffect | None:
        target = self.state.raptor(target_actor)
        carcass_id = target.consumption_pending
        if carcass_id is None:
            return None
        target.consumption_pending = None
        if carcass_id == "carcass_center":
            central = self.state.central_carcass
            central.status = "active"
            central.pending_actor = None
            central.reactivate_on_turn = None
        else:
            lateral_index = 0 if carcass_id == "carcass_left_a" else 1
            lateral = self.state.lateral_carcasses[lateral_index]
            lateral.status = "available"
            lateral.pending_actor = None
        return ConsumptionInterruptedEffect(
            actor=actor,
            target=target_actor,
            carcass_id=carcass_id,
        )

    def _bite(self) -> ActionTransition:
        state = self.state
        actor = state.active_actor
        target_actor = self._opponent(actor)
        attacker = state.raptor(actor)
        target = state.raptor(target_actor)
        state.claim_main_action()
        attacker.movement_points -= 1
        attacker.endurance -= 2
        target.hp = max(0, target.hp - 2)
        effects: list[DamageDealtEffect | ConsumptionInterruptedEffect] = [
            DamageDealtEffect(actor=actor, target=target_actor, amount=2)
        ]
        interruption = self._interrupt_consumption(actor, target_actor)
        if interruption is not None:
            effects.append(interruption)
        if target.hp == 0:
            state.terminal = True
            state.winner = actor
            state.end_reason = EndReason.KO
        return ActionTransition(
            action=Action.BITE,
            actor=actor,
            cost=ActionCost(movement=1, endurance=2),
            effects=tuple(effects),
            turn_ended=False,
        )

    def _shove(self) -> ActionTransition:
        state = self.state
        actor = state.active_actor
        target_actor = self._opponent(actor)
        attacker = state.raptor(actor)
        target = state.raptor(target_actor)
        state.claim_main_action()
        attacker.movement_points -= 1
        attacker.endurance -= 1
        previous_position = target.position
        row_delta = target.position[0] - attacker.position[0]
        column_delta = target.position[1] - attacker.position[1]
        collision = False
        for _ in range(2):
            next_position = (
                target.position[0] + row_delta,
                target.position[1] + column_delta,
            )
            row, column = next_position
            if (
                not (0 <= row < self._arena.rows and 0 <= column < self._arena.columns)
                or next_position in self._arena.walls
            ):
                collision = True
                break
            target.position = next_position
            if target.position in self._arena.mud:
                break
        effects: list[TargetShovedEffect | DamageDealtEffect | ConsumptionInterruptedEffect] = [
            TargetShovedEffect(
                actor=actor,
                target=target_actor,
                from_position=previous_position,
                to_position=target.position,
            )
        ]
        if collision:
            target.hp = max(0, target.hp - 1)
            effects.append(DamageDealtEffect(actor=actor, target=target_actor, amount=1))
        interruption = self._interrupt_consumption(actor, target_actor)
        if interruption is not None:
            effects.append(interruption)
        if target.hp == 0:
            state.terminal = True
            state.winner = actor
            state.end_reason = EndReason.KO
        return ActionTransition(
            action=Action.SHOVE,
            actor=actor,
            cost=ActionCost(movement=1, endurance=1),
            effects=tuple(effects),
            turn_ended=False,
        )

    def _rest(self) -> ActionTransition:
        actor = self.state.active_actor
        self.state.claim_main_action()
        self.state.raptor(actor).rest_pending = True
        automatic_effects = self._end_turn()
        return ActionTransition(
            action=Action.REST,
            actor=actor,
            cost=ActionCost(movement=0, endurance=0),
            effects=(RestStartedEffect(actor=actor), TurnEndedEffect(actor=actor)),
            turn_ended=True,
            automatic_effects=automatic_effects,
        )

    def _feed(self) -> ActionTransition:
        actor_id = self.state.active_actor
        actor = self.state.raptor(actor_id)
        carcass_id = self._carcass_at_active_actor()
        if carcass_id is None:
            raise RuntimeError("legal FEED action has no carcass under active actor")
        self.state.claim_main_action()
        actor.endurance -= 2
        actor.consumption_pending = carcass_id
        if carcass_id == "carcass_center":
            central = self.state.central_carcass
            central.status = "pending"
            central.pending_actor = actor_id
        else:
            lateral_index = 0 if carcass_id == "carcass_left_a" else 1
            lateral = self.state.lateral_carcasses[lateral_index]
            lateral.status = "pending"
            lateral.pending_actor = actor_id
        automatic_effects = self._end_turn()
        return ActionTransition(
            action=Action.FEED,
            actor=actor_id,
            cost=ActionCost(movement=0, endurance=2),
            effects=(
                ConsumptionStartedEffect(actor=actor_id, carcass_id=carcass_id),
                TurnEndedEffect(actor=actor_id),
            ),
            turn_ended=True,
            automatic_effects=automatic_effects,
        )

    def step(self, action: Action) -> ActionTransition:
        """Apply one legal atomic action and return its immutable transition."""

        if not isinstance(action, Action) or not self.legal_actions()[action]:
            raise IllegalActionError(f"Action is not legal: {action!r}")

        actor_id = self.state.active_actor
        if action is Action.END_TURN:
            automatic_effects = self._end_turn()
            return ActionTransition(
                action=action,
                actor=actor_id,
                cost=ActionCost(movement=0, endurance=0),
                effects=(TurnEndedEffect(actor=actor_id),),
                turn_ended=True,
                automatic_effects=automatic_effects,
            )

        if action is Action.BITE:
            return self._bite()

        if action is Action.SHOVE:
            return self._shove()

        if action is Action.FEED:
            return self._feed()

        if action is Action.REST:
            return self._rest()

        actor = self.state.raptor(actor_id)
        previous_position = actor.position
        movement_cost = 2 if actor.position in self._arena.mud else 1
        actor.position = self._movement_destination(action)
        actor.movement_points -= movement_cost
        actor.voluntary_move_done = True
        return ActionTransition(
            action=action,
            actor=actor_id,
            cost=ActionCost(movement=movement_cost, endurance=0),
            effects=(
                ActorMovedEffect(
                    actor=actor_id,
                    from_position=previous_position,
                    to_position=actor.position,
                ),
            ),
            turn_ended=False,
        )
