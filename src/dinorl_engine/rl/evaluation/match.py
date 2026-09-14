"""Real-engine policy matches with in-memory behavioural telemetry and optional replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sb3_contrib import MaskablePPO

from dinorl_engine.controllers.protocol import Controller
from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID, RandomLegalController
from dinorl_engine.controllers.scripted import (
    SCRIPTED_CONTROLLER_IDS,
    create_scripted_controller,
)
from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.events import (
    CarcassPointAwardedEffect,
    ConsumptionInterruptedEffect,
    ConsumptionStartedEffect,
    DamageDealtEffect,
    EnduranceRestoredEffect,
    TargetShovedEffect,
)
from dinorl_engine.core.maps import load_map
from dinorl_engine.match.replay import ReplayControllerDescriptor, ReplayRecorder, replay_sha256
from dinorl_engine.match.runner import MAX_ACTIONS_PER_TURN, MAX_MATCH_ACTIONS, MatchExecutionError
from dinorl_engine.rl.evaluation.metrics import OutcomeCounts
from dinorl_engine.rl.evaluation.policy import MaskablePolicyController
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec, derive_evaluation_seed

__all__ = ["EvaluationGameResult", "run_evaluation_game"]

_ARENA = load_map(MAP_ID)


@dataclass(frozen=True, slots=True)
class EvaluationGameResult:
    """One completed game, retaining aggregates and an optional diagnostic replay only."""

    outcomes: OutcomeCounts
    metrics: dict[str, object]
    replay: Mapping[str, object] | None
    replay_sha256: str | None


class _BehaviourMetrics:
    """Accumulate the complete RL-L7 behaviour catalogue for the learner only."""

    def __init__(self, learner_actor: Actor, opponent_id: str) -> None:
        self._learner = learner_actor
        self._opponent_id = opponent_id
        self._actions = [0] * len(Action)
        self._legal_total = 0
        self._legal_count = 0
        self._bites_attempted = 0
        self._bites_legal = 0
        self._bites_hit = 0
        self._damage_dealt = 0
        self._damage_received = 0
        self._shoves_attempted = 0
        self._shoves_succeeded = 0
        self._shoves_mud = 0
        self._shoves_wall = 0
        self._consumption_attempted = 0
        self._consumption_validated = 0
        self._consumption_interrupted = 0
        self._lateral_points = 0
        self._central_points = 0
        self._rests_started = 0
        self._rests_resolved = 0
        self._movement_points: list[int] = []
        self._endurance: list[int] = []
        self._mud_entries = 0
        self._mud_exits = 0
        self._mud_actions = 0
        self._last_in_mud = False
        self._victory_routes: dict[str, int] = {}

    @staticmethod
    def _raptor_value(state: DinoRLEnv, actor: Actor, attribute: str) -> int:
        value = getattr(state.state.raptor(actor), attribute)
        if type(value) is not int:
            raise RuntimeError(f"engine raptor {attribute} is not an integer")
        return value

    def observe_before_learner_action(
        self, environment: DinoRLEnv, legal_actions: tuple[bool, ...]
    ) -> None:
        self._legal_total += sum(legal_actions)
        self._legal_count += 1
        self._movement_points.append(
            self._raptor_value(environment, self._learner, "movement_points")
        )
        self._endurance.append(self._raptor_value(environment, self._learner, "endurance"))
        in_mud = environment.state.raptor(self._learner).position in _ARENA.mud
        self._mud_actions += int(in_mud)
        self._last_in_mud = in_mud

    def observe_transition(
        self, environment: DinoRLEnv, action: Action, effects: tuple[object, ...]
    ) -> None:
        if action is not Action.END_TURN:
            self._actions[int(action)] += 1
        if action is Action.BITE:
            self._bites_attempted += 1
            self._bites_legal += 1
        if action is Action.SHOVE:
            self._shoves_attempted += 1
        if action is Action.FEED:
            self._consumption_attempted += 1
        if action is Action.REST:
            self._rests_started += 1
        for effect in effects:
            if isinstance(effect, DamageDealtEffect):
                if effect.actor is self._learner:
                    self._damage_dealt += effect.amount
                    if action is Action.BITE:
                        self._bites_hit += 1
                    if action is Action.SHOVE:
                        self._shoves_wall += 1
                elif effect.target is self._learner:
                    self._damage_received += effect.amount
            elif isinstance(effect, TargetShovedEffect) and effect.actor is self._learner:
                self._shoves_succeeded += 1
                if effect.to_position in _ARENA.mud:
                    self._shoves_mud += 1
            elif isinstance(effect, ConsumptionStartedEffect) and effect.actor is self._learner:
                self._consumption_validated += 1
            elif (
                isinstance(effect, ConsumptionInterruptedEffect) and effect.target is self._learner
            ):
                self._consumption_interrupted += 1
            elif isinstance(effect, CarcassPointAwardedEffect) and effect.actor is self._learner:
                if effect.carcass_id == "carcass_center":
                    self._central_points += effect.amount
                else:
                    self._lateral_points += effect.amount
            elif isinstance(effect, EnduranceRestoredEffect) and effect.actor is self._learner:
                self._rests_resolved += 1
        in_mud = environment.state.raptor(self._learner).position in _ARENA.mud
        if in_mud and not self._last_in_mud:
            self._mud_entries += 1
        if self._last_in_mud and not in_mud:
            self._mud_exits += 1
        self._last_in_mud = in_mud

    @staticmethod
    def _summary(values: list[int], final_value: int) -> dict[str, float | int]:
        if not values:
            return {"average": 0.0, "minimum": 0, "final": final_value}
        return {
            "average": sum(values) / len(values),
            "minimum": min(values),
            "final": final_value,
        }

    def finish(self, environment: DinoRLEnv) -> dict[str, object]:
        result = environment.result
        reason = result.reason.name.lower()
        self._victory_routes[reason] = self._victory_routes.get(reason, 0) + 1
        return {
            "action_distribution": {
                action.name.lower(): self._actions[int(action)] for action in Action
            },
            "bites": {
                "attempted": self._bites_attempted,
                "legal": self._bites_legal,
                "hits": self._bites_hit,
            },
            "damage": {"dealt": self._damage_dealt, "received": self._damage_received},
            "shoves": {
                "attempted": self._shoves_attempted,
                "succeeded": self._shoves_succeeded,
                "mud_interrupted": self._shoves_mud,
                "wall_collisions": self._shoves_wall,
            },
            "consumption": {
                "attempted": self._consumption_attempted,
                "validated": self._consumption_validated,
                "interrupted": self._consumption_interrupted,
            },
            "carcass_points": {"lateral": self._lateral_points, "central": self._central_points},
            "rests": {"started": self._rests_started, "resolved": self._rests_resolved},
            "movement_points": self._summary(
                self._movement_points,
                self._raptor_value(environment, self._learner, "movement_points"),
            ),
            "endurance": self._summary(
                self._endurance, self._raptor_value(environment, self._learner, "endurance")
            ),
            "mud": {
                "entries": self._mud_entries,
                "exits": self._mud_exits,
                "learner_actions": self._mud_actions,
            },
            "victory_route": self._victory_routes,
            "opponent_transition_share": {self._opponent_id: 1.0},
            "average_legal_actions": self._legal_total / self._legal_count,
            "rounds": result.rounds_completed,
            "actions": result.actions,
        }


def _opponent_controller(specification: EvaluationGameSpec) -> Controller:
    if specification.opponent_id == RANDOM_LEGAL_CONTROLLER_ID:
        return RandomLegalController(
            seed=derive_evaluation_seed(
                specification.seed,
                suite="opponent-random",
                opponent_id=specification.opponent_id,
                index=int(specification.first_actor),
            )
        )
    if specification.opponent_id not in SCRIPTED_CONTROLLER_IDS:
        raise ValueError("evaluation opponent is not in the versioned controller catalog")
    return create_scripted_controller(specification.opponent_id)


def _replay_descriptor(opponent_id: str) -> ReplayControllerDescriptor:
    if opponent_id in SCRIPTED_CONTROLLER_IDS:
        return ReplayControllerDescriptor("scripted", opponent_id)
    return ReplayControllerDescriptor("sequence", opponent_id)


def run_evaluation_game(
    model: MaskablePPO,
    specification: EvaluationGameSpec,
    *,
    deterministic: bool,
    diagnostic_replay: bool,
) -> EvaluationGameResult:
    """Run one fixed game and keep full transition details only when explicitly requested."""

    environment = DinoRLEnv(map_id=MAP_ID, seed=specification.seed, replay=diagnostic_replay)
    state = environment.reset(first_actor=specification.first_actor)
    learner = MaskablePolicyController(
        model,
        learner_actor=specification.learner_actor,
        first_actor=specification.first_actor,
        deterministic=deterministic,
        stochastic_seed=derive_evaluation_seed(
            specification.seed,
            suite="policy-sampling",
            opponent_id=specification.opponent_id,
            index=int(specification.learner_actor),
        ),
    )
    opponent = _opponent_controller(specification)
    recorder = (
        ReplayRecorder(
            state,
            controller_a=(
                ReplayControllerDescriptor("sequence", "evaluation-policy")
                if specification.learner_actor is Actor.A
                else _replay_descriptor(specification.opponent_id)
            ),
            controller_b=(
                ReplayControllerDescriptor("sequence", "evaluation-policy")
                if specification.learner_actor is Actor.B
                else _replay_descriptor(specification.opponent_id)
            ),
        )
        if diagnostic_replay
        else None
    )
    metrics = _BehaviourMetrics(specification.learner_actor, specification.opponent_id)
    counted_turn = state.turn
    actions_this_turn = 0
    total_actions = 0
    while not environment.is_terminal:
        if environment.state.turn != counted_turn:
            counted_turn = environment.state.turn
            actions_this_turn = 0
        actions_this_turn += 1
        total_actions += 1
        if actions_this_turn > MAX_ACTIONS_PER_TURN or total_actions > MAX_MATCH_ACTIONS:
            raise MatchExecutionError("evaluation match exceeded its engine safety bound")
        active = environment.state.active_actor
        legal_actions = environment.legal_actions()
        controller = learner if active is specification.learner_actor else opponent
        if active is specification.learner_actor:
            metrics.observe_before_learner_action(environment, legal_actions)
        action = controller.choose_action(environment.snapshot_public(), legal_actions)
        transition = environment.step(action)
        metrics.observe_transition(
            environment,
            action if active is specification.learner_actor else Action.END_TURN,
            (*transition.effects, *transition.automatic_effects),
        )
        if recorder is not None:
            recorder.record_action(transition, environment.state)
    winner = environment.result.winner
    outcomes = OutcomeCounts(
        wins=int(winner is specification.learner_actor),
        draws=int(winner == "draw"),
        losses=int(winner is not specification.learner_actor and winner != "draw"),
    )
    replay = None if recorder is None else recorder.finish(environment.result)
    return EvaluationGameResult(
        outcomes=outcomes,
        metrics=metrics.finish(environment),
        replay=replay,
        replay_sha256=None if replay is None else replay_sha256(replay),
    )
