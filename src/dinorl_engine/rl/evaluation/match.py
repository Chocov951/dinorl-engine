"""Real-engine policy matches with in-memory behavioural telemetry and optional replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

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
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec, derive_game_stream_seed
from dinorl_engine.rl.rewards.runtime import CompiledReward
from dinorl_engine.rl.rewards.transition import public_reward_transition
from dinorl_engine.rl.s5c.a3 import EpisodeAuxiliaryBudget, bounded_safe_feed_adjustment
from dinorl_engine.rl.s5c.evidence import analyse_shove_transition
from dinorl_engine.rl.s5c.rewards import specialist_reward_terms

__all__ = ["EvaluationGameResult", "run_evaluation_game"]

_ARENA = load_map(MAP_ID)


@dataclass(frozen=True, slots=True)
class EvaluationGameResult:
    """One completed game, retaining aggregates and an optional diagnostic replay only."""

    outcomes: OutcomeCounts
    metrics: dict[str, object]
    replay: Mapping[str, object] | None
    replay_sha256: str | None
    record: Mapping[str, object] = field(default_factory=dict)


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
        self._distances: list[int] = []
        self._pending_distance = 0
        self._pending_feed_legal = False
        self._feed_opportunities_ignored = 0
        self._pursuit_moves_without_attack = 0
        self._actions_without_tactical_event = 0
        self._first_damage_round: int | None = None
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
        learner_position = environment.state.raptor(self._learner).position
        opponent = Actor.B if self._learner is Actor.A else Actor.A
        opponent_position = environment.state.raptor(opponent).position
        self._pending_distance = abs(learner_position[0] - opponent_position[0]) + abs(
            learner_position[1] - opponent_position[1]
        )
        self._distances.append(self._pending_distance)
        self._pending_feed_legal = legal_actions[Action.FEED]
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
        if self._pending_feed_legal and action is not Action.FEED:
            self._feed_opportunities_ignored += 1
        opponent = Actor.B if self._learner is Actor.A else Actor.A
        learner_position = environment.state.raptor(self._learner).position
        opponent_position = environment.state.raptor(opponent).position
        distance_after = abs(learner_position[0] - opponent_position[0]) + abs(
            learner_position[1] - opponent_position[1]
        )
        if action.name.startswith("MOVE_") and distance_after < self._pending_distance:
            self._pursuit_moves_without_attack += 1
        tactical = False
        for effect in effects:
            if isinstance(effect, DamageDealtEffect):
                tactical = True
                if self._first_damage_round is None:
                    self._first_damage_round = environment.state.round
                if effect.actor is self._learner:
                    self._damage_dealt += effect.amount
                    if action is Action.BITE:
                        self._bites_hit += 1
                    if action is Action.SHOVE:
                        self._shoves_wall += 1
                elif effect.target is self._learner:
                    self._damage_received += effect.amount
            elif isinstance(effect, TargetShovedEffect) and effect.actor is self._learner:
                tactical = True
                self._shoves_succeeded += 1
                if effect.to_position in _ARENA.mud:
                    self._shoves_mud += 1
            elif isinstance(effect, ConsumptionStartedEffect) and effect.actor is self._learner:
                tactical = True
                self._consumption_validated += 1
            elif (
                isinstance(effect, ConsumptionInterruptedEffect) and effect.target is self._learner
            ):
                tactical = True
                self._consumption_interrupted += 1
            elif isinstance(effect, CarcassPointAwardedEffect) and effect.actor is self._learner:
                tactical = True
                if effect.carcass_id == "carcass_center":
                    self._central_points += effect.amount
                else:
                    self._lateral_points += effect.amount
            elif isinstance(effect, EnduranceRestoredEffect) and effect.actor is self._learner:
                self._rests_resolved += 1
        self._actions_without_tactical_event += int(not tactical)
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
            "predator_diagnostic": {
                "average_distance": (
                    sum(self._distances) / len(self._distances) if self._distances else 0.0
                ),
                "initial_endurance": self._endurance[0] if self._endurance else 0,
                "final_endurance": self._raptor_value(environment, self._learner, "endurance"),
                "pursuit_moves_without_attack": self._pursuit_moves_without_attack,
                "feed_opportunities_ignored": self._feed_opportunities_ignored,
                "actions_without_tactical_event": self._actions_without_tactical_event,
                "first_damage_round": self._first_damage_round,
            },
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


def _opponent_controller(specification: EvaluationGameSpec, *, game_id: str) -> Controller:
    if specification.opponent_id == RANDOM_LEGAL_CONTROLLER_ID:
        return RandomLegalController(
            seed=derive_game_stream_seed(game_id, policy_id=f"opponent:{specification.opponent_id}")
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
    learner_policy_id: str = "evaluation-policy",
    checkpoint_id: str = "unspecified-checkpoint",
    checkpoint_sha256: str | None = None,
    configuration_sha256: str | None = None,
    training_seed: int = 0,
    reward_program: CompiledReward | None = None,
    safe_feed_episode_cap: float | None = None,
) -> EvaluationGameResult:
    """Run one fixed game and keep full transition details only when explicitly requested."""

    game_id = specification.game_id or (
        f"legacy/{specification.seed}/{specification.opponent_id}/"
        f"{specification.learner_actor.name}/{specification.first_actor.name}"
    )
    engine_seed = (
        specification.seed
        if not specification.game_id
        else derive_game_stream_seed(specification.pair_id or game_id, policy_id="engine")
    )
    environment = DinoRLEnv(map_id=MAP_ID, seed=engine_seed, replay=diagnostic_replay)
    state = environment.reset(first_actor=specification.first_actor)
    learner = MaskablePolicyController(
        model,
        learner_actor=specification.learner_actor,
        first_actor=specification.first_actor,
        deterministic=deterministic,
        stochastic_seed=derive_game_stream_seed(game_id, policy_id=f"learner:{learner_policy_id}"),
    )
    opponent = _opponent_controller(specification, game_id=game_id)
    recorder = (
        ReplayRecorder(
            state,
            controller_a=(
                ReplayControllerDescriptor("sequence", learner_policy_id)
                if specification.learner_actor is Actor.A
                else _replay_descriptor(specification.opponent_id)
            ),
            controller_b=(
                ReplayControllerDescriptor("sequence", learner_policy_id)
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
    action_trace: list[str] = []
    event_records: list[dict[str, object]] = []
    terminal_return = 0.0
    auxiliary_return = 0.0
    auxiliary_by_rule: dict[str, float] = {}
    safe_feed_budget = (
        EpisodeAuxiliaryBudget(safe_feed_episode_cap) if safe_feed_episode_cap is not None else None
    )
    resource_generations: dict[str, int] = {}
    style_events = {
        "move": 0,
        "bite_attempted": 0,
        "bite_hit": 0,
        "bite_missed": 0,
        "damage_dealt": 0,
        "damage_received": 0,
        "feed_started": 0,
        "feed_completed": 0,
        "feed_interrupted": 0,
        "carcass_point": 0,
        "rest": 0,
        "shove_legal_opportunity": 0,
        "shove_attempted": 0,
        "shove_legal": 0,
        "shove_succeeded": 0,
        "shove_distance": 0,
        "shove_wall_collision": 0,
        "shove_mud_interrupted": 0,
        "shove_into_mud": 0,
        "shove_feed_interrupted": 0,
        "shove_away_from_active_carcass": 0,
        "shove_favourable_carcass_access": 0,
        "useful_shove": 0,
    }
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
        if active is specification.learner_actor and legal_actions[Action.SHOVE]:
            style_events["shove_legal_opportunity"] += 1
        controller = learner if active is specification.learner_actor else opponent
        if active is specification.learner_actor:
            metrics.observe_before_learner_action(environment, legal_actions)
        before = environment.snapshot_public()
        action = controller.choose_action(before, legal_actions)
        transition = environment.step(action)
        after = environment.snapshot_public()
        action_trace.append(f"{active.name}:{action.name}")
        effects = (*transition.effects, *transition.automatic_effects)
        event_records.append(
            {
                "actor": active.name,
                "action": action.name,
                "round": int(before["round"]),
                "turn": int(before["turn"]),
                "effects": [
                    {
                        key: (value.name if isinstance(value, Actor) else list(value))
                        if isinstance(value, tuple | Actor)
                        else value
                        for key, value in asdict(effect).items()
                    }
                    for effect in effects
                ],
            }
        )
        if active is specification.learner_actor:
            if action.name.startswith("MOVE_"):
                style_events["move"] += 1
            elif action is Action.BITE:
                style_events["bite_attempted"] += 1
                hit = any(
                    isinstance(effect, DamageDealtEffect)
                    and effect.actor is specification.learner_actor
                    for effect in effects
                )
                style_events["bite_hit" if hit else "bite_missed"] += 1
            elif action is Action.FEED:
                style_events["feed_started"] += 1
            elif action is Action.REST:
                style_events["rest"] += 1
            elif action is Action.SHOVE:
                shove = analyse_shove_transition(
                    before=before,
                    after=after,
                    transition=transition,
                    learner_actor=specification.learner_actor,
                    shove_was_legal=legal_actions[Action.SHOVE],
                )
                for source, target in (
                    ("attempted", "shove_attempted"),
                    ("legal", "shove_legal"),
                    ("succeeded", "shove_succeeded"),
                    ("distance", "shove_distance"),
                    ("wall_collision", "shove_wall_collision"),
                    ("mud_interrupted", "shove_mud_interrupted"),
                    ("pushed_into_mud", "shove_into_mud"),
                    ("feed_interrupted", "shove_feed_interrupted"),
                    ("away_from_active_carcass", "shove_away_from_active_carcass"),
                    ("favourable_carcass_access", "shove_favourable_carcass_access"),
                    ("useful", "useful_shove"),
                ):
                    style_events[target] += shove[source]
        for effect in effects:
            if isinstance(effect, DamageDealtEffect):
                target = (
                    "damage_dealt"
                    if effect.actor is specification.learner_actor
                    else "damage_received"
                )
                style_events[target] += effect.amount
            elif isinstance(effect, ConsumptionStartedEffect) and (
                effect.actor is specification.learner_actor
            ):
                style_events["feed_started"] += int(action is not Action.FEED)
            elif isinstance(effect, ConsumptionInterruptedEffect):
                style_events["feed_interrupted"] += int(effect.actor is specification.learner_actor)
            elif isinstance(effect, CarcassPointAwardedEffect) and (
                effect.actor is specification.learner_actor
            ):
                style_events["feed_completed"] += effect.amount
                style_events["carcass_point"] += effect.amount
        if reward_program is not None:
            reward_transition = public_reward_transition(
                before=before,
                after=after,
                transition=transition,
                learner_actor=specification.learner_actor,
                first_actor=specification.first_actor,
                result=environment.result if environment.is_terminal else None,
                legal_actions=legal_actions,
            )
            value, terms = reward_program.evaluate_vm_terms(reward_transition)
            safe_feed_adjustment = 0.0
            if safe_feed_budget is not None:
                safe_feed_adjustment = bounded_safe_feed_adjustment(
                    transition=transition,
                    reward_transition=reward_transition,
                    learner_actor=specification.learner_actor,
                    budget=safe_feed_budget,
                    generations=resource_generations,
                )
                value += safe_feed_adjustment
            terminal = {
                "WIN": 1.0,
                "LOSS": -1.0,
                "DRAW": 0.0,
                "ONGOING": 0.0,
            }[str(reward_transition["outcome"])]
            terminal_return += terminal
            auxiliary = value - terminal
            auxiliary_return += auxiliary
            root_terms = specialist_reward_terms(reward_program, reward_transition)
            if root_terms and safe_feed_budget is not None:
                root_terms["safe_feed"] = root_terms.get("safe_feed", 0.0) + safe_feed_adjustment
            if not root_terms:
                root_terms = {
                    name: number
                    for name, number in terms.items()
                    if name.startswith("reward.") and name != "reward.terminal"
                }
            if root_terms:
                for name, number in root_terms.items():
                    auxiliary_by_rule[name] = auxiliary_by_rule.get(name, 0.0) + number
            elif auxiliary:
                auxiliary_by_rule["unattributed"] = (
                    auxiliary_by_rule.get("unattributed", 0.0) + auxiliary
                )
        metrics.observe_transition(
            environment,
            action if active is specification.learner_actor else Action.END_TURN,
            effects,
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
    if replay is not None:
        replay = {
            **replay,
            "game_id": game_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_sha256": checkpoint_sha256,
            "policy_id": learner_policy_id,
            "reward_sha256": None if reward_program is None else reward_program.cache_key,
            "configuration_sha256": configuration_sha256,
            "configuration": {
                "deterministic": deterministic,
                "evaluation_seed": specification.evaluation_seed,
                "training_seed": training_seed,
            },
        }
    outcome = "win" if outcomes.wins else "draw" if outcomes.draws else "loss"
    reason = environment.result.reason.name.lower()
    metric_summary = metrics.finish(environment)
    learner_state = environment.state.raptor(specification.learner_actor)
    opponent_actor = Actor.B if specification.learner_actor is Actor.A else Actor.A
    opponent_state = environment.state.raptor(opponent_actor)
    record: dict[str, object] = {
        "format": "s5c-a2-game-v1",
        "game_id": game_id,
        "pair_id": specification.pair_id,
        "engine_seed": engine_seed,
        "evaluation_seed": (
            specification.evaluation_seed
            if specification.evaluation_seed is not None
            else specification.seed
        ),
        "training_seed": training_seed,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "configuration_sha256": configuration_sha256,
        "learner_policy_id": learner_policy_id,
        "opponent_id": specification.opponent_id,
        "learner_role": (
            "first" if specification.learner_actor is specification.first_actor else "second"
        ),
        "learner_side": specification.learner_actor.name,
        "first_actor": specification.first_actor.name,
        "outcome": outcome,
        "terminal_reason": reason,
        "victory_mode": reason,
        "rounds": environment.result.rounds_completed,
        "actions": environment.result.actions,
        "score": 1.0 if outcomes.wins else 0.5 if outcomes.draws else 0.0,
        "final_score": {
            "learner_carcass": learner_state.carcass_score,
            "opponent_carcass": opponent_state.carcass_score,
            "learner_hp": learner_state.hp,
            "opponent_hp": opponent_state.hp,
        },
        "terminal_return": float(terminal_return),
        "auxiliary_return": float(auxiliary_return),
        "auxiliary_by_rule": auxiliary_by_rule,
        "reward_sha256": None if reward_program is None else reward_program.cache_key,
        "action_distribution": metric_summary["action_distribution"],
        "predator_diagnostic": metric_summary["predator_diagnostic"],
        "style_events": style_events,
        "events": event_records,
        "action_trace": action_trace,
    }
    return EvaluationGameResult(
        outcomes=outcomes,
        metrics=metric_summary,
        replay=replay,
        replay_sha256=None if replay is None else replay_sha256(replay),
        record=record,
    )
