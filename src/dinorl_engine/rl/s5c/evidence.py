"""Auditable per-game aggregation and shove diagnostics for RL-S5c-A2."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.events import (
    ActionTransition,
    ConsumptionInterruptedEffect,
    DamageDealtEffect,
    TargetShovedEffect,
)
from dinorl_engine.core.maps import load_map

__all__ = [
    "aggregate_game_records",
    "analyse_shove_transition",
    "classify_reward_hacking",
    "controller_zero_shove_failure",
    "select_diagnostic_records",
    "validate_absolute_style",
]

_ARENA = load_map(MAP_ID)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _summary(records: list[Mapping[str, object]]) -> dict[str, object]:
    outcomes = [str(record["outcome"]) for record in records]
    wins = outcomes.count("win")
    draws = outcomes.count("draw")
    losses = outcomes.count("loss")
    games = len(records)
    auxiliary_by_rule: dict[str, float] = {}
    style_events: dict[str, int] = {}
    action_distribution: dict[str, int] = {}
    for record in records:
        for name, value in _mapping(record.get("auxiliary_by_rule")).items():
            auxiliary_by_rule[name] = auxiliary_by_rule.get(name, 0.0) + float(value)
        for name, value in _mapping(record.get("style_events")).items():
            style_events[name] = style_events.get(name, 0) + int(value)
        for name, value in _mapping(record.get("action_distribution")).items():
            action_distribution[name] = action_distribution.get(name, 0) + int(value)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / games if games else 0.0,
        "round_limit_rate": (
            sum(record.get("victory_mode") == "round_limit" for record in records) / games
            if games
            else 0.0
        ),
        "average_rounds": _mean([float(record["rounds"]) for record in records]),
        "average_terminal_return": _mean([float(record["terminal_return"]) for record in records]),
        "average_auxiliary_return": _mean(
            [float(record["auxiliary_return"]) for record in records]
        ),
        "auxiliary_by_rule": auxiliary_by_rule,
        "style_events": style_events,
        "action_distribution": action_distribution,
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _group(records: list[Mapping[str, object]], field: str) -> dict[str, dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        groups.setdefault(str(record[field]), []).append(record)
    return {name: _summary(values) for name, values in sorted(groups.items())}


def aggregate_game_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Produce every mandatory A2 conditioning from immutable per-game rows."""

    materialized = list(records)
    if not materialized:
        raise ValueError("at least one game record is required")
    return {
        "global": _summary(materialized),
        "by_training_seed": _group(materialized, "training_seed"),
        "by_evaluation_seed": _group(materialized, "evaluation_seed"),
        "by_role": _group(materialized, "learner_role"),
        "by_side": _group(materialized, "learner_side"),
        "by_outcome": _group(materialized, "outcome"),
        "by_victory_mode": _group(materialized, "victory_mode"),
        "by_opponent": _group(materialized, "opponent_id"),
    }


def classify_reward_hacking(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Flag outcome inversion, dominant terms, loss-only style, and round limits."""

    materialized = list(records)
    if not materialized:
        raise ValueError("reward-hacking analysis requires game records")
    warnings: list[str] = []
    by_outcome = {
        outcome: [record for record in materialized if record.get("outcome") == outcome]
        for outcome in ("win", "draw", "loss")
    }
    total_return = {
        outcome: _mean(
            [
                float(record["terminal_return"]) + float(record["auxiliary_return"])
                for record in values
            ]
        )
        for outcome, values in by_outcome.items()
    }
    if by_outcome["win"] and by_outcome["loss"] and total_return["loss"] > total_return["win"]:
        warnings.append("loss_total_exceeds_win")
    if (
        sum(record.get("victory_mode") == "round_limit" for record in materialized)
        / len(materialized)
        > 0.10
    ):
        warnings.append("round_limit_above_10_percent")
    rule_totals: dict[str, float] = {}
    for record in materialized:
        for name, value in _mapping(record.get("auxiliary_by_rule")).items():
            rule_totals[name] = rule_totals.get(name, 0.0) + abs(float(value))
    total_rules = sum(rule_totals.values())
    if total_rules and max(rule_totals.values()) / total_rules > 0.90:
        warnings.append("single_auxiliary_rule_dominates")
    action_auxiliary: dict[str, float] = {}
    action_counts: dict[str, int] = {}
    for record in materialized:
        auxiliary = abs(float(record.get("auxiliary_return", 0.0)))
        distribution = _mapping(record.get("action_distribution"))
        for name, value in distribution.items():
            count = int(value)
            action_counts[name] = action_counts.get(name, 0) + count
            if count:
                action_auxiliary[name] = action_auxiliary.get(name, 0.0) + auxiliary
    if total_rules and action_auxiliary:
        dominant = max(action_auxiliary, key=action_auxiliary.get)  # type: ignore[arg-type]
        if action_counts.get(dominant, 0) > 0 and action_auxiliary[dominant] / total_rules > 0.90:
            warnings.append("repeatable_action_reward_risk")
    style_by_outcome: dict[str, int] = {}
    for outcome, values in by_outcome.items():
        style_by_outcome[outcome] = sum(
            sum(int(value) for value in _mapping(record.get("style_events")).values())
            for record in values
        )
    if style_by_outcome["loss"] > style_by_outcome["win"] + style_by_outcome["draw"]:
        warnings.append("style_signature_mainly_in_losses")
    return {
        "format": "s5c-a2-reward-hacking-v1",
        "average_total_return_by_outcome": total_return,
        "auxiliary_absolute_by_rule": rule_totals,
        "style_event_total_by_outcome": style_by_outcome,
        "warnings": warnings,
    }


def select_diagnostic_records(
    records: Iterable[Mapping[str, object]], *, ordinary_sample: int
) -> dict[str, list[Mapping[str, object]]]:
    """Select deterministic evidence categories without re-running a policy."""

    materialized = sorted(records, key=lambda record: str(record["game_id"]))
    if type(ordinary_sample) is not int or ordinary_sample < 0:
        raise ValueError("ordinary_sample must be a non-negative integer")
    by_auxiliary = sorted(
        materialized, key=lambda record: float(record["auxiliary_return"]), reverse=True
    )
    return {
        "highest_auxiliary": by_auxiliary[: max(1, ordinary_sample)],
        "loss_high_auxiliary": [
            record for record in by_auxiliary if record.get("outcome") == "loss"
        ][: max(1, ordinary_sample)],
        "round_limit": [
            record for record in materialized if record.get("victory_mode") == "round_limit"
        ][: max(1, ordinary_sample)],
        "deterministic_failures": [
            record
            for record in materialized
            if record.get("opponent_id") != "random-legal-v1" and record.get("outcome") == "loss"
        ][: max(1, ordinary_sample) * 3],
        "style_representative": sorted(
            materialized,
            key=lambda record: sum(
                int(value) for value in _mapping(record.get("style_events")).values()
            ),
            reverse=True,
        )[: max(1, ordinary_sample)],
        "prudent_failures": [
            record
            for record in materialized
            if record.get("opponent_id") == "prudent-v1" and record.get("outcome") != "win"
        ][: max(1, ordinary_sample)],
        "rewarded_shoves": [
            record
            for record in materialized
            if float(_mapping(record.get("auxiliary_by_rule")).get("useful_shove", 0.0)) > 0
        ][: max(1, ordinary_sample)],
        "ordinary": materialized[:ordinary_sample],
    }


def controller_zero_shove_failure(
    observations: Mapping[int, Mapping[int, tuple[int, int]]],
) -> bool:
    """Stop only when all three seeds had legal chances but zero attempts at 5 and 10."""

    if set(observations) != {19, 20, 21}:
        return False
    for checkpoints in observations.values():
        if set(checkpoints) < {5, 10}:
            return False
        for unit in (5, 10):
            attempts, opportunities = checkpoints[unit]
            if attempts != 0 or opportunities <= 0:
                return False
    return True


def validate_absolute_style(
    archetype: str, records: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    """Apply only within-archetype style thresholds; comparisons remain separate."""

    materialized = list(records)
    wins = [record for record in materialized if record.get("outcome") == "win"]
    if archetype == "scavenger":
        point_wins = sum(record.get("victory_mode") == "carcass_score" for record in wins)
        ratio = point_wins / len(wins) if wins else 0.0
        metric = "point_win_rate"
        passed = bool(wins) and ratio >= 0.60
    elif archetype == "predator":
        ko_wins = sum(record.get("victory_mode") == "ko" for record in wins)
        ratio = ko_wins / len(wins) if wins else 0.0
        metric = "ko_win_rate"
        passed = bool(wins) and ratio >= 0.60
    elif archetype == "controller":
        succeeded = sum(
            _as_int(_mapping(record.get("style_events")).get("shove_succeeded", 0))
            for record in materialized
        )
        useful = sum(
            _as_int(_mapping(record.get("style_events")).get("useful_shove", 0))
            for record in materialized
        )
        ratio = useful / succeeded if succeeded else 0.0
        metric = "useful_shove_rate"
        passed = succeeded > 0 and ratio >= 0.50
    else:
        raise ValueError("unknown specialist archetype")
    return {
        "format": "s5c-a2-style-validation-v1",
        "archetype": archetype,
        "metric": metric,
        "value": ratio,
        "minimum": 0.50 if archetype == "controller" else 0.60,
        "passed": passed,
        "comparative_threshold_pending": True,
    }


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return 0


def _raptor(snapshot: Mapping[str, object], actor: Actor) -> Mapping[str, object]:
    raptors = snapshot.get("raptors")
    if not isinstance(raptors, Mapping):
        raise ValueError("snapshot has no raptors")
    raptor = raptors.get(actor.name)
    if not isinstance(raptor, Mapping):
        raise ValueError("snapshot has no requested raptor")
    return raptor


def _position(snapshot: Mapping[str, object], actor: Actor) -> tuple[int, int]:
    value = _raptor(snapshot, actor).get("position")
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("raptor position is invalid")
    return int(value[0]), int(value[1])


def _active_carcasses(snapshot: Mapping[str, object]) -> tuple[tuple[int, int], ...]:
    states = snapshot.get("carcasses")
    if not isinstance(states, Mapping):
        raise ValueError("snapshot has no carcasses")
    positions = []
    for carcass in _ARENA.carcasses:
        state = states.get(carcass.carcass_id)
        if isinstance(state, Mapping) and state.get("status") in {"active", "available"}:
            positions.append(carcass.position)
    return tuple(positions)


def _minimum_distance(position: tuple[int, int], targets: tuple[tuple[int, int], ...]) -> int:
    return min(abs(position[0] - target[0]) + abs(position[1] - target[1]) for target in targets)


def analyse_shove_transition(
    *,
    before: Mapping[str, object],
    after: Mapping[str, object],
    transition: ActionTransition,
    learner_actor: Actor,
    shove_was_legal: bool,
) -> dict[str, int]:
    """Compute all controller predicates from public state and emitted effects."""

    if transition.action is not Action.SHOVE or transition.actor is not learner_actor:
        raise ValueError("controller diagnostic requires a learner shove")
    opponent = Actor.B if learner_actor is Actor.A else Actor.A
    target_effect = next(
        (
            effect
            for effect in transition.effects
            if isinstance(effect, TargetShovedEffect) and effect.actor is learner_actor
        ),
        None,
    )
    if target_effect is None:
        raise ValueError("shove emitted no target displacement effect")
    distance = abs(target_effect.to_position[0] - target_effect.from_position[0]) + abs(
        target_effect.to_position[1] - target_effect.from_position[1]
    )
    wall = any(
        isinstance(effect, DamageDealtEffect) and effect.actor is learner_actor
        for effect in transition.effects
    )
    interrupted = any(
        isinstance(effect, ConsumptionInterruptedEffect) and effect.actor is learner_actor
        for effect in transition.effects
    )
    pushed_into_mud = (
        target_effect.from_position not in _ARENA.mud and target_effect.to_position in _ARENA.mud
    )
    carcasses = _active_carcasses(before)
    before_target = _position(before, opponent)
    after_target = _position(after, opponent)
    learner_position = _position(after, learner_actor)
    away = bool(
        carcasses
        and _minimum_distance(after_target, carcasses) > _minimum_distance(before_target, carcasses)
    )
    favourable = bool(
        carcasses
        and _minimum_distance(learner_position, carcasses)
        < _minimum_distance(after_target, carcasses)
        and _minimum_distance(learner_position, carcasses)
        >= _minimum_distance(before_target, carcasses)
    )
    useful = wall or pushed_into_mud or interrupted or away or favourable
    return {
        "attempted": 1,
        "legal": int(shove_was_legal),
        "succeeded": int(distance > 0),
        "distance": distance,
        "movement_cost": transition.cost.movement,
        "endurance_cost": transition.cost.endurance,
        "wall_collision": int(wall),
        "mud_interrupted": int(pushed_into_mud),
        "pushed_into_mud": int(pushed_into_mud),
        "feed_interrupted": int(interrupted),
        "away_from_active_carcass": int(away),
        "favourable_carcass_access": int(favourable),
        "useful": int(useful),
    }
