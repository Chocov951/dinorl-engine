"""Verification boundary for the immutable strong RL-S5b pool."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.protocol import derive_game_stream_seed
from dinorl_engine.rl.evaluation.tournament import PolicyDuelOutcome, run_policy_duel
from dinorl_engine.rl.s5b.crossplay import LoadedPolicy, _load_policy
from dinorl_engine.rl.s5b.pool import read_verified_pool

__all__ = [
    "EXPECTED_RL_S5B_POOL_SHA256",
    "evaluate_against_strong_pool",
    "verify_strong_pool",
]

EXPECTED_RL_S5B_POOL_SHA256: Final = (
    "f60caef5f68b68cd06c29a2794cdd5054ae353039100e75dbe73e662aa1113db"
)


def verify_strong_pool(path: Path, *, expected_sha256: str) -> dict[str, object]:
    """Reject a substituted pool before loading any checkpoint weights."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"frozen RL-S5b pool cannot be read: {path}") from error
    if not isinstance(value, dict) or value.get("pool_sha256") != expected_sha256:
        raise ValueError("frozen RL-S5b pool hash does not match the A2 protocol")
    return read_verified_pool(path)


def _summary(records: list[Mapping[str, object]]) -> dict[str, object]:
    games = len(records)
    wins = sum(record["outcome"] == "win" for record in records)
    draws = sum(record["outcome"] == "draw" for record in records)
    losses = games - wins - draws
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / games if games else 0.0,
        "round_limit_rate": (
            sum(record["terminal_reason"] == "round_limit" for record in records) / games
            if games
            else 0.0
        ),
        "average_rounds": (
            statistics.fmean(float(record["rounds"]) for record in records) if games else 0.0
        ),
        "average_wall_seconds": (
            statistics.fmean(float(record["wall_seconds"]) for record in records) if games else 0.0
        ),
    }


def _ci95(values: list[float]) -> list[float]:
    if len(values) <= 1:
        value = values[0] if values else 0.0
        return [value, value]
    mean = statistics.fmean(values)
    radius = 1.96 * statistics.stdev(values) / len(values) ** 0.5
    return [max(0.0, mean - radius), min(1.0, mean + radius)]


def evaluate_against_strong_pool(
    model: object,
    *,
    pool_path: Path,
    expected_sha256: str,
    evaluation_seeds: tuple[int, ...],
    confrontations: int,
    learner_policy_id: str,
    opponent_ids: tuple[str, ...] | None = None,
    duel: Callable[..., PolicyDuelOutcome] = run_policy_duel,
    loader: Callable[[Mapping[str, object]], LoadedPolicy] = _load_policy,
    progress: Callable[[str], None] | None = None,
    evaluation_label: str = "S5c strength",
) -> dict[str, object]:
    """Run balanced four-position matches against every frozen strong finalist."""

    pool = verify_strong_pool(pool_path, expected_sha256=expected_sha256)
    entries = pool["entries"]
    final_architectures = pool["final_architectures"]
    if not isinstance(entries, list) or not isinstance(final_architectures, list):
        raise ValueError("frozen RL-S5b pool has invalid finalist metadata")
    opponents = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("kind") == "checkpoint"
        and entry.get("status") == "available"
        and entry.get("unit") == 147
        and (opponent_ids is not None or entry.get("architecture") in final_architectures)
        and (opponent_ids is None or entry.get("id") in opponent_ids)
    ]
    if not opponents:
        raise ValueError("frozen RL-S5b pool has no available strong finalists")
    if opponent_ids is not None and {str(entry["id"]) for entry in opponents} != set(opponent_ids):
        raise ValueError("requested opponent subset is absent from the strong finalists")
    records: list[dict[str, object]] = []
    confrontation_scores: list[float] = []
    total_games = len(opponents) * len(evaluation_seeds) * confrontations * 4
    started_evaluation = time.perf_counter()
    for opponent in opponents:
        opponent_id = str(opponent["id"])
        loaded = loader(opponent)
        try:
            for evaluation_seed in evaluation_seeds:
                for confrontation in range(confrontations):
                    pair_id = (
                        f"s5c-a2-strength/{learner_policy_id}/{opponent_id}/"
                        f"{evaluation_seed}/{confrontation}"
                    )
                    game_seed = derive_game_stream_seed(pair_id, policy_id="engine")
                    paired: list[float] = []
                    for learner_actor in Actor:
                        for first_actor in Actor:
                            game_id = (
                                f"{pair_id}/side-{learner_actor.name}/first-{first_actor.name}"
                            )
                            started = time.perf_counter()
                            outcome = duel(
                                model,
                                loaded.model,
                                seed=game_seed,
                                left_actor=learner_actor,
                                first_actor=first_actor,
                                deterministic=False,
                                matchup_id=f"{learner_policy_id}--{opponent_id}",
                                game_id=game_id,
                                left_policy_id=learner_policy_id,
                                right_policy_id=opponent_id,
                            )
                            wall_seconds = time.perf_counter() - started
                            point = 1.0 if outcome.wins else 0.5 if outcome.draws else 0.0
                            paired.append(point)
                            telemetry = outcome.telemetry or {}
                            records.append(
                                {
                                    "game_id": game_id,
                                    "pair_id": pair_id,
                                    "evaluation_seed": evaluation_seed,
                                    "opponent_id": opponent_id,
                                    "learner_side": learner_actor.name,
                                    "learner_role": (
                                        "first" if learner_actor is first_actor else "second"
                                    ),
                                    "first_actor": first_actor.name,
                                    "outcome": (
                                        "win"
                                        if outcome.wins
                                        else "draw"
                                        if outcome.draws
                                        else "loss"
                                    ),
                                    "score": point,
                                    "terminal_reason": telemetry.get("terminal_reason"),
                                    "victory_mode": telemetry.get("terminal_reason"),
                                    "rounds": outcome.rounds,
                                    "actions": outcome.actions,
                                    "wall_seconds": wall_seconds,
                                }
                            )
                            completed_games = len(records)
                            if progress is not None and (
                                completed_games == total_games or completed_games % 100 == 0
                            ):
                                elapsed = time.perf_counter() - started_evaluation
                                rate = completed_games / elapsed if elapsed else 0.0
                                eta = (total_games - completed_games) / rate if rate else 0.0
                                progress(
                                    f"{evaluation_label}: {completed_games}/{total_games} games "
                                    f"({100 * completed_games / total_games:.1f}%) "
                                    f"rate={rate:.2f} game/s ETA={eta:.0f}s"
                                )
                    confrontation_scores.append(statistics.fmean(paired))
        finally:
            loaded.close()
    by_opponent = {
        opponent_id: _summary(
            [record for record in records if record["opponent_id"] == opponent_id]
        )
        for opponent_id in sorted({str(record["opponent_id"]) for record in records})
    }
    opponent_scores = [float(summary["score"]) for summary in by_opponent.values()]
    quartiles = (
        statistics.quantiles(opponent_scores, n=4, method="inclusive")
        if len(opponent_scores) > 1
        else [opponent_scores[0]] * 3
    )
    global_summary = _summary(records)
    global_score = float(global_summary["score"])
    return {
        "format": "s5c-a2-strength-v1",
        "pool_sha256": pool["pool_sha256"],
        "records": records,
        "global": global_summary,
        "score_ci95": _ci95(confrontation_scores),
        "by_opponent": by_opponent,
        "by_role": {
            role: _summary([record for record in records if record["learner_role"] == role])
            for role in ("first", "second")
        },
        "by_side": {
            side: _summary([record for record in records if record["learner_side"] == side])
            for side in ("A", "B")
        },
        "by_terminal_reason": {
            reason: _summary([record for record in records if record["terminal_reason"] == reason])
            for reason in sorted({str(record["terminal_reason"]) for record in records})
        },
        "worst_opponent": min(by_opponent, key=lambda key: float(by_opponent[key]["score"])),
        "lower_quartile_score": quartiles[0],
        "target_band": {"minimum": 0.25, "target": [0.35, 0.45], "maximum": 0.50},
        "band_validation": {
            "passed": 0.25 <= global_score <= 0.50,
            "in_target": 0.35 <= global_score <= 0.45,
            "score": global_score,
        },
    }
