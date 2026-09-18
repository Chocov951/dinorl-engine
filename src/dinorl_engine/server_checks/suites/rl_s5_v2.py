"""Local MLP-only RL-S5 V2 benchmark with a distinct-policy tournament."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnv

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.env.vectorization import create_vector_environment
from dinorl_engine.rl.evaluation.protocol import derive_evaluation_seed
from dinorl_engine.rl.evaluation.tournament import PolicyDuelOutcome, run_policy_duel
from dinorl_engine.rl.policies.factory import PolicyArchitecture
from dinorl_engine.rl.policies.local_mlp_v2 import (
    LOCAL_MLP_V2_ARCHITECTURES,
    local_mlp_v2_specification,
)
from dinorl_engine.rl.training.runner import AtomicRecoveryStore
from dinorl_engine.rl.training.unit import load_maskable_ppo
from dinorl_engine.server_checks.suites.rl_s3 import SELECTED_CONFIGURATION
from dinorl_engine.server_checks.suites.rl_s5 import (
    DEVELOPMENT_SEEDS,
    UNITS_PER_SEED,
    ProgressCallback,
    S5Progress,
    run_candidate_measurement,
)

__all__ = [
    "TOURNAMENT_CONFRONTATIONS",
    "aggregate_candidate_seeds",
    "run_measurement",
    "tournament_pairs",
]

TOURNAMENT_CONFRONTATIONS = 100
_BASELINE_ID = PolicyArchitecture.MLP.value
_TRANSITIONS_PER_UNIT = 2048


def tournament_pairs() -> tuple[tuple[str, str], ...]:
    """Return every distinct pair; self-play is deliberately absent."""

    identifiers = (
        _BASELINE_ID,
        *(architecture.value for architecture in LOCAL_MLP_V2_ARCHITECTURES),
    )
    return tuple(combinations(identifiers, 2))


def _summary(values: Sequence[float]) -> dict[str, object]:
    if len(values) != len(DEVELOPMENT_SEEDS):
        raise ValueError("V2 aggregates require exactly three seed values")
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "iqr": [ordered[0], ordered[-1]],
    }


def aggregate_candidate_seeds(seeds: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Keep central tendency and spread instead of silently selecting one seed."""

    transitions = [seed.get("transitions_to_gate") for seed in seeds]
    if any(value is None for value in transitions):
        transition_summary: dict[str, object] = {"median": None, "iqr": None}
    else:
        integer_transitions: list[int] = []
        for value in transitions:
            if type(value) is not int or value <= 0:
                raise ValueError("transitions_to_gate values are invalid")
            integer_transitions.append(value)
        ordered = sorted(integer_transitions)
        transition_summary = {
            "median": int(statistics.median(ordered)),
            "iqr": [ordered[0], ordered[-1]],
        }

    def numbers(field: str) -> list[float]:
        values = [seed.get(field) for seed in seeds]
        result: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{field} values are invalid")
            result.append(float(value))
        return result

    return {
        "transitions_to_gate": transition_summary,
        "wall_seconds": _summary(numbers("wall_seconds")),
        "final_score": _summary(numbers("final_score")),
        "inference_seconds": _summary(numbers("inference_seconds")),
    }


def _read_baseline(report_path: Path) -> dict[str, object]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read the MLP V1 baseline report: {report_path}") from error
    if (
        not isinstance(report, dict)
        or report.get("format") != "dinorl-local-rl-s5-diagnostic-v1"
        or not isinstance(report.get("measurement"), dict)
        or not isinstance(report["measurement"].get("candidates"), list)
    ):
        raise ValueError("MLP V1 baseline report has an invalid format")
    candidates = report["measurement"]["candidates"]
    baseline = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("architecture") == _BASELINE_ID
        ),
        None,
    )
    if not isinstance(baseline, dict):
        raise ValueError("MLP V1 baseline report does not contain mlp-v1")
    return baseline


def _load_final_model(run_directory: Path) -> tuple[MaskablePPO, VecEnv]:
    record = AtomicRecoveryStore(run_directory / "recovery").load_latest()
    if record is None or record.sequence != UNITS_PER_SEED:
        raise ValueError(f"completed unit {UNITS_PER_SEED} is missing: {run_directory}")
    environment = create_vector_environment(SELECTED_CONFIGURATION)
    try:
        model = load_maskable_ppo(record.directory / "model.zip", environment)
    except Exception:
        environment.close()
        raise
    return model, environment


def _outcomes(values: Sequence[PolicyDuelOutcome]) -> dict[str, object]:
    wins = sum(value.wins for value in values)
    draws = sum(value.draws for value in values)
    losses = sum(value.losses for value in values)
    games = wins + draws + losses
    return {
        "games": games,
        "wins_left": wins,
        "draws": draws,
        "wins_right": losses,
        "score_left": (wins + 0.5 * draws) / games,
        "average_rounds": statistics.fmean(value.rounds for value in values),
        "average_actions": statistics.fmean(value.actions for value in values),
    }


def _duel_seed(
    left_model: MaskablePPO,
    right_model: MaskablePPO,
    *,
    training_seed: int,
    left_id: str,
    right_id: str,
    progress: Callable[[int], None] | None,
) -> dict[str, object]:
    matchup_id = f"{left_id}-vs-{right_id}"
    deterministic: list[PolicyDuelOutcome] = []
    positions = (
        (Actor.A, Actor.A),
        (Actor.A, Actor.B),
        (Actor.B, Actor.A),
        (Actor.B, Actor.B),
    )
    for index, (left_actor, first_actor) in enumerate(positions):
        deterministic.append(
            run_policy_duel(
                left_model,
                right_model,
                seed=derive_evaluation_seed(
                    training_seed,
                    suite="local-v2-tournament-deterministic",
                    opponent_id=matchup_id,
                    index=index,
                ),
                left_actor=left_actor,
                first_actor=first_actor,
                deterministic=True,
                matchup_id=matchup_id,
            )
        )
    if progress is not None:
        progress(len(deterministic))

    stochastic: list[PolicyDuelOutcome] = []
    for confrontation in range(TOURNAMENT_CONFRONTATIONS):
        game_seed = derive_evaluation_seed(
            training_seed,
            suite="local-v2-tournament-stochastic",
            opponent_id=matchup_id,
            index=confrontation,
        )
        for left_actor, first_actor in positions:
            stochastic.append(
                run_policy_duel(
                    left_model,
                    right_model,
                    seed=game_seed,
                    left_actor=left_actor,
                    first_actor=first_actor,
                    deterministic=False,
                    matchup_id=matchup_id,
                )
            )
        if progress is not None and (
            (confrontation + 1) % 10 == 0 or confrontation + 1 == TOURNAMENT_CONFRONTATIONS
        ):
            progress(len(deterministic) + len(stochastic))
    return {
        "seed": training_seed,
        "deterministic": _outcomes(deterministic),
        "stochastic": _outcomes(stochastic),
    }


def _tournament(
    *,
    work_directory: Path,
    baseline_work_directory: Path,
    progress: ProgressCallback | None,
) -> dict[str, object]:
    pairs = tournament_pairs()
    matchup_seeds: dict[tuple[str, str], list[dict[str, object]]] = {pair: [] for pair in pairs}
    games_per_seed = 4 + 4 * TOURNAMENT_CONFRONTATIONS
    for seed in DEVELOPMENT_SEEDS:
        loaded: dict[str, tuple[MaskablePPO, VecEnv]] = {}
        try:
            loaded[_BASELINE_ID] = _load_final_model(
                baseline_work_directory / f"{_BASELINE_ID}-{seed}"
            )
            for architecture in LOCAL_MLP_V2_ARCHITECTURES:
                loaded[architecture.value] = _load_final_model(
                    work_directory / f"{architecture.value}-{seed}"
                )
            for pair_index, (left_id, right_id) in enumerate(pairs):

                def emit(
                    seed_completed: int,
                    *,
                    current_seed: int = seed,
                    current_left: str = left_id,
                    current_right: str = right_id,
                    current_pair_index: int = pair_index,
                ) -> None:
                    if progress is not None:
                        seed_index = DEVELOPMENT_SEEDS.index(current_seed)
                        completed_matchup_seeds = seed_index * len(pairs) + current_pair_index
                        progress(
                            S5Progress(
                                architecture=f"{current_left}-vs-{current_right}",
                                seed=current_seed,
                                completed_units=seed_completed,
                                total_units=games_per_seed,
                                completed_runs=completed_matchup_seeds,
                                total_runs=len(pairs) * len(DEVELOPMENT_SEEDS),
                                phase="tournament",
                            )
                        )

                matchup_seeds[(left_id, right_id)].append(
                    _duel_seed(
                        loaded[left_id][0],
                        loaded[right_id][0],
                        training_seed=seed,
                        left_id=left_id,
                        right_id=right_id,
                        progress=emit,
                    )
                )
        finally:
            for _model, environment in loaded.values():
                environment.close()

    matchups: list[dict[str, object]] = []
    ranking_scores: dict[str, list[float]] = {
        identifier: []
        for identifier in (_BASELINE_ID, *(item.value for item in LOCAL_MLP_V2_ARCHITECTURES))
    }
    for left_id, right_id in pairs:
        seeds = matchup_seeds[(left_id, right_id)]
        stochastic_scores = [float(seed["stochastic"]["score_left"]) for seed in seeds]  # type: ignore[index]
        left_score = statistics.fmean(stochastic_scores)
        ranking_scores[left_id].append(left_score)
        ranking_scores[right_id].append(1.0 - left_score)
        matchups.append(
            {
                "left": left_id,
                "right": right_id,
                "seeds": seeds,
                "aggregate": {
                    "stochastic_score_left": _summary(stochastic_scores),
                },
            }
        )
    ranking_values = sorted(
        ((identifier, statistics.fmean(scores)) for identifier, scores in ranking_scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    ranking = [
        {"architecture": identifier, "mean_stochastic_score": score}
        for identifier, score in ranking_values
    ]
    return {
        "self_play": False,
        "confrontations_per_seed": TOURNAMENT_CONFRONTATIONS,
        "position_configurations_per_confrontation": 4,
        "matchups": matchups,
        "ranking": ranking,
    }


def run_measurement(
    work_directory: Path,
    *,
    baseline_work_directory: Path,
    baseline_report_path: Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Train the three local MLP candidates, then run all six distinct pairings."""

    baseline = _read_baseline(baseline_report_path)
    candidates: list[dict[str, object]] = []
    total_runs = len(LOCAL_MLP_V2_ARCHITECTURES) * len(DEVELOPMENT_SEEDS)
    for architecture_index, architecture in enumerate(LOCAL_MLP_V2_ARCHITECTURES):
        seeds = [
            run_candidate_measurement(
                work_directory,
                architecture,
                seed,
                configuration=SELECTED_CONFIGURATION,
                completed_runs=architecture_index * len(DEVELOPMENT_SEEDS) + seed_index,
                total_runs=total_runs,
                progress=progress,
            )
            for seed_index, seed in enumerate(DEVELOPMENT_SEEDS)
        ]
        specification = local_mlp_v2_specification(architecture)
        candidates.append(
            {
                "architecture": architecture.value,
                "encoder_parameters": specification.encoder_parameters,
                "encoder_multiply_accumulates": specification.encoder_multiply_accumulates,
                "policy_head_parameters": 585,
                "value_head_parameters": 65,
                "total_parameters": specification.encoder_parameters + 650,
                "seeds": seeds,
                "aggregate": aggregate_candidate_seeds(seeds),
            }
        )
    tournament = _tournament(
        work_directory=work_directory,
        baseline_work_directory=baseline_work_directory,
        progress=progress,
    )
    return {
        "baseline": baseline,
        "candidates": candidates,
        "tournament": tournament,
        "training_transitions_per_seed": UNITS_PER_SEED * _TRANSITIONS_PER_UNIT,
    }
