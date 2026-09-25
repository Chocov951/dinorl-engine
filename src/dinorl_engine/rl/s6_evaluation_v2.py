"""Paired, statistically explicit RL-S6 re-evaluation primitives.

This module never trains or mutates the RL-S6 v1 campaign.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, cast

import numpy as np

from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec, derive_game_stream_seed

__all__ = [
    "PAIRED_BOOTSTRAP_REPLICATES",
    "PAIRED_CONFRONTATIONS",
    "paired_gate_v2",
    "paired_specs_v2",
]

PAIRED_CONFRONTATIONS: Final = 200
PAIRED_BOOTSTRAP_REPLICATES: Final = 10_000
_ALPHA: Final = 0.05


def paired_specs_v2(
    *,
    seed: int,
    opponent_id: str,
    confrontations: int = PAIRED_CONFRONTATIONS,
    start_repetition: int = 0,
) -> tuple[EvaluationGameSpec, ...]:
    """Return four seat/initiative games per shared map seed and repetition."""

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("evaluation seed must be an unsigned 32-bit integer")
    if type(confrontations) is not int or confrontations <= 0:
        raise ValueError("confrontations must be a positive integer")
    if type(start_repetition) is not int or start_repetition < 0:
        raise ValueError("start repetition must be a non-negative integer")
    games: list[EvaluationGameSpec] = []
    positions = ((Actor.A, Actor.A), (Actor.A, Actor.B), (Actor.B, Actor.A), (Actor.B, Actor.B))
    for repetition in range(start_repetition, start_repetition + confrontations):
        pair_id = f"s6-v2/{seed}/{opponent_id}/{repetition}"
        game_seed = derive_game_stream_seed(pair_id, policy_id="engine")
        for position, (learner_actor, first_actor) in enumerate(positions):
            games.append(
                EvaluationGameSpec(
                    seed=game_seed,
                    opponent_id=opponent_id,
                    learner_actor=learner_actor,
                    first_actor=first_actor,
                    game_id=f"{pair_id}/position-{position}",
                    evaluation_seed=seed,
                    pair_id=pair_id,
                )
            )
    return tuple(games)


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"paired record {field} must be numeric")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"paired record {field} must be in [0, 1]")
    return number


def _paired_means(
    records: Sequence[Mapping[str, object]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for record in records:
        opponent = record.get("opponent_id")
        pair_id = record.get("pair_id")
        if not isinstance(opponent, str) or not isinstance(pair_id, str):
            raise ValueError("paired record identity is invalid")
        grouped.setdefault(opponent, {}).setdefault(pair_id, []).append(
            (
                _number(record.get("deterministic_score"), "deterministic_score"),
                _number(record.get("stochastic_score"), "stochastic_score"),
            )
        )
    expected = {*SCRIPTED_CONTROLLER_IDS, RANDOM_LEGAL_CONTROLLER_ID}
    if set(grouped) != expected:
        raise ValueError("paired records must cover the complete opponent catalog")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for opponent, pairs in grouped.items():
        if not pairs or any(len(values) != 4 for values in pairs.values()):
            raise ValueError("every paired repetition must contain exactly four positions")
        deterministic = np.asarray(
            [np.mean([item[0] for item in values]) for values in pairs.values()]
        )
        stochastic = np.asarray(
            [np.mean([item[1] for item in values]) for values in pairs.values()]
        )
        result[opponent] = deterministic, stochastic
    return result


def _ci(values: np.ndarray) -> tuple[float, float]:
    return (float(np.quantile(values, _ALPHA / 2)), float(np.quantile(values, 1 - _ALPHA / 2)))


def paired_gate_v2(
    records: Sequence[Mapping[str, object]],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    """Apply fixed paired-bootstrap acceptance, rejection, or extension decision."""

    if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed < 2**32:
        raise ValueError("bootstrap seed must be an unsigned 32-bit integer")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1_000:
        raise ValueError("bootstrap replicates must be an integer >= 1000")
    values = _paired_means(records)
    generator = np.random.default_rng(bootstrap_seed)
    sampled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for opponent, (deterministic, stochastic) in values.items():
        indexes = generator.integers(
            0, len(deterministic), size=(bootstrap_replicates, len(deterministic))
        )
        sampled[opponent] = (deterministic[indexes].mean(axis=1), stochastic[indexes].mean(axis=1))
    report: dict[str, object] = {}
    for opponent, (deterministic, stochastic) in sampled.items():
        report[opponent] = {
            "deterministic_score": float(deterministic.mean()),
            "stochastic_score": float(stochastic.mean()),
            "stochastic_ci95": list(_ci(stochastic)),
            "drop": float((deterministic - stochastic).mean()),
            "drop_ci95": list(_ci(deterministic - stochastic)),
            "pairs": len(values[opponent][0]),
            "positions_per_pair": 4,
        }
    scripted_stochastic = np.stack([sampled[name][1] for name in SCRIPTED_CONTROLLER_IDS]).mean(
        axis=0
    )
    scripted_drop = np.stack(
        [sampled[name][0] - sampled[name][1] for name in SCRIPTED_CONTROLLER_IDS]
    ).mean(axis=0)
    main = {
        "stochastic_score": float(scripted_stochastic.mean()),
        "stochastic_ci95": list(_ci(scripted_stochastic)),
        "drop": float(scripted_drop.mean()),
        "drop_ci95": list(_ci(scripted_drop)),
    }

    def interval(value: object) -> tuple[float, float]:
        if not isinstance(value, list) or len(value) != 2:
            raise RuntimeError("paired bootstrap confidence interval is invalid")
        return float(value[0]), float(value[1])

    random_ci = interval(
        cast(dict[str, object], report[RANDOM_LEGAL_CONTROLLER_ID])["stochastic_ci95"]
    )
    script_cis = [
        interval(cast(dict[str, object], report[name])["stochastic_ci95"])
        for name in SCRIPTED_CONTROLLER_IDS
    ]
    drop_cis = [
        interval(cast(dict[str, object], report[name])["drop_ci95"])
        for name in SCRIPTED_CONTROLLER_IDS
    ]
    main_score_ci = interval(main["stochastic_ci95"])
    main_drop_ci = interval(main["drop_ci95"])
    acceptance = {
        "random_score": random_ci[0] >= 0.85,
        "main_score": main_score_ci[0] > 0.55,
        "individual_scores": all(interval[0] > 0.40 for interval in script_cis),
        "main_drop": main_drop_ci[1] <= 0.10,
        "individual_drops": all(interval[1] <= 0.15 for interval in drop_cis),
    }
    rejection = {
        "random_score": random_ci[1] < 0.85,
        "main_score": main_score_ci[1] <= 0.55,
        "individual_scores": any(interval[1] <= 0.40 for interval in script_cis),
        "main_drop": main_drop_ci[0] > 0.10,
        "individual_drops": any(interval[0] > 0.15 for interval in drop_cis),
    }
    passed = all(acceptance.values())
    failed = any(rejection.values())
    decision = "pass" if passed else "fail" if failed else "inconclusive"
    return {
        "format": "rl-s6-publication-evaluation-v2",
        "bootstrap": {
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "alpha": _ALPHA,
        },
        "opponents": report,
        "main": main,
        "acceptance_criteria": acceptance,
        "rejection_criteria": rejection,
        "failed_criteria": [name for name, value in rejection.items() if value],
        "decision": decision,
        "extension_required": decision == "inconclusive",
    }
