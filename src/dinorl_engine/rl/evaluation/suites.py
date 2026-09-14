"""Complete deterministic and stochastic evaluation suites for one PPO checkpoint."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from sb3_contrib import MaskablePPO

from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.match.replay import canonical_replay_json
from dinorl_engine.rl.evaluation.deterministic import deterministic_game_specs
from dinorl_engine.rl.evaluation.match import EvaluationGameResult, run_evaluation_game
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec
from dinorl_engine.rl.evaluation.stochastic import paired_game_specs

__all__ = [
    "PUBLICATION_CONFRONTATIONS",
    "RANDOM_DETERMINISTIC_CONFRONTATIONS",
    "evaluate_deterministic",
    "evaluate_publication",
]

PUBLICATION_CONFRONTATIONS = 200
RANDOM_DETERMINISTIC_CONFRONTATIONS = 200


def _sum_fields(values: Iterable[Mapping[str, object]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for value in values:
        for key, item in value.items():
            if type(item) is not int:
                raise RuntimeError("behaviour counter is not an integer")
            totals[key] = totals.get(key, 0) + item
    return totals


def _groups(metrics: list[dict[str, object]], name: str) -> list[Mapping[str, object]]:
    values: list[Mapping[str, object]] = []
    for metric in metrics:
        value = metric.get(name)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"evaluation metric group is missing: {name}")
        values.append(value)
    return values


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"evaluation metric is not numeric: {field_name}")
    return float(value)


def _integer(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"evaluation metric is not an integer: {field_name}")
    return value


def _aggregate_metrics(results: list[EvaluationGameResult]) -> dict[str, object]:
    if not results:
        raise ValueError("evaluation metrics require at least one game")
    metrics = [result.metrics for result in results]
    action_distribution = _sum_fields(_groups(metrics, "action_distribution"))
    groups = ("bites", "damage", "shoves", "consumption", "carcass_points", "rests", "mud")
    aggregate: dict[str, object] = {"action_distribution": action_distribution}
    for group in groups:
        aggregate[group] = _sum_fields(_groups(metrics, group))
    for group in ("movement_points", "endurance"):
        values = _groups(metrics, group)
        averages = [_number(value.get("average"), f"{group}.average") for value in values]
        minimums = [_integer(value.get("minimum"), f"{group}.minimum") for value in values]
        finals = [_integer(value.get("final"), f"{group}.final") for value in values]
        aggregate[group] = {
            "average": sum(averages) / len(averages),
            "minimum": min(minimums),
            "final_average": sum(finals) / len(finals),
        }
    aggregate["average_legal_actions"] = sum(
        _number(value.get("average_legal_actions"), "average_legal_actions") for value in metrics
    ) / len(metrics)
    aggregate["average_rounds"] = sum(
        _integer(value.get("rounds"), "rounds") for value in metrics
    ) / len(metrics)
    aggregate["average_actions"] = sum(
        _integer(value.get("actions"), "actions") for value in metrics
    ) / len(metrics)
    aggregate["victory_route"] = _sum_fields(_groups(metrics, "victory_route"))
    return aggregate


def _write_diagnostic(
    directory: Path,
    *,
    suite: str,
    specification: EvaluationGameSpec,
    result: EvaluationGameResult,
) -> dict[str, object]:
    if result.replay is None or result.replay_sha256 is None:
        raise RuntimeError("diagnostic evaluation did not produce a replay")
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"{suite}-{specification.opponent_id}-{specification.seed}-"
        f"{specification.learner_actor.name}-{specification.first_actor.name}.json"
    )
    path = directory / name
    if not path.exists():
        path.write_bytes(canonical_replay_json(result.replay))
    return {"file": name, "sha256": result.replay_sha256}


def _evaluate(
    model: MaskablePPO,
    *,
    suite: str,
    games_by_opponent: Mapping[str, tuple[EvaluationGameSpec, ...]],
    deterministic: bool,
    diagnostic_directory: Path | None,
) -> dict[str, object]:
    opponent_reports: dict[str, object] = {}
    diagnostics: list[dict[str, object]] = []
    for opponent_id, games in games_by_opponent.items():
        outcomes = {"wins": 0, "draws": 0, "losses": 0}
        results: list[EvaluationGameResult] = []
        captured_diagnostic = False
        for specification in games:
            result = run_evaluation_game(
                model, specification, deterministic=deterministic, diagnostic_replay=False
            )
            results.append(result)
            outcomes["wins"] += result.outcomes.wins
            outcomes["draws"] += result.outcomes.draws
            outcomes["losses"] += result.outcomes.losses
            if (
                diagnostic_directory is not None
                and result.outcomes.losses
                and not captured_diagnostic
            ):
                diagnostic = run_evaluation_game(
                    model, specification, deterministic=deterministic, diagnostic_replay=True
                )
                diagnostics.append(
                    _write_diagnostic(
                        diagnostic_directory,
                        suite=suite,
                        specification=specification,
                        result=diagnostic,
                    )
                )
                captured_diagnostic = True
        games_count = outcomes["wins"] + outcomes["draws"] + outcomes["losses"]
        opponent_reports[opponent_id] = {
            **outcomes,
            "games": games_count,
            "score": (outcomes["wins"] + 0.5 * outcomes["draws"]) / games_count,
            "metrics": _aggregate_metrics(results),
        }
    scores = {
        opponent: float(report["score"])
        for opponent, report in opponent_reports.items()
        if isinstance(report, Mapping)
    }
    return {
        "suite": suite,
        "opponents": opponent_reports,
        "scores": scores,
        "games": sum(
            int(report["games"])
            for report in opponent_reports.values()
            if isinstance(report, Mapping)
        ),
        "diagnostic_replays": diagnostics,
    }


def evaluate_deterministic(
    model: MaskablePPO,
    *,
    seed: int,
    diagnostic_directory: Path | None = None,
    random_confrontations: int = RANDOM_DETERMINISTIC_CONFRONTATIONS,
) -> dict[str, object]:
    """Evaluate four fixed games per primary bot and 200 paired random-legal confrontations."""

    games: dict[str, tuple[EvaluationGameSpec, ...]] = {
        opponent: deterministic_game_specs(seed=seed, opponent_id=opponent)
        for opponent in SCRIPTED_CONTROLLER_IDS
    }
    games[RANDOM_LEGAL_CONTROLLER_ID] = paired_game_specs(
        seed=seed,
        opponent_id=RANDOM_LEGAL_CONTROLLER_ID,
        confrontations=random_confrontations,
    )
    return _evaluate(
        model,
        suite="deterministic",
        games_by_opponent=games,
        deterministic=True,
        diagnostic_directory=diagnostic_directory,
    )


def evaluate_publication(
    model: MaskablePPO,
    *,
    seed: int,
    diagnostic_directory: Path | None = None,
    confrontations: int = PUBLICATION_CONFRONTATIONS,
) -> dict[str, object]:
    """Evaluate sampled temperature-1 inference on paired confrontations for every opponent."""

    games = {
        opponent: paired_game_specs(seed=seed, opponent_id=opponent, confrontations=confrontations)
        for opponent in (*SCRIPTED_CONTROLLER_IDS, RANDOM_LEGAL_CONTROLLER_ID)
    }
    return _evaluate(
        model,
        suite="publication",
        games_by_opponent=games,
        deterministic=False,
        diagnostic_directory=diagnostic_directory,
    )
