"""Resumable, position-balanced RL-S5b cross-play evaluation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from pathlib import Path
from typing import Final

from sb3_contrib import MaskablePPO

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.evaluation.protocol import derive_evaluation_seed
from dinorl_engine.rl.evaluation.tournament import PolicyDuelOutcome, run_policy_duel
from dinorl_engine.rl.s5b.pool import PoolError, read_verified_pool
from dinorl_engine.rl.training.unit import load_maskable_ppo

__all__ = ["CrossplayError", "LoadedPolicy", "cross_evaluate"]

_FORMAT: Final = "s5b-crossplay-v1"
_POSITIONS: Final = tuple(
    (left_actor, first_actor) for left_actor in Actor for first_actor in Actor
)


class CrossplayError(RuntimeError):
    """Raised when the official paired cross-play cannot be completed safely."""


type Duel = Callable[..., PolicyDuelOutcome]


@dataclass(slots=True)
class LoadedPolicy:
    """A policy plus the temporary inference environment needed to load it."""

    model: object
    close: Callable[[], None]


def _sha256_model(model: object) -> str:
    if not isinstance(model, MaskablePPO):
        raise CrossplayError("cross-play model must be a MaskablePPO")
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise CrossplayError(f"cross-play {field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise CrossplayError(f"cross-play {field} must be a finite number")
    return float(value)


def _load_policy(entry: Mapping[str, object]) -> LoadedPolicy:
    provenance = entry.get("provenance")
    if not isinstance(provenance, str):
        raise CrossplayError("pool checkpoint provenance is invalid")
    directory = Path(provenance)
    try:
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CrossplayError(
            f"checkpoint recovery state cannot be read: {entry.get('id')}"
        ) from error
    if not isinstance(state, dict) or not isinstance(state.get("configuration"), dict):
        raise CrossplayError("checkpoint recovery state has no vector configuration")
    configuration = state["configuration"]
    try:
        vector = VectorEnvironmentConfig(
            backend=VectorBackend(configuration["backend"]),
            n_envs=configuration["n_envs"],
            seed=configuration["seed"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CrossplayError("checkpoint vector configuration is incompatible") from error
    environment = create_vector_environment(vector)
    try:
        model = load_maskable_ppo(directory / "model.zip", environment)
    except Exception:
        environment.close()
        raise
    return LoadedPolicy(model=model, close=environment.close)


def _score(wins: int, draws: int, games: int) -> float:
    if games <= 0:
        raise CrossplayError("cross-play aggregate has no games")
    return (wins + 0.5 * draws) / games


def _iqr(values: Iterable[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        raise CrossplayError("cannot compute an aggregate without samples")
    if len(ordered) == 1:
        return [ordered[0], ordered[0]]
    cuts = statistics.quantiles(ordered, n=4, method="inclusive")
    return [cuts[0], cuts[2]]


def _ci95(values: list[float]) -> list[float]:
    if not values:
        raise CrossplayError("cannot compute a confidence interval without confrontations")
    average = statistics.mean(values)
    if len(values) == 1:
        return [average, average]
    radius = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [max(0.0, average - radius), min(1.0, average + radius)]


def _empty_position(left_actor: Actor, first_actor: Actor) -> dict[str, object]:
    return {
        "actions": 0,
        "draws": 0,
        "first_actor": first_actor.name,
        "games": 0,
        "left_actor": left_actor.name,
        "losses": 0,
        "rounds": 0,
        "wins": 0,
    }


def _record_id(pool_sha256: str, left_id: str, right_id: str, evaluation_seed: int) -> str:
    raw = f"{pool_sha256}/{left_id}/{right_id}/{evaluation_seed}".encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_json_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            )
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_record(path: Path, *, pool_sha256: str) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CrossplayError(f"cross-play record cannot be read: {path}") from error
    if not isinstance(record, dict) or record.get("pool_sha256") != pool_sha256:
        raise CrossplayError("cross-play record belongs to a different immutable pool")
    return record


def _evaluate_record(
    *,
    pool_sha256: str,
    left: Mapping[str, object],
    right: Mapping[str, object],
    evaluation_seed: int,
    confrontations: int,
    left_model: object,
    right_model: object,
    duel: Duel,
) -> dict[str, object]:
    left_id, right_id = left.get("id"), right.get("id")
    if not isinstance(left_id, str) or not isinstance(right_id, str):
        raise CrossplayError("pool checkpoint identifier is invalid")
    matchup_id = f"{left_id}--{right_id}"
    positions = {
        f"{left_actor.name}-{first_actor.name}": _empty_position(left_actor, first_actor)
        for left_actor, first_actor in _POSITIONS
    }
    confrontation_scores: list[float] = []
    first_player_scores: list[float] = []
    for confrontation in range(confrontations):
        game_seed = derive_evaluation_seed(
            evaluation_seed,
            suite="s5b-crossplay",
            opponent_id=matchup_id,
            index=confrontation,
        )
        paired_scores: list[float] = []
        for left_actor, first_actor in _POSITIONS:
            outcome = duel(
                left_model,
                right_model,
                seed=game_seed,
                left_actor=left_actor,
                first_actor=first_actor,
                deterministic=False,
                matchup_id=matchup_id,
            )
            if (
                min(outcome.wins, outcome.draws, outcome.losses, outcome.rounds, outcome.actions)
                < 0
            ):
                raise CrossplayError("cross-play duel returned a negative metric")
            if outcome.wins + outcome.draws + outcome.losses != 1:
                raise CrossplayError("each cross-play duel must describe exactly one game")
            position = positions[f"{left_actor.name}-{first_actor.name}"]
            for name, value in (
                ("wins", outcome.wins),
                ("draws", outcome.draws),
                ("losses", outcome.losses),
                ("rounds", outcome.rounds),
                ("actions", outcome.actions),
            ):
                existing = position[name]
                assert type(existing) is int
                position[name] = existing + value
            games = position["games"]
            assert type(games) is int
            position["games"] = games + 1
            point = _score(outcome.wins, outcome.draws, 1)
            paired_scores.append(point)
            first_player_scores.append(point if left_actor is first_actor else 1.0 - point)
        confrontation_scores.append(statistics.mean(paired_scores))
    games = confrontations * len(_POSITIONS)
    wins = sum(_integer(position["wins"], "position.wins") for position in positions.values())
    draws = sum(_integer(position["draws"], "position.draws") for position in positions.values())
    losses = sum(_integer(position["losses"], "position.losses") for position in positions.values())
    scores_by_role: dict[str, list[float]] = {"first": [], "second": []}
    for position in positions.values():
        position_games = _integer(position["games"], "position.games")
        position_wins = _integer(position["wins"], "position.wins")
        position_draws = _integer(position["draws"], "position.draws")
        position_actions = _integer(position["actions"], "position.actions")
        position_rounds = _integer(position["rounds"], "position.rounds")
        position_score = _score(position_wins, position_draws, position_games)
        position["average_actions"] = position_actions / position_games
        position["average_rounds"] = position_rounds / position_games
        position["score"] = position_score
        role = "first" if position["left_actor"] == position["first_actor"] else "second"
        scores_by_role[role].append(position_score)
    first_role_score = statistics.mean(scores_by_role["first"])
    second_role_score = statistics.mean(scores_by_role["second"])
    return {
        "average_actions": sum(
            _integer(position["actions"], "position.actions") for position in positions.values()
        )
        / games,
        "average_rounds": sum(
            _integer(position["rounds"], "position.rounds") for position in positions.values()
        )
        / games,
        "confrontation_scores": confrontation_scores,
        "evaluation_seed": evaluation_seed,
        "first_player_advantage": statistics.mean(first_player_scores) - 0.5,
        "first_player_score": statistics.mean(first_player_scores),
        "games": games,
        "left": {
            "architecture": left["architecture"],
            "checkpoint_id": left_id,
            "seed": left["seed"],
        },
        "outcomes": {"draws": draws, "losses": losses, "wins": wins},
        "pool_sha256": pool_sha256,
        "positions": positions,
        "right": {
            "architecture": right["architecture"],
            "checkpoint_id": right_id,
            "seed": right["seed"],
        },
        "score": _score(wins, draws, games),
        "score_by_role": {"first": first_role_score, "second": second_role_score},
        "score_ci95": _ci95(confrontation_scores),
    }


def _reverse_record(record: Mapping[str, object]) -> dict[str, object]:
    outcomes = record["outcomes"]
    positions = record["positions"]
    assert isinstance(outcomes, Mapping) and isinstance(positions, Mapping)
    reverse_positions: dict[str, object] = {}
    for _name, raw_position in positions.items():
        assert isinstance(raw_position, Mapping)
        left_actor = Actor[raw_position["left_actor"]]
        first_actor = Actor[raw_position["first_actor"]]
        right_actor = Actor.B if left_actor is Actor.A else Actor.A
        reverse_positions[f"{right_actor.name}-{first_actor.name}"] = {
            "actions": raw_position["actions"],
            "draws": raw_position["draws"],
            "first_actor": first_actor.name,
            "games": raw_position["games"],
            "left_actor": right_actor.name,
            "losses": raw_position["wins"],
            "rounds": raw_position["rounds"],
            "wins": raw_position["losses"],
        }
    games = record["games"]
    assert type(games) is int
    return {
        "architecture": record["right"],
        "checkpoint": record["right"],
        "evaluation_seed": record["evaluation_seed"],
        "games": games,
        "positions": reverse_positions,
        "score": 1.0 - _number(record["score"], "score"),
    }


def _side_samples(records: Iterable[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    samples: dict[str, list[dict[str, object]]] = {}
    for record in records:
        left = record["left"]
        right = record["right"]
        assert isinstance(left, Mapping) and isinstance(right, Mapping)
        left_id = left["checkpoint_id"]
        right_id = right["checkpoint_id"]
        assert isinstance(left_id, str) and isinstance(right_id, str)
        left_sample = {
            "architecture": left["architecture"],
            "checkpoint_id": left_id,
            "seed": left["seed"],
            "score": record["score"],
        }
        samples.setdefault(left_id, []).append(left_sample)
        if left_id != right_id:
            samples.setdefault(right_id, []).append(
                {
                    "architecture": right["architecture"],
                    "checkpoint_id": right_id,
                    "seed": right["seed"],
                    "score": 1.0 - _number(record["score"], "score"),
                }
            )
    return samples


def _summary(values: list[float]) -> dict[str, object]:
    return {
        "iqr": _iqr(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "samples": len(values),
    }


def _aggregates(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    samples = _side_samples(records)
    by_checkpoint: dict[str, object] = {}
    by_architecture: dict[str, dict[int, list[float]]] = {}
    for checkpoint_id, observations in sorted(samples.items()):
        scores = [_number(observation["score"], "checkpoint.score") for observation in observations]
        architecture = observations[0]["architecture"]
        seed = observations[0]["seed"]
        assert isinstance(architecture, str) and type(seed) is int
        by_checkpoint[checkpoint_id] = {
            "architecture": architecture,
            "seed": seed,
            "score": _summary(scores),
        }
        by_architecture.setdefault(architecture, {}).setdefault(seed, []).extend(scores)
    architecture_summary: dict[str, object] = {}
    for architecture, seeds in sorted(by_architecture.items()):
        seed_scores = {str(seed): statistics.mean(values) for seed, values in sorted(seeds.items())}
        values = list(seed_scores.values())
        lower_count = max(1, math.ceil(len(values) / 4))
        architecture_summary[architecture] = {
            "lower_quartile_mean": statistics.mean(sorted(values)[:lower_count]),
            "score": _summary(values),
            "seed_scores": seed_scores,
            "worst_seed": min(seed_scores, key=seed_scores.__getitem__),
        }
    return {"architectures": architecture_summary, "checkpoints": by_checkpoint}


def _cycles(records: Iterable[Mapping[str, object]]) -> list[list[str]]:
    scores: dict[tuple[str, str], list[float]] = {}
    for record in records:
        left, right = record["left"], record["right"]
        assert isinstance(left, Mapping) and isinstance(right, Mapping)
        left_architecture, right_architecture = left["architecture"], right["architecture"]
        if not isinstance(left_architecture, str) or not isinstance(right_architecture, str):
            continue
        if left_architecture == right_architecture:
            continue
        scores.setdefault((left_architecture, right_architecture), []).append(
            _number(record["score"], "score")
        )
    architectures = sorted({architecture for pair in scores for architecture in pair})
    cycles: list[list[str]] = []
    for first, second, third in combinations(architectures, 3):

        def mean_for(left: str, right: str) -> float:
            direct = scores.get((left, right))
            if direct:
                return statistics.mean(direct)
            inverse = scores.get((right, left))
            return 1.0 - statistics.mean(inverse) if inverse else 0.5

        if (
            mean_for(first, second) > 0.5
            and mean_for(second, third) > 0.5
            and mean_for(third, first) > 0.5
        ):
            cycles.append([first, second, third, first])
        elif (
            mean_for(first, third) > 0.5
            and mean_for(third, second) > 0.5
            and mean_for(second, first) > 0.5
        ):
            cycles.append([first, third, second, first])
    return cycles


def _write_csv(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "left_checkpoint",
            "left_architecture",
            "right_checkpoint",
            "right_architecture",
            "evaluation_seed",
            "games",
            "left_score",
            "score_ci95_low",
            "score_ci95_high",
            "first_player_score",
        ),
    )
    writer.writeheader()
    for record in records:
        left, right, ci = record["left"], record["right"], record["score_ci95"]
        assert isinstance(left, Mapping) and isinstance(right, Mapping) and isinstance(ci, list)
        writer.writerow(
            {
                "left_checkpoint": left["checkpoint_id"],
                "left_architecture": left["architecture"],
                "right_checkpoint": right["checkpoint_id"],
                "right_architecture": right["architecture"],
                "evaluation_seed": record["evaluation_seed"],
                "games": record["games"],
                "left_score": record["score"],
                "score_ci95_low": ci[0],
                "score_ci95_high": ci[1],
                "first_player_score": record["first_player_score"],
            }
        )
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")


def _write_report(path: Path, result: Mapping[str, object]) -> None:
    aggregates = result["aggregates"]
    cycles = result["cycles"]
    assert isinstance(aggregates, Mapping) and isinstance(cycles, list)
    architectures = aggregates["architectures"]
    assert isinstance(architectures, Mapping)
    lines = [
        "# RL-S5b cross-play report",
        "",
        f"- Immutable pool: `{result['pool_sha256']}`",
        f"- Official games completed: `{result['games_completed']}`",
        f"- Matrix records: `{result['records_completed']}`",
        "",
        "## Architecture robustness",
        "",
        "| Architecture | Mean score | Median | Worst seed | Lower-quartile mean |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for architecture, raw in architectures.items():
        assert isinstance(raw, Mapping)
        score = raw["score"]
        assert isinstance(score, Mapping)
        lines.append(
            f"| `{architecture}` | {float(score['mean']):.3f} | {float(score['median']):.3f} | "
            f"`{raw['worst_seed']}` | {float(raw['lower_quartile_mean']):.3f} |"
        )
    lines.extend(("", "## Non-transitive cycles", ""))
    if cycles:
        lines.extend(f"- {' > '.join(cycle)}" for cycle in cycles)
    else:
        lines.append("- No strict three-architecture cycle was observed in the aggregate matrix.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _update_campaign_manifest(
    *, output_directory: Path, pool: Mapping[str, object], result: Mapping[str, object]
) -> None:
    """Record durable phase-A progress without changing the frozen pool."""

    required = ("config_sha256", "git_commit", "pool_sha256", "run_id", "evaluation_seeds")
    if any(name not in pool for name in required):
        raise CrossplayError("pool lacks required campaign provenance")
    path = output_directory / "manifest.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CrossplayError("campaign manifest cannot be read") from error
        if not isinstance(existing, dict) or existing.get("pool_sha256") != pool["pool_sha256"]:
            raise CrossplayError("campaign manifest belongs to another immutable pool")
        phases = existing.get("phases")
        if not isinstance(phases, dict):
            raise CrossplayError("campaign manifest phases are invalid")
    else:
        existing = {
            "config_sha256": pool["config_sha256"],
            "format": "s5b-run-manifest-v1",
            "git_commit": pool["git_commit"],
            "phases": {},
            "pool_sha256": pool["pool_sha256"],
            "run_id": pool["run_id"],
            "seeds": pool["evaluation_seeds"],
        }
        phases = existing["phases"]
        assert isinstance(phases, dict)
    phases["cross-evaluate"] = {
        "games_completed": result["games_completed"],
        "records_completed": result["records_completed"],
        "status": "completed",
    }
    _atomic_json_write(path, existing)


def cross_evaluate(
    *,
    pool_path: Path,
    output_directory: Path,
    model_loader: Callable[[Mapping[str, object]], LoadedPolicy] | None = None,
    duel: Duel | None = None,
) -> dict[str, object]:
    """Complete or resume the full checkpoint matrix for one immutable pool.

    Every record contains one evaluation seed, one unordered checkpoint pair
    (including a defined diagonal auto-game), and all four learner-seat / first
    player configurations.  Completed records are immutable and idempotently
    reused on resume.
    """

    try:
        pool = read_verified_pool(pool_path)
    except PoolError as error:
        raise CrossplayError(str(error)) from error
    pool_sha256 = pool["pool_sha256"]
    confrontations = pool["confrontations"]
    entries = pool["entries"]
    seeds = pool["evaluation_seeds"]
    if (
        not isinstance(pool_sha256, str)
        or type(confrontations) is not int
        or not isinstance(entries, list)
        or not isinstance(seeds, list)
    ):
        raise CrossplayError("verified pool has invalid matrix settings")
    checkpoints = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("kind") == "checkpoint"
        and entry.get("status") == "available"
        and entry.get("unit") == 147
    ]
    if not checkpoints:
        raise CrossplayError("pool has no available final checkpoints")
    loader = _load_policy if model_loader is None else model_loader
    duel_runner = run_policy_duel if duel is None else duel
    records_directory = output_directory / "crossplay" / "records"
    loaded: dict[str, LoadedPolicy] = {}
    try:
        for entry in checkpoints:
            identifier = entry["id"]
            assert isinstance(identifier, str)
            loaded[identifier] = loader(entry)
        for left, right in combinations_with_replacement(checkpoints, 2):
            left_id, right_id = left["id"], right["id"]
            assert isinstance(left_id, str) and isinstance(right_id, str)
            for seed in seeds:
                if type(seed) is not int:
                    raise CrossplayError("pool evaluation seed is invalid")
                path = (
                    records_directory / f"{_record_id(pool_sha256, left_id, right_id, seed)}.json"
                )
                if _read_record(path, pool_sha256=pool_sha256) is not None:
                    continue
                left_model = loaded[left_id].model
                right_model = loaded[right_id].model
                before_left = (
                    _sha256_model(left_model) if isinstance(left_model, MaskablePPO) else None
                )
                before_right = (
                    _sha256_model(right_model) if isinstance(right_model, MaskablePPO) else None
                )
                record = _evaluate_record(
                    pool_sha256=pool_sha256,
                    left=left,
                    right=right,
                    evaluation_seed=seed,
                    confrontations=confrontations,
                    left_model=left_model,
                    right_model=right_model,
                    duel=duel_runner,
                )
                if before_left is not None and before_left != _sha256_model(left_model):
                    raise CrossplayError("evaluation mutated the left policy weights")
                if before_right is not None and before_right != _sha256_model(right_model):
                    raise CrossplayError("evaluation mutated the right policy weights")
                _atomic_json_write(path, record)
    finally:
        for policy in loaded.values():
            policy.close()
    records: list[dict[str, object]] = []
    for path in sorted(records_directory.glob("*.json")):
        completed_record = _read_record(path, pool_sha256=pool_sha256)
        if completed_record is not None:
            records.append(completed_record)
    expected_records = len(checkpoints) * (len(checkpoints) + 1) // 2 * len(seeds)
    if len(records) != expected_records:
        raise CrossplayError("cross-play matrix is incomplete after evaluation")
    ordered_records = tuple(records)
    result = {
        "aggregates": _aggregates(ordered_records),
        "confrontations_per_record": confrontations,
        "cycles": _cycles(ordered_records),
        "format": _FORMAT,
        "games_completed": sum(_integer(record["games"], "games") for record in ordered_records),
        "pool_sha256": pool_sha256,
        "records": records,
        "records_completed": len(records),
    }
    crossplay_directory = output_directory / "crossplay"
    _atomic_json_write(crossplay_directory / "crossplay.json", result)
    _write_csv(crossplay_directory / "crossplay.csv", ordered_records)
    _write_report(crossplay_directory / "crossplay-report.md", result)
    _update_campaign_manifest(output_directory=output_directory, pool=pool, result=result)
    return result
