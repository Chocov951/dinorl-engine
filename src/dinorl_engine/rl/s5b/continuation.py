"""RL-S5b-B controlled local continuation from the immutable phase-A pool."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnv

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.evaluation.protocol import derive_evaluation_seed
from dinorl_engine.rl.evaluation.suites import evaluate_deterministic
from dinorl_engine.rl.evaluation.tournament import run_policy_duel
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.s5b.crossplay import _load_policy
from dinorl_engine.rl.s5b.opponents import FrozenOpponentPool
from dinorl_engine.rl.s5b.pool import read_verified_pool
from dinorl_engine.rl.training.resume import (
    capture_process_rng_state,
    restore_process_rng_state,
    restore_vector_recovery_state,
)
from dinorl_engine.rl.training.runner import (
    AtomicPPOUnitRunner,
    AtomicRecoveryStore,
    _restore_ppo_resume_arrays,
)
from dinorl_engine.rl.training.unit import PPOUnitMetrics, load_maskable_ppo

__all__ = ["ContinuationError", "continue_phase_b"]

_FORMAT: Final = "s5b-continuation-v1"
_RECOVERY_FORMAT: Final = "s5b-continuation-recovery-v1"
_FINAL_ARCHITECTURES: Final = frozenset({"mlp-compact-v2", "mlp-deep-v2"})
_DEVELOPMENT_SEEDS: Final = frozenset({19, 20, 21})
_PARENT_UNIT: Final = 147
_PHASE_B_UNITS: Final = 50
_EVALUATION_INTERVAL: Final = 10
_RANDOM_EVALUATION_CONFRONTATIONS: Final = 20
_FROZEN_EVALUATION_CONFRONTATIONS: Final = 100


class ContinuationError(RuntimeError):
    """Raised when a controlled phase-B continuation cannot be verified."""


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _configuration(state: Mapping[str, object]) -> VectorEnvironmentConfig:
    raw = state.get("configuration")
    if not isinstance(raw, Mapping):
        raise ContinuationError("parent checkpoint has no vector configuration")
    try:
        configuration = VectorEnvironmentConfig(
            backend=VectorBackend(raw["backend"]), n_envs=raw["n_envs"], seed=raw["seed"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContinuationError("parent checkpoint vector configuration is invalid") from error
    if configuration.backend is not VectorBackend.DUMMY:
        raise ContinuationError("RL-S5b-B requires the reproducible DummyVecEnv parent")
    return configuration


def _parent_state(entry: Mapping[str, object]) -> tuple[Path, dict[str, object]]:
    provenance = entry.get("provenance")
    if not isinstance(provenance, str):
        raise ContinuationError("finalist checkpoint has invalid provenance")
    directory = Path(provenance)
    try:
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContinuationError("finalist checkpoint state cannot be read") from error
    if (
        not isinstance(state, dict)
        or state.get("format") != "dinorl-ppo-unit-recovery-v1"
        or not isinstance(state.get("environments"), list)
    ):
        raise ContinuationError("finalist checkpoint is not an exact PPO recovery")
    return directory, state


def _finalists(pool: Mapping[str, object]) -> list[Mapping[str, object]]:
    entries = pool.get("entries")
    if not isinstance(entries, list):
        raise ContinuationError("pool entries are invalid")
    finalists = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("kind") == "checkpoint"
        and entry.get("status") == "available"
        and entry.get("unit") == _PARENT_UNIT
        and entry.get("architecture") in _FINAL_ARCHITECTURES
        and entry.get("seed") in _DEVELOPMENT_SEEDS
    ]
    identifiers = [entry.get("id") for entry in finalists]
    if len(finalists) != 6 or len(set(identifiers)) != 6:
        raise ContinuationError("phase B requires compact and deep finalists for seeds 19, 20, 21")
    return sorted(finalists, key=lambda item: (str(item["architecture"]), int(item["seed"])))


def _metadata(
    pool: Mapping[str, object],
    entry: Mapping[str, object],
    opponent_pool: FrozenOpponentPool,
    *,
    random_only: bool,
) -> dict[str, object]:
    identifier = entry.get("id")
    architecture = entry.get("architecture")
    digest = entry.get("artifact_sha256")
    if not all(isinstance(value, str) for value in (identifier, architecture, digest)):
        raise ContinuationError("finalist identity is invalid")
    return {
        "architecture": architecture,
        "effective_training_weights": opponent_pool.effective_weights,
        "format": _RECOVERY_FORMAT,
        "parent_artifact_sha256": digest,
        "parent_checkpoint_id": identifier,
        "pool_sha256": pool["pool_sha256"],
        "random_only_control": random_only,
        "reward_compatibility": pool["compatibility"],
    }


def _environment_factory(
    opponents: FrozenOpponentPool,
) -> Callable[[VectorEnvironmentConfig], VecEnv]:
    return lambda configuration: create_vector_environment(
        configuration, training_opponent_pool=opponents
    )


def _restore_parent(
    directory: Path,
    state: Mapping[str, object],
    configuration: VectorEnvironmentConfig,
    opponents: FrozenOpponentPool,
) -> tuple[MaskablePPO, VecEnv]:
    environment = _environment_factory(opponents)(configuration)
    try:
        restore_vector_recovery_state(environment, state["environments"])
        model = load_maskable_ppo(directory / "model.zip", environment)
        _restore_ppo_resume_arrays(model, directory / "ppo_state.npz")
        restore_process_rng_state(state.get("process_rng"))
    except Exception:
        environment.close()
        raise
    return model, environment


def _metrics(metrics: PPOUnitMetrics) -> dict[str, object]:
    return {
        "checkpoint_seconds": None,
        "diagnostics": metrics.diagnostics,
        "engine_actions": metrics.engine_actions,
        "epochs": metrics.epochs,
        "learner_transitions": metrics.learner_transitions,
        "optimizer_steps": metrics.optimizer_steps,
        "role_counts": metrics.role_counts,
        "role_proportions": {
            name: (count / sum(metrics.role_counts.values()) if metrics.role_counts else 0.0)
            for name, count in metrics.role_counts.items()
        },
        "training_seconds": None,
    }


def _frozen_evaluation(
    model: MaskablePPO, finalists: list[Mapping[str, object]], *, seed: int
) -> dict[str, object]:
    """Evaluate the live learner against each frozen final snapshot in four positions."""

    results: dict[str, object] = {}
    for entry in finalists:
        identifier = entry.get("id")
        if not isinstance(identifier, str):
            raise ContinuationError("frozen evaluation checkpoint identifier is invalid")
        loaded = _load_policy(entry)
        try:
            wins = draws = losses = games = 0
            for confrontation in range(_FROZEN_EVALUATION_CONFRONTATIONS):
                game_seed = derive_evaluation_seed(
                    seed,
                    suite="s5b-b-frozen-evaluation",
                    opponent_id=identifier,
                    index=confrontation,
                )
                for learner_actor in Actor:
                    for first_actor in Actor:
                        outcome = run_policy_duel(
                            model,
                            loaded.model,
                            seed=game_seed,
                            left_actor=learner_actor,
                            first_actor=first_actor,
                            deterministic=False,
                            matchup_id=f"continuation--{identifier}",
                        )
                        wins += outcome.wins
                        draws += outcome.draws
                        losses += outcome.losses
                        games += 1
            results[identifier] = {
                "draws": draws,
                "games": games,
                "losses": losses,
                "score": (wins + 0.5 * draws) / games,
                "wins": wins,
            }
        finally:
            loaded.close()
    return results


def _evaluation(
    model: MaskablePPO, finalists: list[Mapping[str, object]], *, seed: int
) -> dict[str, object]:
    # Loading opponent checkpoints and sampled match play must not perturb the
    # learner's process streams before its next PPO unit.
    process_rng = capture_process_rng_state()
    try:
        return {
            "deterministic_bots": evaluate_deterministic(
                model, seed=seed, random_confrontations=_RANDOM_EVALUATION_CONFRONTATIONS
            ),
            "frozen_finalists": _frozen_evaluation(model, finalists, seed=seed),
        }
    finally:
        restore_process_rng_state(process_rng)


def _mean_score(evaluation: Mapping[str, object], field: str) -> float:
    raw = evaluation.get(field)
    if field == "deterministic_bots":
        if not isinstance(raw, Mapping) or not isinstance(raw.get("scores"), Mapping):
            raise ContinuationError("phase-B deterministic evaluation is invalid")
        scores = raw["scores"].values()
    else:
        if not isinstance(raw, Mapping):
            raise ContinuationError("phase-B frozen evaluation is invalid")
        scores = [item.get("score") for item in raw.values() if isinstance(item, Mapping)]
    values = [float(value) for value in scores if isinstance(value, int | float)]
    if not values:
        raise ContinuationError("phase-B evaluation has no scores")
    return sum(values) / len(values)


def _write_campaign_report(destination: Path, results: list[Mapping[str, object]]) -> None:
    """Write a compact comparative decision report without copying model artifacts."""

    lines = [
        "# RL-S5b-B — continuation contrôlée",
        "",
        "Les six finalistes Compact/Deep ont été poursuivis exactement 50 unités depuis "
        "l'unité 147. Le contrôle `random-only` poursuit Compact seed 19 contre le seul "
        "adversaire aléatoire. Les adversaires PPO employés pour l'entraînement et "
        "l'évaluation sont des instantanés immuables du pool phase A.",
        "",
        "| Branche | Pool entraînement | Δ bots déterministes | Δ finalistes figés |",
        "| --- | --- | ---: | ---: |",
    ]
    for result in results:
        label = result.get("label")
        if not isinstance(label, str):
            raise ContinuationError("phase-B branch result has no label")
        evaluations_path = destination / "continuation" / label / "evaluations.json"
        try:
            evaluations = json.loads(evaluations_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ContinuationError("phase-B branch evaluations cannot be read") from error
        if not isinstance(evaluations, Mapping):
            raise ContinuationError("phase-B branch evaluations are invalid")
        initial = evaluations.get(str(_PARENT_UNIT))
        final = evaluations.get(str(_PARENT_UNIT + _PHASE_B_UNITS))
        if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
            raise ContinuationError("phase-B branch is missing an endpoint evaluation")
        bot_delta = _mean_score(final, "deterministic_bots") - _mean_score(
            initial, "deterministic_bots"
        )
        frozen_delta = _mean_score(final, "frozen_finalists") - _mean_score(
            initial, "frozen_finalists"
        )
        metadata = result.get("metadata")
        random_only = isinstance(metadata, Mapping) and metadata.get("random_only_control") is True
        training = "random-only control" if random_only else "frozen mixed pool"
        lines.append(f"| `{label}` | {training} | {bot_delta:+.3f} | {frozen_delta:+.3f} |")
    lines.extend(
        (
            "",
            "Les scores sont des scores de match : victoire = 1, nul = 0,5, défaite = 0. "
            "Les variations mesurent 197 moins 147. Les confrontations PPO sont équilibrées "
            "sur les quatre positions, avec 100 confrontations par adversaire figé.",
            "",
        )
    )
    (destination / "continuation" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_one(
    *,
    pool: Mapping[str, object],
    entry: Mapping[str, object],
    output_directory: Path,
    random_only: bool,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    parent_directory, parent_state = _parent_state(entry)
    configuration = _configuration(parent_state)
    opponents = FrozenOpponentPool(pool, random_only=random_only)
    metadata = _metadata(pool, entry, opponents, random_only=random_only)
    parent_id = metadata["parent_checkpoint_id"]
    assert isinstance(parent_id, str)
    label = f"{parent_id}{'-random-only' if random_only else ''}"
    branch = output_directory / "continuation" / label
    recovery_store = AtomicRecoveryStore(branch / "recovery")
    ledger = UnitLedger(branch / "ledger.sqlite3")
    run_id = f"s5b-b-{hashlib.sha256(label.encode()).hexdigest()[:16]}"
    ledger.reserve(run_id=run_id, units=_PHASE_B_UNITS, idempotency_key=f"{run_id}-reserve")
    _atomic_json(
        branch / "branch.json",
        {
            "configuration": {
                "backend": configuration.backend.value,
                "n_envs": configuration.n_envs,
                "seed": configuration.seed,
            },
            "format": _FORMAT,
            "metadata": metadata,
            "parent_directory": str(parent_directory.resolve()),
            "phase": "B",
            "units": _PHASE_B_UNITS,
        },
    )
    latest = recovery_store.load_latest()
    if latest is None:
        model, environment = _restore_parent(
            parent_directory, parent_state, configuration, opponents
        )
        runner = AtomicPPOUnitRunner(
            run_id=run_id,
            configuration=configuration,
            model=model,
            environment=environment,
            recovery_store=recovery_store,
            ledger=ledger,
            recovery_metadata=metadata,
        )
        completed = 0
    else:
        runner, _last_metrics, latest = AtomicPPOUnitRunner.restore_latest(
            run_id=run_id,
            configuration=configuration,
            recovery_store=recovery_store,
            ledger=ledger,
            environment_factory=_environment_factory(opponents),
            recovery_metadata=metadata,
        )
        completed = latest.sequence
    if completed > _PHASE_B_UNITS:
        runner.environment.close()
        opponents.close()
        raise ContinuationError("phase-B recovery exceeds its fixed 50-unit budget")
    evaluations_path = branch / "evaluations.json"
    try:
        try:
            evaluations = json.loads(evaluations_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            evaluations = {}
        except json.JSONDecodeError as error:
            raise ContinuationError("phase-B evaluations file is invalid") from error
        if not isinstance(evaluations, dict):
            raise ContinuationError("phase-B evaluations file is invalid")
        if str(_PARENT_UNIT) not in evaluations:
            if progress is not None:
                progress(f"{label}: evaluating frozen benchmark at unit {_PARENT_UNIT}")
            evaluations[str(_PARENT_UNIT)] = _evaluation(
                runner.model,
                finalists=_finalists(pool),
                seed=derive_evaluation_seed(
                    configuration.seed,
                    suite="s5b-b-evaluation",
                    opponent_id=label,
                    index=0,
                ),
            )
            _atomic_json(evaluations_path, evaluations)
        for unit in range(completed + 1, _PHASE_B_UNITS + 1):
            result = runner.run_unit(unit_id=f"unit-{_PARENT_UNIT + unit}", sequence=unit)
            cycle = _metrics(result.metrics)
            cycle["checkpoint_seconds"] = result.checkpoint_seconds
            cycle["training_seconds"] = result.training_seconds
            cycle["unit"] = _PARENT_UNIT + unit
            _atomic_json(branch / "cycles" / f"unit-{_PARENT_UNIT + unit}.json", cycle)
            if progress is not None:
                progress(f"{label}: {unit}/{_PHASE_B_UNITS} units completed")
            if unit % _EVALUATION_INTERVAL == 0:
                if progress is not None:
                    progress(f"{label}: evaluating frozen benchmark at unit {_PARENT_UNIT + unit}")
                evaluations[str(_PARENT_UNIT + unit)] = _evaluation(
                    runner.model,
                    finalists=_finalists(pool),
                    seed=derive_evaluation_seed(
                        configuration.seed,
                        suite="s5b-b-evaluation",
                        opponent_id=label,
                        index=unit,
                    ),
                )
                _atomic_json(evaluations_path, evaluations)
        summary = {
            "completed_units": _PHASE_B_UNITS,
            "effective_training_weights": opponents.effective_weights,
            "evaluation_units": sorted(int(unit) for unit in evaluations),
            "format": _FORMAT,
            "label": label,
            "metadata": metadata,
            "status": "completed",
        }
        _atomic_json(branch / "result.json", summary)
        return summary
    finally:
        runner.environment.close()
        opponents.close()


def continue_phase_b(
    *,
    pool_path: Path,
    output_directory: Path | None = None,
    units: int = _PHASE_B_UNITS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run or resume all fixed RL-S5b-B branches and one random-only control.

    Phase B is intentionally fixed at 50 units.  Extending by the optional 97
    units is a later decision after these results have been reviewed.
    """

    if units != _PHASE_B_UNITS:
        raise ContinuationError("phase B is fixed to 50 units pending result review")
    pool = read_verified_pool(pool_path)
    destination = pool_path.parent.parent if output_directory is None else output_directory
    finalists = _finalists(pool)
    results = [
        _run_one(
            pool=pool,
            entry=entry,
            output_directory=destination,
            random_only=False,
            progress=progress,
        )
        for entry in finalists
    ]
    compact_seed_19 = next(
        entry
        for entry in finalists
        if entry["architecture"] == "mlp-compact-v2" and entry["seed"] == 19
    )
    results.append(
        _run_one(
            pool=pool,
            entry=compact_seed_19,
            output_directory=destination,
            random_only=True,
            progress=progress,
        )
    )
    result = {
        "branches": results,
        "format": _FORMAT,
        "pool_sha256": pool["pool_sha256"],
        "status": "completed",
    }
    _atomic_json(destination / "continuation" / "result.json", result)
    _write_campaign_report(destination, results)
    return result
