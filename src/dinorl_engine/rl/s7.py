"""RL-S7 server calibration suite, branched immutably from RL-S6 V3 evidence.

This module deliberately has no automatic invocation.  ``preflight`` creates
independent recovery roots and ``run_seed`` is the explicit, resumable server
operation.  The local ticket checks use only ``dry_run`` and pure analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Final, cast

from dinorl_engine.rl.contracts import canonical_json_bytes
from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.evaluation.suites import evaluate_deterministic
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.rewards.runtime import compile_reward
from dinorl_engine.rl.s6 import _UniformScriptedPool
from dinorl_engine.rl.s6_config import S6Config
from dinorl_engine.rl.s7_calibration import (
    CalibrationError,
    extract_curve_times,
    propose_daily_budget,
)
from dinorl_engine.rl.s7_config import S7Config
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore

__all__ = ["S7Error", "analyze", "run_preflight", "run_seed"]

_FORMAT: Final = "rl-s7-manifest-v1"


class S7Error(RuntimeError):
    """A server calibration input, branch, or measurement is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S7Error(f"invalid JSON: {path}") from error


def _atomic_json(path: Path, value: object) -> None:
    content = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _main_score(scores: object) -> float:
    if not isinstance(scores, Mapping):
        raise S7Error("deterministic evaluation lacks scores")
    identifiers = ("aggressive-v1", "prudent-v1", "opportunist-v1")
    values = [scores.get(identifier) for identifier in identifiers]
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        raise S7Error("deterministic evaluation lacks a scripted-bot score")
    result = sum(float(cast(float, value)) for value in values) / len(values)
    if not 0 <= result <= 1:
        raise S7Error("deterministic main score is outside [0, 1]")
    return result


def _source_candidates(source_v3: Path, config: S7Config) -> dict[int, Mapping[str, object]]:
    value = _read_json(source_v3 / "campaign-result.json")
    if not isinstance(value, Mapping) or value.get("decision") != "RL_S6_V3_PASSED":
        raise S7Error("RL-S7 requires a passed immutable RL-S6 V3 campaign result")
    candidates = value.get("seeds")
    if not isinstance(candidates, list):
        raise S7Error("RL-S6 V3 candidate list is invalid")
    by_seed: dict[int, Mapping[str, object]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise S7Error("RL-S6 V3 candidate record is invalid")
        seed = candidate.get("seed")
        unit = candidate.get("candidate_unit")
        checkpoint = candidate.get("checkpoint_sha256")
        if type(seed) is not int or type(unit) is not int or not isinstance(checkpoint, str):
            raise S7Error("RL-S6 V3 candidate identity is invalid")
        by_seed[seed] = candidate
    if set(config.seeds) - set(by_seed):
        raise S7Error("RL-S6 V3 lacks one or more requested RL-S7 seeds")
    return by_seed


def _source_curve(source_branch: Path) -> list[dict[str, object]]:
    records = _read_json(source_branch / "evaluations.json")
    if not isinstance(records, list):
        raise S7Error("RL-S6 source evaluations are invalid")
    result: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping) or type(record.get("unit")) is not int:
            raise S7Error("RL-S6 source evaluation record is invalid")
        deterministic = record.get("deterministic")
        if not isinstance(deterministic, Mapping):
            raise S7Error("RL-S6 source deterministic evaluation is invalid")
        result.append(
            {
                "unit": record["unit"],
                "score": _main_score(deterministic.get("scores")),
                "source": "rl-s6-v1",
                "checkpoint_sha256": record.get("checkpoint_sha256"),
            }
        )
    return result


def _metadata(s6: S6Config, seed: int) -> dict[str, object]:
    return {
        "phase": "RL-S7",
        "architecture": s6.architecture,
        "seed": seed,
        "starter_weights_sha256": s6.starter_weights_sha256,
        "reward_sha256": compile_reward(s6.reward_dsl).cache_key,
        "training_opponents": list(s6.training_opponents),
    }


def run_preflight(
    config: S7Config,
    *,
    source_v3: Path,
    source_s6_config: S6Config,
    output_directory: Path,
    dry_run: bool,
) -> dict[str, object]:
    """Verify V3 provenance and branch selected PPO recoveries without mutation."""

    if source_s6_config.architecture != "mlp-compact-v2":
        raise S7Error("RL-S7 source must use mlp-compact-v2")
    candidates = _source_candidates(source_v3, config)
    branch_records: list[dict[str, object]] = []
    for seed in config.seeds:
        candidate = candidates[seed]
        unit = candidate["candidate_unit"]
        if type(unit) is not int or unit >= config.max_units:
            raise S7Error("RL-S7 maximum must exceed every source candidate unit")
        source_branch = source_s6_config.output_directory / "seeds" / f"seed-{seed}"
        source_unit = source_branch / "recovery" / "units" / f"unit-{unit}"
        model = source_unit / "model.zip"
        state_path = source_unit / "state.json"
        ppo_state = source_unit / "ppo_state.npz"
        if not model.is_file() or not state_path.is_file() or not ppo_state.is_file():
            raise S7Error(f"RL-S6 source recovery is incomplete for seed {seed}, unit {unit}")
        expected = candidate.get("checkpoint_sha256")
        if not isinstance(expected, str) or _sha256(model) != expected:
            raise S7Error(f"RL-S6 V3 checkpoint hash differs for seed {seed}")
        branch_records.append(
            {
                "seed": seed,
                "source_unit": unit,
                "checkpoint_sha256": expected,
                "source_curve_points": len(_source_curve(source_branch)),
            }
        )
    manifest = {
        "format": _FORMAT,
        "status": "READY",
        "config_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "format": "dinorl-s7-config-v1",
                    "seeds": list(config.seeds),
                    "max_units": config.max_units,
                    "visible_score": config.visible_score,
                    "gate_score": config.gate_score,
                    "plateau_delta": config.plateau_delta,
                }
            )
        ).hexdigest(),
        "source_v3_result_sha256": _sha256(source_v3 / "campaign-result.json"),
        "source_s6_config_sha256": source_s6_config.sha256,
        "architecture": source_s6_config.architecture,
        "architecture_dimensions": list(source_s6_config.architecture_dimensions),
        "seeds": branch_records,
        "max_units": config.max_units,
        "evaluation_interval_units": source_s6_config.evaluation_interval_units,
        "metric": "mean deterministic score against aggressive-v1, prudent-v1, opportunist-v1",
        "definitions": {
            "T_visible": "first of two consecutive scores >= visible_score",
            "T_gate": "first score >= gate_score",
            "T_plateau": (
                "first post-increment point in three consecutive score improvements "
                "<= plateau_delta"
            ),
        },
    }
    path = output_directory / "manifest.json"
    expected_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if path.exists() and path.read_bytes() != expected_bytes:
        raise S7Error("existing RL-S7 manifest differs from frozen inputs")
    if dry_run:
        return {**manifest, "dry_run": True}
    if not path.exists():
        _atomic_json(path, manifest)
    for seed in config.seeds:
        candidate = candidates[seed]
        unit = cast(int, candidate["candidate_unit"])
        source_branch = source_s6_config.output_directory / "seeds" / f"seed-{seed}"
        destination = output_directory / "seeds" / f"seed-{seed}"
        recovery = AtomicRecoveryStore(destination / "recovery")
        latest = recovery.load_latest()
        if latest is None:
            source_unit = source_branch / "recovery" / "units" / f"unit-{unit}"
            state = _read_json(source_unit / "state.json")
            if not isinstance(state, dict):
                raise S7Error("RL-S6 source recovery state is invalid")
            state["run_id"] = f"rl-s7-{seed}"
            state["metadata"] = _metadata(source_s6_config, seed)

            def copy_files(directory: Path, source: Path = source_unit) -> tuple[str, ...]:
                return _copy_recovery_files(directory, source)

            recovery.commit_json_state(
                unit_id=f"unit-{unit}",
                sequence=unit,
                state=state,
                write_files=copy_files,
            )
            _atomic_json(destination / "curve.json", _source_curve(source_branch))
        elif latest.sequence != unit:
            raise S7Error("RL-S7 recovery does not match its frozen source checkpoint")
    return manifest


def _copy_recovery_files(directory: Path, source: Path) -> tuple[str, ...]:
    shutil.copy2(source / "model.zip", directory / "model.zip")
    shutil.copy2(source / "ppo_state.npz", directory / "ppo_state.npz")
    return "model.zip", "ppo_state.npz"


def _verified_manifest(config: S7Config, output_directory: Path) -> Mapping[str, object]:
    value = _read_json(output_directory / "manifest.json")
    expected = hashlib.sha256(
        canonical_json_bytes(
            {
                "format": "dinorl-s7-config-v1",
                "seeds": list(config.seeds),
                "max_units": config.max_units,
                "visible_score": config.visible_score,
                "gate_score": config.gate_score,
                "plateau_delta": config.plateau_delta,
            }
        )
    ).hexdigest()
    if (
        not isinstance(value, Mapping)
        or value.get("format") != _FORMAT
        or value.get("status") != "READY"
        or value.get("config_sha256") != expected
    ):
        raise S7Error("run s7 preflight before training")
    return value


def run_seed(
    config: S7Config,
    *,
    source_s6_config: S6Config,
    output_directory: Path,
    seed: int,
    resume: bool,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Resume exactly one frozen RL-S7 recovery branch through ``max_units``."""

    _verified_manifest(config, output_directory)
    if seed not in config.seeds:
        raise S7Error("seed is not part of the frozen RL-S7 campaign")
    branch = output_directory / "seeds" / f"seed-{seed}"
    result_path = branch / "result.json"
    if result_path.is_file():
        value = _read_json(result_path)
        if not resume:
            raise S7Error("completed RL-S7 seed requires no further execution")
        if not isinstance(value, dict):
            raise S7Error("RL-S7 seed result is invalid")
        return value
    pool = _UniformScriptedPool()
    vector = VectorEnvironmentConfig(VectorBackend.DUMMY, 2, seed)
    recovery = AtomicRecoveryStore(branch / "recovery")
    latest = recovery.load_latest()
    if latest is None:
        raise S7Error("run s7 preflight before training")
    if not resume and latest.sequence > 0:
        raise S7Error("existing RL-S7 branch requires --resume")
    ledger = UnitLedger(branch / "ledger.sqlite3")
    run_id = f"rl-s7-{seed}"
    ledger.reserve(run_id=run_id, units=config.max_units, idempotency_key=f"{run_id}-reserve")
    runner, _, latest = AtomicPPOUnitRunner.restore_latest(
        run_id=run_id,
        configuration=vector,
        recovery_store=recovery,
        ledger=ledger,
        environment_factory=lambda value: create_vector_environment(
            value,
            training_opponent_pool=pool,
            reward_program=compile_reward(source_s6_config.reward_dsl),
        ),
        recovery_metadata=_metadata(source_s6_config, seed),
    )
    started = time.perf_counter()
    try:
        curve = _read_json(branch / "curve.json")
        if not isinstance(curve, list):
            raise S7Error("RL-S7 curve is invalid")
        for unit in range(latest.sequence + 1, config.max_units + 1):
            trained = runner.run_unit(unit_id=f"unit-{unit}", sequence=unit)
            elapsed = max(time.perf_counter() - started, 0.001)
            if progress is not None:
                rate = (unit - latest.sequence) / elapsed
                eta = (config.max_units - unit) / rate if rate else 0.0
                progress(
                    f"RL-S7 seed {seed}: {unit}/{config.max_units} units "
                    f"({100 * unit / config.max_units:.1f}%) rate={rate:.3f} unit/s "
                    f"ETA={eta:.0f}s"
                )
            if unit % source_s6_config.evaluation_interval_units:
                continue
            deterministic = evaluate_deterministic(
                runner.model, seed=source_s6_config.evaluation_seed
            )
            curve.append(
                {
                    "unit": unit,
                    "score": _main_score(deterministic.get("scores")),
                    "source": "rl-s7",
                    "checkpoint_sha256": trained.recovery.files.get("model.zip"),
                    "deterministic": deterministic,
                }
            )
            _atomic_json(branch / "curve.json", curve)
        result = {
            "format": "rl-s7-seed-result-v1",
            "seed": seed,
            "status": "completed",
            "completed_units": config.max_units,
        }
        _atomic_json(result_path, result)
        return result
    finally:
        runner.environment.close()


def analyze(config: S7Config, *, output_directory: Path) -> dict[str, object]:
    """Derive capacity measurements solely from completed server curve artifacts."""

    manifest = _verified_manifest(config, output_directory)
    per_seed: list[dict[str, object]] = []
    for seed in config.seeds:
        curve = _read_json(output_directory / "seeds" / f"seed-{seed}" / "curve.json")
        if not isinstance(curve, list):
            raise S7Error("RL-S7 curve is invalid")
        try:
            times = extract_curve_times(
                cast(Sequence[Mapping[str, object]], curve),
                visible_score=config.visible_score,
                gate_score=config.gate_score,
                plateau_delta=config.plateau_delta,
            )
        except CalibrationError as error:
            raise S7Error("RL-S7 curve cannot be calibrated") from error
        per_seed.append({"seed": seed, "points": len(curve), **times})
    visible = [item["T_visible"] for item in per_seed]
    plateau = [item["T_plateau"] for item in per_seed]
    daily: int | None = None
    if all(type(value) is int for value in visible) and all(
        type(value) is int for value in plateau
    ):
        try:
            daily = propose_daily_budget(
                visible=max(cast(list[int], visible)), plateau=min(cast(list[int], plateau))
            )
        except CalibrationError:
            daily = None
    result: dict[str, object] = {
        "format": "rl-s7-capacity-report-v1",
        "status": "MEASURED",
        "manifest_sha256": _sha256(output_directory / "manifest.json"),
        "config": {
            "max_units": config.max_units,
            "seeds": list(config.seeds),
            "visible_score": config.visible_score,
            "gate_score": config.gate_score,
            "plateau_delta": config.plateau_delta,
        },
        "per_seed": per_seed,
        "typical": {
            key: median([cast(int, item[key]) for item in per_seed if type(item[key]) is int])
            if any(type(item[key]) is int for item in per_seed)
            else None
            for key in ("T_visible", "T_gate", "T_plateau")
        },
        "proposed_daily_budget": daily,
        "daily_budget_rule": "max(T_visible) <= D < min(T_plateau) / 2",
        "source_manifest": manifest,
    }
    _atomic_json(output_directory / "capacity-report.json", result)
    return result
