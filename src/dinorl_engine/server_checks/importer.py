"""Validation and installation of light-weight server gate archives."""

from __future__ import annotations

import json
import math
from pathlib import Path

from dinorl_engine.server_checks.manifests import ArchiveValidationError, read_archive
from dinorl_engine.server_checks.runner import ServerCheckError, collect_provenance

__all__ = ["import_archive"]

_RL_S0_CONFIGURATION = {
    "warmups": 1,
    "repetitions": 1,
    "seed": 19,
    "rollout_transitions": 2048,
    "epochs": 4,
    "batch_size": 256,
}


def _require_exact_keys(value: dict[str, object], expected: set[str], field_name: str) -> None:
    if set(value) != expected:
        raise ServerCheckError(f"server archive {field_name} has unexpected or missing fields")


def _expect_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ServerCheckError(f"server archive {field_name} must be an object")
    return value


def _expect_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ServerCheckError(f"server archive {field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ServerCheckError(f"server archive {field_name} must be finite")
    return number


def _validate_rl_s0_result(
    result: dict[str, object], manifest: dict[str, object], current_provenance: dict[str, object]
) -> tuple[str, str]:
    """Validate the complete fixed RL-S0 result and its local compatibility."""

    suite = result.get("suite")
    run_id = result.get("run_id")
    if suite != "RL-S0" or not isinstance(run_id, str) or not run_id:
        raise ServerCheckError("server archive is not an RL-S0 result with a run identifier")
    if manifest.get("suite") != suite or manifest.get("run_id") != run_id:
        raise ServerCheckError("server archive manifest and result identity differ")
    if result.get("format") != "dinorl-server-result-v1":
        raise ServerCheckError("server archive result format is not supported")
    _require_exact_keys(
        result,
        {"format", "suite", "run_id", "provenance", "configuration", "warmup", "repetitions"},
        "result",
    )
    if result.get("configuration") != _RL_S0_CONFIGURATION:
        raise ServerCheckError("server archive RL-S0 configuration is incomplete or unexpected")
    provenance = _expect_mapping(result.get("provenance"), "provenance")
    _require_exact_keys(
        provenance,
        {
            "git_commit",
            "lock_sha256",
            "installed_packages",
            "python",
            "platform",
            "processor",
            "cpu_count",
        },
        "provenance",
    )
    for key in ("git_commit", "lock_sha256", "installed_packages"):
        if provenance.get(key) != current_provenance.get(key):
            raise ServerCheckError(f"server archive provenance mismatch: {key}")
    warmup = _expect_mapping(result.get("warmup"), "warmup")
    _require_exact_keys(warmup, {"duration_seconds", "status"}, "warmup")
    if warmup.get("status") != "passed":
        raise ServerCheckError("server archive warmup did not pass")
    _expect_finite_number(warmup.get("duration_seconds"), "warmup.duration_seconds")
    repetitions = result.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != 1:
        raise ServerCheckError("server archive RL-S0 repetitions are incomplete")
    measurement = _expect_mapping(repetitions[0], "repetitions[0]")
    _require_exact_keys(
        measurement,
        {"duration_seconds", "peak_memory_bytes", "save_load_round_trip", "metrics"},
        "repetitions[0]",
    )
    _expect_finite_number(measurement.get("duration_seconds"), "repetitions[0].duration_seconds")
    peak_memory = measurement.get("peak_memory_bytes")
    if peak_memory is not None and (
        isinstance(peak_memory, bool) or not isinstance(peak_memory, int)
    ):
        raise ServerCheckError("server archive repetitions[0].peak_memory_bytes is invalid")
    if measurement.get("save_load_round_trip") is not True:
        raise ServerCheckError("server archive save/load check did not pass")
    metrics = _expect_mapping(measurement.get("metrics"), "repetitions[0].metrics")
    _require_exact_keys(
        metrics,
        {"learner_transitions", "engine_actions", "epochs", "optimizer_steps", "diagnostics"},
        "repetitions[0].metrics",
    )
    expected_metrics = {
        "learner_transitions": 2048,
        "epochs": 4,
        "optimizer_steps": 32,
    }
    for key, expected in expected_metrics.items():
        if metrics.get(key) != expected:
            raise ServerCheckError(f"server archive metric mismatch: {key}")
    engine_actions = metrics.get("engine_actions")
    if (
        isinstance(engine_actions, bool)
        or not isinstance(engine_actions, int)
        or engine_actions < 2048
    ):
        raise ServerCheckError("server archive metric engine_actions is invalid")
    diagnostics = _expect_mapping(metrics.get("diagnostics"), "repetitions[0].metrics.diagnostics")
    if not diagnostics:
        raise ServerCheckError("server archive contains no PPO diagnostics")
    for key, value in diagnostics.items():
        _expect_finite_number(value, f"repetitions[0].metrics.diagnostics.{key}")
    return suite, run_id


def import_archive(*, archive_path: Path, root: Path) -> dict[str, object]:
    """Verify then install an RL-S0 server archive under ``benchmarks/server``."""

    if not archive_path.is_file():
        raise ServerCheckError(f"server archive does not exist: {archive_path}")
    try:
        archive = read_archive(archive_path)
    except ArchiveValidationError as error:
        raise ServerCheckError(str(error)) from error
    suite, run_id = _validate_rl_s0_result(
        archive.result,
        archive.manifest,
        collect_provenance(root),
    )
    destination = root / "benchmarks" / "server" / suite / run_id
    if destination.exists():
        raise ServerCheckError(f"server archive has already been imported: {destination}")
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text(
        json.dumps(archive.manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (destination / "result.json").write_text(
        json.dumps(archive.result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (destination / "summary.md").write_text(archive.summary, encoding="utf-8")
    return {
        "archive_sha256": archive.sha256,
        "destination": str(destination),
        "run_id": run_id,
        "status": "imported",
        "suite": suite,
    }
