"""Validation and installation of light-weight server gate archives."""

from __future__ import annotations

import json
import math
from pathlib import Path

from dinorl_engine.server_checks.manifests import ArchiveValidationError, read_archive
from dinorl_engine.server_checks.profiles import ProfileError, get_profile
from dinorl_engine.server_checks.runner import ServerCheckError, verify_import_provenance
from dinorl_engine.server_checks.suites.rl_s1 import (
    REPEATED_MEASUREMENTS,
    build_decision,
    candidate_configurations,
)

__all__ = ["import_archive"]

_RL_S0_CONFIGURATION = {
    "warmups": 1,
    "repetitions": 1,
    "seed": 19,
    "rollout_transitions": 2048,
    "epochs": 4,
    "batch_size": 256,
}


def _verify_result_provenance(result: dict[str, object], root: Path) -> None:
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
    profile_name = result.get("dependency_profile")
    if not isinstance(profile_name, str):
        raise ServerCheckError("server archive dependency profile is invalid")
    try:
        profile = get_profile(profile_name)
    except ProfileError as error:
        raise ServerCheckError(str(error)) from error
    verify_import_provenance(root, profile, provenance)


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
    result: dict[str, object], manifest: dict[str, object], root: Path
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
        {
            "format",
            "suite",
            "dependency_profile",
            "run_id",
            "provenance",
            "configuration",
            "warmup",
            "repetitions",
        },
        "result",
    )
    if result.get("configuration") != _RL_S0_CONFIGURATION:
        raise ServerCheckError("server archive RL-S0 configuration is incomplete or unexpected")
    _verify_result_provenance(result, root)
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


def _validate_rl_s1_measurement(value: object, field_name: str) -> None:
    measurement = _expect_mapping(value, field_name)
    _require_exact_keys(
        measurement,
        {
            "duration_seconds",
            "cpu_seconds",
            "peak_memory_bytes",
            "phase_seconds",
            "learner_transitions",
            "engine_actions",
            "epochs",
            "optimizer_steps",
            "diagnostics",
        },
        field_name,
    )
    if (
        _expect_finite_number(measurement.get("duration_seconds"), f"{field_name}.duration_seconds")
        <= 0
    ):
        raise ServerCheckError(f"server archive {field_name}.duration_seconds must be positive")
    if _expect_finite_number(measurement.get("cpu_seconds"), f"{field_name}.cpu_seconds") < 0:
        raise ServerCheckError(f"server archive {field_name}.cpu_seconds must be non-negative")
    phases = _expect_mapping(measurement.get("phase_seconds"), f"{field_name}.phase_seconds")
    _require_exact_keys(phases, {"setup", "training", "closing"}, f"{field_name}.phase_seconds")
    for phase_name, phase_duration in phases.items():
        if _expect_finite_number(phase_duration, f"{field_name}.phase_seconds.{phase_name}") < 0:
            raise ServerCheckError(
                f"server archive {field_name}.phase_seconds.{phase_name} must be non-negative"
            )
    if measurement.get("learner_transitions") != 2048:
        raise ServerCheckError(f"server archive metric mismatch: {field_name}.learner_transitions")
    if measurement.get("epochs") != 4 or measurement.get("optimizer_steps") != 32:
        raise ServerCheckError(f"server archive metric mismatch: {field_name}.optimizer_steps")
    engine_actions = measurement.get("engine_actions")
    if (
        isinstance(engine_actions, bool)
        or not isinstance(engine_actions, int)
        or engine_actions < 2048
    ):
        raise ServerCheckError(f"server archive {field_name}.engine_actions is invalid")
    diagnostics = _expect_mapping(measurement.get("diagnostics"), f"{field_name}.diagnostics")
    if not diagnostics:
        raise ServerCheckError(f"server archive {field_name}.diagnostics is empty")
    for key, diagnostic in diagnostics.items():
        _expect_finite_number(diagnostic, f"{field_name}.diagnostics.{key}")


def _validate_rl_s1_result(
    result: dict[str, object], manifest: dict[str, object], root: Path
) -> tuple[str, str]:
    suite, run_id = result.get("suite"), result.get("run_id")
    if suite != "RL-S1" or not isinstance(run_id, str) or not run_id:
        raise ServerCheckError("server archive is not an RL-S1 result with a run identifier")
    if manifest.get("suite") != suite or manifest.get("run_id") != run_id:
        raise ServerCheckError("server archive manifest and result identity differ")
    _require_exact_keys(
        result,
        {
            "format",
            "suite",
            "dependency_profile",
            "run_id",
            "provenance",
            "configuration",
            "candidates",
            "decision",
        },
        "result",
    )
    if result.get("format") != "dinorl-server-result-v1":
        raise ServerCheckError("server archive result format is not supported")
    expected_candidates = [
        {"backend": item.backend.value, "n_envs": item.n_envs, "n_steps": item.n_steps}
        for item in candidate_configurations(seed=19)
    ]
    expected_configuration = {
        "warmups": 1,
        "repetitions": REPEATED_MEASUREMENTS,
        "seed": 19,
        "rollout_transitions": 2048,
        "epochs": 4,
        "batch_size": 256,
        "candidates": expected_candidates,
    }
    if result.get("configuration") != expected_configuration:
        raise ServerCheckError("server archive RL-S1 configuration is incomplete or unexpected")
    _verify_result_provenance(result, root)
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(expected_candidates):
        raise ServerCheckError("server archive RL-S1 candidates are incomplete")
    for index, candidate in enumerate(candidates):
        candidate_map = _expect_mapping(candidate, f"candidates[{index}]")
        _require_exact_keys(
            candidate_map,
            {"backend", "n_envs", "n_steps", "warmup", "repetitions"},
            f"candidates[{index}]",
        )
        if {
            key: candidate_map[key] for key in ("backend", "n_envs", "n_steps")
        } != expected_candidates[index]:
            raise ServerCheckError("server archive RL-S1 candidate matrix is unexpected")
        _validate_rl_s1_measurement(candidate_map.get("warmup"), f"candidates[{index}].warmup")
        repetitions = candidate_map.get("repetitions")
        if not isinstance(repetitions, list) or len(repetitions) != REPEATED_MEASUREMENTS:
            raise ServerCheckError("server archive RL-S1 repetitions are incomplete")
        for repetition_index, measurement in enumerate(repetitions):
            _validate_rl_s1_measurement(
                measurement, f"candidates[{index}].repetitions[{repetition_index}]"
            )
    try:
        expected_decision = build_decision(candidates)
    except ValueError as error:
        raise ServerCheckError(str(error)) from error
    if result.get("decision") != expected_decision:
        raise ServerCheckError("server archive RL-S1 decision does not match its measurements")
    return suite, run_id


def import_archive(*, archive_path: Path, root: Path) -> dict[str, object]:
    """Verify then install an RL-S0 server archive under ``benchmarks/server``."""

    if not archive_path.is_file():
        raise ServerCheckError(f"server archive does not exist: {archive_path}")
    try:
        archive = read_archive(archive_path)
    except ArchiveValidationError as error:
        raise ServerCheckError(str(error)) from error
    if archive.result.get("suite") == "RL-S0":
        suite, run_id = _validate_rl_s0_result(archive.result, archive.manifest, root)
    elif archive.result.get("suite") == "RL-S1":
        suite, run_id = _validate_rl_s1_result(archive.result, archive.manifest, root)
    else:
        raise ServerCheckError("server archive suite is not supported")
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
    if suite == "RL-S1":
        decision = _expect_mapping(archive.result["decision"], "decision")
        (destination / "decision.md").write_text(
            "# RL-S1 vectorization decision\n\n"
            f"- Backend: `{decision['backend']}`\n"
            f"- Environments: `{decision['n_envs']}`\n"
            f"- Steps per environment: `{decision['n_steps']}`\n"
            "- Median end-to-end throughput: "
            f"`{decision['median_transitions_per_second']}` transitions/s\n"
            "- Selection: highest median end-to-end throughput across the complete matrix.\n",
            encoding="utf-8",
        )
    return {
        "archive_sha256": archive.sha256,
        "destination": str(destination),
        "run_id": run_id,
        "status": "imported",
        "suite": suite,
    }
