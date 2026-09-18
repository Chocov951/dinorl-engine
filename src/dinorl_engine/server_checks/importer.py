"""Validation and installation of light-weight server gate archives."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from dinorl_engine.server_checks.manifests import ArchiveValidationError, read_archive
from dinorl_engine.server_checks.profiles import ProfileError, get_profile
from dinorl_engine.server_checks.runner import ServerCheckError, verify_import_provenance
from dinorl_engine.server_checks.suites.rl_s1 import (
    REPEATED_MEASUREMENTS,
    build_decision,
    candidate_configurations,
)
from dinorl_engine.server_checks.suites.rl_s2 import (
    CORPUS_TRANSITIONS,
    ITERATIONS,
    MAX_OVERHEAD_RATIO,
)
from dinorl_engine.server_checks.suites.rl_s2 import (
    REPEATED_MEASUREMENTS as RL_S2_REPEATED_MEASUREMENTS,
)
from dinorl_engine.server_checks.suites.rl_s3 import SELECTED_CONFIGURATION
from dinorl_engine.server_checks.suites.rl_s3 import build_decision as build_s3_decision
from dinorl_engine.server_checks.suites.rl_s4 import (
    ARCHITECTURES,
    UNITS_PER_ARCHITECTURE,
    smoke_passed,
)
from dinorl_engine.server_checks.suites.rl_s5 import (
    ARCHITECTURES as RL_S5_ARCHITECTURES,
)
from dinorl_engine.server_checks.suites.rl_s5 import (
    DEVELOPMENT_SEEDS,
    UNITS_PER_SEED,
    comparison_passed,
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
_RL_S2_CONFIGURATION = {
    "warmups": 1,
    "repetitions": RL_S2_REPEATED_MEASUREMENTS,
    "seed": 19,
    "corpus_transitions": CORPUS_TRANSITIONS,
    "iterations": ITERATIONS,
    "max_overhead_ratio": MAX_OVERHEAD_RATIO,
}
_RL_S3_CONFIGURATION = {
    "seed": SELECTED_CONFIGURATION.seed,
    "backend": SELECTED_CONFIGURATION.backend.value,
    "n_envs": SELECTED_CONFIGURATION.n_envs,
    "n_steps": SELECTED_CONFIGURATION.n_steps,
    "rollout_transitions": 2048,
    "priority_modes": ["unit_boundary", "separate_worker"],
}
_RL_S4_CONFIGURATION = {
    "seed": SELECTED_CONFIGURATION.seed,
    "backend": SELECTED_CONFIGURATION.backend.value,
    "n_envs": SELECTED_CONFIGURATION.n_envs,
    "n_steps": SELECTED_CONFIGURATION.n_steps,
    "rollout_transitions": 2048,
    "epochs": 4,
    "batch_size": 256,
    "units_per_architecture": UNITS_PER_ARCHITECTURE,
    "architectures": [architecture.value for architecture in ARCHITECTURES],
}
_RL_S5_CONFIGURATION = {
    "seed": SELECTED_CONFIGURATION.seed,
    "backend": SELECTED_CONFIGURATION.backend.value,
    "n_envs": SELECTED_CONFIGURATION.n_envs,
    "n_steps": SELECTED_CONFIGURATION.n_steps,
    "rollout_transitions": 2048,
    "epochs": 4,
    "batch_size": 256,
    "units_per_seed": UNITS_PER_SEED,
    "seeds": list(DEVELOPMENT_SEEDS),
    "architectures": [architecture.value for architecture in RL_S5_ARCHITECTURES],
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


def _validate_rl_s2_measurement(value: object, field_name: str) -> float:
    measurement = _expect_mapping(value, field_name)
    _require_exact_keys(
        measurement,
        {"ast_seconds", "native_seconds", "vm_seconds", "overhead_ratio", "transitions", "exact"},
        field_name,
    )
    for key in ("ast_seconds", "native_seconds", "vm_seconds"):
        if _expect_finite_number(measurement.get(key), f"{field_name}.{key}") <= 0:
            raise ServerCheckError(f"server archive {field_name}.{key} must be positive")
    overhead = _expect_finite_number(
        measurement.get("overhead_ratio"), f"{field_name}.overhead_ratio"
    )
    native_seconds = _expect_finite_number(
        measurement.get("native_seconds"), f"{field_name}.native_seconds"
    )
    vm_seconds = _expect_finite_number(measurement.get("vm_seconds"), f"{field_name}.vm_seconds")
    if not math.isclose(overhead, vm_seconds / native_seconds - 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ServerCheckError(f"server archive {field_name}.overhead_ratio is inconsistent")
    if measurement.get("transitions") != CORPUS_TRANSITIONS or measurement.get("exact") is not True:
        raise ServerCheckError(f"server archive {field_name} does not prove required exact parity")
    return overhead


def _validate_rl_s2_result(
    result: dict[str, object], manifest: dict[str, object], root: Path
) -> tuple[str, str]:
    suite, run_id = result.get("suite"), result.get("run_id")
    if suite != "RL-S2" or not isinstance(run_id, str) or not run_id:
        raise ServerCheckError("server archive is not an RL-S2 result with a run identifier")
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
            "warmup",
            "repetitions",
            "decision",
        },
        "result",
    )
    if result.get("format") != "dinorl-server-result-v1":
        raise ServerCheckError("server archive result format is not supported")
    if result.get("configuration") != _RL_S2_CONFIGURATION:
        raise ServerCheckError("server archive RL-S2 configuration is incomplete or unexpected")
    _verify_result_provenance(result, root)
    _validate_rl_s2_measurement(result.get("warmup"), "warmup")
    repetitions = result.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != RL_S2_REPEATED_MEASUREMENTS:
        raise ServerCheckError("server archive RL-S2 repetitions are incomplete")
    overheads = [
        _validate_rl_s2_measurement(repetition, f"repetitions[{index}]")
        for index, repetition in enumerate(repetitions)
    ]
    decision = _expect_mapping(result.get("decision"), "decision")
    _require_exact_keys(
        decision, {"max_overhead_ratio", "median_overhead_ratio", "passed"}, "decision"
    )
    median_overhead = statistics.median(overheads)
    expected_decision = {
        "max_overhead_ratio": MAX_OVERHEAD_RATIO,
        "median_overhead_ratio": median_overhead,
        "passed": median_overhead <= MAX_OVERHEAD_RATIO,
    }
    if decision != expected_decision:
        raise ServerCheckError("server archive RL-S2 decision does not match its measurements")
    return suite, run_id


def _validate_rl_s3_timing(value: object, field_name: str) -> None:
    timing = _expect_mapping(value, field_name)
    _require_exact_keys(
        timing,
        {"collection_seconds", "optimization_seconds", "checkpoint_seconds"},
        field_name,
    )
    for name, seconds in timing.items():
        if _expect_finite_number(seconds, f"{field_name}.{name}") < 0:
            raise ServerCheckError(f"server archive {field_name}.{name} must be non-negative")


def _validate_rl_s3_result(
    result: dict[str, object], manifest: dict[str, object], root: Path
) -> tuple[str, str]:
    suite, run_id = result.get("suite"), result.get("run_id")
    if suite != "RL-S3" or not isinstance(run_id, str) or not run_id:
        raise ServerCheckError("server archive is not an RL-S3 result with a run identifier")
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
            "resume",
            "crash",
            "priority_candidates",
            "decision",
            "passed",
        },
        "result",
    )
    if result.get("format") != "dinorl-server-result-v1":
        raise ServerCheckError("server archive result format is not supported")
    if result.get("configuration") != _RL_S3_CONFIGURATION:
        raise ServerCheckError("server archive RL-S3 configuration is incomplete or unexpected")
    _verify_result_provenance(result, root)
    resume = _expect_mapping(result.get("resume"), "resume")
    _require_exact_keys(
        resume,
        {
            "continuous_weights_sha256",
            "resumed_weights_sha256",
            "exact",
            "continuous",
            "resumed",
            "processes_cleaned_up",
        },
        "resume",
    )
    for name in ("continuous_weights_sha256", "resumed_weights_sha256"):
        digest = resume.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ServerCheckError(f"server archive resume.{name} is not a SHA-256 digest")
    if not isinstance(resume.get("exact"), bool) or not isinstance(
        resume.get("processes_cleaned_up"), bool
    ):
        raise ServerCheckError("server archive resume flags are invalid")
    _validate_rl_s3_timing(resume.get("continuous"), "resume.continuous")
    _validate_rl_s3_timing(resume.get("resumed"), "resume.resumed")
    crash = _expect_mapping(result.get("crash"), "crash")
    _require_exact_keys(
        crash,
        {"before_rename_preserved", "post_rename_recovered", "no_duplicate_debit"},
        "crash",
    )
    if not all(isinstance(value, bool) for value in crash.values()):
        raise ServerCheckError("server archive crash flags are invalid")
    candidates = result.get("priority_candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ServerCheckError("server archive RL-S3 priority candidates are incomplete")
    decision_inputs: list[dict[str, object]] = []
    for index, candidate_value in enumerate(candidates):
        candidate = _expect_mapping(candidate_value, f"priority_candidates[{index}]")
        _require_exact_keys(
            candidate,
            {
                "mode",
                "play_latency_seconds",
                "training_transitions_per_second",
                "interactive_served",
            },
            f"priority_candidates[{index}]",
        )
        if not isinstance(candidate.get("interactive_served"), bool):
            raise ServerCheckError(f"server archive priority_candidates[{index}] has invalid flag")
        decision_inputs.append(
            {
                "mode": candidate["mode"],
                "play_latency_seconds": candidate["play_latency_seconds"],
                "training_transitions_per_second": candidate["training_transitions_per_second"],
            }
        )
    try:
        expected_decision = build_s3_decision(decision_inputs)
    except ValueError as error:
        raise ServerCheckError(str(error)) from error
    if result.get("decision") != expected_decision:
        raise ServerCheckError("server archive RL-S3 decision does not match its measurements")
    expected_passed = (
        resume["exact"] is True
        and resume["processes_cleaned_up"] is True
        and crash
        == {
            "before_rename_preserved": True,
            "post_rename_recovered": True,
            "no_duplicate_debit": True,
        }
        and all(
            _expect_mapping(candidate, "priority candidate")["interactive_served"] is True
            for candidate in candidates
        )
    )
    if result.get("passed") is not expected_passed:
        raise ServerCheckError("server archive RL-S3 passed flag does not match its evidence")
    return suite, run_id


def _validate_rl_s4_result(
    result: dict[str, object], manifest: dict[str, object], root: Path
) -> tuple[str, str]:
    """Accept only a complete, comparable RL-S4 smoke archive."""

    suite, run_id = result.get("suite"), result.get("run_id")
    if suite != "RL-S4" or not isinstance(run_id, str) or not run_id:
        raise ServerCheckError("server archive is not an RL-S4 result with a run identifier")
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
            "measurement",
            "passed",
        },
        "result",
    )
    if result.get("format") != "dinorl-server-result-v1":
        raise ServerCheckError("server archive result format is not supported")
    if result.get("configuration") != _RL_S4_CONFIGURATION:
        raise ServerCheckError("server archive RL-S4 configuration is incomplete or unexpected")
    _verify_result_provenance(result, root)
    expected_passed = smoke_passed(result.get("measurement"))
    if result.get("passed") is not expected_passed:
        raise ServerCheckError("server archive RL-S4 passed flag does not match its evidence")
    return suite, run_id


def _validate_rl_s5_result(
    result: dict[str, object], manifest: dict[str, object], root: Path
) -> tuple[str, str]:
    suite, run_id = result.get("suite"), result.get("run_id")
    if suite != "RL-S5" or not isinstance(run_id, str) or not run_id:
        raise ServerCheckError("server archive is not an RL-S5 result with a run identifier")
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
            "measurement",
            "passed",
        },
        "result",
    )
    if result.get("format") != "dinorl-server-result-v1":
        raise ServerCheckError("server archive result format is not supported")
    if result.get("configuration") != _RL_S5_CONFIGURATION:
        raise ServerCheckError("server archive RL-S5 configuration is incomplete or unexpected")
    _verify_result_provenance(result, root)
    expected_passed = comparison_passed(result.get("measurement"))
    if result.get("passed") is not expected_passed:
        raise ServerCheckError("server archive RL-S5 passed flag does not match its evidence")
    return suite, run_id


def _validate_rl_s5b_a_result(
    result: dict[str, object], manifest: dict[str, object], root: Path
) -> tuple[str, str]:
    """Validate the light-weight evidence produced at server checkpoint RL-S5b-A."""

    del root  # S5b carries its immutable pool and Git evidence in the campaign manifest.
    suite, run_id = result.get("suite"), result.get("run_id")
    if suite != "RL-S5b-A" or not isinstance(run_id, str) or not run_id:
        raise ServerCheckError("server archive is not an RL-S5b-A result with a run identifier")
    if manifest.get("suite") != suite or manifest.get("run_id") != run_id:
        raise ServerCheckError("server archive manifest and result identity differ")
    _require_exact_keys(result, {"campaign", "crossplay", "format", "run_id", "suite"}, "result")
    if result.get("format") != "dinorl-s5b-server-result-v1":
        raise ServerCheckError("server archive RL-S5b-A format is not supported")
    campaign = _expect_mapping(result.get("campaign"), "campaign")
    crossplay = _expect_mapping(result.get("crossplay"), "crossplay")
    _require_exact_keys(
        campaign,
        {"config_sha256", "format", "git_commit", "phases", "pool_sha256", "run_id", "seeds"},
        "campaign",
    )
    phases = campaign.get("phases")
    if (
        campaign.get("format") != "s5b-run-manifest-v1"
        or campaign.get("run_id") != run_id
        or not isinstance(campaign.get("pool_sha256"), str)
        or not isinstance(phases, dict)
    ):
        raise ServerCheckError("server archive RL-S5b-A campaign is invalid")
    phase = phases.get("cross-evaluate")
    if not isinstance(phase, dict) or phase.get("status") != "completed":
        raise ServerCheckError("server archive RL-S5b-A cross-play is incomplete")
    records_completed = crossplay.get("records_completed")
    games_completed = crossplay.get("games_completed")
    if (
        crossplay.get("format") != "s5b-crossplay-v1"
        or crossplay.get("pool_sha256") != campaign["pool_sha256"]
        or type(records_completed) is not int
        or type(games_completed) is not int
        or not isinstance(crossplay.get("records"), list)
    ):
        raise ServerCheckError("server archive RL-S5b-A cross-play evidence is invalid")
    if records_completed <= 0 or games_completed <= 0:
        raise ServerCheckError("server archive RL-S5b-A cross-play has no completed games")
    return suite, run_id


def _validate_rl_s5c_a_result(
    result: dict[str, object], manifest: dict[str, object], root: Path
) -> tuple[str, str]:
    """Validate compact calibration evidence without importing specialist weights."""

    del root
    suite, run_id = result.get("suite"), result.get("run_id")
    if suite != "RL-S5c-A" or not isinstance(run_id, str) or not run_id:
        raise ServerCheckError("server archive is not an RL-S5c-A result with a run identifier")
    if manifest.get("suite") != suite or manifest.get("run_id") != run_id:
        raise ServerCheckError("server archive manifest and result identity differ")
    _require_exact_keys(result, {"calibration", "campaign", "format", "run_id", "suite"}, "result")
    if result.get("format") != "dinorl-s5c-server-result-v1":
        raise ServerCheckError("server archive RL-S5c-A format is not supported")
    campaign = _expect_mapping(result.get("campaign"), "campaign")
    calibration = _expect_mapping(result.get("calibration"), "calibration")
    if (
        campaign.get("format") != "s5c-run-manifest-v1"
        or campaign.get("run_id") != run_id
        or set(calibration) != {"scavenger", "predator", "controller"}
    ):
        raise ServerCheckError("server archive RL-S5c-A campaign is invalid")
    for archetype, item in calibration.items():
        value = _expect_mapping(item, f"calibration.{archetype}")
        if (
            value.get("format") != "s5c-calibration-v1"
            or value.get("archetype") != archetype
            or not isinstance(value.get("seeds"), list)
            or len(value["seeds"]) != 3
        ):
            raise ServerCheckError("server archive RL-S5c-A calibration is incomplete")
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
    elif archive.result.get("suite") == "RL-S2":
        suite, run_id = _validate_rl_s2_result(archive.result, archive.manifest, root)
    elif archive.result.get("suite") == "RL-S3":
        suite, run_id = _validate_rl_s3_result(archive.result, archive.manifest, root)
    elif archive.result.get("suite") == "RL-S4":
        suite, run_id = _validate_rl_s4_result(archive.result, archive.manifest, root)
    elif archive.result.get("suite") == "RL-S5":
        suite, run_id = _validate_rl_s5_result(archive.result, archive.manifest, root)
    elif archive.result.get("suite") == "RL-S5b-A":
        suite, run_id = _validate_rl_s5b_a_result(archive.result, archive.manifest, root)
    elif archive.result.get("suite") == "RL-S5c-A":
        suite, run_id = _validate_rl_s5c_a_result(archive.result, archive.manifest, root)
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
    elif suite == "RL-S2":
        decision = _expect_mapping(archive.result["decision"], "decision")
        status = "passed" if decision["passed"] else "blocked: profile before native extension"
        (destination / "decision.md").write_text(
            "# RL-S2 Reward VM decision\n\n"
            "- Exact AST/VM/native parity: `passed`\n"
            f"- Median VM overhead: `{decision['median_overhead_ratio']}`\n"
            f"- Maximum permitted overhead: `{decision['max_overhead_ratio']}`\n"
            f"- Status: {status}\n",
            encoding="utf-8",
        )
    elif suite == "RL-S3":
        decision = _expect_mapping(archive.result["decision"], "decision")
        status = "passed" if archive.result["passed"] is True else "failed"
        (destination / "decision.md").write_text(
            "# RL-S3 interactive priority decision\n\n"
            f"- Mode: `{decision['mode']}`\n"
            f"- Play latency: `{decision['play_latency_seconds']}` seconds\n"
            "- Selection: minimum play latency, then maximum learning throughput.\n"
            f"- Status: {status}\n",
            encoding="utf-8",
        )
    elif suite == "RL-S4":
        status = "passed" if archive.result["passed"] is True else "failed"
        (destination / "smoke.md").write_text(
            "# RL-S4 architecture smoke\n\n"
            f"- Units per architecture: `{UNITS_PER_ARCHITECTURE}`\n"
            "- Both MLP and small CNN use the same seed and PPO budget.\n"
            "- No architecture is selected at this checkpoint; selection is deferred to RL-S5.\n"
            f"- Status: {status}\n",
            encoding="utf-8",
        )
    elif suite == "RL-S5":
        measurement = _expect_mapping(archive.result["measurement"], "measurement")
        decision = _expect_mapping(measurement["decision"], "measurement.decision")
        status = "passed" if archive.result["passed"] is True else "failed"
        if decision.get("status") == "selected":
            content = (
                "# RL-S5 architecture decision\n\n"
                f"- Selected architecture: `{decision['architecture']}`\n"
                "- Criteria converged: speed, wall time and final score.\n"
                f"- Status: {status}\n"
            )
        else:
            content = (
                "# RL-S5 architecture decision\n\n"
                "- Status: collective decision required.\n"
                f"- Metric winners: `{decision.get('winners')}`\n"
                "- RL-L8 must not start before an explicit user decision.\n"
            )
        (destination / "architecture-decision.md").write_text(content, encoding="utf-8")
    elif suite == "RL-S5b-A":
        crossplay = _expect_mapping(archive.result["crossplay"], "crossplay")
        (destination / "crossplay.json").write_text(
            json.dumps(crossplay, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        (destination / "crossplay-report.md").write_text(archive.summary, encoding="utf-8")
    elif suite == "RL-S5c-A":
        calibration = _expect_mapping(archive.result["calibration"], "calibration")
        (destination / "calibration.json").write_text(
            json.dumps(calibration, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return {
        "archive_sha256": archive.sha256,
        "destination": str(destination),
        "run_id": run_id,
        "status": "imported",
        "suite": suite,
    }
