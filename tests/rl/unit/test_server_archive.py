"""RL-S0 archive integrity and import tests."""

from __future__ import annotations

from pathlib import Path

import dinorl_engine.server_checks.importer as importer
from dinorl_engine.rl.policies.factory import architecture_specification
from dinorl_engine.server_checks.importer import import_archive
from dinorl_engine.server_checks.manifests import create_archive, read_archive
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
from dinorl_engine.server_checks.suites.rl_s4 import ARCHITECTURES, UNITS_PER_ARCHITECTURE


def _result(provenance: dict[str, object]) -> dict[str, object]:
    return {
        "format": "dinorl-server-result-v1",
        "suite": "RL-S0",
        "dependency_profile": "standard",
        "run_id": "test-run-id",
        "provenance": provenance,
        "configuration": {
            "warmups": 1,
            "repetitions": 1,
            "seed": 19,
            "rollout_transitions": 2048,
            "epochs": 4,
            "batch_size": 256,
        },
        "warmup": {"duration_seconds": 0.25, "status": "passed"},
        "repetitions": [
            {
                "duration_seconds": 12.5,
                "peak_memory_bytes": 1024,
                "save_load_round_trip": True,
                "metrics": {
                    "learner_transitions": 2048,
                    "engine_actions": 3000,
                    "epochs": 4,
                    "optimizer_steps": 32,
                    "diagnostics": {"train/loss": 0.5},
                },
            }
        ],
    }


def test_rl_s0_archive_round_trips_and_imports_only_after_provenance_validation(
    tmp_path: Path, monkeypatch: object
) -> None:
    provenance = {
        "git_commit": "a" * 40,
        "lock_sha256": {"requirements.lock": "b" * 64},
        "installed_packages": {"torch": "2.14.0"},
        "python": "3.12.13",
        "platform": "test-platform",
        "processor": "test-processor",
        "cpu_count": 1,
    }
    archive_path = create_archive(
        output_dir=tmp_path,
        suite="RL-S0",
        run_id="test-run-id",
        result=_result(provenance),
        summary="# RL-S0\n",
    )

    archive = read_archive(archive_path)
    assert archive.result["run_id"] == "test-run-id"
    monkeypatch.setattr(importer, "verify_import_provenance", lambda *_args: None)  # type: ignore[attr-defined]

    imported = import_archive(archive_path=archive_path, root=tmp_path / "repository")

    destination = Path(str(imported["destination"]))
    assert imported["status"] == "imported"
    assert (destination / "manifest.json").is_file()
    assert (destination / "result.json").is_file()
    assert (destination / "summary.md").read_text(encoding="utf-8") == "# RL-S0\n"


def test_rl_s1_archive_imports_the_server_selected_decision(
    tmp_path: Path, monkeypatch: object
) -> None:
    configurations = candidate_configurations(seed=19)
    measurement = {
        "duration_seconds": 2.0,
        "cpu_seconds": 1.5,
        "peak_memory_bytes": 1024,
        "phase_seconds": {"setup": 0.1, "training": 1.8, "closing": 0.1},
        "learner_transitions": 2048,
        "engine_actions": 3000,
        "epochs": 4,
        "optimizer_steps": 32,
        "diagnostics": {"train/loss": 0.5},
    }
    candidates = [
        {
            "backend": configuration.backend.value,
            "n_envs": configuration.n_envs,
            "n_steps": configuration.n_steps,
            "warmup": measurement,
            "repetitions": [measurement for _ in range(REPEATED_MEASUREMENTS)],
        }
        for configuration in configurations
    ]
    result = {
        "format": "dinorl-server-result-v1",
        "suite": "RL-S1",
        "dependency_profile": "standard",
        "run_id": "test-s1-run-id",
        "provenance": {
            "git_commit": "a" * 40,
            "lock_sha256": {"requirements.lock": "b" * 64},
            "installed_packages": {"torch": "2.14.0"},
            "python": "3.12.13",
            "platform": "test-platform",
            "processor": "test-processor",
            "cpu_count": 1,
        },
        "configuration": {
            "warmups": 1,
            "repetitions": REPEATED_MEASUREMENTS,
            "seed": 19,
            "rollout_transitions": 2048,
            "epochs": 4,
            "batch_size": 256,
            "candidates": [
                {
                    "backend": configuration.backend.value,
                    "n_envs": configuration.n_envs,
                    "n_steps": configuration.n_steps,
                }
                for configuration in configurations
            ],
        },
        "candidates": candidates,
        "decision": build_decision(candidates),
    }
    archive_path = create_archive(
        output_dir=tmp_path,
        suite="RL-S1",
        run_id="test-s1-run-id",
        result=result,
        summary="# RL-S1\n",
    )
    monkeypatch.setattr(importer, "verify_import_provenance", lambda *_args: None)  # type: ignore[attr-defined]

    imported = import_archive(archive_path=archive_path, root=tmp_path / "repository")

    decision = Path(str(imported["destination"])) / "decision.md"
    assert decision.is_file()
    assert "highest median end-to-end throughput" in decision.read_text(encoding="utf-8")


def test_rl_s2_archive_imports_a_complete_blocked_gate(tmp_path: Path, monkeypatch: object) -> None:
    measurement = {
        "ast_seconds": 2.0,
        "native_seconds": 1.0,
        "vm_seconds": 1.2,
        "overhead_ratio": 0.2,
        "transitions": CORPUS_TRANSITIONS,
        "exact": True,
    }
    result = {
        "format": "dinorl-server-result-v1",
        "suite": "RL-S2",
        "dependency_profile": "standard",
        "run_id": "test-s2-run-id",
        "provenance": {
            "git_commit": "a" * 40,
            "lock_sha256": {"requirements.lock": "b" * 64},
            "installed_packages": {"torch": "2.14.0"},
            "python": "3.12.13",
            "platform": "test-platform",
            "processor": "test-processor",
            "cpu_count": 1,
        },
        "configuration": {
            "warmups": 1,
            "repetitions": RL_S2_REPEATED_MEASUREMENTS,
            "seed": 19,
            "corpus_transitions": CORPUS_TRANSITIONS,
            "iterations": ITERATIONS,
            "max_overhead_ratio": MAX_OVERHEAD_RATIO,
        },
        "warmup": measurement,
        "repetitions": [measurement for _ in range(RL_S2_REPEATED_MEASUREMENTS)],
        "decision": {
            "max_overhead_ratio": MAX_OVERHEAD_RATIO,
            "median_overhead_ratio": 0.2,
            "passed": False,
        },
    }
    archive_path = create_archive(
        output_dir=tmp_path,
        suite="RL-S2",
        run_id="test-s2-run-id",
        result=result,
        summary="# RL-S2\n",
    )
    monkeypatch.setattr(importer, "verify_import_provenance", lambda *_args: None)  # type: ignore[attr-defined]

    imported = import_archive(archive_path=archive_path, root=tmp_path / "repository")

    decision = Path(str(imported["destination"])) / "decision.md"
    assert "blocked: profile before native extension" in decision.read_text(encoding="utf-8")


def test_rl_s3_archive_imports_the_versioned_interactive_priority_decision(
    tmp_path: Path, monkeypatch: object
) -> None:
    candidates = [
        {
            "mode": "unit_boundary",
            "play_latency_seconds": 2.0,
            "training_transitions_per_second": 100.0,
            "interactive_served": True,
        },
        {
            "mode": "separate_worker",
            "play_latency_seconds": 0.1,
            "training_transitions_per_second": 90.0,
            "interactive_served": True,
        },
    ]
    result = {
        "format": "dinorl-server-result-v1",
        "suite": "RL-S3",
        "dependency_profile": "standard",
        "run_id": "test-s3-run-id",
        "provenance": {
            "git_commit": "a" * 40,
            "lock_sha256": {"requirements.lock": "b" * 64},
            "installed_packages": {"torch": "2.14.0"},
            "python": "3.12.13",
            "platform": "test-platform",
            "processor": "test-processor",
            "cpu_count": 1,
        },
        "configuration": {
            "seed": SELECTED_CONFIGURATION.seed,
            "backend": SELECTED_CONFIGURATION.backend.value,
            "n_envs": SELECTED_CONFIGURATION.n_envs,
            "n_steps": SELECTED_CONFIGURATION.n_steps,
            "rollout_transitions": 2048,
            "priority_modes": ["unit_boundary", "separate_worker"],
        },
        "resume": {
            "continuous_weights_sha256": "a" * 64,
            "resumed_weights_sha256": "a" * 64,
            "exact": True,
            "continuous": {
                "collection_seconds": 1.0,
                "optimization_seconds": 1.0,
                "checkpoint_seconds": 0.1,
            },
            "resumed": {
                "collection_seconds": 1.0,
                "optimization_seconds": 1.0,
                "checkpoint_seconds": 0.1,
            },
            "processes_cleaned_up": True,
        },
        "crash": {
            "before_rename_preserved": True,
            "post_rename_recovered": True,
            "no_duplicate_debit": True,
        },
        "priority_candidates": candidates,
        "decision": build_s3_decision(
            [
                {
                    key: candidate[key]
                    for key in ("mode", "play_latency_seconds", "training_transitions_per_second")
                }
                for candidate in candidates
            ]
        ),
        "passed": True,
    }
    archive_path = create_archive(
        output_dir=tmp_path,
        suite="RL-S3",
        run_id="test-s3-run-id",
        result=result,
        summary="# RL-S3\n",
    )
    monkeypatch.setattr(importer, "verify_import_provenance", lambda *_args: None)  # type: ignore[attr-defined]

    imported = import_archive(archive_path=archive_path, root=tmp_path / "repository")

    decision = Path(str(imported["destination"])) / "decision.md"
    assert "separate_worker" in decision.read_text(encoding="utf-8")


def test_rl_s4_archive_imports_a_smoke_without_an_architecture_decision(
    tmp_path: Path, monkeypatch: object
) -> None:
    candidates: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        specification = architecture_specification(architecture)
        candidates.append(
            {
                "architecture": architecture.value,
                "encoder_parameters": specification.encoder_parameters,
                "policy_head_parameters": 585,
                "value_head_parameters": 65,
                "total_parameters": specification.encoder_parameters + 650,
                "approximate_inference_multiply_accumulates": (
                    specification.encoder_multiply_accumulates + 640
                ),
                "inference_seconds_per_observation": 0.001,
                "units": [
                    {
                        "learner_transitions": 2048,
                        "engine_actions": 3_000,
                        "epochs": 4,
                        "optimizer_steps": 32,
                        "diagnostics_finite": True,
                        "legal_actions_only": True,
                        "training_seconds": 1.0,
                        "checkpoint_seconds": 0.1,
                    }
                    for _ in range(UNITS_PER_ARCHITECTURE)
                ],
            }
        )
    result = {
        "format": "dinorl-server-result-v1",
        "suite": "RL-S4",
        "dependency_profile": "standard",
        "run_id": "test-s4-run-id",
        "provenance": {
            "git_commit": "a" * 40,
            "lock_sha256": {"requirements.lock": "b" * 64},
            "installed_packages": {"torch": "2.14.0"},
            "python": "3.12.13",
            "platform": "test-platform",
            "processor": "test-processor",
            "cpu_count": 1,
        },
        "configuration": {
            "seed": SELECTED_CONFIGURATION.seed,
            "backend": SELECTED_CONFIGURATION.backend.value,
            "n_envs": SELECTED_CONFIGURATION.n_envs,
            "n_steps": SELECTED_CONFIGURATION.n_steps,
            "rollout_transitions": 2048,
            "epochs": 4,
            "batch_size": 256,
            "units_per_architecture": UNITS_PER_ARCHITECTURE,
            "architectures": [architecture.value for architecture in ARCHITECTURES],
        },
        "measurement": {"candidates": candidates},
        "passed": True,
    }
    archive_path = create_archive(
        output_dir=tmp_path,
        suite="RL-S4",
        run_id="test-s4-run-id",
        result=result,
        summary="# RL-S4\n",
    )
    monkeypatch.setattr(importer, "verify_import_provenance", lambda *_args: None)  # type: ignore[attr-defined]

    imported = import_archive(archive_path=archive_path, root=tmp_path / "repository")

    destination = Path(str(imported["destination"]))
    assert (destination / "smoke.md").is_file()
    assert not (destination / "decision.md").exists()
