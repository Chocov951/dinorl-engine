"""RL-S0 archive integrity and import tests."""

from __future__ import annotations

from pathlib import Path

import dinorl_engine.server_checks.importer as importer
from dinorl_engine.server_checks.importer import import_archive
from dinorl_engine.server_checks.manifests import create_archive, read_archive
from dinorl_engine.server_checks.suites.rl_s1 import (
    REPEATED_MEASUREMENTS,
    build_decision,
    candidate_configurations,
)


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
