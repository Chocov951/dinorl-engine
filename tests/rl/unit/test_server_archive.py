"""RL-S0 archive integrity and import tests."""

from __future__ import annotations

from pathlib import Path

import dinorl_engine.server_checks.importer as importer
from dinorl_engine.server_checks.importer import import_archive
from dinorl_engine.server_checks.manifests import create_archive, read_archive


def _result(provenance: dict[str, object]) -> dict[str, object]:
    return {
        "format": "dinorl-server-result-v1",
        "suite": "RL-S0",
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
    monkeypatch.setattr(importer, "collect_provenance", lambda _root: provenance)  # type: ignore[attr-defined]

    imported = import_archive(archive_path=archive_path, root=tmp_path / "repository")

    destination = Path(str(imported["destination"]))
    assert imported["status"] == "imported"
    assert (destination / "manifest.json").is_file()
    assert (destination / "result.json").is_file()
    assert (destination / "summary.md").read_text(encoding="utf-8") == "# RL-S0\n"
