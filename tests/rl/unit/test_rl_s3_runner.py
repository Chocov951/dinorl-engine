"""RL-S3 server-runner archive construction, independent of the target host."""

from __future__ import annotations

from pathlib import Path

import dinorl_engine.server_checks.runner as runner
from dinorl_engine.server_checks.manifests import read_archive
from dinorl_engine.server_checks.profiles import STANDARD_PROFILE


def test_rl_s3_runner_archives_only_the_decision_supported_by_its_measurement(
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
    monkeypatch.setattr(  # type: ignore[attr-defined]
        runner,
        "run_s3_measurement",
        lambda: {
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
            "decision": {
                "mode": "separate_worker",
                "criterion": "minimum_play_latency_then_maximum_training_throughput",
                "play_latency_seconds": 0.1,
                "training_transitions_per_second": 90.0,
            },
        },
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        runner, "collect_provenance", lambda *_args: {"test": "provenance"}
    )

    result = runner._run_rl_s3(profile=STANDARD_PROFILE, root=tmp_path, output_dir=tmp_path)

    archive = read_archive(Path(str(result["archive"])))
    assert result["status"] == "passed"
    assert archive.result["decision"] == {
        "mode": "separate_worker",
        "criterion": "minimum_play_latency_then_maximum_training_throughput",
        "play_latency_seconds": 0.1,
        "training_transitions_per_second": 90.0,
    }
