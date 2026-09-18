"""RL-S4 server-runner archive construction, independent of the target host."""

from __future__ import annotations

import json
from pathlib import Path

import dinorl_engine.server_checks.runner as runner
from dinorl_engine.rl.policies.factory import architecture_specification
from dinorl_engine.server_checks.manifests import read_archive
from dinorl_engine.server_checks.profiles import STANDARD_PROFILE
from dinorl_engine.server_checks.suites.rl_s4 import ARCHITECTURES, UNITS_PER_ARCHITECTURE


def _candidate(architecture_index: int) -> dict[str, object]:
    architecture = ARCHITECTURES[architecture_index]
    specification = architecture_specification(architecture)
    return {
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


def test_rl_s4_runner_archives_a_comparable_smoke_without_a_selection(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        runner,
        "run_s4_measurement",
        lambda: {"candidates": [_candidate(0), _candidate(1)]},
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        runner, "collect_provenance", lambda *_args: {"test": "provenance"}
    )

    result = runner._run_rl_s4(profile=STANDARD_PROFILE, root=tmp_path, output_dir=tmp_path)

    archive = read_archive(Path(str(result["archive"])))
    assert result["status"] == "passed"
    assert archive.result["passed"] is True
    assert "decision" not in archive.result
    assert "selection is deferred" in archive.summary


def test_local_rl_s5_diagnostic_is_not_an_importable_server_result(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        runner,
        "run_s5_measurement",
        lambda *_args, **_kwargs: {"candidates": [], "decision": {"status": "test"}},
    )

    result = runner.run_local_rl_s5_diagnostic(output_dir=tmp_path)

    report = json.loads(Path(str(result["report"])).read_text(encoding="utf-8"))
    assert result["classification"] == "diagnostic_only_not_server_evidence"
    assert report["classification"] == "diagnostic_only_not_server_evidence"
    assert report["format"] == "dinorl-local-rl-s5-diagnostic-v1"


def test_local_rl_s5_v2_report_is_non_importable_and_excludes_self_play(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        runner,
        "run_s5_v2_measurement",
        lambda *_args, **_kwargs: {"tournament": {"self_play": False}},
    )

    result = runner.run_local_rl_s5_v2_diagnostic(
        output_dir=tmp_path,
        baseline_work_directory=tmp_path / "baseline",
    )

    report = json.loads(Path(str(result["report"])).read_text(encoding="utf-8"))
    assert report["format"] == "dinorl-local-rl-s5-v2-diagnostic-v1"
    assert report["classification"] == "diagnostic_only_not_server_evidence"
    assert report["configuration"]["candidate_architectures"] == [
        "mlp-compact-v2",
        "mlp-balanced-v2",
        "mlp-deep-v2",
    ]
    assert report["configuration"]["self_play"] is False
