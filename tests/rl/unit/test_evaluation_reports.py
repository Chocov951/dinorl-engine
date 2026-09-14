"""RL-L7 JSONL and final-report persistence contracts."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from dinorl_engine.rl.evaluation.reports import RunReportWriter


def test_report_writer_only_persists_cycles_and_emits_a_schema_valid_final_report(
    tmp_path: Path,
) -> None:
    writer = RunReportWriter(tmp_path, run_id="run-19", architecture="mlp-v1", seed=19)
    writer.record_cycle(
        {
            "unit": 1,
            "learner_transitions": 2048,
            "engine_actions": 3000,
            "epochs": 4,
            "optimizer_steps": 32,
            "wall_seconds": 1.0,
            "cpu_seconds": 0.8,
        }
    )
    writer.record_evaluation(
        {
            "unit": 1,
            "suite": "deterministic",
            "score": 0.5,
            "games": 12,
            "diagnostic_replays": [],
        }
    )
    report_path = writer.write_final_report(final_score=0.5, inference_seconds=0.01)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    schema = json.loads(
        (
            Path(__file__).parents[3] / "schemas" / "rl" / "comparison-report-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert list(Draft202012Validator(schema).iter_errors(report)) == []
    assert len((tmp_path / "cycles.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len((tmp_path / "eval.jsonl").read_text(encoding="utf-8").splitlines()) == 1
