"""Fast contract tests for the Python match benchmark."""

import json
from pathlib import Path

import pytest
from benchmarks.benchmark_matches import (
    REQUIRED_PROCESS_COUNTS,
    main,
    run_baseline,
    write_report,
)

ROOT = Path(__file__).parents[2]
BASELINE_PATH = ROOT / "benchmarks" / "results" / "baseline-2026-09-10.json"


def test_quick_baseline_exposes_every_required_metric_with_replay_disabled() -> None:
    report = run_baseline(
        process_counts=(1,),
        seeds=(0,),
        warmup_repetitions=1,
        measured_repetitions=1,
    )

    assert report["benchmark"] == "dinorl-match-baseline"
    assert report["workload"]["map_id"] == "arena_mvp_v1"
    assert report["workload"]["include_replay"] is False
    assert len(report["workload"]["controller_pairs"]) == 9
    measurement = report["measurements"][0]
    assert measurement["processes"] == 1
    assert measurement["matches_per_second"] > 0
    assert measurement["actions_per_second"] > 0
    assert measurement["mean_match_seconds"] > 0
    assert measurement["p95_match_seconds"] > 0
    assert measurement["speedup"] == 1.0
    assert measurement["peak_memory_bytes_per_process"] > 0
    assert report["environment"]["python_version"]
    assert report["environment"]["cpu"]
    assert report["environment"]["system"]


def test_report_export_is_machine_readable_json(tmp_path: Path) -> None:
    report = run_baseline(
        process_counts=(1,),
        seeds=(1,),
        warmup_repetitions=0,
        measured_repetitions=1,
    )
    output = tmp_path / "baseline.json"

    write_report(report, output)

    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_cli_refuses_a_non_baseline_run_with_fewer_than_five_repetitions() -> None:
    with pytest.raises(SystemExit) as captured:
        main(["--processes", "1", "--repetitions", "4"])

    assert captured.value.code == 2


def test_archived_baseline_contains_the_required_process_matrix() -> None:
    report = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert [entry["processes"] for entry in report["measurements"]] == list(REQUIRED_PROCESS_COUNTS)
    assert report["workload"]["measured_repetitions"] >= 5
    assert report["workload"]["include_replay"] is False
