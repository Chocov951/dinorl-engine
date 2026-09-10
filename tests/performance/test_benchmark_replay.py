"""Fast contract tests for replay overhead and size measurement."""

import json
from pathlib import Path

import pytest
from benchmarks.benchmark_replay import (
    MAX_REPLAY_BYTES,
    MAX_RESPONSE_BYTES,
    main,
    run_replay_benchmark,
    write_report,
)

ROOT = Path(__file__).parents[2]
REPORT_PATH = ROOT / "benchmarks" / "results" / "replay-cost-2026-09-10.json"


def test_quick_replay_benchmark_compares_throughput_and_all_size_percentiles() -> None:
    report = run_replay_benchmark(
        seeds=(0,),
        warmup_repetitions=0,
        measured_repetitions=1,
    )

    assert report["benchmark"] == "dinorl-replay-cost"
    without_replay, with_replay = report["measurements"]
    assert without_replay["include_replay"] is False
    assert with_replay["include_replay"] is True
    assert without_replay["matches_per_second"] > 0
    assert with_replay["matches_per_second"] > 0
    assert without_replay["actions_per_second"] > 0
    assert with_replay["actions_per_second"] > 0
    for size_name in ("replay_size_bytes", "response_size_bytes"):
        sizes = with_replay[size_name]
        assert 0 < sizes["p50"] <= sizes["p95"] <= sizes["max"]
    assert with_replay["replay_size_bytes"]["max"] <= MAX_REPLAY_BYTES
    assert with_replay["response_size_bytes"]["max"] <= MAX_RESPONSE_BYTES
    assert report["comparison"]["throughput_ratio_with_over_without"] > 0
    assert report["limits"]["all_samples_compliant"] is True


def test_cli_refuses_a_report_with_fewer_than_five_repetitions() -> None:
    with pytest.raises(SystemExit) as captured:
        main(["--repetitions", "4"])

    assert captured.value.code == 2


def test_replay_report_export_is_machine_readable_json(tmp_path: Path) -> None:
    report = run_replay_benchmark(
        seeds=(1,),
        warmup_repetitions=0,
        measured_repetitions=1,
    )
    output = tmp_path / "replay-cost.json"

    write_report(report, output)

    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_archived_replay_report_has_five_measured_repetitions_and_limits() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["workload"]["measured_repetitions"] >= 5
    assert report["limits"] == {
        "all_samples_compliant": True,
        "replay_max_bytes": MAX_REPLAY_BYTES,
        "response_max_bytes": MAX_RESPONSE_BYTES,
    }
    with_replay = report["measurements"][1]
    assert with_replay["replay_size_bytes"]["max"] <= MAX_REPLAY_BYTES
    assert with_replay["response_size_bytes"]["max"] <= MAX_RESPONSE_BYTES
