"""Measure replay throughput overhead and serialized response sizes."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Final

from benchmarks.benchmark_matches import environment_metadata, write_report
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.core.constants import (
    ENGINE_VERSION,
    MAP_ID,
    PROTOCOL_VERSION,
    REPLAY_VERSION,
    RULES_VERSION,
)
from dinorl_engine.match.replay import canonical_replay_json
from dinorl_engine.match.runner import run_match

MAX_REPLAY_BYTES: Final = 480 * 1024
MAX_RESPONSE_BYTES: Final = 512 * 1024
DEFAULT_SEEDS: Final = tuple(range(20))
DEFAULT_REPETITIONS: Final = 5
DEFAULT_WARMUP_REPETITIONS: Final = 1
_BENCHMARK_REQUEST_ID: Final = "00000000-0000-4000-8000-000000000000"

type MatchCase = tuple[str, str, int]


def _nearest_rank(values: Sequence[int], percentile: int) -> int:
    ordered = sorted(values)
    index = max(0, (percentile * len(ordered) + 99) // 100 - 1)
    return ordered[index]


def _size_summary(values: Sequence[int]) -> dict[str, int]:
    return {
        "p50": _nearest_rank(values, 50),
        "p95": _nearest_rank(values, 95),
        "max": max(values),
    }


def _response_document(
    *,
    result: Mapping[str, object],
    replay: Mapping[str, object],
    replay_sha256: str,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": _BENCHMARK_REQUEST_ID,
        "engine_version": ENGINE_VERSION,
        "rules_version": RULES_VERSION,
        "map_id": MAP_ID,
        "replay_version": REPLAY_VERSION,
        "result": result,
        "replay_sha256": replay_sha256,
        "replay": replay,
    }


def _measure(
    cases: tuple[MatchCase, ...],
    *,
    include_replay: bool,
    warmup_repetitions: int,
    measured_repetitions: int,
) -> dict[str, object]:
    for _ in range(warmup_repetitions):
        for controller_a_id, controller_b_id, seed in cases:
            run_match(
                map_id=MAP_ID,
                seed=seed,
                controller_a_id=controller_a_id,
                controller_b_id=controller_b_id,
                include_replay=include_replay,
            )

    action_count = 0
    match_durations: list[float] = []
    replay_sizes: list[int] = []
    response_sizes: list[int] = []
    repetition_times: list[float] = []
    for _ in range(measured_repetitions):
        repetition_started = perf_counter()
        for controller_a_id, controller_b_id, seed in cases:
            match_started = perf_counter()
            outcome = run_match(
                map_id=MAP_ID,
                seed=seed,
                controller_a_id=controller_a_id,
                controller_b_id=controller_b_id,
                include_replay=include_replay,
            )
            if include_replay:
                if outcome.replay is None or outcome.replay_sha256 is None:
                    raise AssertionError("replay-enabled matches must return replay data")
                replay_json = canonical_replay_json(outcome.replay)
                response_json = canonical_replay_json(
                    _response_document(
                        result=outcome.summary(),
                        replay=outcome.replay,
                        replay_sha256=outcome.replay_sha256,
                    )
                )
                replay_sizes.append(len(replay_json))
                response_sizes.append(len(response_json))
            elif outcome.replay is not None or outcome.replay_sha256 is not None:
                raise AssertionError("no-replay matches returned replay data")
            match_durations.append(perf_counter() - match_started)
            action_count += outcome.result.actions
        repetition_times.append(perf_counter() - repetition_started)

    elapsed = sum(repetition_times)
    match_count = len(cases) * measured_repetitions
    measurement: dict[str, object] = {
        "include_replay": include_replay,
        "measured_matches": match_count,
        "measured_actions": action_count,
        "elapsed_seconds": elapsed,
        "matches_per_second": match_count / elapsed,
        "actions_per_second": action_count / elapsed,
        "mean_match_seconds": statistics.fmean(match_durations),
        "p95_match_seconds": _nearest_rank_seconds(match_durations, 95),
        "repetition_seconds": repetition_times,
    }
    if include_replay:
        measurement["replay_size_bytes"] = _size_summary(replay_sizes)
        measurement["response_size_bytes"] = _size_summary(response_sizes)
    return measurement


def _nearest_rank_seconds(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    index = max(0, (percentile * len(ordered) + 99) // 100 - 1)
    return ordered[index]


def run_replay_benchmark(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    warmup_repetitions: int = DEFAULT_WARMUP_REPETITIONS,
    measured_repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, object]:
    """Compare identical fixed workloads with replay disabled and enabled."""

    if not seeds:
        raise ValueError("seeds must not be empty")
    if warmup_repetitions < 0 or measured_repetitions < 1:
        raise ValueError("repetition counts are invalid")
    controller_pairs = list(product(SCRIPTED_CONTROLLER_IDS, repeat=2))
    cases = tuple(
        (controller_a, controller_b, seed)
        for controller_a, controller_b in controller_pairs
        for seed in seeds
    )
    without_replay = _measure(
        cases,
        include_replay=False,
        warmup_repetitions=warmup_repetitions,
        measured_repetitions=measured_repetitions,
    )
    with_replay = _measure(
        cases,
        include_replay=True,
        warmup_repetitions=warmup_repetitions,
        measured_repetitions=measured_repetitions,
    )
    if without_replay["measured_actions"] != with_replay["measured_actions"]:
        raise AssertionError("replay changed the deterministic workload")
    without_rate = float(without_replay["matches_per_second"])
    with_rate = float(with_replay["matches_per_second"])
    replay_max = int(with_replay["replay_size_bytes"]["max"])
    response_max = int(with_replay["response_size_bytes"]["max"])
    return {
        "schema_version": "1.0.0",
        "benchmark": "dinorl-replay-cost",
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(),
        "workload": {
            "map_id": MAP_ID,
            "processes": 1,
            "controller_pairs": [list(pair) for pair in controller_pairs],
            "seeds": list(seeds),
            "warmup_repetitions": warmup_repetitions,
            "measured_repetitions": measured_repetitions,
            "matches_per_repetition": len(cases),
        },
        "measurements": [without_replay, with_replay],
        "comparison": {
            "throughput_ratio_with_over_without": with_rate / without_rate,
            "slowdown_percent": (1.0 - with_rate / without_rate) * 100.0,
        },
        "limits": {
            "all_samples_compliant": (
                replay_max <= MAX_REPLAY_BYTES and response_max <= MAX_RESPONSE_BYTES
            ),
            "replay_max_bytes": MAX_REPLAY_BYTES,
            "response_max_bytes": MAX_RESPONSE_BYTES,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DinoRL replay cost")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmup-repetitions", type=int, default=DEFAULT_WARMUP_REPETITIONS)
    parser.add_argument("--seed-count", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/replay-cost.json"),
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run and export replay costs; short measured reports are refused."""

    options = _parser().parse_args(arguments)
    if options.repetitions < 5:
        _parser().error("--repetitions must be at least 5 for a report")
    if options.warmup_repetitions < 1:
        _parser().error("--warmup-repetitions must be at least 1")
    if options.seed_count < 1:
        _parser().error("--seed-count must be positive")
    report = run_replay_benchmark(
        seeds=tuple(range(options.seed_count)),
        warmup_repetitions=options.warmup_repetitions,
        measured_repetitions=options.repetitions,
    )
    if not report["limits"]["all_samples_compliant"]:
        raise RuntimeError("a serialized replay or response exceeds its size limit")
    write_report(report, options.output)
    print(options.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
