"""Measure deterministic matches without replay across several processes."""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing
import os
import platform
import statistics
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Final

from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.core.constants import ENGINE_VERSION, MAP_ID, RULES_VERSION
from dinorl_engine.match.runner import run_match

REQUIRED_PROCESS_COUNTS: Final = (1, 2, 4, 8)
DEFAULT_SEEDS: Final = tuple(range(20))
DEFAULT_REPETITIONS: Final = 5
DEFAULT_WARMUP_REPETITIONS: Final = 1

type MatchCase = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class _MatchSample:
    process_id: int
    duration_seconds: float
    actions: int
    peak_memory_bytes: int


def _peak_memory_bytes() -> int:
    """Return the process peak resident set size with platform-native stdlib APIs."""

    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        process = get_current_process()
        if not get_process_memory_info(process, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _run_case(case: MatchCase) -> _MatchSample:
    controller_a_id, controller_b_id, seed = case
    started = perf_counter()
    outcome = run_match(
        map_id=MAP_ID,
        seed=seed,
        controller_a_id=controller_a_id,
        controller_b_id=controller_b_id,
        include_replay=False,
    )
    return _MatchSample(
        process_id=os.getpid(),
        duration_seconds=perf_counter() - started,
        actions=outcome.result.actions,
        peak_memory_bytes=_peak_memory_bytes(),
    )


def _percentile_95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return ordered[index]


def _measure_process_count(
    cases: tuple[MatchCase, ...],
    process_count: int,
    warmup_repetitions: int,
    measured_repetitions: int,
) -> dict[str, object]:
    measured: list[_MatchSample] = []
    elapsed_total = 0.0
    repetition_times: list[float] = []

    def consume(executor: ProcessPoolExecutor | None) -> list[_MatchSample]:
        if executor is None:
            return [_run_case(case) for case in cases]
        return list(executor.map(_run_case, cases, chunksize=1))

    executor_context = ProcessPoolExecutor(max_workers=process_count) if process_count > 1 else None
    try:
        for _ in range(warmup_repetitions):
            consume(executor_context)
        for _ in range(measured_repetitions):
            started = perf_counter()
            samples = consume(executor_context)
            elapsed = perf_counter() - started
            repetition_times.append(elapsed)
            elapsed_total += elapsed
            measured.extend(samples)
    finally:
        if executor_context is not None:
            executor_context.shutdown()

    match_count = len(cases) * measured_repetitions
    action_count = sum(sample.actions for sample in measured)
    durations = [sample.duration_seconds for sample in measured]
    peak_by_process: dict[int, int] = {}
    for sample in measured:
        peak_by_process[sample.process_id] = max(
            peak_by_process.get(sample.process_id, 0), sample.peak_memory_bytes
        )
    return {
        "processes": process_count,
        "measured_matches": match_count,
        "measured_actions": action_count,
        "elapsed_seconds": elapsed_total,
        "matches_per_second": match_count / elapsed_total,
        "actions_per_second": action_count / elapsed_total,
        "mean_match_seconds": statistics.fmean(durations),
        "p95_match_seconds": _percentile_95(durations),
        "speedup": 0.0,
        "peak_memory_bytes_per_process": max(peak_by_process.values()),
        "observed_worker_processes": len(peak_by_process),
        "repetition_seconds": repetition_times,
    }


def environment_metadata() -> dict[str, object]:
    """Describe the runtime used by a benchmark report."""

    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "engine_version": ENGINE_VERSION,
        "rules_version": RULES_VERSION,
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "system": platform.platform(),
        "machine": platform.machine(),
        "process_start_method": multiprocessing.get_start_method(),
    }


def run_baseline(
    *,
    process_counts: tuple[int, ...] = REQUIRED_PROCESS_COUNTS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    warmup_repetitions: int = DEFAULT_WARMUP_REPETITIONS,
    measured_repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, object]:
    """Run the fixed no-replay workload and return a JSON-compatible report."""

    if not process_counts or any(count < 1 for count in process_counts):
        raise ValueError("process_counts must contain positive integers")
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
    measurements = [
        _measure_process_count(
            cases,
            process_count,
            warmup_repetitions,
            measured_repetitions,
        )
        for process_count in process_counts
    ]
    baseline_rate = float(measurements[0]["matches_per_second"])
    for measurement in measurements:
        measurement["speedup"] = float(measurement["matches_per_second"]) / baseline_rate
    return {
        "schema_version": "1.0.0",
        "benchmark": "dinorl-match-baseline",
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(),
        "workload": {
            "map_id": MAP_ID,
            "include_replay": False,
            "controller_pairs": [list(pair) for pair in controller_pairs],
            "seeds": list(seeds),
            "warmup_repetitions": warmup_repetitions,
            "measured_repetitions": measured_repetitions,
            "matches_per_repetition": len(cases),
        },
        "measurements": measurements,
    }


def write_report(report: dict[str, object], output: Path) -> None:
    """Write a benchmark report as stable, readable UTF-8 JSON."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark complete DinoRL matches")
    parser.add_argument("--processes", type=int, nargs="+", default=REQUIRED_PROCESS_COUNTS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmup-repetitions", type=int, default=DEFAULT_WARMUP_REPETITIONS)
    parser.add_argument("--seed-count", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/baseline.json"),
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run and export a baseline; short measured runs are intentionally refused."""

    options = _parser().parse_args(arguments)
    if options.repetitions < 5:
        _parser().error("--repetitions must be at least 5 for a baseline")
    if options.warmup_repetitions < 1:
        _parser().error("--warmup-repetitions must be at least 1")
    if options.seed_count < 1:
        _parser().error("--seed-count must be positive")
    report = run_baseline(
        process_counts=tuple(options.processes),
        seeds=tuple(range(options.seed_count)),
        warmup_repetitions=options.warmup_repetitions,
        measured_repetitions=options.repetitions,
    )
    write_report(report, options.output)
    print(options.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
