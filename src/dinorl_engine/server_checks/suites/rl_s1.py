"""RL-S1: server-only comparison of the fixed vectorization candidates."""

from __future__ import annotations

import importlib
import math
import platform
import statistics
import time
from collections.abc import Sequence

from dinorl_engine.rl.env.vectorization import (
    VECTOR_ENV_COUNTS,
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.training.unit import create_maskable_ppo, train_one_unit

__all__ = [
    "REPEATED_MEASUREMENTS",
    "build_decision",
    "candidate_configurations",
    "run_warmup",
    "run_with_measurement",
]

REPEATED_MEASUREMENTS = 5


def candidate_configurations(*, seed: int) -> tuple[VectorEnvironmentConfig, ...]:
    """Return the complete, server-owned RL-S1 comparison matrix."""

    return tuple(
        VectorEnvironmentConfig(backend=backend, n_envs=n_envs, seed=seed)
        for backend in VectorBackend
        for n_envs in VECTOR_ENV_COUNTS
    )


def _peak_memory_bytes() -> int | None:
    try:
        resource_module = importlib.import_module("resource")
    except ModuleNotFoundError:
        return None
    usage = resource_module.getrusage(resource_module.RUSAGE_SELF)
    peak = getattr(usage, "ru_maxrss", None)
    if not isinstance(peak, int) or peak <= 0:
        return None
    return peak if platform.system() == "Darwin" else peak * 1024


def _cpu_seconds() -> float:
    """Return CPU consumed by this process and its reaped worker children."""

    try:
        resource_module = importlib.import_module("resource")
    except ModuleNotFoundError:
        return time.process_time()
    total = 0.0
    for scope in (resource_module.RUSAGE_SELF, resource_module.RUSAGE_CHILDREN):
        usage = resource_module.getrusage(scope)
        user = getattr(usage, "ru_utime", None)
        system = getattr(usage, "ru_stime", None)
        if isinstance(user, int | float) and isinstance(system, int | float):
            total += float(user) + float(system)
    return total


def _run_unit(configuration: VectorEnvironmentConfig) -> dict[str, object]:
    started = time.perf_counter()
    cpu_started = _cpu_seconds()
    environment = create_vector_environment(configuration)
    try:
        model = create_maskable_ppo(
            environment, seed=configuration.seed, n_steps=configuration.n_steps
        )
        setup_seconds = time.perf_counter() - started
        training_started = time.perf_counter()
        unit = train_one_unit(model, environment)
        training_seconds = time.perf_counter() - training_started
    finally:
        closing_started = time.perf_counter()
        environment.close()
        closing_seconds = time.perf_counter() - closing_started
    duration = time.perf_counter() - started
    cpu_duration = _cpu_seconds() - cpu_started
    if duration <= 0.0 or cpu_duration < 0.0:
        raise RuntimeError("RL-S1 clock returned an invalid duration")
    return {
        "duration_seconds": duration,
        "cpu_seconds": cpu_duration,
        "peak_memory_bytes": _peak_memory_bytes(),
        "phase_seconds": {
            "setup": setup_seconds,
            "training": training_seconds,
            "closing": closing_seconds,
        },
        "learner_transitions": unit.metrics.learner_transitions,
        "engine_actions": unit.metrics.engine_actions,
        "epochs": unit.metrics.epochs,
        "optimizer_steps": unit.metrics.optimizer_steps,
        "diagnostics": unit.metrics.diagnostics,
    }


def run_warmup(configuration: VectorEnvironmentConfig) -> dict[str, object]:
    """Run one excluded unit to initialize one backend candidate."""

    return _run_unit(configuration)


def run_with_measurement(configuration: VectorEnvironmentConfig) -> dict[str, object]:
    """Run one measured global PPO unit for one candidate."""

    return _run_unit(configuration)


def _measurement_throughput(measurement: dict[str, object]) -> float:
    duration = measurement.get("duration_seconds")
    transitions = measurement.get("learner_transitions")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int | float)
        or not math.isfinite(float(duration))
        or duration <= 0
        or isinstance(transitions, bool)
        or not isinstance(transitions, int)
        or transitions != 2048
    ):
        raise ValueError("RL-S1 measurement has invalid duration or learner transitions")
    return transitions / float(duration)


def build_decision(candidates: Sequence[dict[str, object]]) -> dict[str, object]:
    """Select the fastest complete candidate by median end-to-end throughput."""

    ranked: list[tuple[float, float, str, int, int]] = []
    for candidate in candidates:
        backend = candidate.get("backend")
        n_envs = candidate.get("n_envs")
        n_steps = candidate.get("n_steps")
        repetitions = candidate.get("repetitions")
        if (
            not isinstance(backend, str)
            or isinstance(n_envs, bool)
            or not isinstance(n_envs, int)
            or isinstance(n_steps, bool)
            or not isinstance(n_steps, int)
            or not isinstance(repetitions, list)
            or len(repetitions) != REPEATED_MEASUREMENTS
            or not all(isinstance(item, dict) for item in repetitions)
        ):
            raise ValueError("RL-S1 candidate is structurally invalid")
        throughputs = [_measurement_throughput(item) for item in repetitions]
        durations = [float(item["duration_seconds"]) for item in repetitions]
        ranked.append(
            (
                statistics.median(throughputs),
                statistics.median(durations),
                backend,
                n_envs,
                n_steps,
            )
        )
    if not ranked:
        raise ValueError("RL-S1 requires at least one candidate")
    throughput, duration, backend, n_envs, n_steps = min(
        ranked, key=lambda item: (-item[0], item[1], item[2], item[3])
    )
    return {
        "backend": backend,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "median_duration_seconds": duration,
        "median_transitions_per_second": throughput,
        "criterion": "highest_median_end_to_end_transitions_per_second",
    }
