"""RL-S5 fixed-comparison contract: 147 PPO units across three common seeds."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path

from sb3_contrib.common.maskable.utils import get_action_masks

from dinorl_engine.rl.env.vectorization import create_vector_environment
from dinorl_engine.rl.evaluation.comparison import select_architecture
from dinorl_engine.rl.evaluation.gates import deterministic_gate, publication_gate
from dinorl_engine.rl.evaluation.reports import RunReportWriter
from dinorl_engine.rl.evaluation.suites import evaluate_deterministic, evaluate_publication
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.policies.factory import PolicyArchitecture
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore
from dinorl_engine.rl.training.unit import create_maskable_ppo
from dinorl_engine.server_checks.suites.rl_s3 import SELECTED_CONFIGURATION

__all__ = [
    "ARCHITECTURES",
    "DEVELOPMENT_SEEDS",
    "UNITS_PER_SEED",
    "comparison_passed",
    "run_measurement",
]

ARCHITECTURES = (PolicyArchitecture.MLP, PolicyArchitecture.SMALL_CNN)
DEVELOPMENT_SEEDS = (19, 20, 21)
UNITS_PER_SEED = 147
_TRANSITIONS_PER_UNIT = 2048
_EPOCHS_PER_UNIT = 4
_OPTIMIZER_STEPS_PER_UNIT = 32
_EVALUATION_INTERVAL = 5


def _non_negative_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and value >= 0
    )


def _number(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise RuntimeError("RL-S5 aggregate is not a non-negative finite number")
    return float(value)


def _seed_passed(seed_result: object, seed: int) -> bool:
    if not isinstance(seed_result, Mapping) or set(seed_result) != {
        "seed",
        "units",
        "learner_transitions",
        "engine_actions",
        "epochs",
        "optimizer_steps",
        "wall_seconds",
        "cpu_seconds",
        "inference_seconds",
        "final_score",
        "transitions_to_gate",
        "deterministic_gate",
        "publication_gate",
    }:
        return False
    transitions_to_gate = seed_result.get("transitions_to_gate")
    return (
        seed_result.get("seed") == seed
        and seed_result.get("units") == UNITS_PER_SEED
        and seed_result.get("learner_transitions") == UNITS_PER_SEED * _TRANSITIONS_PER_UNIT
        and isinstance(seed_result.get("engine_actions"), int)
        and seed_result["engine_actions"] >= UNITS_PER_SEED * _TRANSITIONS_PER_UNIT
        and seed_result.get("epochs") == UNITS_PER_SEED * _EPOCHS_PER_UNIT
        and seed_result.get("optimizer_steps") == UNITS_PER_SEED * _OPTIMIZER_STEPS_PER_UNIT
        and _non_negative_number(seed_result.get("wall_seconds"))
        and _non_negative_number(seed_result.get("cpu_seconds"))
        and _non_negative_number(seed_result.get("inference_seconds"))
        and _non_negative_number(seed_result.get("final_score"))
        and seed_result["final_score"] <= 1
        and (
            transitions_to_gate is None
            or (type(transitions_to_gate) is int and transitions_to_gate > 0)
        )
        and isinstance(seed_result.get("deterministic_gate"), bool)
        and isinstance(seed_result.get("publication_gate"), bool)
    )


def _candidate_passed(candidate: object, architecture: PolicyArchitecture) -> bool:
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "architecture",
        "seeds",
        "aggregate",
    }:
        return False
    seeds = candidate.get("seeds")
    aggregate = candidate.get("aggregate")
    if (
        candidate.get("architecture") != architecture.value
        or not isinstance(seeds, list)
        or len(seeds) != len(DEVELOPMENT_SEEDS)
        or not isinstance(aggregate, Mapping)
        or set(aggregate)
        != {
            "transitions_to_gate",
            "wall_seconds",
            "final_score",
            "inference_seconds",
        }
        or not all(
            _seed_passed(seed_result, seed)
            for seed_result, seed in zip(seeds, DEVELOPMENT_SEEDS, strict=True)
        )
    ):
        return False
    transitions_to_gate = aggregate.get("transitions_to_gate")
    return (
        (
            transitions_to_gate is None
            or (type(transitions_to_gate) is int and transitions_to_gate > 0)
        )
        and _non_negative_number(aggregate.get("wall_seconds"))
        and _non_negative_number(aggregate.get("inference_seconds"))
        and _non_negative_number(aggregate.get("final_score"))
        and aggregate["final_score"] <= 1
    )


def comparison_passed(measurement: object) -> bool:
    """Validate complete RL-S5 evidence and its derived non-arbitrary decision."""

    if not isinstance(measurement, Mapping) or set(measurement) != {"candidates", "decision"}:
        return False
    candidates = measurement.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(ARCHITECTURES):
        return False
    if not all(
        _candidate_passed(candidate, architecture)
        for candidate, architecture in zip(candidates, ARCHITECTURES, strict=True)
    ):
        return False
    comparison = {
        candidate["architecture"]: candidate["aggregate"]
        for candidate in candidates
        if isinstance(candidate, Mapping)
    }
    try:
        return measurement.get("decision") == select_architecture(comparison)
    except ValueError:
        return False


def _inference_seconds(runner: AtomicPPOUnitRunner) -> float:
    observation = runner.model._last_obs
    if not isinstance(observation, dict):
        raise RuntimeError("RL-S5 model has no last observation")
    masks = get_action_masks(runner.environment)
    started = time.perf_counter()
    for _ in range(64):
        actions, _state = runner.model.predict(observation, action_masks=masks, deterministic=True)
        if not all(bool(mask[int(action)]) for mask, action in zip(masks, actions, strict=True)):
            raise RuntimeError("RL-S5 inference selected an illegal action")
    return (time.perf_counter() - started) / (64 * runner.environment.num_envs)


def _scores(report: object) -> dict[str, float]:
    if not isinstance(report, Mapping):
        raise RuntimeError("evaluation report is invalid")
    scores = report.get("scores")
    if not isinstance(scores, Mapping):
        raise RuntimeError("evaluation scores are invalid")
    return {key: float(value) for key, value in scores.items() if isinstance(key, str)}


def _candidate_run(
    directory: Path, architecture: PolicyArchitecture, seed: int
) -> dict[str, object]:
    identifier = f"{architecture.value}-{seed}"
    run_directory = directory / identifier
    reports = RunReportWriter(
        run_directory, run_id=identifier, architecture=architecture.value, seed=seed
    )
    store = AtomicRecoveryStore(run_directory / "recovery")
    ledger = UnitLedger(run_directory / "ledger.sqlite3")
    ledger.reserve(run_id=identifier, units=UNITS_PER_SEED, idempotency_key=f"reserve-{identifier}")
    latest = store.load_latest()
    if latest is None:
        environment = create_vector_environment(SELECTED_CONFIGURATION)
        runner = AtomicPPOUnitRunner(
            run_id=identifier,
            configuration=SELECTED_CONFIGURATION,
            model=create_maskable_ppo(
                environment,
                seed=seed,
                n_steps=SELECTED_CONFIGURATION.n_steps,
                architecture=architecture,
            ),
            environment=environment,
            recovery_store=store,
            ledger=ledger,
        )
        completed = 0
    else:
        runner, _metrics, latest = AtomicPPOUnitRunner.restore_latest(
            run_id=identifier,
            configuration=SELECTED_CONFIGURATION,
            recovery_store=store,
            ledger=ledger,
        )
        completed = latest.sequence
    try:
        previous_threshold = False
        for prior_record in reports.evaluation_records():
            gate = prior_record.get("deterministic_gate")
            if isinstance(gate, Mapping) and gate.get("thresholds_passed") is True:
                previous_threshold = True
        if not reports.has_evaluation(0):
            initial = evaluate_deterministic(
                runner.model, seed=seed, diagnostic_directory=run_directory / "diagnostic_replays"
            )
            gate = deterministic_gate(_scores(initial), previous_passed=False)
            reports.record_evaluation(
                {"unit": 0, "deterministic": initial, "deterministic_gate": gate}
            )
            previous_threshold = gate["thresholds_passed"] is True
        for sequence in range(completed + 1, UNITS_PER_SEED + 1):
            cpu_started = time.process_time()
            result = runner.run_unit(unit_id=f"unit-{sequence}", sequence=sequence)
            reports.record_cycle(
                {
                    "unit": sequence,
                    "learner_transitions": result.metrics.learner_transitions,
                    "engine_actions": result.metrics.engine_actions,
                    "epochs": result.metrics.epochs,
                    "optimizer_steps": result.metrics.optimizer_steps,
                    "wall_seconds": result.training_seconds + result.checkpoint_seconds,
                    "cpu_seconds": time.process_time() - cpu_started,
                }
            )
            store.discard_before(sequence)
            if (
                sequence % _EVALUATION_INTERVAL == 0 or sequence == UNITS_PER_SEED
            ) and not reports.has_evaluation(sequence):
                deterministic = evaluate_deterministic(
                    runner.model,
                    seed=seed,
                    diagnostic_directory=run_directory / "diagnostic_replays",
                )
                gate = deterministic_gate(
                    _scores(deterministic), previous_passed=previous_threshold
                )
                evaluation_record: dict[str, object] = {
                    "unit": sequence,
                    "deterministic": deterministic,
                    "deterministic_gate": gate,
                }
                if sequence == UNITS_PER_SEED:
                    publication = evaluate_publication(
                        runner.model,
                        seed=seed,
                        diagnostic_directory=run_directory / "diagnostic_replays",
                    )
                    evaluation_record["publication"] = publication
                    evaluation_record["publication_gate"] = publication_gate(
                        _scores(deterministic), _scores(publication)
                    )
                reports.record_evaluation(evaluation_record)
                previous_threshold = gate["thresholds_passed"] is True
        final_records = [
            record
            for record in reports.evaluation_records()
            if record.get("unit") == UNITS_PER_SEED
        ]
        if len(final_records) != 1:
            raise RuntimeError("RL-S5 final evaluation is missing")
        final = final_records[0]
        final_deterministic = final.get("deterministic")
        deterministic_result = final.get("deterministic_gate")
        publication_result = final.get("publication_gate")
        if not isinstance(deterministic_result, Mapping) or not isinstance(
            publication_result, Mapping
        ):
            raise RuntimeError("RL-S5 final gates are missing")
        transitions_to_gate: int | None = None
        for record in reports.evaluation_records():
            gate = record.get("deterministic_gate")
            unit = record.get("unit")
            if isinstance(gate, Mapping) and gate.get("passed") is True and type(unit) is int:
                transitions_to_gate = unit * _TRANSITIONS_PER_UNIT
                break
        cycles = [
            json.loads(line)
            for line in (run_directory / "cycles.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        wall_seconds = sum(float(cycle["wall_seconds"]) for cycle in cycles)
        cpu_seconds = sum(float(cycle["cpu_seconds"]) for cycle in cycles)
        final_scores = _scores(final_deterministic)
        primary_score = (
            sum(
                final_scores[architecture_id]
                for architecture_id in ("aggressive-v1", "prudent-v1", "opportunist-v1")
            )
            / 3
        )
        reports.write_final_report(
            final_score=primary_score, inference_seconds=_inference_seconds(runner)
        )
        return {
            "seed": seed,
            "units": UNITS_PER_SEED,
            "learner_transitions": UNITS_PER_SEED * _TRANSITIONS_PER_UNIT,
            "engine_actions": sum(int(cycle["engine_actions"]) for cycle in cycles),
            "epochs": UNITS_PER_SEED * _EPOCHS_PER_UNIT,
            "optimizer_steps": UNITS_PER_SEED * _OPTIMIZER_STEPS_PER_UNIT,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "inference_seconds": _inference_seconds(runner),
            "final_score": primary_score,
            "transitions_to_gate": transitions_to_gate,
            "deterministic_gate": deterministic_result.get("passed") is True,
            "publication_gate": publication_result.get("passed") is True,
        }
    finally:
        runner.environment.close()


def run_measurement(work_directory: Path) -> dict[str, object]:
    """Run or resume the complete six-run RL-S5 comparison in ``work_directory``."""

    candidates: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        seeds = [_candidate_run(work_directory, architecture, seed) for seed in DEVELOPMENT_SEEDS]
        aggregates: dict[str, object] = {
            "transitions_to_gate": next(
                (
                    item["transitions_to_gate"]
                    for item in seeds
                    if item["transitions_to_gate"] is not None
                ),
                None,
            ),
            "wall_seconds": sum(_number(item.get("wall_seconds")) for item in seeds) / len(seeds),
            "final_score": sum(_number(item.get("final_score")) for item in seeds) / len(seeds),
            "inference_seconds": sum(_number(item.get("inference_seconds")) for item in seeds)
            / len(seeds),
        }
        candidates.append(
            {"architecture": architecture.value, "seeds": seeds, "aggregate": aggregates}
        )
    comparison: dict[str, Mapping[str, object]] = {}
    for candidate in candidates:
        architecture_id = candidate.get("architecture")
        aggregate = candidate.get("aggregate")
        if not isinstance(architecture_id, str) or not isinstance(aggregate, Mapping):
            raise RuntimeError("RL-S5 candidate result is invalid")
        comparison[architecture_id] = aggregate
    decision = select_architecture(comparison)
    return {"candidates": candidates, "decision": decision}
