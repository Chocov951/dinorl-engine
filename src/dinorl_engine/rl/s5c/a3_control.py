"""Resumable generalist policy control for RL-S5c-A3."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.evaluation.gates import deterministic_gate, publication_gate
from dinorl_engine.rl.evaluation.suites import evaluate_deterministic, evaluate_publication
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.policies.local_mlp_v2 import LocalMLPV2Architecture
from dinorl_engine.rl.s5c.a3_config import A3Config
from dinorl_engine.rl.s5c.calibration import _atomic_json
from dinorl_engine.rl.s5c.strength import evaluate_against_strong_pool
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore
from dinorl_engine.rl.training.unit import create_maskable_ppo


def _read_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def run_policy_control(
    config: A3Config,
    *,
    resume: bool,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    branch = config.output_directory / "policy-control"
    result_path = branch / "result.json"
    if result_path.is_file():
        if not resume:
            raise RuntimeError("validated A3 policy control cannot be resumed without --resume")
        value = _read_json(result_path, {})
        assert isinstance(value, dict)
        return value
    recovery = AtomicRecoveryStore(branch / "recovery")
    latest = recovery.load_latest()
    if latest is not None and not resume:
        raise RuntimeError("existing A3 policy control requires --resume")
    vector = VectorEnvironmentConfig(
        backend=VectorBackend.DUMMY, n_envs=2, seed=config.control_seed
    )
    ledger = UnitLedger(branch / "ledger.sqlite3")
    run_id = "s5c-a3-policy-control-seed-20"
    ledger.reserve(
        run_id=run_id, units=config.control_max_units, idempotency_key=f"{run_id}-reserve"
    )
    metadata = {
        "phase": "A3-CONTROL",
        "architecture": config.architecture,
        "reward": "historical-generalist-reference",
        "opponent": "random-legal-v1",
    }
    if latest is None:
        environment = create_vector_environment(vector)
        model = create_maskable_ppo(
            environment,
            seed=config.control_seed,
            n_steps=vector.n_steps,
            architecture=LocalMLPV2Architecture.COMPACT,
        )
        runner = AtomicPPOUnitRunner(
            run_id=run_id,
            configuration=vector,
            model=model,
            environment=environment,
            recovery_store=recovery,
            ledger=ledger,
            recovery_metadata=metadata,
        )
        completed = 0
    else:
        runner, _, latest = AtomicPPOUnitRunner.restore_latest(
            run_id=run_id,
            configuration=vector,
            recovery_store=recovery,
            ledger=ledger,
            environment_factory=create_vector_environment,
            recovery_metadata=metadata,
        )
        completed = latest.sequence
    evaluations = _read_json(branch / "evaluations.json", [])
    if not isinstance(evaluations, list):
        raise RuntimeError("A3 control evaluations are invalid")
    previous = bool(evaluations and evaluations[-1].get("gate", {}).get("thresholds_passed"))
    stable_unit = next(
        (int(item["unit"]) for item in evaluations if item.get("gate", {}).get("passed") is True),
        None,
    )
    try:
        for unit in range(completed + 1, config.control_max_units + 1):
            trained = runner.run_unit(unit_id=f"unit-{unit}", sequence=unit)
            if progress is not None:
                progress(
                    f"S5c-A3 CONTROL: {unit}/{config.control_max_units} units "
                    f"({100 * unit / config.control_max_units:.1f}%)"
                )
            if unit % 5:
                continue
            report = evaluate_deterministic(runner.model, seed=config.evaluation_seeds[0])
            scores = report.get("scores")
            if not isinstance(scores, dict):
                raise RuntimeError("A3 control deterministic evaluation is incomplete")
            gate = deterministic_gate(scores, previous_passed=previous)
            evaluations.append(
                {
                    "unit": unit,
                    "scores": scores,
                    "gate": gate,
                    "checkpoint_sha256": trained.recovery.files.get("model.zip"),
                }
            )
            _atomic_json(branch / "evaluations.json", evaluations)
            previous = gate["thresholds_passed"] is True
            if gate["passed"] is True and stable_unit is None:
                stable_unit = unit
        strength = evaluate_against_strong_pool(
            runner.model,
            pool_path=config.rl_s5b_pool,
            expected_sha256=config.rl_s5b_pool_sha256,
            evaluation_seeds=config.evaluation_seeds,
            confrontations=20,
            learner_policy_id="s5c-a3-policy-control-s20-u147",
        )
        _atomic_json(branch / "strength.json", strength)
        final_deterministic = evaluate_deterministic(
            runner.model,
            seed=config.evaluation_seeds[0],
            diagnostic_directory=branch / "replays" / "deterministic",
        )
        stochastic = evaluate_publication(
            runner.model,
            seed=config.evaluation_seeds[0],
            diagnostic_directory=branch / "replays" / "stochastic",
        )
        stochastic_gate = publication_gate(
            final_deterministic["scores"],
            stochastic["scores"],  # type: ignore[arg-type]
        )
        _atomic_json(
            branch / "final-evaluation.json",
            {
                "deterministic": final_deterministic,
                "stochastic": stochastic,
                "stochastic_gate": stochastic_gate,
            },
        )
        score = float(strength["global"]["score"])
        coherent = 0.25 <= score <= 0.75
        passed = stable_unit is not None and coherent and stochastic_gate["passed"] is True
        decision = "PASSED_POLICY_CONTROL" if passed else "FAILED_POLICY_CONTROL"
        result = {
            "format": "s5c-a3-policy-control-v1",
            "decision": decision,
            "completed_units": config.control_max_units,
            "historical_stable_gate": stable_unit,
            "strong_pool_score": score,
            "learning_observed": stable_unit is not None,
            "stochastic_gate": stochastic_gate,
        }
        _atomic_json(result_path, result)
        return result
    finally:
        runner.environment.close()
