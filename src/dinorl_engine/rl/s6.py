"""Server-owned RL-S6 campaign: five independent common-starter runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Any, Final, cast

from safetensors import safe_open
from safetensors.torch import load_file
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnv

from dinorl_engine.controllers.protocol import Controller
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS, create_scripted_controller
from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.artifacts.snapshots import assess_s6_publication, publish_policy_snapshot
from dinorl_engine.rl.env.single_agent import TrainingOpponentPool
from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.evaluation.gates import deterministic_gate, publication_gate
from dinorl_engine.rl.evaluation.suites import evaluate_deterministic, evaluate_publication
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.policies.local_mlp_v2 import LocalMLPV2Architecture
from dinorl_engine.rl.rewards.runtime import compile_reward
from dinorl_engine.rl.s5_closure import ClosureError, audit_closure
from dinorl_engine.rl.s5c.provenance import initialization_record, policy_weights_sha256
from dinorl_engine.rl.s6_config import S6Config
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore
from dinorl_engine.rl.training.unit import create_maskable_ppo, load_maskable_ppo

__all__ = ["S6Error", "run_preflight", "run_seed", "summarize_campaign"]

_FORMAT: Final = "rl-s6-run-manifest-v1"


class S6Error(RuntimeError):
    """A frozen RL-S6 input, state, or result is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    content = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class _UniformScriptedPool:
    """Exactly uniform, deterministic sampling among the three frozen scripted bots."""

    def _identifier(self, selection_seed: int) -> str:
        if type(selection_seed) is not int or not 0 <= selection_seed < 2**63:
            raise S6Error("training opponent selection seed is invalid")
        return Random(selection_seed).choice(SCRIPTED_CONTROLLER_IDS)

    def select(
        self, *, selection_seed: int, learner_actor: Actor, first_actor: Actor
    ) -> tuple[str, Controller]:
        del learner_actor, first_actor
        identifier = self._identifier(selection_seed)
        return identifier, create_scripted_controller(identifier)

    def restore(
        self,
        *,
        opponent_id: str,
        selection_seed: int,
        learner_actor: Actor,
        first_actor: Actor,
        controller_state: object,
    ) -> Controller:
        del learner_actor, first_actor
        if opponent_id != self._identifier(selection_seed) or controller_state is not None:
            raise S6Error("RL-S6 scripted opponent recovery is invalid")
        return create_scripted_controller(opponent_id)


def _read_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise S6Error(f"invalid JSON: {path}") from error


def run_preflight(config: S6Config, *, repository: Path) -> dict[str, object]:
    """Freeze verified input hashes before any unit or evaluation is executed."""

    try:
        closure = audit_closure(repository)
    except ClosureError as error:
        raise S6Error("RL-S5 closure is not valid") from error
    weights = config.starter_directory / "weights.safetensors"
    starter_manifest = config.starter_directory / "manifest.json"
    if not weights.is_file() or _sha256(weights) != config.starter_weights_sha256:
        raise S6Error("starter-zero-v1 weights hash differs")
    starter = _read_json(starter_manifest, {})
    if not isinstance(starter, dict) or starter.get("snapshot_id") != "starter-zero-v1":
        raise S6Error("starter-zero-v1 manifest is invalid")
    with safe_open(weights, framework="pt") as tensors:
        if not tensors.keys():
            raise S6Error("starter-zero-v1 has no tensors")
    pool = _read_json(config.beta_validation_pool, {})
    if not isinstance(pool, dict) or pool.get("pool_sha256") != config.beta_validation_pool_sha256:
        raise S6Error("beta-validation-pool-v1 hash differs")
    reward = compile_reward(config.reward_dsl)
    manifest = {
        "format": _FORMAT,
        "status": "READY",
        "config_sha256": config.sha256,
        "starter_weights_sha256": config.starter_weights_sha256,
        "beta_validation_pool_sha256": config.beta_validation_pool_sha256,
        "reward_sha256": reward.cache_key,
        "architecture": config.architecture,
        "architecture_dimensions": list(config.architecture_dimensions),
        "training_seeds": list(config.training_seeds),
        "training_opponents": list(config.training_opponents),
        "closure_status": closure["status"],
    }
    path = config.output_directory / "manifest.json"
    if (
        path.exists()
        and path.read_bytes()
        != (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ):
        raise S6Error("existing RL-S6 manifest differs from frozen inputs")
    if not path.exists():
        _atomic_json(path, manifest)
    return manifest


def _verified_manifest(config: S6Config) -> None:
    value = _read_json(config.output_directory / "manifest.json", {})
    if (
        not isinstance(value, Mapping)
        or value.get("format") != _FORMAT
        or value.get("status") != "READY"
    ):
        raise S6Error("run s6 preflight before training")
    if value.get("config_sha256") != config.sha256:
        raise S6Error("RL-S6 config differs from its frozen preflight")


def _starter_model(
    config: S6Config, *, seed: int, pool: TrainingOpponentPool
) -> tuple[VectorEnvironmentConfig, VecEnv, MaskablePPO]:
    vector = VectorEnvironmentConfig(VectorBackend.DUMMY, 2, seed)
    environment = create_vector_environment(
        vector, training_opponent_pool=pool, reward_program=compile_reward(config.reward_dsl)
    )
    model = create_maskable_ppo(
        environment, seed=seed, n_steps=vector.n_steps, architecture=LocalMLPV2Architecture.COMPACT
    )
    tensors = load_file(config.starter_directory / "weights.safetensors", device="cpu")
    model.policy.load_state_dict(tensors, strict=True)
    return vector, environment, model


def run_seed(
    config: S6Config, *, seed: int, resume: bool, progress: Callable[[str], None] | None = None
) -> dict[str, object]:
    """Run/resume one independent seed; no target or beta-pool policy enters training."""

    _verified_manifest(config)
    if seed not in config.training_seeds:
        raise S6Error("seed is not part of the frozen RL-S6 campaign")
    branch = config.output_directory / "seeds" / f"seed-{seed}"
    result_path = branch / "result.json"
    if result_path.is_file():
        value = _read_json(result_path, {})
        if not resume:
            raise S6Error("completed seed requires no further execution")
        if not isinstance(value, dict):
            raise S6Error("RL-S6 seed result is invalid")
        return value
    pool = _UniformScriptedPool()
    vector = VectorEnvironmentConfig(VectorBackend.DUMMY, 2, seed)
    recovery = AtomicRecoveryStore(branch / "recovery")
    latest = recovery.load_latest()
    if latest is not None and not resume:
        raise S6Error("existing RL-S6 seed requires --resume")
    ledger = UnitLedger(branch / "ledger.sqlite3")
    run_id = f"rl-s6-{seed}"
    ledger.reserve(run_id=run_id, units=config.max_units, idempotency_key=f"{run_id}-reserve")
    metadata = {
        "phase": "RL-S6",
        "architecture": config.architecture,
        "seed": seed,
        "starter_weights_sha256": config.starter_weights_sha256,
        "reward_sha256": compile_reward(config.reward_dsl).cache_key,
        "training_opponents": list(config.training_opponents),
    }
    if latest is None:
        _, environment, model = _starter_model(config, seed=seed, pool=pool)
        initialization = {
            **initialization_record(cast(Any, model), seed=seed),
            "source_snapshot": "starter-zero-v1",
            "source_weights_sha256": config.starter_weights_sha256,
        }
        if initialization["weights_sha256"] != policy_weights_sha256(cast(Any, model)):
            raise S6Error("starter policy initialization is not deterministic")
        _atomic_json(branch / "initialization.json", initialization)
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
            environment_factory=lambda value: create_vector_environment(
                value, training_opponent_pool=pool, reward_program=compile_reward(config.reward_dsl)
            ),
            recovery_metadata=metadata,
        )
        completed = latest.sequence
    evaluations = _read_json(branch / "evaluations.json", [])
    if not isinstance(evaluations, list):
        raise S6Error("RL-S6 evaluations are invalid")
    previous = bool(
        evaluations
        and isinstance(evaluations[-1], Mapping)
        and evaluations[-1].get("gate", {}).get("thresholds_passed")
    )
    try:
        for unit in range(completed + 1, config.max_units + 1):
            trained = runner.run_unit(unit_id=f"unit-{unit}", sequence=unit)
            if progress is not None:
                progress(
                    f"RL-S6 seed {seed}: {unit}/{config.max_units} units "
                    f"({100 * unit / config.max_units:.1f}%)"
                )
            if unit % config.evaluation_interval_units:
                continue
            deterministic = evaluate_deterministic(runner.model, seed=config.evaluation_seed)
            scores = deterministic.get("scores")
            if not isinstance(scores, Mapping):
                raise S6Error("deterministic evaluation has no complete scores")
            gate = deterministic_gate(scores, previous_passed=previous)
            evidence: dict[str, object] = {
                "unit": unit,
                "deterministic": deterministic,
                "gate": gate,
                "checkpoint_sha256": trained.recovery.files.get("model.zip"),
            }
            evaluations.append(evidence)
            _atomic_json(branch / "evaluations.json", evaluations)
            previous = gate["thresholds_passed"] is True
            if gate["passed"] is not True:
                continue
            stochastic = evaluate_publication(
                runner.model,
                seed=config.stochastic_evaluation_seed,
                diagnostic_directory=branch / "replays" / f"unit-{unit}" / "stochastic",
            )
            stochastic_scores = stochastic.get("scores")
            if not isinstance(stochastic_scores, Mapping):
                raise S6Error("stochastic evaluation has no complete scores")
            stochastic_gate = publication_gate(scores, stochastic_scores)
            extension = None
            if stochastic_gate["near_boundary"] is True:
                extension = evaluate_publication(
                    runner.model,
                    seed=config.stochastic_extension_seed,
                    diagnostic_directory=branch / "replays" / f"unit-{unit}" / "extension",
                )
                extension_scores = extension.get("scores")
                if not isinstance(extension_scores, Mapping):
                    raise S6Error("stochastic extension has no complete scores")
                extension_gate = publication_gate(scores, extension_scores)
                if extension_gate["passed"] is not True:
                    stochastic_gate = {
                        **stochastic_gate,
                        "passed": False,
                        "extension": extension_gate,
                    }
            candidate = {
                "seed": seed,
                "unit": unit,
                "deterministic_evaluations": [evaluations[-2]["deterministic"]["scores"], scores],
                "stochastic_evaluation": stochastic_scores,
                "stochastic_extension": None if extension is None else extension["scores"],
            }
            _atomic_json(branch / "candidate.json", candidate)
            result = {
                "format": "rl-s6-seed-result-v1",
                "seed": seed,
                "status": "passed" if stochastic_gate["passed"] else "stochastic_failed",
                "candidate": candidate,
                "stochastic_gate": stochastic_gate,
                "completed_units": unit,
            }
            _atomic_json(result_path, result)
            return result
        last = evaluations[-1] if evaluations else None
        deterministic_tail: list[object] = []
        for evaluation in evaluations[-2:]:
            if isinstance(evaluation, Mapping):
                report = evaluation.get("deterministic")
                if isinstance(report, Mapping) and isinstance(report.get("scores"), Mapping):
                    deterministic_tail.append(dict(report["scores"]))
        result = {
            "format": "rl-s6-seed-result-v1",
            "seed": seed,
            "status": "budget_exhausted",
            "completed_units": config.max_units,
            "last_evaluation": last,
            "deterministic_evaluations": deterministic_tail,
        }
        _atomic_json(result_path, result)
        return result
    finally:
        runner.environment.close()


def summarize_campaign(config: S6Config) -> dict[str, object]:
    """Aggregate only completed seed evidence; never alter gates after beta diagnostics."""

    _verified_manifest(config)
    results = [
        _read_json(config.output_directory / "seeds" / f"seed-{seed}" / "result.json", None)
        for seed in config.training_seeds
    ]
    if not all(isinstance(value, Mapping) for value in results):
        raise S6Error("all five RL-S6 seed results are required")
    seed_evidence: list[Mapping[str, object]] = []
    for value in results:
        if not isinstance(value, Mapping):
            raise S6Error("RL-S6 seed result is invalid")
        if value.get("status") in {"passed", "stochastic_failed"}:
            candidate = value.get("candidate")
            if not isinstance(candidate, Mapping):
                raise S6Error("RL-S6 candidate seed lacks evidence")
            seed_evidence.append(dict(candidate))
        else:
            deterministic = value.get("deterministic_evaluations")
            if not isinstance(deterministic, list) or len(deterministic) != 2:
                raise S6Error("non-candidate seed lacks two deterministic evaluations")
            seed_evidence.append(
                {
                    "seed": value.get("seed"),
                    "deterministic_evaluations": deterministic,
                    "stochastic_evaluation": None,
                }
            )
    evidence = assess_s6_publication(seed_evidence)
    if evidence["eligible"] is not True:
        result = {
            "format": "rl-s6-campaign-result-v1",
            "decision": "FAILED_RL_S6",
            "passed_seeds": evidence["conforming_seed_count"],
            "required_passed_seeds": 4,
            "gate": evidence,
            "seed_results": results,
        }
        _atomic_json(config.output_directory / "campaign-result.json", result)
        return result
    snapshots: list[dict[str, object]] = []
    evaluation_id = f"rl-s6-evaluation-{config.sha256[:16]}"
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    for value in results:
        if not isinstance(value, Mapping) or value.get("status") != "passed":
            continue
        candidate = value.get("candidate")
        if not isinstance(candidate, Mapping):
            raise S6Error("passing RL-S6 seed lacks candidate evidence")
        seed = candidate.get("seed")
        unit = candidate.get("unit")
        if type(seed) is not int or type(unit) is not int:
            raise S6Error("passing RL-S6 candidate identity is invalid")
        pool = _UniformScriptedPool()
        vector = VectorEnvironmentConfig(VectorBackend.DUMMY, 2, seed)
        environment = create_vector_environment(
            vector,
            training_opponent_pool=pool,
            reward_program=compile_reward(config.reward_dsl),
        )
        try:
            checkpoint = (
                config.output_directory
                / "seeds"
                / f"seed-{seed}"
                / "recovery"
                / "units"
                / f"unit-{unit}"
                / "model.zip"
            )
            if not checkpoint.is_file():
                raise S6Error("candidate internal checkpoint is absent")
            model = load_maskable_ppo(checkpoint, environment)
            snapshot_id = f"rl-s6-seed-{seed}-unit-{unit}"
            manifest = publish_policy_snapshot(
                policy=model,
                destination=config.output_directory / "snapshots" / snapshot_id,
                snapshot_id=snapshot_id,
                owner_id="server",
                source_checkpoint_id=f"rl-s6-seed-{seed}-unit-{unit}",
                evaluation=evidence,
                evaluation_id=evaluation_id,
                created_at=created_at,
            )
            snapshots.append(
                {
                    "snapshot_id": snapshot_id,
                    "weights_sha256": manifest["files"]["weights.safetensors"],  # type: ignore[index]
                }
            )
        finally:
            environment.close()
    result = {
        "format": "rl-s6-campaign-result-v1",
        "decision": "RL_S6_GATE_PASSED",
        "passed_seeds": evidence["conforming_seed_count"],
        "required_passed_seeds": 4,
        "gate": evidence,
        "seed_results": results,
        "snapshots": snapshots,
        "beta_validation_pool": "diagnostic_required_non_blocking",
    }
    _atomic_json(config.output_directory / "campaign-result.json", result)
    return result
