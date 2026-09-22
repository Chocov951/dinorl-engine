"""Resumable four-variant RL-S5c-A3 pilot and its audit report."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.rl.env.single_agent import TrainingOpponentPool
from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.evaluation.gates import publication_gate
from dinorl_engine.rl.evaluation.suites import evaluate_publication
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.policies.local_mlp_v2 import LocalMLPV2Architecture
from dinorl_engine.rl.rewards.runtime import CompiledReward
from dinorl_engine.rl.s5b.opponents import FrozenOpponentPool
from dinorl_engine.rl.s5c.a3 import (
    A3ProtocolError,
    CompositeEvidence,
    CurriculumPhase,
    composite_gate,
    select_variant,
)
from dinorl_engine.rl.s5c.a3_config import A3Config
from dinorl_engine.rl.s5c.a3_curriculum import (
    A3CurriculumOpponentPool,
    CurriculumSchedule,
)
from dinorl_engine.rl.s5c.calibration import _atomic_json
from dinorl_engine.rl.s5c.evaluation import (
    evaluate_gate_checkpoint,
    write_evaluation_evidence,
)
from dinorl_engine.rl.s5c.evidence import validate_absolute_style
from dinorl_engine.rl.s5c.provenance import initialization_record
from dinorl_engine.rl.s5c.rewards import a3_rewards
from dinorl_engine.rl.s5c.strength import (
    evaluate_against_strong_pool,
    verify_strong_pool,
)
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore
from dinorl_engine.rl.training.unit import create_maskable_ppo

_VARIANTS = (
    "scavenger-a2-control",
    "scavenger-a3-curriculum",
    "predator-a2-control",
    "predator-a3-balanced",
)
_CURRICULUM = frozenset(_VARIANTS[1:])
_CONTROL_CHECKPOINT_SHA256 = "086fc0f4f3a7988e2064a88f90fce63edea917f53d95bee5414162b561b11e8e"
_TRAINING_SPLIT_SHA256 = "8c5b9aa690eb088eb74fabc0b5c8346c4d1567a9bc6c31eff17a297cdc6cfd61"
_HELDOUT_SPLIT_SHA256 = "0b9713c29ba6d7c93bb1a42b08b4618f8eb483a5f3efa799f0fd095955889b66"
_EVALUATION_INTERVAL = 5
_RANDOM_CONFRONTATIONS = 200
_STRONG_CONFRONTATIONS = 20
_TERMINAL = frozenset({"composite_stable_gate", "failed_bootstrap", "budget_exhausted"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _profile(variant: str) -> str:
    if variant not in _VARIANTS:
        raise A3ProtocolError("unknown A3 pilot variant")
    return variant.split("-", 1)[0]


@dataclass(slots=True)
class PilotVariantState:
    variant: str
    profile: str
    phase: str
    completed_units: int = 0
    bootstrap_consecutive: int = 0
    bootstrap_completed_unit: int | None = None
    composite_consecutive: int = 0
    gate_unit: int | None = None
    last_evaluation_unit: int = 0
    status: str = "training"

    @classmethod
    def new(cls, variant: str) -> PilotVariantState:
        return cls(
            variant=variant,
            profile=_profile(variant),
            phase="bootstrap" if variant in _CURRICULUM else "random_control",
        )

    @classmethod
    def from_mapping(cls, value: object, *, variant: str) -> PilotVariantState:
        if not isinstance(value, Mapping) or value.get("variant") != variant:
            raise A3ProtocolError("pilot state is incompatible")
        try:
            state = cls(
                variant=variant,
                profile=str(value["profile"]),
                phase=str(value["phase"]),
                completed_units=int(value["completed_units"]),
                bootstrap_consecutive=int(value["bootstrap_consecutive"]),
                bootstrap_completed_unit=(
                    None
                    if value.get("bootstrap_completed_unit") is None
                    else int(value["bootstrap_completed_unit"])
                ),
                composite_consecutive=int(value["composite_consecutive"]),
                gate_unit=(None if value.get("gate_unit") is None else int(value["gate_unit"])),
                last_evaluation_unit=int(value["last_evaluation_unit"]),
                status=str(value["status"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise A3ProtocolError("pilot state is incompatible") from error
        if state.profile != _profile(variant) or state.status not in {*_TERMINAL, "training"}:
            raise A3ProtocolError("pilot state is incompatible")
        return state

    def record_elementary(self, *, unit: int, random_score: float) -> None:
        self.last_evaluation_unit = unit
        if self.phase != "bootstrap":
            return
        self.bootstrap_consecutive = self.bootstrap_consecutive + 1 if random_score >= 0.90 else 0
        if self.bootstrap_consecutive >= 2:
            self.phase = "robustify"
            self.bootstrap_completed_unit = unit

    def record_composite(self, *, unit: int, thresholds_passed: bool) -> None:
        self.composite_consecutive = self.composite_consecutive + 1 if thresholds_passed else 0
        if self.composite_consecutive >= 2:
            self.gate_unit = unit
            self.status = "composite_stable_gate"

    def mapping(self) -> dict[str, object]:
        return {"format": "s5c-a3-pilot-state-v1", **asdict(self)}


@dataclass(frozen=True, slots=True)
class _EvaluationConfig:
    evaluation_seeds: tuple[int, ...]
    resolved_sha256: str
    random_confrontations: int = _RANDOM_CONFRONTATIONS


@dataclass(slots=True)
class _Branch:
    variant: str
    profile: str
    directory: Path
    state: PilotVariantState
    reward: CompiledReward
    pool: TrainingOpponentPool
    runner: AtomicPPOUnitRunner
    cycles: list[dict[str, object]]
    checkpoint_sha256: str | None


def _split(manifest: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    value = manifest.get("opponent_split")
    if not isinstance(value, Mapping):
        raise A3ProtocolError("A3 opponent split is absent")
    training = value.get("training_ids")
    heldout = value.get("heldout_ids")
    if (
        not isinstance(training, list)
        or not isinstance(heldout, list)
        or not all(isinstance(item, str) for item in (*training, *heldout))
        or value.get("training_sha256") != _TRAINING_SPLIT_SHA256
        or value.get("heldout_sha256") != _HELDOUT_SPLIT_SHA256
        or set(training) & set(heldout)
    ):
        raise A3ProtocolError("A3 opponent split hashes or disjointness are invalid")
    return tuple(training), tuple(heldout)


def verify_pilot_preflight(config: A3Config) -> dict[str, object]:
    manifest_value = _read_json(config.output_directory / "manifest.json", {})
    control_value = _read_json(config.output_directory / "policy-control" / "result.json", {})
    if (
        not isinstance(manifest_value, Mapping)
        or manifest_value.get("config_sha256") != config.sha256
    ):
        raise A3ProtocolError("A3 manifest/config provenance is invalid")
    if (
        not isinstance(control_value, Mapping)
        or control_value.get("decision") != "PASSED_POLICY_CONTROL"
    ):
        raise A3ProtocolError("FAILED_POLICY_CONTROL")
    control_checkpoint = (
        config.output_directory / "policy-control" / "recovery" / "units" / "unit-147" / "model.zip"
    )
    actual_control_hash = _sha256(control_checkpoint)
    if actual_control_hash != _CONTROL_CHECKPOINT_SHA256:
        raise A3ProtocolError("CONTROL unit-147 checkpoint hash differs")
    pool = verify_strong_pool(config.rl_s5b_pool, expected_sha256=config.rl_s5b_pool_sha256)
    if pool.get("pool_sha256") != config.rl_s5b_pool_sha256:
        raise A3ProtocolError("RL-S5b pool hash differs")
    training_ids, heldout_ids = _split(manifest_value)
    rewards = a3_rewards()
    declared_rewards = manifest_value.get("reward_sha256")
    reward_hashes = {name: reward.cache_key for name, reward in rewards.items()}
    if declared_rewards != reward_hashes:
        raise A3ProtocolError("A3 reward hashes differ from the frozen manifest")
    result = {
        "format": "s5c-a3-pilot-preflight-v1",
        "status": "PASSED",
        "control_checkpoint_sha256": actual_control_hash,
        "pool_sha256": config.rl_s5b_pool_sha256,
        "training_split_sha256": _TRAINING_SPLIT_SHA256,
        "heldout_split_sha256": _HELDOUT_SPLIT_SHA256,
        "training_ids": list(training_ids),
        "heldout_ids": list(heldout_ids),
        "overlap": sorted(set(training_ids) & set(heldout_ids)),
        "config_sha256": config.sha256,
        "reward_sha256": reward_hashes,
    }
    _atomic_json(config.output_directory / "pilot" / "preflight.json", result)
    return result


def _schedule(
    state: PilotVariantState, training: tuple[str, ...], heldout: tuple[str, ...]
) -> CurriculumSchedule:
    phase = CurriculumPhase.BOOTSTRAP if state.phase == "bootstrap" else CurriculumPhase.ROBUSTIFY
    robustification_unit = (
        0
        if state.bootstrap_completed_unit is None
        else max(1, state.completed_units - state.bootstrap_completed_unit + 1)
    )
    return CurriculumSchedule(phase, robustification_unit, training, heldout)


def _open_branch(
    config: A3Config,
    *,
    variant: str,
    pool_document: Mapping[str, object],
    training_ids: tuple[str, ...],
    heldout_ids: tuple[str, ...],
    resume: bool,
) -> _Branch:
    directory = config.output_directory / "pilot" / variant
    state_path = directory / "state.json"
    state_value = _read_json(state_path, None)
    state = (
        PilotVariantState.new(variant)
        if state_value is None
        else PilotVariantState.from_mapping(state_value, variant=variant)
    )
    recovery = AtomicRecoveryStore(directory / "recovery")
    latest = recovery.load_latest()
    if latest is not None and not resume:
        raise A3ProtocolError(f"existing pilot branch requires --resume: {variant}")
    if latest is not None and latest.sequence != state.completed_units:
        # A unit checkpoint is the durable authority; evaluation/state can be rebuilt.
        state.completed_units = latest.sequence
    reward = a3_rewards()[variant]
    if variant in _CURRICULUM:
        pool: TrainingOpponentPool = A3CurriculumOpponentPool(
            pool_document, schedule=_schedule(state, training_ids, heldout_ids)
        )
    else:
        pool = FrozenOpponentPool(pool_document, random_only=True)
    vector = VectorEnvironmentConfig(VectorBackend.DUMMY, 2, config.pilot_seed)
    safe_cap = config.safe_feed_episode_cap if variant == "scavenger-a3-curriculum" else None

    def environment_factory(configuration: VectorEnvironmentConfig):  # type: ignore[no-untyped-def]
        return create_vector_environment(
            configuration,
            training_opponent_pool=pool,
            reward_program=reward,
            safe_feed_episode_cap=safe_cap,
        )

    ledger = UnitLedger(directory / "ledger.sqlite3")
    run_id = f"s5c-a3-pilot-{variant}-s{config.pilot_seed}"
    ledger.reserve(
        run_id=run_id,
        units=config.specialist_max_units,
        idempotency_key=f"{run_id}-reserve",
    )
    metadata = {
        "phase": "A3-PILOT",
        "variant": variant,
        "profile": state.profile,
        "architecture": config.architecture,
        "seed": config.pilot_seed,
        "reward_sha256": reward.cache_key,
        "safe_feed_episode_cap": safe_cap,
        "training_split_sha256": _TRAINING_SPLIT_SHA256,
        "heldout_split_sha256": _HELDOUT_SPLIT_SHA256,
    }
    if latest is None:
        environment = environment_factory(vector)
        model = create_maskable_ppo(
            environment,
            seed=config.pilot_seed,
            n_steps=vector.n_steps,
            architecture=LocalMLPV2Architecture.COMPACT,
        )
        _atomic_json(
            directory / "initialization.json",
            {**initialization_record(model, seed=config.pilot_seed), "variant": variant},
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
        checkpoint_sha256 = None
    else:
        runner, _, latest = AtomicPPOUnitRunner.restore_latest(
            run_id=run_id,
            configuration=vector,
            recovery_store=recovery,
            ledger=ledger,
            environment_factory=environment_factory,
            recovery_metadata=metadata,
        )
        checkpoint_sha256 = latest.files.get("model.zip")
    cycles_value = _read_json(directory / "cycles.json", [])
    if not isinstance(cycles_value, list) or not all(
        isinstance(item, dict) for item in cycles_value
    ):
        raise A3ProtocolError("pilot training cycles are invalid")
    return _Branch(
        variant,
        state.profile,
        directory,
        state,
        reward,
        pool,
        runner,
        list(cycles_value),
        checkpoint_sha256,
    )


def _pool_counts(pool: TrainingOpponentPool) -> dict[str, int]:
    value = getattr(pool, "actual_counts", None)
    return value if isinstance(value, dict) else {"random": 0, "training_pool": 0}


def _train_unit(branch: _Branch, *, config: A3Config) -> dict[str, object]:
    unit = branch.state.completed_units + 1
    if isinstance(branch.pool, A3CurriculumOpponentPool):
        branch.pool.set_schedule(
            _schedule(
                branch.state, branch.pool.schedule.training_ids, branch.pool.schedule.heldout_ids
            )
        )
    before = _pool_counts(branch.pool)
    started = time.perf_counter()
    trained = branch.runner.run_unit(unit_id=f"unit-{unit}", sequence=unit)
    elapsed = time.perf_counter() - started
    after = _pool_counts(branch.pool)
    actual = {name: after[name] - before[name] for name in before}
    if branch.variant not in _CURRICULUM:
        episodes = (
            trained.metrics.role_counts["learner_first"]
            + trained.metrics.role_counts["learner_second"]
        )
        actual = {"random": episodes, "training_pool": 0}
    theoretical = (
        branch.pool.effective_weights
        if isinstance(branch.pool, A3CurriculumOpponentPool)
        else {"random": 1.0, "training_pool": 0.0}
    )
    cycle = {
        "unit": unit,
        "training_seconds": trained.training_seconds,
        "checkpoint_seconds": trained.checkpoint_seconds,
        "wall_seconds": elapsed,
        "checkpoint_sha256": trained.recovery.files.get("model.zip"),
        "learner_transitions": trained.metrics.learner_transitions,
        "engine_actions": trained.metrics.engine_actions,
        "optimizer_steps": trained.metrics.optimizer_steps,
        "diagnostics": trained.metrics.diagnostics,
        "role_counts": trained.metrics.role_counts,
        "curriculum_phase": branch.state.phase,
        "curriculum_theoretical": theoretical,
        "curriculum_actual_episodes": actual,
    }
    branch.cycles.append(cycle)
    branch.state.completed_units = unit
    branch.checkpoint_sha256 = trained.recovery.files.get("model.zip")
    _atomic_json(branch.directory / "cycles.json", branch.cycles)
    _atomic_json(branch.directory / "state.json", branch.state.mapping())
    return cycle


def _style_metrics(records: list[Mapping[str, object]]) -> dict[str, float]:
    wins = [record for record in records if record.get("outcome") == "win"]
    point_wins = sum(record.get("victory_mode") == "carcass_score" for record in wins)
    damage_all = [
        float(record.get("style_events", {}).get("damage_dealt", 0.0))  # type: ignore[union-attr]
        for record in records
    ]
    damage_wins = [
        float(record.get("style_events", {}).get("damage_dealt", 0.0))  # type: ignore[union-attr]
        for record in wins
    ]
    return {
        "point_win_rate_conditioned": point_wins / len(wins) if wins else 0.0,
        "point_win_rate_unconditioned": point_wins / len(records) if records else 0.0,
        "damage_mean": statistics.fmean(damage_all) if damage_all else 0.0,
        "damage_mean_conditioned_on_win": (statistics.fmean(damage_wins) if damage_wins else 0.0),
    }


def _elementary(
    branch: _Branch,
    *,
    config: A3Config,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    unit = branch.state.completed_units
    directory = branch.directory / "evaluations" / f"unit-{unit}" / "elementary"
    existing = _read_json(directory / "summary.json", None)
    games_path = directory / "games.jsonl"
    if isinstance(existing, dict) and games_path.is_file():
        records = [
            json.loads(line) for line in games_path.read_text(encoding="utf-8").splitlines() if line
        ]
        expected_games = len(config.evaluation_seeds) * (
            len(SCRIPTED_CONTROLLER_IDS) * 4 + _RANDOM_CONFRONTATIONS * 2
        )
        if len(records) != expected_games:
            raise A3ProtocolError(
                f"pilot elementary evaluation has {len(records)}/{expected_games} games"
            )
        return {**existing, "records": records}
    eval_config = _EvaluationConfig(config.evaluation_seeds, config.sha256)
    evaluation = evaluate_gate_checkpoint(
        branch.runner.model,
        config=eval_config,  # type: ignore[arg-type]
        archetype=branch.profile,
        training_seed=config.pilot_seed,
        unit=unit,
        checkpoint_sha256=branch.checkpoint_sha256,
        progress=progress,
        reward_program=branch.reward,
        safe_feed_episode_cap=(
            config.safe_feed_episode_cap if branch.variant == "scavenger-a3-curriculum" else None
        ),
        evaluation_label=f"S5c-A3 {branch.variant}",
    )
    records_value = evaluation.get("records")
    assert isinstance(records_value, list)
    records = [record for record in records_value if isinstance(record, Mapping)]
    evaluation["absolute_style"] = validate_absolute_style(branch.profile, records)
    evaluation["style_metrics"] = _style_metrics(records)
    write_evaluation_evidence(directory, evaluation, diagnostic_replay_sample=5)
    return evaluation


def _comparative(
    variant: str,
    elementary: Mapping[str, Mapping[str, object]],
) -> tuple[bool, float, dict[str, object]]:
    own_value = elementary[variant].get("style_metrics")
    if not isinstance(own_value, Mapping):
        return False, 0.0, {}
    other_profile = "predator" if _profile(variant) == "scavenger" else "scavenger"
    opponents = [
        value.get("style_metrics")
        for name, value in elementary.items()
        if _profile(name) == other_profile and isinstance(value.get("style_metrics"), Mapping)
    ]
    if not opponents:
        return False, 0.0, {}
    if _profile(variant) == "scavenger":
        fields = ("point_win_rate_conditioned", "point_win_rate_unconditioned")
    else:
        fields = ("damage_mean", "damage_mean_conditioned_on_win")
    margins = [
        min(float(own_value[field]) - float(opponent[field]) for field in fields)
        for opponent in opponents
        if isinstance(opponent, Mapping)
    ]
    best_margin = max(margins)
    return (
        best_margin > 0.0,
        best_margin,
        {
            "fields": list(fields),
            "own": {field: float(own_value[field]) for field in fields},
            "best_pairwise_margin": best_margin,
            "rule": "strictly greater than at least one opposite-profile pilot variant",
        },
    )


def _full_composite(
    branch: _Branch,
    *,
    config: A3Config,
    heldout_ids: tuple[str, ...],
    elementary: Mapping[str, object],
    comparative_passed: bool,
    style_margin: float,
) -> dict[str, object]:
    unit = branch.state.completed_units
    directory = branch.directory / "evaluations" / f"unit-{unit}"
    heldout_path = directory / "heldout.json"
    heldout_value = _read_json(heldout_path, None)
    if isinstance(heldout_value, dict):
        heldout = heldout_value
    else:
        heldout = evaluate_against_strong_pool(
            branch.runner.model,
            pool_path=config.rl_s5b_pool,
            expected_sha256=config.rl_s5b_pool_sha256,
            evaluation_seeds=config.evaluation_seeds,
            confrontations=_STRONG_CONFRONTATIONS,
            learner_policy_id=f"s5c-a3-{branch.variant}-s20-u{unit}-heldout",
            opponent_ids=heldout_ids,
        )
        _atomic_json(heldout_path, heldout)
    stochastic_path = directory / "stochastic.json"
    stochastic_value = _read_json(stochastic_path, None)
    if isinstance(stochastic_value, dict):
        stochastic = stochastic_value
    else:
        publication = evaluate_publication(
            branch.runner.model,
            seed=config.evaluation_seeds[0],
            diagnostic_directory=directory / "stochastic-replays",
        )
        scores = elementary.get("scores")
        if not isinstance(scores, Mapping) or not isinstance(publication.get("scores"), Mapping):
            raise A3ProtocolError("pilot stochastic evaluation is incomplete")
        stochastic = {
            "evaluation": publication,
            "gate": publication_gate(scores, publication["scores"]),
        }
        _atomic_json(stochastic_path, stochastic)
    aggregates = elementary.get("aggregates")
    global_summary = aggregates.get("global") if isinstance(aggregates, Mapping) else None
    reward_hacking = elementary.get("reward_hacking")
    warnings = reward_hacking.get("warnings") if isinstance(reward_hacking, Mapping) else []
    warnings = warnings if isinstance(warnings, list) else []
    blocking_warnings = [item for item in warnings if item != "round_limit_above_10_percent"]
    round_limit_rate = (
        float(global_summary.get("round_limit_rate", 1.0))
        if isinstance(global_summary, Mapping)
        else 1.0
    )
    absolute = elementary.get("absolute_style")
    scores = elementary.get("scores")
    deterministic_scores = (
        tuple(float(scores[name]) for name in SCRIPTED_CONTROLLER_IDS)
        if isinstance(scores, Mapping)
        else (0.0, 0.0, 0.0)
    )
    random_score = (
        float(scores.get(RANDOM_LEGAL_CONTROLLER_ID, 0.0)) if isinstance(scores, Mapping) else 0.0
    )
    stochastic_gate = stochastic.get("gate")
    evidence = CompositeEvidence(
        random_score=random_score,
        deterministic_scores=deterministic_scores,
        absolute_style_passed=isinstance(absolute, Mapping) and absolute.get("passed") is True,
        comparative_style_passed=comparative_passed,
        heldout_score=float(heldout["global"]["score"]),  # type: ignore[index]
        reward_hacking_blocking=bool(blocking_warnings),
        round_limit_strategy=round_limit_rate > 0.10,
        stochastic_collapse=(
            not isinstance(stochastic_gate, Mapping) or stochastic_gate.get("passed") is not True
        ),
        replay_identity_visible=(
            isinstance(absolute, Mapping)
            and absolute.get("passed") is True
            and (directory / "elementary" / "diagnostics" / "style_representative").is_dir()
        ),
    )
    gate = composite_gate(evidence, previous_passed=branch.state.composite_consecutive == 1)
    result = {
        "format": "s5c-a3-composite-evaluation-v1",
        "unit": unit,
        "evidence": asdict(evidence),
        "gate": gate,
        "style_margin": style_margin,
        "blocking_reward_hacking_warnings": blocking_warnings,
        "heldout": heldout,
        "stochastic": stochastic,
    }
    _atomic_json(directory / "composite.json", result)
    return result


def _latest_elementary(branch: _Branch) -> dict[str, object] | None:
    unit = branch.state.last_evaluation_unit
    if not unit:
        return None
    directory = branch.directory / "evaluations" / f"unit-{unit}" / "elementary"
    summary = _read_json(directory / "summary.json", None)
    games = directory / "games.jsonl"
    if not isinstance(summary, dict) or not games.is_file():
        return None
    records = [json.loads(line) for line in games.read_text(encoding="utf-8").splitlines() if line]
    return {**summary, "records": records}


def _totals(cycles: list[dict[str, object]]) -> dict[str, object]:
    roles = {name: 0 for name in ("learner_first", "learner_second", "learner_a", "learner_b")}
    curriculum = {"random": 0, "training_pool": 0}
    for cycle in cycles:
        for name, value in cycle.get("role_counts", {}).items():  # type: ignore[union-attr]
            roles[name] += int(value)
        for name, value in cycle.get("curriculum_actual_episodes", {}).items():  # type: ignore[union-attr]
            curriculum[name] += int(value)
    episodes = sum(curriculum.values())
    return {
        "role_counts": roles,
        "curriculum_actual_episodes": curriculum,
        "curriculum_actual_proportions": {
            name: value / episodes if episodes else 0.0 for name, value in curriculum.items()
        },
    }


def _candidate(branch: _Branch) -> dict[str, object]:
    unit = branch.state.gate_unit or branch.state.last_evaluation_unit
    composite = _read_json(branch.directory / "evaluations" / f"unit-{unit}" / "composite.json", {})
    heldout = composite.get("heldout") if isinstance(composite, Mapping) else None
    global_value = heldout.get("global") if isinstance(heldout, Mapping) else None
    by_opponent = heldout.get("by_opponent") if isinstance(heldout, Mapping) else None
    opponent_scores = (
        [float(value["score"]) for value in by_opponent.values() if isinstance(value, Mapping)]
        if isinstance(by_opponent, Mapping)
        else []
    )
    return {
        "id": branch.variant,
        "profile": branch.profile,
        "eligible": branch.state.status == "composite_stable_gate",
        "status": branch.state.status,
        "gate_unit": branch.state.gate_unit or config_specialist_sentinel(branch.state),
        "heldout_score": (
            float(global_value.get("score", 0.0)) if isinstance(global_value, Mapping) else 0.0
        ),
        "opponent_variance": statistics.pvariance(opponent_scores) if opponent_scores else 1.0,
        "round_limit_rate": (
            float(global_value.get("round_limit_rate", 1.0))
            if isinstance(global_value, Mapping)
            else 1.0
        ),
        "style_margin": (
            float(composite.get("style_margin", 0.0)) if isinstance(composite, Mapping) else 0.0
        ),
        "checkpoint_sha256": branch.checkpoint_sha256,
        "completed_units": branch.state.completed_units,
        "totals": _totals(branch.cycles),
        "composite": composite,
    }


def config_specialist_sentinel(state: PilotVariantState) -> int:
    return state.completed_units + 1_000_000


def _render_report(report: Mapping[str, object]) -> str:
    lines = [
        "# RL-S5c-A3 — rapport PILOT",
        "",
        f"Décision : `{report['decision']}`",
        f"A3-CONFIRM proposable : `{'oui' if report['a3_confirm_proposable'] else 'non'}`",
        "",
        "## Variantes",
        "",
    ]
    variants = report.get("variants")
    if isinstance(variants, list):
        for item in variants:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"- `{item.get('id')}` : statut `{item.get('status')}`, "
                f"éligible `{item.get('eligible')}`, unité gate `{item.get('gate_unit')}`, "
                f"score held-out `{float(item.get('heldout_score', 0.0)):.4f}`."
            )
    lines.extend(["", "## Sélection par profil", ""])
    decisions = report.get("profile_decisions")
    if isinstance(decisions, Mapping):
        for profile, decision in decisions.items():
            lines.append(f"- `{profile}` : `{decision}`")
    hashes = report.get("hashes")
    lines.extend(["", "## Hashes", ""])
    if isinstance(hashes, Mapping):
        for name, value in hashes.items():
            lines.append(f"- `{name}` : `{value}`")
    return "\n".join(lines) + "\n"


def _report(
    config: A3Config,
    *,
    branches: list[_Branch],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    candidates = [_candidate(branch) for branch in branches]
    selected: dict[str, object] = {}
    profile_decisions: dict[str, str] = {}
    for profile in ("scavenger", "predator"):
        family = [item for item in candidates if item["profile"] == profile]
        try:
            choice = dict(select_variant(family))
        except A3ProtocolError:
            profile_decisions[profile] = "NO_ELIGIBLE_VARIANT"
        else:
            selected_branch = next(branch for branch in branches if branch.variant == choice["id"])
            full_pool_path = selected_branch.directory / "full-pool-diagnostic.json"
            full_pool_value = _read_json(full_pool_path, None)
            if not isinstance(full_pool_value, dict):
                full_pool_value = evaluate_against_strong_pool(
                    selected_branch.runner.model,
                    pool_path=config.rl_s5b_pool,
                    expected_sha256=config.rl_s5b_pool_sha256,
                    evaluation_seeds=config.evaluation_seeds,
                    confrontations=_STRONG_CONFRONTATIONS,
                    learner_policy_id=(
                        f"s5c-a3-{selected_branch.variant}-s20-"
                        f"u{selected_branch.state.gate_unit}-full-pool"
                    ),
                )
                _atomic_json(full_pool_path, full_pool_value)
            selected[profile] = choice
            choice["full_pool_diagnostic"] = full_pool_value
            profile_decisions[profile] = f"SELECTED:{choice['id']}"
    proposable = len(selected) == 2
    decision = "ELIGIBLE_VARIANTS_SELECTED" if selected else "NO_ELIGIBLE_VARIANT"
    report = {
        "format": "s5c-a3-pilot-report-v1",
        "phase": "A3-PILOT",
        "decision": decision,
        "a3_confirm_proposable": proposable,
        "profile_decisions": profile_decisions,
        "selected": selected,
        "variants": candidates,
        "preflight": dict(preflight),
        "hashes": {
            "control_checkpoint_sha256": _CONTROL_CHECKPOINT_SHA256,
            "pool_sha256": config.rl_s5b_pool_sha256,
            "training_split_sha256": _TRAINING_SPLIT_SHA256,
            "heldout_split_sha256": _HELDOUT_SPLIT_SHA256,
            "config_sha256": config.sha256,
            "reward_sha256": {name: reward.cache_key for name, reward in a3_rewards().items()},
        },
        "prohibited_phases_launched": [],
    }
    report_directory = config.output_directory / "report"
    json_path = report_directory / "rl-s5c-a3-pilot-report.json"
    markdown_path = report_directory / "rl-s5c-a3-pilot-report.md"
    _atomic_json(json_path, report)
    markdown_path.write_text(_render_report(report), encoding="utf-8")
    selection = {
        "format": "s5c-a3-pilot-selection-v1",
        "decision": decision,
        "a3_confirm_proposable": proposable,
        "selected": selected,
    }
    _atomic_json(config.output_directory / "pilot" / "selection.json", selection)
    return {
        "decision": decision,
        "a3_confirm_proposable": proposable,
        "report": str(json_path),
        "markdown": str(markdown_path),
        "selection": str(config.output_directory / "pilot" / "selection.json"),
    }


def _evaluate_checkpoints(
    checkpoint_branches: list[_Branch],
    *,
    branches: list[_Branch],
    config: A3Config,
    heldout_ids: tuple[str, ...],
    progress: Callable[[str], None] | None,
) -> None:
    elementary_latest: dict[str, Mapping[str, object]] = {}
    for branch in branches:
        if branch in checkpoint_branches:
            elementary_latest[branch.variant] = _elementary(
                branch, config=config, progress=progress
            )
        else:
            previous = _latest_elementary(branch)
            if previous is not None:
                elementary_latest[branch.variant] = previous
    for branch in checkpoint_branches:
        evaluation = elementary_latest[branch.variant]
        scores = evaluation.get("scores")
        random_score = (
            float(scores.get(RANDOM_LEGAL_CONTROLLER_ID, 0.0))
            if isinstance(scores, Mapping)
            else 0.0
        )
        branch.state.record_elementary(unit=branch.state.completed_units, random_score=random_score)
        comparative_passed, style_margin, comparative = _comparative(
            branch.variant, elementary_latest
        )
        _atomic_json(
            branch.directory
            / "evaluations"
            / f"unit-{branch.state.completed_units}"
            / "comparative-style.json",
            comparative,
        )
        absolute = evaluation.get("absolute_style")
        gate = evaluation.get("gate")
        curriculum_ready = branch.variant not in _CURRICULUM or (
            branch.state.bootstrap_completed_unit is not None
            and branch.state.completed_units - branch.state.bootstrap_completed_unit >= 20
        )
        preliminary = (
            curriculum_ready
            and branch.state.phase != "bootstrap"
            and isinstance(gate, Mapping)
            and gate.get("thresholds_passed") is True
            and isinstance(absolute, Mapping)
            and absolute.get("passed") is True
            and comparative_passed
        )
        if preliminary:
            composite = _full_composite(
                branch,
                config=config,
                heldout_ids=heldout_ids,
                elementary=evaluation,
                comparative_passed=comparative_passed,
                style_margin=style_margin,
            )
            composite_gate_value = composite.get("gate")
            thresholds = (
                isinstance(composite_gate_value, Mapping)
                and composite_gate_value.get("thresholds_passed") is True
            )
            branch.state.record_composite(
                unit=branch.state.completed_units, thresholds_passed=thresholds
            )
        else:
            branch.state.record_composite(
                unit=branch.state.completed_units, thresholds_passed=False
            )
        _atomic_json(branch.directory / "state.json", branch.state.mapping())


def run_pilot(
    config: A3Config,
    *,
    resume: bool,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run all four common-seed branches in synchronized five-unit blocks."""

    selection_path = config.output_directory / "pilot" / "selection.json"
    if selection_path.is_file():
        if not resume:
            raise A3ProtocolError("completed A3 pilot requires --resume for idempotent read")
        report_path = config.output_directory / "report" / "rl-s5c-a3-pilot-report.json"
        value = _read_json(report_path, {})
        if not isinstance(value, Mapping):
            raise A3ProtocolError("completed pilot report is invalid")
        return {
            "decision": value.get("decision"),
            "a3_confirm_proposable": value.get("a3_confirm_proposable"),
            "report": str(report_path),
            "markdown": str(report_path.with_suffix(".md")),
            "selection": str(selection_path),
        }
    preflight = verify_pilot_preflight(config)
    manifest = _read_json(config.output_directory / "manifest.json", {})
    assert isinstance(manifest, Mapping)
    training_ids, heldout_ids = _split(manifest)
    pool_document = verify_strong_pool(
        config.rl_s5b_pool, expected_sha256=config.rl_s5b_pool_sha256
    )
    branches = [
        _open_branch(
            config,
            variant=variant,
            pool_document=pool_document,
            training_ids=training_ids,
            heldout_ids=heldout_ids,
            resume=resume,
        )
        for variant in _VARIANTS
    ]
    initialization_values = [
        _read_json(branch.directory / "initialization.json", {}) for branch in branches
    ]
    if not all(isinstance(value, Mapping) for value in initialization_values):
        raise A3ProtocolError("pilot initialization provenance is invalid")
    initial_hashes = {
        str(value.get("weights_sha256"))
        for value in initialization_values
        if isinstance(value, Mapping)
    }
    if len(initial_hashes) != 1:
        raise A3ProtocolError("common seed did not produce identical zero-start weights")
    try:
        pending = [
            branch
            for branch in branches
            if branch.state.status not in _TERMINAL
            and branch.state.completed_units % _EVALUATION_INTERVAL == 0
            and branch.state.completed_units > branch.state.last_evaluation_unit
        ]
        if pending:
            _evaluate_checkpoints(
                pending,
                branches=branches,
                config=config,
                heldout_ids=heldout_ids,
                progress=progress,
            )
        while any(branch.state.status not in _TERMINAL for branch in branches):
            target = min(
                branch.state.completed_units + 1
                for branch in branches
                if branch.state.status not in _TERMINAL
            )
            trained_this_round: list[_Branch] = []
            for branch in branches:
                if branch.state.status in _TERMINAL or branch.state.completed_units >= target:
                    continue
                if (
                    branch.state.phase == "bootstrap"
                    and branch.state.completed_units >= config.bootstrap_max_units
                ):
                    branch.state.status = "failed_bootstrap"
                    _atomic_json(branch.directory / "state.json", branch.state.mapping())
                    continue
                cycle = _train_unit(branch, config=config)
                trained_this_round.append(branch)
                if progress is not None:
                    percentage = 100 * branch.state.completed_units / config.specialist_max_units
                    progress(
                        f"S5c-A3 PILOT {branch.variant}: {branch.state.completed_units}/"
                        f"{config.specialist_max_units} units "
                        f"({percentage:.1f}%) "
                        f"phase={branch.state.phase} train={float(cycle['training_seconds']):.2f}s"
                    )
            checkpoint_branches = [
                branch
                for branch in trained_this_round
                if branch.state.completed_units % _EVALUATION_INTERVAL == 0
            ]
            if checkpoint_branches:
                _evaluate_checkpoints(
                    checkpoint_branches,
                    branches=branches,
                    config=config,
                    heldout_ids=heldout_ids,
                    progress=progress,
                )
            for branch in branches:
                if (
                    branch.state.status == "training"
                    and branch.state.completed_units >= config.specialist_max_units
                ):
                    branch.state.status = "budget_exhausted"
                    _atomic_json(branch.directory / "state.json", branch.state.mapping())
        return _report(config, branches=branches, preflight=preflight)
    finally:
        for branch in branches:
            branch.runner.environment.close()
            close = getattr(branch.pool, "close", None)
            if callable(close):
                close()


def run_confirm(config: A3Config, *, resume: bool) -> dict[str, object]:
    del config, resume
    raise A3ProtocolError(
        "A3-CONFIRM is intentionally blocked until a new explicit user checkpoint"
    )
