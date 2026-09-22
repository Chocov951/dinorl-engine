"""Resumable corrective audit and continuation for RL-S5c-A3-PILOT-R2."""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID
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
from dinorl_engine.rl.s5c.a3 import A3ProtocolError
from dinorl_engine.rl.s5c.a3_config import A3Config
from dinorl_engine.rl.s5c.a3_curriculum import (
    A3R2CurriculumOpponentPool,
    R2CurriculumSchedule,
)
from dinorl_engine.rl.s5c.a3_r2 import (
    R2CompositeEvidence,
    composite_gate_r2,
    not_evaluated_metrics,
    reward_semantics_descriptor,
    reward_semantics_sha256,
    reward_warning_is_blocking,
)
from dinorl_engine.rl.s5c.calibration import _atomic_json
from dinorl_engine.rl.s5c.evaluation import evaluate_gate_checkpoint, write_evaluation_evidence
from dinorl_engine.rl.s5c.evidence import validate_absolute_style
from dinorl_engine.rl.s5c.provenance import initialization_record
from dinorl_engine.rl.s5c.rewards import a3_rewards
from dinorl_engine.rl.s5c.strength import evaluate_against_strong_pool, verify_strong_pool
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore
from dinorl_engine.rl.training.unit import create_maskable_ppo, load_maskable_ppo

_SOURCE_VARIANTS = (
    "scavenger-a2-control",
    "scavenger-a3-curriculum",
    "predator-a2-control",
    "predator-a3-balanced",
)
_SCAVENGERS = _SOURCE_VARIANTS[:2]
_PREDATORS = _SOURCE_VARIANTS[2:]
_SOFT_SCAVENGER = "scavenger-a3-soft"
_CONTROL_CHECKPOINT_SHA256 = "086fc0f4f3a7988e2064a88f90fce63edea917f53d95bee5414162b561b11e8e"
_TRAINING_SPLIT_SHA256 = "8c5b9aa690eb088eb74fabc0b5c8346c4d1567a9bc6c31eff17a297cdc6cfd61"
_HELDOUT_SPLIT_SHA256 = "0b9713c29ba6d7c93bb1a42b08b4618f8eb483a5f3efa799f0fd095955889b66"
_BOOTSTRAP_MAX_UNITS = 147
_MAX_UNITS = 197
_SCREENING_CONFRONTATIONS = 2
_FULL_CONFRONTATIONS = 20
_NEAR_BAND = (0.20, 0.55)


@dataclass(frozen=True, slots=True)
class _EvaluationConfig:
    evaluation_seeds: tuple[int, ...]
    resolved_sha256: str
    random_confrontations: int = 200


@dataclass(slots=True)
class R2TrainingState:
    variant: str
    profile: str
    completed_units: int
    phase: str = "bootstrap"
    bootstrap_consecutive: int = 0
    bootstrap_completed_unit: int | None = None
    last_evaluation_unit: int = 0
    previous_composite_passed: bool = False
    gate_unit: int | None = None
    status: str = "training"
    force_full_next: bool = False

    @classmethod
    def from_mapping(cls, value: object) -> R2TrainingState:
        if not isinstance(value, Mapping):
            raise A3ProtocolError("PILOT-R2 training state is invalid")
        fields = {field.name for field in __import__("dataclasses").fields(cls)}
        try:
            return cls(**{name: value[name] for name in fields})  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as error:
            raise A3ProtocolError("PILOT-R2 training state is invalid") from error

    def mapping(self) -> dict[str, object]:
        return {"format": "s5c-a3-pilot-r2-state-v1", **asdict(self)}


@dataclass(slots=True)
class _TrainingBranch:
    state: R2TrainingState
    directory: Path
    reward: CompiledReward
    pool: TrainingOpponentPool
    runner: AtomicPPOUnitRunner
    cycles: list[dict[str, object]]
    checkpoint_sha256: str | None
    safe_feed_cap: float | None


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


def _split(manifest: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    split = manifest.get("opponent_split")
    if not isinstance(split, Mapping):
        raise A3ProtocolError("PILOT-R2 opponent split is absent")
    training = split.get("training_ids")
    heldout = split.get("heldout_ids")
    if (
        not isinstance(training, list)
        or not isinstance(heldout, list)
        or not all(isinstance(item, str) for item in (*training, *heldout))
        or split.get("training_sha256") != _TRAINING_SPLIT_SHA256
        or split.get("heldout_sha256") != _HELDOUT_SPLIT_SHA256
        or set(training) & set(heldout)
    ):
        raise A3ProtocolError("PILOT-R2 split hashes or disjointness are invalid")
    return tuple(training), tuple(heldout)


def _checkpoint(source: Path, unit: int) -> tuple[Path, dict[str, object]]:
    directory = source / "recovery" / "units" / f"unit-{unit}"
    manifest_value = _read_json(directory / "manifest.json", {})
    if not isinstance(manifest_value, dict) or manifest_value.get("sequence") != unit:
        raise A3ProtocolError(f"invalid source checkpoint unit {unit}: {source.name}")
    files = manifest_value.get("files")
    if not isinstance(files, dict):
        raise A3ProtocolError("source checkpoint manifest has no hashes")
    for name, expected in files.items():
        path = directory / str(name)
        if not isinstance(expected, str) or not path.is_file() or _sha256(path) != expected:
            raise A3ProtocolError(f"source checkpoint hash mismatch: {source.name}/unit-{unit}")
    return directory, manifest_value


def _reward_semantics(variant: str, config: A3Config) -> tuple[float | None, bool]:
    bounded = variant in {"scavenger-a3-curriculum", _SOFT_SCAVENGER}
    return (config.safe_feed_episode_cap if bounded else None, bounded)


def _preflight(config: A3Config) -> dict[str, object]:
    root = config.output_directory
    manifest = _read_json(root / "manifest.json", {})
    control = _read_json(root / "policy-control" / "result.json", {})
    if not isinstance(manifest, Mapping) or not isinstance(control, Mapping):
        raise A3ProtocolError("PILOT-R2 source manifest or CONTROL is absent")
    if control.get("decision") != "PASSED_POLICY_CONTROL":
        raise A3ProtocolError("FAILED_POLICY_CONTROL")
    control_model = root / "policy-control" / "recovery" / "units" / "unit-147" / "model.zip"
    if _sha256(control_model) != _CONTROL_CHECKPOINT_SHA256:
        raise A3ProtocolError("CONTROL checkpoint hash differs")
    pool = verify_strong_pool(config.rl_s5b_pool, expected_sha256=config.rl_s5b_pool_sha256)
    if pool.get("pool_sha256") != config.rl_s5b_pool_sha256:
        raise A3ProtocolError("RL-S5b pool hash differs")
    training, heldout = _split(manifest)
    source_hashes = manifest.get("reward_sha256")
    if not isinstance(source_hashes, Mapping):
        raise A3ProtocolError("source reward hashes are absent")
    rewards = a3_rewards()
    provenance: dict[str, object] = {}
    for variant in _SOURCE_VARIANTS:
        cap, once = _reward_semantics(variant, config)
        source = root / "pilot" / variant
        unit = 197 if variant in _SCAVENGERS else 100
        checkpoint, checkpoint_manifest = _checkpoint(source, unit)
        state = _read_json(checkpoint / "state.json", {})
        metadata = state.get("metadata") if isinstance(state, Mapping) else None
        legacy_hash = rewards[variant].cache_key
        valid = (
            source_hashes.get(variant) == legacy_hash
            and isinstance(metadata, Mapping)
            and metadata.get("reward_sha256") == legacy_hash
            and metadata.get("safe_feed_episode_cap") == cap
        )
        provenance[variant] = {
            "status": "valid" if valid else "invalid_reward_provenance",
            "source_unit": unit,
            "source_checkpoint_sha256": checkpoint_manifest["files"]["model.zip"],
            "legacy_dsl_sha256": legacy_hash,
            "semantic_descriptor": reward_semantics_descriptor(
                rewards[variant], safe_feed_cap=cap, once_per_opportunity=once
            ),
            "semantic_sha256": reward_semantics_sha256(
                rewards[variant], safe_feed_cap=cap, once_per_opportunity=once
            ),
        }
    result = {
        "format": "s5c-a3-pilot-r2-preflight-v1",
        "status": (
            "PASSED"
            if all(item["status"] == "valid" for item in provenance.values())  # type: ignore[index,union-attr]
            else "INVALID_REWARD_PROVENANCE"
        ),
        "control_checkpoint_sha256": _CONTROL_CHECKPOINT_SHA256,
        "pool_sha256": config.rl_s5b_pool_sha256,
        "training_split_sha256": _TRAINING_SPLIT_SHA256,
        "heldout_split_sha256": _HELDOUT_SPLIT_SHA256,
        "overlap": sorted(set(training) & set(heldout)),
        "bootstrap_max_units": _BOOTSTRAP_MAX_UNITS,
        "specialist_max_units": _MAX_UNITS,
        "source_config_sha256": config.sha256,
        "reward_provenance": provenance,
    }
    output = root / "pilot-r2"
    _atomic_json(output / "preflight.json", result)
    _atomic_json(output / "reward-provenance.json", provenance)
    return result


def _load_model(source_branch: Path, *, unit: int) -> tuple[object, object, str]:
    checkpoint, manifest = _checkpoint(source_branch, unit)
    vector = VectorEnvironmentConfig(VectorBackend.DUMMY, 2, 20)
    environment = create_vector_environment(vector)
    model = load_maskable_ppo(checkpoint / "model.zip", environment)
    files = manifest["files"]
    assert isinstance(files, dict)
    return model, environment, str(files["model.zip"])


def _source_elementary(root: Path, variant: str, unit: int) -> dict[str, object] | None:
    directory = root / "pilot" / variant / "evaluations" / f"unit-{unit}" / "elementary"
    summary = _read_json(directory / "summary.json", None)
    games = directory / "games.jsonl"
    if not isinstance(summary, dict) or not games.is_file():
        return None
    records = [json.loads(line) for line in games.read_text(encoding="utf-8").splitlines() if line]
    return {**summary, "records": records}


def _elementary_for_checkpoint(
    *,
    config: A3Config,
    variant: str,
    profile: str,
    unit: int,
    model: object,
    checkpoint_sha256: str,
    reward: CompiledReward,
    destination: Path,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, object], str]:
    reused = _source_elementary(config.output_directory, variant, unit)
    if reused is not None:
        return reused, "reused_source_elementary"
    existing_summary = _read_json(destination / "summary.json", None)
    games = destination / "games.jsonl"
    if isinstance(existing_summary, dict) and games.is_file():
        records = [
            json.loads(line) for line in games.read_text(encoding="utf-8").splitlines() if line
        ]
        return {**existing_summary, "records": records}, "reused_r2_elementary"
    cap, _ = _reward_semantics(variant, config)
    evaluation = evaluate_gate_checkpoint(
        model,
        config=_EvaluationConfig(config.evaluation_seeds, config.sha256),  # type: ignore[arg-type]
        archetype=profile,
        training_seed=config.pilot_seed,
        unit=unit,
        checkpoint_sha256=checkpoint_sha256,
        progress=progress,
        reward_program=reward,
        safe_feed_episode_cap=cap,
        evaluation_label=f"S5c-A3 PILOT-R2 {variant} unit {unit}",
    )
    records = evaluation.get("records")
    if not isinstance(records, list):
        raise A3ProtocolError("PILOT-R2 elementary evaluation has no records")
    evaluation["absolute_style"] = validate_absolute_style(profile, records)
    write_evaluation_evidence(destination, evaluation, diagnostic_replay_sample=5)
    return evaluation, "new_r2_elementary"


def _strength(
    *,
    path: Path,
    model: object,
    config: A3Config,
    heldout_ids: tuple[str, ...],
    variant: str,
    unit: int,
    confrontations: int,
    label: str,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, object], str]:
    existing = _read_json(path, None)
    if isinstance(existing, dict):
        return existing, f"reused_r2_{label}"
    result = evaluate_against_strong_pool(
        model,
        pool_path=config.rl_s5b_pool,
        expected_sha256=config.rl_s5b_pool_sha256,
        evaluation_seeds=config.evaluation_seeds,
        confrontations=confrontations,
        learner_policy_id=f"s5c-a3-pilot-r2-{variant}-s20-u{unit}-{label}",
        opponent_ids=heldout_ids,
        progress=progress,
        evaluation_label=f"S5c-A3 PILOT-R2 {label} {variant} unit {unit}",
    )
    _atomic_json(path, result)
    return result, f"new_r2_{label}"


def _scores(elementary: Mapping[str, object]) -> tuple[bool, Mapping[str, object]]:
    scores = elementary.get("scores")
    gate = elementary.get("gate")
    if not isinstance(scores, Mapping) or not isinstance(gate, Mapping):
        raise A3ProtocolError("PILOT-R2 elementary competence evidence is incomplete")
    return gate.get("thresholds_passed") is True, scores


def _full_evaluation(
    *,
    config: A3Config,
    variant: str,
    profile: str,
    unit: int,
    model: object,
    elementary: Mapping[str, object],
    heldout_ids: tuple[str, ...],
    directory: Path,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, object], list[str]]:
    actions: list[str] = []
    heldout, action = _strength(
        path=directory / "heldout-full.json",
        model=model,
        config=config,
        heldout_ids=heldout_ids,
        variant=variant,
        unit=unit,
        confrontations=_FULL_CONFRONTATIONS,
        label="full",
        progress=progress,
    )
    actions.append(action)
    records_value = heldout.get("records")
    if not isinstance(records_value, list):
        raise A3ProtocolError("PILOT-R2 held-out evaluation has no records")
    heldout_records = [record for record in records_value if isinstance(record, Mapping)]
    heldout_style = validate_absolute_style(profile, heldout_records)
    stochastic_path = directory / "stochastic.json"
    stochastic_value = _read_json(stochastic_path, None)
    if isinstance(stochastic_value, dict):
        stochastic = stochastic_value
        actions.append("reused_r2_stochastic")
    else:
        publication = evaluate_publication(
            model,
            seed=config.evaluation_seeds[0],
            diagnostic_directory=directory / "stochastic-replays",
        )
        _, scores = _scores(elementary)
        stochastic = {
            "evaluation": publication,
            "gate": publication_gate(scores, publication["scores"]),
        }  # type: ignore[arg-type]
        _atomic_json(stochastic_path, stochastic)
        actions.append("new_r2_stochastic")
    hacking = elementary.get("reward_hacking")
    warnings_value = hacking.get("warnings") if isinstance(hacking, Mapping) else []
    warnings = [str(item) for item in warnings_value] if isinstance(warnings_value, list) else []
    cap, bounded = _reward_semantics(variant, config)
    blocking = [
        warning
        for warning in warnings
        if reward_warning_is_blocking(
            warning,
            bounded=bounded,
            exploitable=(warning == "repeatable_action_reward_risk" and cap is None),
        )
    ]
    historical, _ = _scores(elementary)
    global_value = heldout.get("global")
    if not isinstance(global_value, Mapping):
        raise A3ProtocolError("PILOT-R2 held-out global summary is absent")
    heldout_score = float(global_value["score"])
    round_limit_rate = float(global_value["round_limit_rate"])
    stochastic_gate = stochastic.get("gate")
    evidence = R2CompositeEvidence(
        historical_gate_passed=historical,
        heldout_score=heldout_score,
        heldout_style_passed=heldout_style.get("passed") is True,
        reward_hacking_blocking=bool(blocking),
        round_limit_strategy=round_limit_rate > 0.10,
        stochastic_collapse=(
            not isinstance(stochastic_gate, Mapping) or stochastic_gate.get("passed") is not True
        ),
    )
    by_opponent = heldout.get("by_opponent")
    opponent_scores = (
        [float(value["score"]) for value in by_opponent.values() if isinstance(value, Mapping)]
        if isinstance(by_opponent, Mapping)
        else []
    )
    style_value = float(heldout_style.get("value", 0.0))
    result = {
        "format": "s5c-a3-pilot-r2-composite-v1",
        "evaluation_status": "evaluated",
        "variant": variant,
        "profile": profile,
        "unit": unit,
        "gate": composite_gate_r2(evidence, previous_passed=False),
        "heldout": heldout,
        "heldout_style": heldout_style,
        "heldout_score": heldout_score,
        "opponent_variance": (
            statistics.pvariance(opponent_scores) if len(opponent_scores) > 1 else 0.0
        ),
        "round_limit_rate": round_limit_rate,
        "style_margin": style_value - 0.60,
        "reward_hacking": hacking,
        "blocking_reward_hacking_warnings": blocking,
        "stochastic": stochastic,
    }
    _atomic_json(directory / "composite.json", result)
    return result, actions


def _finalize_stability(checkpoints: list[dict[str, object]]) -> int | None:
    previous = False
    gate_unit: int | None = None
    for checkpoint in checkpoints:
        gate = checkpoint.get("gate")
        if not isinstance(gate, dict):
            previous = False
            continue
        evidence_value = gate.get("evidence")
        if not isinstance(evidence_value, Mapping):
            previous = False
            continue
        evidence = R2CompositeEvidence(**evidence_value)  # type: ignore[arg-type]
        corrected = composite_gate_r2(evidence, previous_passed=previous)
        checkpoint["gate"] = corrected
        previous = corrected["thresholds_passed"] is True
        if corrected["passed"] is True and gate_unit is None:
            gate_unit = int(checkpoint["unit"])
    return gate_unit


def _retrospective_variant(
    config: A3Config,
    *,
    variant: str,
    start_unit: int,
    heldout_ids: tuple[str, ...],
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    root = config.output_directory
    source = root / "pilot" / variant
    destination = root / "pilot-r2" / "retrospective" / variant
    profile = "scavenger"
    reward = a3_rewards()[variant]
    units = list(range(start_unit, 198, 5))
    if units[-1] != 197:
        units.append(197)
    screenings: dict[int, dict[str, object]] = {}
    elementary_by_unit: dict[int, dict[str, object]] = {}
    actions: list[dict[str, object]] = []
    for unit in units:
        if progress is not None:
            progress(f"S5c-A3 PILOT-R2 retrospective {variant}: checkpoint {unit}/197")
        model, environment, checkpoint_hash = _load_model(source, unit=unit)
        try:
            elementary, elementary_action = _elementary_for_checkpoint(
                config=config,
                variant=variant,
                profile=profile,
                unit=unit,
                model=model,
                checkpoint_sha256=checkpoint_hash,
                reward=reward,
                destination=destination / f"unit-{unit}" / "elementary",
                progress=progress,
            )
            elementary_by_unit[unit] = elementary
            screening, screening_action = _strength(
                path=destination / f"unit-{unit}" / "heldout-screening.json",
                model=model,
                config=config,
                heldout_ids=heldout_ids,
                variant=variant,
                unit=unit,
                confrontations=_SCREENING_CONFRONTATIONS,
                label="screening",
                progress=progress,
            )
            screenings[unit] = screening
            actions.append(
                {"unit": unit, "elementary": elementary_action, "screening": screening_action}
            )
        finally:
            environment.close()  # type: ignore[union-attr]
    full_units: set[int] = set()
    for unit, screening in screenings.items():
        global_value = screening.get("global")
        score = float(global_value["score"]) if isinstance(global_value, Mapping) else -1.0
        if _NEAR_BAND[0] <= score <= _NEAR_BAND[1]:
            full_units.add(unit)
            index = units.index(unit)
            if index + 1 < len(units):
                full_units.add(units[index + 1])
    complete: list[dict[str, object]] = []
    for unit in units:
        screening = screenings[unit]
        global_value = screening.get("global")
        if unit not in full_units:
            complete.append(
                {
                    "unit": unit,
                    "evaluation_status": "screening_only",
                    "screening_score": (
                        float(global_value["score"]) if isinstance(global_value, Mapping) else None
                    ),
                    "gate": None,
                    "heldout_score": None,
                    "opponent_variance": None,
                    "round_limit_rate": None,
                    "style_margin": None,
                }
            )
            continue
        model, environment, _ = _load_model(source, unit=unit)
        try:
            evaluated, new_actions = _full_evaluation(
                config=config,
                variant=variant,
                profile=profile,
                unit=unit,
                model=model,
                elementary=elementary_by_unit[unit],
                heldout_ids=heldout_ids,
                directory=destination / f"unit-{unit}",
                progress=progress,
            )
            complete.append(evaluated)
            actions.append({"unit": unit, "full": new_actions})
        finally:
            environment.close()  # type: ignore[union-attr]
    gate_unit = _finalize_stability(complete)
    for checkpoint in complete:
        if checkpoint.get("evaluation_status") == "evaluated":
            _atomic_json(destination / f"unit-{checkpoint['unit']}" / "composite.json", checkpoint)
    result = {
        "variant": variant,
        "profile": profile,
        "status": "eligible" if gate_unit is not None else "not_eligible",
        "eligible": gate_unit is not None,
        "gate_unit": gate_unit,
        "evaluation_status": "evaluated",
        "checkpoints": complete,
        "calculations": actions,
        "training_reused": True,
        "training_executed": False,
    }
    curve = []
    for checkpoint in complete:
        unit = int(checkpoint["unit"])
        elementary = elementary_by_unit[unit]
        curve.append(
            {
                "unit": unit,
                "elementary_scores": elementary.get("scores"),
                "historical_gate": elementary.get("gate"),
                "evaluation_status": checkpoint.get("evaluation_status"),
                "screening_score": checkpoint.get("screening_score"),
                "heldout_score": checkpoint.get("heldout_score"),
                "heldout_style": checkpoint.get("heldout_style"),
                "composite_gate": checkpoint.get("gate"),
            }
        )
    _atomic_json(destination / "curves.json", curve)
    _atomic_json(destination / "result.json", result)
    return result


def _r2_schedule(
    state: R2TrainingState, training_ids: tuple[str, ...], heldout_ids: tuple[str, ...]
) -> R2CurriculumSchedule:
    robustification = (
        0
        if state.bootstrap_completed_unit is None
        else max(1, state.completed_units - state.bootstrap_completed_unit + 1)
    )
    return R2CurriculumSchedule(state.phase, robustification, training_ids, heldout_ids)


def _open_training_branch(
    config: A3Config,
    *,
    variant: str,
    profile: str,
    pool_document: Mapping[str, object],
    training_ids: tuple[str, ...],
    heldout_ids: tuple[str, ...],
    resume: bool,
) -> _TrainingBranch:
    root = config.output_directory
    directory = root / "pilot-r2" / "training" / variant
    state_path = directory / "state.json"
    source_variant = variant if variant in _SOURCE_VARIANTS else None
    source_unit = 100 if variant in _PREDATORS else 0
    reward_key = variant if variant in a3_rewards() else "scavenger-a3-curriculum"
    reward = a3_rewards()[reward_key]
    cap, once = _reward_semantics(variant, config)
    if state_path.is_file():
        if not resume:
            raise A3ProtocolError(f"existing PILOT-R2 branch requires --resume: {variant}")
        state = R2TrainingState.from_mapping(_read_json(state_path, {}))
    else:
        state = R2TrainingState(variant=variant, profile=profile, completed_units=source_unit)
        directory.mkdir(parents=True, exist_ok=True)
        if source_variant is not None:
            source = root / "pilot" / source_variant
            source_checkpoint, _ = _checkpoint(source, source_unit)
            destination = directory / "recovery" / "units" / f"unit-{source_unit}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copytree(source_checkpoint, destination)
            source_cycles = _read_json(source / "cycles.json", [])
            if not isinstance(source_cycles, list) or len(source_cycles) < source_unit:
                raise A3ProtocolError("source training curve is incomplete")
            _atomic_json(directory / "cycles.json", source_cycles[:source_unit])
            state.last_evaluation_unit = source_unit
        _atomic_json(state_path, state.mapping())
    schedule = _r2_schedule(state, training_ids, heldout_ids)
    pool: TrainingOpponentPool = A3R2CurriculumOpponentPool(pool_document, schedule=schedule)
    vector = VectorEnvironmentConfig(VectorBackend.DUMMY, 2, config.pilot_seed)

    def environment_factory(configuration: VectorEnvironmentConfig):  # type: ignore[no-untyped-def]
        return create_vector_environment(
            configuration,
            training_opponent_pool=pool,
            reward_program=reward,
            safe_feed_episode_cap=cap,
        )

    recovery = AtomicRecoveryStore(directory / "recovery")
    latest = recovery.load_latest()
    cycles_value = _read_json(directory / "cycles.json", [])
    if not isinstance(cycles_value, list):
        raise A3ProtocolError("PILOT-R2 cycles are invalid")
    cycles = [dict(item) for item in cycles_value if isinstance(item, Mapping)]
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
        metadata = {
            "phase": "A3-PILOT-R2",
            "variant": variant,
            "profile": profile,
            "architecture": config.architecture,
            "seed": config.pilot_seed,
            "semantic_reward_sha256": reward_semantics_sha256(
                reward, safe_feed_cap=cap, once_per_opportunity=once
            ),
            "safe_feed_episode_cap": cap,
            "training_split_sha256": _TRAINING_SPLIT_SHA256,
            "heldout_split_sha256": _HELDOUT_SPLIT_SHA256,
        }
        run_id = f"s5c-a3-pilot-r2-{variant}-s20"
        ledger = UnitLedger(directory / "ledger.sqlite3")
        ledger.reserve(run_id=run_id, units=_MAX_UNITS, idempotency_key=f"{run_id}-reserve")
        runner = AtomicPPOUnitRunner(
            run_id=run_id,
            configuration=vector,
            model=model,
            environment=environment,
            recovery_store=recovery,
            ledger=ledger,
            recovery_metadata=metadata,
        )
        checkpoint_hash = None
    else:
        source_state = latest.state
        run_id = str(source_state.get("run_id"))
        metadata_value = source_state.get("metadata")
        if not isinstance(metadata_value, Mapping):
            raise A3ProtocolError("PILOT-R2 recovery metadata are absent")
        ledger = UnitLedger(directory / "ledger.sqlite3")
        ledger.reserve(run_id=run_id, units=_MAX_UNITS, idempotency_key=f"{run_id}-reserve")
        runner, _, latest = AtomicPPOUnitRunner.restore_latest(
            run_id=run_id,
            configuration=vector,
            recovery_store=recovery,
            ledger=ledger,
            environment_factory=environment_factory,
            recovery_metadata=metadata_value,
        )
        state.completed_units = latest.sequence
        checkpoint_hash = latest.files.get("model.zip")
    return _TrainingBranch(state, directory, reward, pool, runner, cycles, checkpoint_hash, cap)


def _train_one(
    branch: _TrainingBranch, *, training: tuple[str, ...], heldout: tuple[str, ...]
) -> None:
    if isinstance(branch.pool, A3R2CurriculumOpponentPool):
        branch.pool.set_schedule(_r2_schedule(branch.state, training, heldout))
        before = branch.pool.actual_counts
        theoretical = branch.pool.effective_weights
    else:
        before = {"random": 0, "training_pool": 0}
        theoretical = {"random": 1.0, "training_pool": 0.0}
    unit = branch.state.completed_units + 1
    trained = branch.runner.run_unit(unit_id=f"unit-{unit}", sequence=unit)
    after = (
        branch.pool.actual_counts if isinstance(branch.pool, A3R2CurriculumOpponentPool) else before
    )
    actual = {name: after[name] - before[name] for name in before}
    branch.cycles.append(
        {
            "unit": unit,
            "training_seconds": trained.training_seconds,
            "checkpoint_seconds": trained.checkpoint_seconds,
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
    )
    branch.state.completed_units = unit
    branch.checkpoint_sha256 = trained.recovery.files.get("model.zip")
    _atomic_json(branch.directory / "cycles.json", branch.cycles)
    _atomic_json(branch.directory / "state.json", branch.state.mapping())


def _evaluate_training_checkpoint(
    branch: _TrainingBranch,
    *,
    config: A3Config,
    heldout_ids: tuple[str, ...],
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    state = branch.state
    unit = state.completed_units
    evaluation_dir = branch.directory / "evaluations" / f"unit-{unit}"
    elementary, elementary_action = _elementary_for_checkpoint(
        config=config,
        variant=state.variant,
        profile=state.profile,
        unit=unit,
        model=branch.runner.model,
        checkpoint_sha256=str(branch.checkpoint_sha256),
        reward=branch.reward,
        destination=evaluation_dir / "elementary",
        progress=progress,
    )
    historical, scores = _scores(elementary)
    random_score = float(scores[RANDOM_LEGAL_CONTROLLER_ID])
    state.last_evaluation_unit = unit
    if state.phase == "bootstrap":
        state.bootstrap_consecutive = state.bootstrap_consecutive + 1 if random_score >= 0.90 else 0
        if state.bootstrap_consecutive >= 2:
            state.phase = "robustify"
            state.bootstrap_completed_unit = unit
    checkpoint: dict[str, object] = {
        "unit": unit,
        "evaluation_status": "elementary_only",
        "elementary_action": elementary_action,
        "historical_gate_passed": historical,
        "random_score": random_score,
        **{key: value for key, value in not_evaluated_metrics().items() if key != "gate_unit"},
    }
    if state.phase == "robustify":
        screening, action = _strength(
            path=evaluation_dir / "heldout-screening.json",
            model=branch.runner.model,
            config=config,
            heldout_ids=heldout_ids,
            variant=state.variant,
            unit=unit,
            confrontations=_SCREENING_CONFRONTATIONS,
            label="screening",
            progress=progress,
        )
        global_value = screening.get("global")
        screening_score = (
            float(global_value["score"]) if isinstance(global_value, Mapping) else -1.0
        )
        checkpoint.update(
            evaluation_status="screening_only",
            screening_action=action,
            screening_score=screening_score,
        )
        near = _NEAR_BAND[0] <= screening_score <= _NEAR_BAND[1]
        if near or state.force_full_next:
            full, actions = _full_evaluation(
                config=config,
                variant=state.variant,
                profile=state.profile,
                unit=unit,
                model=branch.runner.model,
                elementary=elementary,
                heldout_ids=heldout_ids,
                directory=evaluation_dir,
                progress=progress,
            )
            gate_value = full.get("gate")
            evidence = gate_value.get("evidence") if isinstance(gate_value, Mapping) else None
            if isinstance(evidence, Mapping):
                corrected = composite_gate_r2(
                    R2CompositeEvidence(**evidence),  # type: ignore[arg-type]
                    previous_passed=state.previous_composite_passed,
                )
                full["gate"] = corrected
                state.previous_composite_passed = corrected["thresholds_passed"] is True
                if corrected["passed"] is True:
                    state.gate_unit = unit
                    state.status = "eligible"
            full["calculation_actions"] = actions
            checkpoint = full
            _atomic_json(evaluation_dir / "composite.json", full)
        else:
            state.previous_composite_passed = False
        state.force_full_next = near
    _atomic_json(branch.directory / "state.json", state.mapping())
    return checkpoint


def _run_training_branch(
    config: A3Config,
    *,
    variant: str,
    profile: str,
    pool_document: Mapping[str, object],
    training_ids: tuple[str, ...],
    heldout_ids: tuple[str, ...],
    resume: bool,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    branch = _open_training_branch(
        config,
        variant=variant,
        profile=profile,
        pool_document=pool_document,
        training_ids=training_ids,
        heldout_ids=heldout_ids,
        resume=resume,
    )
    checkpoints_value = _read_json(branch.directory / "checkpoints.json", [])
    checkpoints = (
        [dict(item) for item in checkpoints_value if isinstance(item, Mapping)]
        if isinstance(checkpoints_value, list)
        else []
    )
    try:
        while branch.state.status == "training" and branch.state.completed_units < _MAX_UNITS:
            _train_one(branch, training=training_ids, heldout=heldout_ids)
            if progress is not None:
                progress(
                    f"S5c-A3 PILOT-R2 {variant}: {branch.state.completed_units}/{_MAX_UNITS} "
                    f"units ({100 * branch.state.completed_units / _MAX_UNITS:.1f}%) "
                    f"phase={branch.state.phase}"
                )
            unit = branch.state.completed_units
            should_evaluate = unit % 5 == 0 or unit in {_BOOTSTRAP_MAX_UNITS, _MAX_UNITS}
            if should_evaluate:
                checkpoints.append(
                    _evaluate_training_checkpoint(
                        branch, config=config, heldout_ids=heldout_ids, progress=progress
                    )
                )
                _atomic_json(branch.directory / "checkpoints.json", checkpoints)
            if (
                branch.state.phase == "bootstrap"
                and branch.state.completed_units >= _BOOTSTRAP_MAX_UNITS
            ):
                branch.state.status = "failed_bootstrap"
            elif branch.state.completed_units >= _MAX_UNITS and branch.state.status == "training":
                branch.state.status = "budget_exhausted"
            _atomic_json(branch.directory / "state.json", branch.state.mapping())
    finally:
        branch.runner.environment.close()
        close = getattr(branch.pool, "close", None)
        if callable(close):
            close()
    result = {
        "variant": variant,
        "profile": profile,
        "status": branch.state.status,
        "eligible": branch.state.status == "eligible",
        "gate_unit": branch.state.gate_unit,
        "evaluation_status": "evaluated" if checkpoints else "not_evaluated",
        "completed_units": branch.state.completed_units,
        "bootstrap_completed_unit": branch.state.bootstrap_completed_unit,
        "checkpoints": checkpoints,
        "training_reused": variant in _PREDATORS,
        "training_executed": True,
    }
    _atomic_json(
        branch.directory / "curves.json",
        {"training": branch.cycles, "evaluations": checkpoints},
    )
    _atomic_json(branch.directory / "result.json", result)
    return result


def _best(profile: str, variants: list[dict[str, object]]) -> dict[str, object] | None:
    candidates = [
        item for item in variants if item.get("profile") == profile and item.get("eligible") is True
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (int(item["gate_unit"]), str(item["variant"])))


def _calculation_inventory(variants: list[dict[str, object]]) -> dict[str, list[str]]:
    reused: set[str] = set()
    executed: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, str):
            if value.startswith("reused_"):
                reused.add(value)
            elif value.startswith("new_"):
                executed.add(value)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for variant in variants:
        visit(variant.get("calculations"))
        visit(variant.get("checkpoints"))
    return {"reused": sorted(reused), "newly_executed": sorted(executed)}


def _report(
    config: A3Config,
    *,
    preflight: Mapping[str, object],
    variants: list[dict[str, object]],
) -> dict[str, object]:
    selected = {profile: _best(profile, variants) for profile in ("scavenger", "predator")}
    proposable = all(value is not None for value in selected.values())
    report = {
        "format": "s5c-a3-pilot-r2-report-v1",
        "decision": "ELIGIBLE_VARIANTS_SELECTED" if proposable else "NO_ELIGIBLE_VARIANT",
        "a3_confirm_proposable": proposable,
        "anomalies_corrected": [
            "numeric_sentinels_replaced_by_null",
            "explicit_not_evaluated_status",
            "scavenger_a2_composite_reconstructed",
            "bootstrap_limit_raised_to_147",
            "reward_hash_covers_dsl_caps_counters_and_runtime",
            "bounded_repeatability_warning_is_not_automatically_blocking",
            "heldout_style_is_gate_evidence",
        ],
        "preflight": dict(preflight),
        "variants": variants,
        "selected": {
            profile: (
                None
                if value is None
                else {"variant": value["variant"], "gate_unit": value["gate_unit"]}
            )
            for profile, value in selected.items()
        },
        "training_avoided": [
            str(item["variant"]) for item in variants if item.get("training_executed") is False
        ],
        "runs_resumed": [
            str(item["variant"])
            for item in variants
            if item.get("training_reused") is True and item.get("training_executed") is True
        ],
        "calculation_inventory": _calculation_inventory(variants),
        "forbidden_phases_started": [],
    }
    report_dir = config.output_directory / "report"
    json_path = report_dir / "rl-s5c-a3-pilot-r2-report.json"
    markdown_path = report_dir / "rl-s5c-a3-pilot-r2-report.md"
    _atomic_json(json_path, report)
    lines = [
        "# RL-S5c-A3 — rapport PILOT-R2",
        "",
        f"Décision : `{report['decision']}`",
        f"A3-CONFIRM proposable : `{'oui' if proposable else 'non'}`",
        "",
        "## Variantes",
        "",
    ]
    for item in variants:
        gate = "null" if item.get("gate_unit") is None else str(item["gate_unit"])
        lines.append(
            f"- `{item['variant']}` : statut `{item['status']}`, "
            f"éligible `{item['eligible']}`, unité gate `{gate}`, "
            f"évaluation `{item['evaluation_status']}`."
        )
    lines.extend(["", "## Sélection", ""])
    for profile, value in selected.items():
        lines.append(
            f"- `{profile}` : "
            + ("aucune variante éligible" if value is None else f"`{value['variant']}`")
        )
    lines.extend(
        [
            "",
            "## Exécution",
            "",
            f"- Entraînements évités : `{report['training_avoided']}`",
            f"- Runs repris : `{report['runs_resumed']}`",
            "- A3-CONFIRM et RL-S5c-B lancés : `non`",
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    _atomic_json(
        config.output_directory / "pilot-r2" / "selection.json",
        {
            "format": "s5c-a3-pilot-r2-selection-v1",
            "decision": report["decision"],
            "a3_confirm_proposable": proposable,
            "selected": report["selected"],
        },
    )
    return report


def run_pilot_r2(
    config: A3Config,
    *,
    resume: bool,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Audit old checkpoints, resume only artificial stops, and report R2 eligibility."""

    report_path = config.output_directory / "report" / "rl-s5c-a3-pilot-r2-report.json"
    if report_path.is_file():
        if not resume:
            raise A3ProtocolError("completed PILOT-R2 requires --resume for idempotent read")
        value = _read_json(report_path, {})
        if not isinstance(value, dict):
            raise A3ProtocolError("PILOT-R2 report is invalid")
        return value
    preflight = _preflight(config)
    if preflight.get("status") != "PASSED":
        invalid = []
        provenance = preflight.get("reward_provenance")
        if isinstance(provenance, Mapping):
            invalid = [
                {
                    "variant": name,
                    "profile": str(name).split("-", 1)[0],
                    "status": "invalid_reward_provenance",
                    "eligible": False,
                    **not_evaluated_metrics(),
                }
                for name, value in provenance.items()
                if isinstance(value, Mapping) and value.get("status") != "valid"
            ]
        return _report(config, preflight=preflight, variants=invalid)
    manifest = _read_json(config.output_directory / "manifest.json", {})
    assert isinstance(manifest, Mapping)
    training_ids, heldout_ids = _split(manifest)
    pool = verify_strong_pool(config.rl_s5b_pool, expected_sha256=config.rl_s5b_pool_sha256)
    variants = [
        _retrospective_variant(
            config,
            variant="scavenger-a2-control",
            start_unit=90,
            heldout_ids=heldout_ids,
            progress=progress,
        ),
        _retrospective_variant(
            config,
            variant="scavenger-a3-curriculum",
            start_unit=80,
            heldout_ids=heldout_ids,
            progress=progress,
        ),
    ]
    for variant in _PREDATORS:
        variants.append(
            _run_training_branch(
                config,
                variant=variant,
                profile="predator",
                pool_document=pool,
                training_ids=training_ids,
                heldout_ids=heldout_ids,
                resume=resume,
                progress=progress,
            )
        )
    if not any(item.get("eligible") is True for item in variants if item["profile"] == "scavenger"):
        variants.append(
            _run_training_branch(
                config,
                variant=_SOFT_SCAVENGER,
                profile="scavenger",
                pool_document=pool,
                training_ids=training_ids,
                heldout_ids=heldout_ids,
                resume=resume,
                progress=progress,
            )
        )
    report = _report(config, preflight=preflight, variants=variants)
    return {
        "decision": report["decision"],
        "a3_confirm_proposable": report["a3_confirm_proposable"],
        "report": str(report_path),
        "markdown": str(report_path.with_suffix(".md")),
    }
