"""Server-only paired re-evaluation of immutable RL-S6 v1 checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, cast

from dinorl_engine.rl.artifacts.snapshots import runtime_contract_hashes
from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.evaluation.match import run_evaluation_game
from dinorl_engine.rl.rewards.runtime import compile_reward
from dinorl_engine.rl.s6_config import S6Config
from dinorl_engine.rl.s6_evaluation_v2 import paired_gate_v2, paired_specs_v2
from dinorl_engine.rl.s6_evaluation_v2_config import S6EvaluationV2Config
from dinorl_engine.rl.training.unit import load_maskable_ppo

__all__ = ["S6V2Error", "reevaluate_v2"]

_FORMAT: Final = "rl-s6-campaign-result-v2"


class S6V2Error(RuntimeError):
    """A v2 measurement attempted to alter or use unverifiable v1 evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S6V2Error(f"cannot read JSON: {path}") from error


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


def _point(outcome: object) -> float:
    outcomes = getattr(outcome, "outcomes", None)
    wins = getattr(outcomes, "wins", None)
    draws = getattr(outcomes, "draws", None)
    if type(wins) is not int or type(draws) is not int:
        raise S6V2Error("evaluation game result is invalid")
    return 1.0 if wins else 0.5 if draws else 0.0


def _candidate(source: Path, seed: int) -> tuple[int, Path, str]:
    result = _read(source / "seeds" / f"seed-{seed}" / "result.json")
    if not isinstance(result, Mapping):
        raise S6V2Error("v1 seed result is invalid")
    candidate = result.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("seed") != seed:
        raise S6V2Error("v1 seed has no candidate checkpoint")
    unit = candidate.get("unit")
    if type(unit) is not int:
        raise S6V2Error("v1 candidate unit is invalid")
    directory = source / "seeds" / f"seed-{seed}" / "recovery" / "units" / f"unit-{unit}"
    manifest = _read(directory / "manifest.json")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("files"), Mapping):
        raise S6V2Error("candidate recovery manifest is invalid")
    checkpoint = directory / "model.zip"
    expected = manifest["files"].get("model.zip")
    actual = _sha256(checkpoint)
    if actual != expected:
        raise S6V2Error("candidate checkpoint hash mismatch")
    return unit, checkpoint, actual


def _manifest(config: S6Config, source: Path, protocol: S6EvaluationV2Config) -> dict[str, object]:
    v1 = _read(source / "campaign-result.json")
    if not isinstance(v1, Mapping) or v1.get("decision") != "FAILED_RL_S6":
        raise S6V2Error("v2 requires the immutable FAILED_RL_S6 v1 campaign")
    checkpoints = {}
    for seed in config.training_seeds:
        unit, _, digest = _candidate(source, seed)
        checkpoints[str(seed)] = {"unit": unit, "model_sha256": digest}
    return {
        "format": "rl-s6-evaluation-v2-manifest-v1",
        "source_campaign": str(source),
        "source_campaign_sha256": _sha256(source / "campaign-result.json"),
        "source_decision": "FAILED_RL_S6",
        "config_sha256": config.sha256,
        "architecture": config.architecture,
        "architecture_dimensions": list(config.architecture_dimensions),
        "reward_sha256": compile_reward(config.reward_dsl).cache_key,
        "beta_validation_pool_sha256": config.beta_validation_pool_sha256,
        "runtime_contracts": runtime_contract_hashes(),
        "checkpoints": checkpoints,
        "confrontations_per_opponent": protocol.confrontations_per_opponent,
        "positions_per_confrontation": 4,
        "bootstrap_seed": protocol.bootstrap_seed,
        "bootstrap_replicates": protocol.bootstrap_replicates,
        "protocol_files_sha256": {
            "s6_evaluation_v2": _sha256(Path(__file__).with_name("s6_evaluation_v2.py")),
            "s6_v2_runner": _sha256(Path(__file__)),
            "evaluation_match": _sha256(Path(__file__).parent / "evaluation" / "match.py"),
            "random_legal": _sha256(
                Path(__file__).parent.parent / "controllers" / "random_legal.py"
            ),
            "scripted": _sha256(Path(__file__).parent.parent / "controllers" / "scripted.py"),
        },
        "no_training": True,
    }


def _records_for_seed(
    checkpoint: Path,
    *,
    seed: int,
    extension: bool,
    protocol: S6EvaluationV2Config,
    progress: Callable[[str], None] | None,
) -> list[dict[str, object]]:
    environment = DinoRLSingleAgentEnv(seed=seed)
    model = load_maskable_ppo(checkpoint, environment)
    records: list[dict[str, object]] = []
    try:
        for opponent in ("aggressive-v1", "prudent-v1", "opportunist-v1", "random-legal-v1"):
            specifications = paired_specs_v2(
                seed=101,
                opponent_id=opponent,
                confrontations=protocol.confrontations_per_opponent,
                start_repetition=protocol.confrontations_per_opponent if extension else 0,
            )
            for index, specification in enumerate(specifications, start=1):
                deterministic = run_evaluation_game(
                    model, specification, deterministic=True, diagnostic_replay=False
                )
                stochastic = run_evaluation_game(
                    model, specification, deterministic=False, diagnostic_replay=False
                )
                records.append(
                    {
                        "seed": seed,
                        "opponent_id": opponent,
                        "pair_id": specification.pair_id,
                        "position": specification.game_id.rsplit("-", 1)[1],
                        "game_id": specification.game_id,
                        "map_seed": specification.seed,
                        "learner_actor": specification.learner_actor.name,
                        "first_actor": specification.first_actor.name,
                        "deterministic_score": _point(deterministic),
                        "stochastic_score": _point(stochastic),
                        "deterministic_metrics": deterministic.metrics,
                        "stochastic_metrics": stochastic.metrics,
                    }
                )
                if progress is not None and (index == len(specifications) or index % 100 == 0):
                    progress(
                        f"RL-S6 v2 seed {seed} {opponent}: {index}/{len(specifications)} games "
                        f"({100 * index / len(specifications):.1f}%)"
                    )
    finally:
        environment.close()
    return records


def reevaluate_v2(
    config: S6Config,
    *,
    source_directory: Path,
    output_directory: Path,
    protocol: S6EvaluationV2Config,
    dry_run: bool,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Measure v2 only; ``dry_run`` verifies inputs and writes no measurement evidence."""

    source = source_directory.resolve()
    output = output_directory.resolve()
    manifest = _manifest(config, source, protocol)
    if dry_run:
        return {"command": "s6 reevaluate-v2", "status": "dry_run", "manifest": manifest}
    if output.exists():
        raise S6V2Error("v2 output directory already exists; immutable evaluation cannot overwrite")
    _atomic_json(output / "manifest.json", manifest)
    seeds: list[dict[str, object]] = []
    for seed in config.training_seeds:
        unit, checkpoint, digest = _candidate(source, seed)
        records = _records_for_seed(
            checkpoint, seed=seed, extension=False, protocol=protocol, progress=progress
        )
        gate = paired_gate_v2(
            records,
            bootstrap_seed=protocol.bootstrap_seed + seed,
            bootstrap_replicates=protocol.bootstrap_replicates,
        )
        extension_records: list[dict[str, object]] = []
        if gate["extension_required"] is True:
            extension_records = _records_for_seed(
                checkpoint, seed=seed, extension=True, protocol=protocol, progress=progress
            )
            gate = paired_gate_v2(
                records + extension_records,
                bootstrap_seed=protocol.bootstrap_seed + seed,
                bootstrap_replicates=protocol.bootstrap_replicates,
            )
        seed_directory = output / "seeds" / f"seed-{seed}"
        _atomic_json(seed_directory / "records.json", {"records": records + extension_records})
        seed_result = {
            "seed": seed,
            "candidate_unit": unit,
            "checkpoint_sha256": digest,
            "gate": gate,
            "extension_games": len(extension_records),
        }
        _atomic_json(seed_directory / "result.json", seed_result)
        seeds.append(seed_result)
    conforming = sum(
        cast(Mapping[str, object], item["gate"]).get("decision") == "pass" for item in seeds
    )
    result: dict[str, object] = {
        "format": _FORMAT,
        "source_campaign": str(source),
        "source_decision": "FAILED_RL_S6",
        "decision": "RL_S6_V2_PASSED" if conforming >= 4 else "FAILED_RL_S6_V2",
        "conforming_seed_count": conforming,
        "required_conforming_seed_count": 4,
        "seeds": seeds,
        "no_training": True,
    }
    _atomic_json(output / "campaign-result.json", result)
    return result
