"""Safe, immutable RL-L8 snapshot publication primitives.

Only an in-memory trusted PPO policy can be exported.  Published snapshots are
strictly safetensors plus JSON; the loader never accepts an SB3 archive,
pickle, or an architecture selected by a snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from safetensors.torch import load_file, save
from sb3_contrib import MaskablePPO

from dinorl_engine.core import actions, constants, engine, events, maps, state
from dinorl_engine.core.constants import ACTION_VERSION, ENGINE_VERSION, MAP_ID, RULES_VERSION
from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.evaluation.gates import deterministic_gate, publication_gate
from dinorl_engine.rl.policies.local_mlp_v2 import (
    DinoRLCompactMLPV2FeaturesExtractor,
    LocalMLPV2Architecture,
)
from dinorl_engine.rl.training.unit import create_maskable_ppo
from dinorl_engine.rl.versions import OBSERVATION_VERSION, SNAPSHOT_VERSION

__all__ = [
    "SnapshotError",
    "assess_s6_publication",
    "load_published_policy",
    "publish_policy_snapshot",
    "replay_policy_actions",
]

_IDENTIFIER: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_HASH: Final = re.compile(r"[a-f0-9]{64}")
_SNAPSHOT_FILES: Final = frozenset(
    {"weights.safetensors", "manifest.json", "evaluation.json", "sha256sums.txt"}
)
_COMPACT_DIMENSIONS: Final[dict[str, object]] = {
    "encoder": [663, 64, 64],
    "action_count": 9,
    "policy_latent": 64,
    "value_latent": 64,
}
_CONTRACTS: Final[dict[str, str]] = {
    "engine_version": ENGINE_VERSION,
    "rules_version": RULES_VERSION,
    "map_id": MAP_ID,
    "action_version": ACTION_VERSION,
    "observation_version": OBSERVATION_VERSION,
}


class SnapshotError(RuntimeError):
    """A published snapshot violates a closed RL-L8 contract."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_path(module: object) -> Path:
    path = getattr(module, "__file__", None)
    if not isinstance(path, str):
        raise SnapshotError("cannot resolve versioned runtime source")
    return Path(path)


def _hash_paths(paths: Sequence[Path]) -> str:
    return _sha256(
        _canonical_json_bytes(
            {path.name: _file_sha256(path) for path in sorted(paths, key=lambda item: str(item))}
        )
    )


def _contracts_with_hashes() -> dict[str, str]:
    """Bind a snapshot to the exact engine, rules, map, action and observation inputs."""

    map_path = _module_path(maps).parents[3] / "fixtures" / "maps" / f"{MAP_ID}.json"
    observation_path = Path(__file__).parents[1] / "env" / "observation.py"
    canonical_path = Path(__file__).parents[1] / "env" / "canonical.py"
    return {
        **_CONTRACTS,
        "engine_sha256": _hash_paths([_module_path(engine)]),
        "rules_sha256": _hash_paths(
            [_module_path(constants), _module_path(events), _module_path(state)]
        ),
        "map_sha256": _file_sha256(map_path),
        "action_sha256": _hash_paths([_module_path(actions)]),
        "observation_sha256": _hash_paths([observation_path, canonical_path]),
    }


def _atomic_write(path: Path, content: bytes) -> None:
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


def _require_scores(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SnapshotError(f"{field} must be an opponent score mapping")
    return value


def _seed_evidence(value: Mapping[str, object]) -> dict[str, object]:
    seed = value.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise SnapshotError("seed must be an unsigned 32-bit integer")
    deterministic = value.get("deterministic_evaluations")
    if not isinstance(deterministic, list) or len(deterministic) != 2:
        raise SnapshotError("exactly two deterministic evaluations are required")
    first = deterministic_gate(
        _require_scores(deterministic[0], "first deterministic"), previous_passed=False
    )
    second = deterministic_gate(
        _require_scores(deterministic[1], "second deterministic"),
        previous_passed=bool(first["thresholds_passed"]),
    )
    if value.get("stochastic_evaluation") is None:
        if second["passed"] is True:
            raise SnapshotError("a deterministic candidate requires stochastic evaluation")
        return {
            "seed": seed,
            "deterministic": [first, second],
            "stochastic": None,
            "stochastic_extension_required": False,
            "stochastic_extension": None,
            "evaluation_status": "not_evaluated",
            "conforming": False,
        }
    stochastic_scores = _require_scores(value.get("stochastic_evaluation"), "stochastic evaluation")
    stochastic = publication_gate(
        _require_scores(deterministic[1], "second deterministic"), stochastic_scores
    )
    extension_value = value.get("stochastic_extension")
    extension_required = bool(stochastic["near_boundary"])
    extension: dict[str, object] | None = None
    if extension_required:
        if extension_value is None:
            raise SnapshotError("stochastic extension is required near a publication boundary")
        extension = publication_gate(
            _require_scores(deterministic[1], "second deterministic"),
            _require_scores(extension_value, "stochastic extension"),
        )
    elif extension_value is not None:
        raise SnapshotError("stochastic extension is only allowed near a publication boundary")
    passed = (
        bool(second["passed"])
        and bool(stochastic["passed"])
        and (extension is None or bool(extension["passed"]))
    )
    return {
        "seed": seed,
        "deterministic": [first, second],
        "stochastic": stochastic,
        "stochastic_extension_required": extension_required,
        "stochastic_extension": extension,
        "evaluation_status": "evaluated",
        "conforming": passed,
    }


def assess_s6_publication(seed_evaluations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Apply the frozen two-pass/stochastic gate to the five RL-S6 seeds."""

    results = [_seed_evidence(value) for value in seed_evaluations]
    seeds = [result["seed"] for result in results]
    if len(results) != 5 or len(set(seeds)) != 5:
        raise SnapshotError("RL-S6 publication requires evidence from exactly five distinct seeds")
    results.sort(key=lambda result: cast(int, result["seed"]))
    conforming = sum(bool(result["conforming"]) for result in results)
    return {
        "format": "rl-s6-publication-evaluation-v1",
        "seed_count": 5,
        "required_conforming_seed_count": 4,
        "conforming_seed_count": conforming,
        "eligible": conforming >= 4,
        "seeds": results,
    }


def _valid_identifier(value: str, field: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise SnapshotError(f"{field} must be an opaque identifier")


def _validated_evaluation(value: Mapping[str, object]) -> dict[str, object]:
    if value.get("format") != "rl-s6-publication-evaluation-v1":
        raise SnapshotError("evaluation has an unsupported format")
    if value.get("eligible") is not True:
        raise SnapshotError("publication requires four conforming RL-S6 seeds")
    seeds = value.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 5:
        raise SnapshotError("evaluation lacks five seed results")
    # The publication endpoint is server-only: this verifies the frozen result
    # envelope before it becomes an immutable artifact.
    if value.get("conforming_seed_count") not in (4, 5):
        raise SnapshotError("evaluation conforming seed count is invalid")
    return dict(value)


def _policy_weights(policy: MaskablePPO) -> bytes:
    tensors = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in sorted(policy.policy.state_dict().items())
    }
    return save(
        tensors, metadata={"format": "dinorl-policy-snapshot-v1", "architecture": "mlp-compact-v2"}
    )


def _manifest(
    *,
    snapshot_id: str,
    owner_id: str,
    source_checkpoint_id: str,
    evaluation_id: str,
    created_at: str,
    weights_hash: str,
    evaluation_hash: str,
) -> dict[str, object]:
    _valid_identifier(snapshot_id, "snapshot_id")
    _valid_identifier(owner_id, "owner_id")
    _valid_identifier(source_checkpoint_id, "source_checkpoint_id")
    _valid_identifier(evaluation_id, "evaluation_id")
    if not created_at.endswith("Z"):
        raise SnapshotError("created_at must be an explicit UTC timestamp")
    return {
        "schema_version": "1.0.0",
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_id": snapshot_id,
        "owner_id": owner_id,
        "source_checkpoint_id": source_checkpoint_id,
        "parent_snapshot_id": None,
        "architecture_id": "mlp-compact-v2",
        "architecture_dimensions": _COMPACT_DIMENSIONS,
        "contracts": _contracts_with_hashes(),
        "torch_version": torch.__version__,
        "exporter_version": "rl-l8-v1",
        "publication_status": "published",
        "evaluation_id": evaluation_id,
        "files": {"weights.safetensors": weights_hash, "evaluation.json": evaluation_hash},
        "created_at": created_at,
    }


def _registry_path(destination: Path) -> Path:
    return destination.parent / "registry.json"


def _update_registry(
    destination: Path, manifest_bytes: bytes, manifest: Mapping[str, object]
) -> None:
    path = _registry_path(destination)
    if path.exists():
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SnapshotError("snapshot registry is unreadable") from error
        if not isinstance(value, dict) or value.get("format") != "rl-snapshot-registry-v1":
            raise SnapshotError("snapshot registry has an unsupported format")
        raw_entries = value.get("entries")
        if not isinstance(raw_entries, list):
            raise SnapshotError("snapshot registry entries are invalid")
        entries = list(raw_entries)
    else:
        entries = []
    entry = {
        "snapshot_id": manifest["snapshot_id"],
        "manifest_sha256": _sha256(manifest_bytes),
        "relative_location": destination.name,
    }
    matches = [
        item
        for item in entries
        if isinstance(item, Mapping) and item.get("snapshot_id") == entry["snapshot_id"]
    ]
    if matches and matches != [entry]:
        raise SnapshotError("immutable registry entry differs")
    if not matches:
        entries.append(entry)
        entries.sort(key=lambda item: str(item["snapshot_id"]))
        _atomic_write(
            path, _canonical_json_bytes({"format": "rl-snapshot-registry-v1", "entries": entries})
        )


def publish_policy_snapshot(
    *,
    policy: MaskablePPO,
    destination: Path,
    snapshot_id: str,
    owner_id: str,
    source_checkpoint_id: str,
    evaluation: Mapping[str, object],
    evaluation_id: str,
    created_at: str,
) -> dict[str, object]:
    """Export a server-trusted compact policy as an immutable safe snapshot."""

    if not isinstance(policy, MaskablePPO):
        raise SnapshotError("only an in-memory trusted MaskablePPO policy may be exported")
    if not isinstance(policy.policy.features_extractor, DinoRLCompactMLPV2FeaturesExtractor):
        raise SnapshotError("only mlp-compact-v2 may be published in V1")
    if destination.name != snapshot_id:
        raise SnapshotError("snapshot destination must use its opaque snapshot_id")
    validated_evaluation = _validated_evaluation(evaluation)
    weights = _policy_weights(policy)
    evaluation_bytes = _canonical_json_bytes(validated_evaluation)
    manifest = _manifest(
        snapshot_id=snapshot_id,
        owner_id=owner_id,
        source_checkpoint_id=source_checkpoint_id,
        evaluation_id=evaluation_id,
        created_at=created_at,
        weights_hash=_sha256(weights),
        evaluation_hash=_sha256(evaluation_bytes),
    )
    manifest_bytes = _canonical_json_bytes(manifest)
    checksums = (
        f"{_sha256(weights)}  weights.safetensors\n"
        f"{_sha256(evaluation_bytes)}  evaluation.json\n"
        f"{_sha256(manifest_bytes)}  manifest.json\n"
    ).encode("ascii")
    expected = {
        destination / "weights.safetensors": weights,
        destination / "evaluation.json": evaluation_bytes,
        destination / "manifest.json": manifest_bytes,
        destination / "sha256sums.txt": checksums,
    }
    if destination.exists():
        raise SnapshotError("immutable snapshot already exists; publish with a new snapshot_id")
    else:
        for path, content in expected.items():
            _atomic_write(path, content)
    _update_registry(destination, manifest_bytes, manifest)
    return manifest


def _read_manifest(destination: Path) -> dict[str, object]:
    files = {path.name for path in destination.iterdir()} if destination.is_dir() else set()
    if files != _SNAPSHOT_FILES:
        raise SnapshotError("unexpected_file_in_snapshot")
    try:
        value: object = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotError("invalid_manifest") from error
    if not isinstance(value, dict):
        raise SnapshotError("invalid_manifest")
    if value.get("snapshot_version") != SNAPSHOT_VERSION:
        raise SnapshotError("incompatible_snapshot_version")
    if value.get("architecture_id") != "mlp-compact-v2":
        raise SnapshotError("unknown_architecture")
    if value.get("architecture_dimensions") != _COMPACT_DIMENSIONS:
        raise SnapshotError("incompatible_architecture_dimensions")
    if value.get("contracts") != _contracts_with_hashes():
        raise SnapshotError("incompatible_contracts")
    files_value = value.get("files")
    if not isinstance(files_value, dict) or set(files_value) != {
        "weights.safetensors",
        "evaluation.json",
    }:
        raise SnapshotError("invalid_manifest_files")
    for name, expected in files_value.items():
        if not isinstance(expected, str) or _HASH.fullmatch(expected) is None:
            raise SnapshotError("invalid_manifest_hash")
        if _file_sha256(destination / name) != expected:
            raise SnapshotError("hash_mismatch")
    sums = (destination / "sha256sums.txt").read_bytes()
    expected_sums = (
        f"{files_value['weights.safetensors']}  weights.safetensors\n"
        f"{files_value['evaluation.json']}  evaluation.json\n"
        f"{_sha256((destination / 'manifest.json').read_bytes())}  manifest.json\n"
    ).encode("ascii")
    if sums != expected_sums:
        raise SnapshotError("hash_mismatch")
    return value


def _require_registered_snapshot(destination: Path, manifest: Mapping[str, object]) -> None:
    path = _registry_path(destination)
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotError("external_import_disabled") from error
    if not isinstance(value, dict) or value.get("format") != "rl-snapshot-registry-v1":
        raise SnapshotError("external_import_disabled")
    entries = value.get("entries")
    expected = {
        "snapshot_id": manifest.get("snapshot_id"),
        "manifest_sha256": _sha256((destination / "manifest.json").read_bytes()),
        "relative_location": destination.name,
    }
    if not isinstance(entries, list) or expected not in entries:
        raise SnapshotError("external_import_disabled")


def load_published_policy(
    destination: Path, *, seed: int
) -> tuple[DinoRLSingleAgentEnv, MaskablePPO]:
    """Construct the sole supported architecture and strictly load safetensors."""

    manifest = _read_manifest(destination)
    _require_registered_snapshot(destination, manifest)
    environment = DinoRLSingleAgentEnv(seed=seed)
    model = create_maskable_ppo(environment, seed=seed, architecture=LocalMLPV2Architecture.COMPACT)
    try:
        tensors = load_file(destination / "weights.safetensors", device="cpu")
        model.policy.load_state_dict(tensors, strict=True)
    except Exception as error:
        environment.close()
        raise SnapshotError("invalid_safetensors") from error
    return environment, model


def replay_policy_actions(
    policy: MaskablePPO,
    observations: Sequence[dict[str, np.ndarray]],
    action_masks: Sequence[np.ndarray],
) -> list[int]:
    """Return a deterministic action trace for a recorded observation replay."""

    if len(observations) != len(action_masks):
        raise SnapshotError("replay observations and action masks have different lengths")
    actions: list[int] = []
    for observation, mask in zip(observations, action_masks, strict=True):
        action, _ = policy.predict(observation, deterministic=True, action_masks=mask)
        if isinstance(action, np.ndarray):
            actions.append(int(action.item()))
        else:
            actions.append(int(action))
    return actions
