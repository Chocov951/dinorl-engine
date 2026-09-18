"""Creation and verification of immutable RL-S5b benchmark pools."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.rl.contracts import canonical_json_bytes, canonical_sha256
from dinorl_engine.rl.s5b.config import S5bConfig
from dinorl_engine.rl.versions import (
    CHECKPOINT_VERSION,
    OBSERVATION_VERSION,
    REWARD_DSL_VERSION,
)

__all__ = ["PoolError", "freeze_pool", "read_verified_pool"]

_FORMAT: Final = "s5b-benchmark-pool-v1"
_HISTORY_UNITS: Final = (32, 64, 96)
_CHECKPOINT_FILES: Final = ("manifest.json", "model.zip", "ppo_state.npz", "state.json")


class PoolError(RuntimeError):
    """Raised when an immutable pool cannot be built or verified."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compatibility() -> dict[str, object]:
    return {
        "actions": "dinorl-actions-v1",
        "checkpoint": CHECKPOINT_VERSION,
        "engine_rules": "duel-de-raptors-rules-v1",
        "observation": OBSERVATION_VERSION,
        "reward_dsl": REWARD_DSL_VERSION,
    }


def _git_commit() -> str:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise PoolError("Git is required to freeze an RL-S5b pool") from error
    commit = process.stdout.strip()
    if process.returncode != 0 or len(commit) != 40:
        raise PoolError("Git commit cannot be determined while freezing an RL-S5b pool")
    return commit


def _controller_entries() -> list[dict[str, object]]:
    identifiers = (RANDOM_LEGAL_CONTROLLER_ID, *SCRIPTED_CONTROLLER_IDS)
    return [
        {
            "architecture": None,
            "artifacts": {},
            "artifact_sha256": canonical_sha256({"controller": identifier}),
            "compatibility": _compatibility(),
            "id": identifier,
            "kind": "controller",
            "provenance": "versioned-controller-catalog",
            "seed": None,
            "status": "available",
            "unit": None,
        }
        for identifier in identifiers
    ]


def _checkpoint_entry(
    *,
    checkpoint_id: str,
    architecture: str,
    seed: int,
    unit: int,
    directory: Path,
    role: str,
) -> dict[str, object]:
    artifacts = {name: directory / name for name in _CHECKPOINT_FILES}
    entry: dict[str, object] = {
        "architecture": architecture,
        "artifacts": {},
        "compatibility": _compatibility(),
        "id": f"{checkpoint_id}-unit-{unit}-{role}",
        "kind": "checkpoint",
        "provenance": str(directory),
        "seed": seed,
        "unit": unit,
    }
    if not all(path.is_file() for path in artifacts.values()):
        return {**entry, "artifact_sha256": None, "status": "missing"}
    hashes = {name: _sha256_file(path) for name, path in artifacts.items()}
    return {
        **entry,
        "artifacts": hashes,
        "artifact_sha256": canonical_sha256({"artifacts": hashes}),
        "status": "available",
    }


def _pool_payload(config: S5bConfig) -> dict[str, object]:
    entries = _controller_entries()
    for checkpoint in config.checkpoints:
        entries.append(
            _checkpoint_entry(
                checkpoint_id=checkpoint.checkpoint_id,
                architecture=checkpoint.architecture,
                seed=checkpoint.seed,
                unit=147,
                directory=checkpoint.directory,
                role="final",
            )
        )
        for unit in _HISTORY_UNITS:
            entries.append(
                _checkpoint_entry(
                    checkpoint_id=checkpoint.checkpoint_id,
                    architecture=checkpoint.architecture,
                    seed=checkpoint.seed,
                    unit=unit,
                    directory=checkpoint.directory.parent / f"unit-{unit}",
                    role="historical",
                )
            )
    return {
        "checkpoints": [checkpoint.checkpoint_id for checkpoint in config.checkpoints],
        "compatibility": _compatibility(),
        "config_sha256": canonical_sha256(config.to_mapping()),
        "confrontations": config.confrontations,
        "entries": entries,
        "evaluation_seeds": list(config.evaluation_seeds),
        "final_architectures": list(config.final_architectures),
        "format": _FORMAT,
        "git_commit": _git_commit(),
        "run_id": config.run_id,
    }


def _with_digest(payload: Mapping[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["pool_sha256"] = canonical_sha256(document)
    return document


def _validate_pool_shape(pool: object) -> dict[str, object]:
    if not isinstance(pool, dict) or set(pool) != {
        "checkpoints",
        "compatibility",
        "config_sha256",
        "confrontations",
        "entries",
        "evaluation_seeds",
        "final_architectures",
        "format",
        "git_commit",
        "pool_sha256",
        "run_id",
    }:
        raise PoolError("pool manifest has missing or unexpected fields")
    if pool.get("format") != _FORMAT:
        raise PoolError("pool manifest format is unsupported")
    supplied_digest = pool["pool_sha256"]
    without_digest = {key: value for key, value in pool.items() if key != "pool_sha256"}
    if not isinstance(supplied_digest, str) or supplied_digest != canonical_sha256(without_digest):
        raise PoolError("pool manifest SHA-256 is invalid")
    entries = pool.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PoolError("pool manifest has no entries")
    return pool


def read_verified_pool(path: Path, *, verify_artifacts: bool = True) -> dict[str, object]:
    """Read a pool and reject any manifest or policy-artifact mutation."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PoolError(f"cannot read pool manifest: {path}") from error
    pool = _validate_pool_shape(raw)
    entries = pool["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "architecture",
            "artifacts",
            "artifact_sha256",
            "compatibility",
            "id",
            "kind",
            "provenance",
            "seed",
            "status",
            "unit",
        }:
            raise PoolError("pool entry has missing or unexpected fields")
        if entry["kind"] != "checkpoint" or entry["status"] != "available":
            continue
        provenance = entry["provenance"]
        digest = entry["artifact_sha256"]
        artifacts = entry["artifacts"]
        if (
            not isinstance(provenance, str)
            or not isinstance(digest, str)
            or not isinstance(artifacts, dict)
            or set(artifacts) != set(_CHECKPOINT_FILES)
            or not all(isinstance(value, str) and len(value) == 64 for value in artifacts.values())
        ):
            raise PoolError("available checkpoint entry is malformed")
        if digest != canonical_sha256({"artifacts": artifacts}):
            raise PoolError("available checkpoint entry has an invalid artifact digest")
        if verify_artifacts and any(
            not (artifact := Path(provenance) / name).is_file()
            or _sha256_file(artifact) != expected
            for name, expected in artifacts.items()
        ):
            raise PoolError(f"pool checkpoint was altered or is unavailable: {entry['id']}")
    return pool


def freeze_pool(config: S5bConfig) -> Path:
    """Freeze the complete checkpoint catalog once, without silently overwriting it."""

    payload = _pool_payload(config)
    entries = payload["entries"]
    assert isinstance(entries, list)
    missing_finals = [
        entry["id"]
        for entry in entries
        if isinstance(entry, dict)
        and entry["kind"] == "checkpoint"
        and entry["unit"] == 147
        and entry["status"] != "available"
    ]
    if missing_finals:
        raise PoolError(f"cannot freeze pool with missing final checkpoints: {missing_finals}")
    pool = _with_digest(payload)
    directory = config.output_directory / "pools"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pool['pool_sha256']}.json"
    content = canonical_json_bytes(pool) + b"\n"
    if path.exists():
        if path.read_bytes() != content:
            raise PoolError("an existing pool path has different immutable contents")
    else:
        path.write_bytes(content)
    campaign = {
        "config_sha256": pool["config_sha256"],
        "format": "s5b-run-manifest-v1",
        "git_commit": pool["git_commit"],
        "phases": {"freeze-pool": {"pool_sha256": pool["pool_sha256"], "status": "completed"}},
        "pool_sha256": pool["pool_sha256"],
        "run_id": config.run_id,
        "seeds": list(config.evaluation_seeds),
    }
    (config.output_directory / "manifest.json").write_bytes(canonical_json_bytes(campaign) + b"\n")
    return path
