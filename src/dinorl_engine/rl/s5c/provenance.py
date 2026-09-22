"""Content-addressed provenance and fresh-initialization evidence for RL-S5c-A2."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

__all__ = [
    "ProvenanceError",
    "build_campaign_provenance",
    "initialization_record",
    "optimizer_state_sha256",
    "policy_weights_sha256",
    "repository_content_sha256",
    "verify_worktree_policy",
    "write_immutable_manifest",
]


class ProvenanceError(RuntimeError):
    """Raised when a campaign cannot be tied to reproducible source inputs."""


class _Policy(Protocol):
    optimizer: torch.optim.Optimizer

    def state_dict(self) -> Mapping[str, object]: ...


class _Model(Protocol):
    policy: _Policy


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


def verify_worktree_policy(*, dirty: bool, allow_dirty: bool) -> str | None:
    """Reject dirty server runs unless the development override is explicit."""

    if dirty and not allow_dirty:
        raise ProvenanceError("dirty worktree refused; pass --allow-dirty only for development")
    if dirty:
        return "WARNING: dirty worktree explicitly allowed; full content hash recorded"
    return None


def _update_hash(digest: _Digest, value: object) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(b"tensor\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(repr(array.shape).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
        return
    if isinstance(value, np.ndarray):
        digest.update(b"ndarray\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(repr(value.shape).encode())
        digest.update(b"\0")
        digest.update(value.tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=str):
            _update_hash(digest, str(key))
            _update_hash(digest, value[key])
        return
    if isinstance(value, list | tuple):
        digest.update(b"sequence\0")
        for item in value:
            _update_hash(digest, item)
        return
    if value is None or isinstance(value, str | bool | int | float):
        digest.update(type(value).__name__.encode())
        digest.update(b"\0")
        digest.update(repr(value).encode())
        digest.update(b"\0")
        return
    raise ProvenanceError(f"unsupported provenance value: {type(value).__name__}")


def _sha256(value: object) -> str:
    digest = hashlib.sha256()
    _update_hash(digest, value)
    return digest.hexdigest()


def policy_weights_sha256(model: _Model) -> str:
    """Hash policy tensors independently of ZIP metadata and filesystem paths."""

    return _sha256(model.policy.state_dict())


def optimizer_state_sha256(model: _Model) -> str:
    """Hash the fresh optimizer state using deterministic tensor serialization."""

    return _sha256(model.policy.optimizer.state_dict())


def initialization_record(model: _Model, *, seed: int) -> dict[str, object]:
    """Describe a zero-start model before its first learner transition."""

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ProvenanceError("initialization seed must be an unsigned 32-bit integer")
    optimizer = model.policy.optimizer.state_dict()
    state = optimizer.get("state")
    if not isinstance(state, Mapping):
        raise ProvenanceError("optimizer state is malformed")
    return {
        "format": "s5c-a2-initialization-v1",
        "seed": seed,
        "provenance": "fresh-maskable-ppo",
        "parent_checkpoint": None,
        "weights_sha256": policy_weights_sha256(model),
        "optimizer_sha256": optimizer_state_sha256(model),
        "optimizer_state_entries": len(state),
    }


def _git(repository: Path, *arguments: str) -> str:
    try:
        process = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ProvenanceError("Git is required for campaign provenance") from error
    if process.returncode != 0:
        raise ProvenanceError(f"Git provenance command failed: {' '.join(arguments)}")
    return process.stdout


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_files(repository: Path) -> dict[str, str]:
    raw = _git(repository, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    paths = sorted(path for path in raw.split("\0") if path)
    result: dict[str, str] = {}
    for relative in paths:
        normalized = relative.replace("\\", "/")
        influences_run = (
            normalized.startswith(("src/", "configs/"))
            or normalized in {"pyproject.toml", "uv.lock"}
            or normalized.startswith("docs/duel-de-raptors-")
            or normalized.endswith("RL-S5c-specification-bots-specialises.md")
        )
        if not influences_run:
            continue
        path = repository / relative
        if path.is_file():
            result[normalized] = _file_sha256(path)
    if not result:
        raise ProvenanceError("repository content manifest is empty")
    return result


def repository_content_sha256(repository: Path) -> str:
    """Hash exactly the tracked and untracked files that can influence an A2 run."""

    return _sha256(_repository_files(repository))


def _combined_hash(repository: Path, relative_paths: tuple[str, ...]) -> str:
    files: dict[str, str] = {}
    for relative in relative_paths:
        path = repository / relative
        if not path.is_file():
            raise ProvenanceError(f"provenance component is missing: {relative}")
        files[relative] = _file_sha256(path)
    return _sha256(files)


def _dependencies() -> dict[str, str]:
    packages = (
        "dinorl-engine",
        "gymnasium",
        "numpy",
        "sb3-contrib",
        "stable-baselines3",
        "torch",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise ProvenanceError(f"critical dependency is unavailable: {package}") from error
    return versions


def build_campaign_provenance(
    *,
    repository: Path,
    config_sha256: str,
    reward_sha256: Mapping[str, str],
    pool_sha256: str,
    allow_dirty: bool,
) -> dict[str, object]:
    """Capture Git plus full relevant content hashes for a replayable A2 run."""

    commit = _git(repository, "rev-parse", "HEAD").strip()
    status_lines = tuple(
        line
        for line in _git(
            repository, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    )
    dirty = bool(status_lines)
    warning = verify_worktree_policy(dirty=dirty, allow_dirty=allow_dirty)
    content = _repository_files(repository)
    content_sha256 = _sha256(content)
    components = {
        "engine": _combined_hash(repository, ("src/dinorl_engine/core/engine.py",)),
        "rules": _combined_hash(
            repository,
            (
                "docs/duel-de-raptors-regles-du-jeu.md",
                "src/dinorl_engine/core/engine.py",
                "src/dinorl_engine/core/events.py",
                "src/dinorl_engine/core/state.py",
            ),
        ),
        "observation": _combined_hash(
            repository,
            (
                "src/dinorl_engine/rl/env/canonical.py",
                "src/dinorl_engine/rl/env/observation.py",
            ),
        ),
        "actions_and_masks": _combined_hash(
            repository,
            (
                "src/dinorl_engine/core/actions.py",
                "src/dinorl_engine/core/engine.py",
                "src/dinorl_engine/rl/env/canonical.py",
                "src/dinorl_engine/rl/env/single_agent.py",
            ),
        ),
        "ppo": _combined_hash(
            repository,
            (
                "src/dinorl_engine/rl/training/unit.py",
                "src/dinorl_engine/rl/policies/local_mlp_v2.py",
            ),
        ),
    }
    ppo_hyperparameters = {
        "batch_size": 256,
        "clip_range": 0.20,
        "ent_coef": 0.01,
        "epochs": 4,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "learning_rate": 0.0003,
        "max_grad_norm": 0.50,
        "rollout_transitions": 2048,
        "vf_coef": 0.50,
    }
    return {
        "format": "s5c-a2-provenance-v1",
        "git_commit": commit,
        "worktree_dirty": dirty,
        "worktree_status": list(status_lines),
        "warning": warning,
        "content_sha256": content_sha256,
        "content_files": content,
        "component_sha256": components,
        "ppo_hyperparameters": ppo_hyperparameters,
        "ppo_hyperparameters_sha256": _sha256(ppo_hyperparameters),
        "reward_sha256": dict(sorted(reward_sha256.items())),
        "config_sha256": config_sha256,
        "rl_s5b_pool_sha256": pool_sha256,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "dependencies": _dependencies(),
    }


def write_immutable_manifest(path: Path, value: Mapping[str, object]) -> None:
    """Atomically create a manifest and reject any later content drift."""

    content = (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    if path.is_file():
        if path.read_bytes() != content:
            raise ProvenanceError("existing A2 manifest differs from resolved inputs")
        return
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
