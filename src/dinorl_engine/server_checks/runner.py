"""Shared safeguards and provenance capture for server-only gate suites."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Final

from dinorl_engine.server_checks.manifests import create_archive
from dinorl_engine.server_checks.suites.rl_s0 import run_warmup, run_with_measurement

__all__ = [
    "DirtyWorktreeError",
    "ServerCheckError",
    "collect_provenance",
    "ensure_clean_tracked_worktree",
    "repository_root",
    "run_suite",
]

_LOCK_FILES: Final = ("requirements.lock", "requirements-dev.lock")
_REQUIRED_MODULES: Final = {
    "gymnasium": "gymnasium",
    "torch": "torch",
    "stable-baselines3": "stable_baselines3",
    "sb3-contrib": "sb3_contrib",
}


class ServerCheckError(RuntimeError):
    """Raised when a server gate cannot produce an auditable result."""


class DirtyWorktreeError(ServerCheckError):
    """Raised when tracked worktree edits would invalidate a server measurement."""


def repository_root(start: Path | None = None) -> Path:
    """Resolve the repository root from the current working directory."""

    current = (Path.cwd() if start is None else start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise ServerCheckError("current directory is not inside a Git repository")


def _git(root: Path, *arguments: str) -> str:
    try:
        process = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ServerCheckError("Git is required for server gate provenance") from error
    if process.returncode != 0:
        message = process.stderr.strip() or "Git command failed"
        raise ServerCheckError(message)
    return process.stdout.strip()


def ensure_clean_tracked_worktree(root: Path) -> None:
    """Refuse a gate run if either staged or unstaged tracked edits exist."""

    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        try:
            process = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise ServerCheckError("Git is required for server gate provenance") from error
        if process.returncode == 1:
            raise DirtyWorktreeError("tracked_worktree_dirty")
        if process.returncode != 0:
            message = process.stderr.strip() or "Git worktree check failed"
            raise ServerCheckError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(" ") or "==" not in line:
            continue
        name, version = line.split("==", maxsplit=1)
        if name and version and ";" not in name:
            versions[name] = version
    return versions


def _verify_runtime_dependencies(root: Path) -> dict[str, str]:
    lock_path = root / "requirements.lock"
    if not lock_path.is_file():
        raise ServerCheckError("requirements.lock is missing")
    installed: dict[str, str] = {}
    for package, expected_version in _locked_versions(lock_path).items():
        try:
            installed_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise ServerCheckError(f"locked package is not installed: {package}") from error
        if installed_version != expected_version:
            raise ServerCheckError(
                "locked package version mismatch: "
                f"{package}={installed_version}, expected {expected_version}"
            )
        installed[package] = installed_version
    for distribution, module in _REQUIRED_MODULES.items():
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise ServerCheckError(f"required module cannot be imported: {module}") from error
        if distribution not in installed:
            raise ServerCheckError(
                f"required module is absent from requirements.lock: {distribution}"
            )
    importlib.import_module("dinorl_engine.core.engine")
    return installed


def collect_provenance(root: Path) -> dict[str, object]:
    """Capture the code, lock, runtime, CPU and OS identity of this gate run."""

    lock_hashes: dict[str, str] = {}
    for filename in _LOCK_FILES:
        path = root / filename
        if not path.is_file():
            raise ServerCheckError(f"required lockfile is missing: {filename}")
        lock_hashes[filename] = _sha256_file(path)
    return {
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "lock_sha256": lock_hashes,
        "installed_packages": _verify_runtime_dependencies(root),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _summary(result: dict[str, object]) -> str:
    repetition = result["repetitions"]
    if not isinstance(repetition, list) or len(repetition) != 1:
        raise ServerCheckError("RL-S0 must contain exactly one measured repetition")
    measurement = repetition[0]
    if not isinstance(measurement, dict):
        raise ServerCheckError("RL-S0 measurement has an invalid format")
    duration = measurement.get("duration_seconds")
    memory = measurement.get("peak_memory_bytes")
    return (
        "# RL-S0 server result\n\n"
        f"- Run: `{result['run_id']}`\n"
        f"- Duration: `{duration}` seconds\n"
        f"- Peak RSS: `{memory}` bytes\n"
        "- Status: passed\n"
    )


def run_suite(*, suite: str, root: Path, output_dir: Path) -> dict[str, object]:
    """Run one supported gate only after verifying a clean tracked worktree."""

    if suite != "RL-S0":
        raise ServerCheckError(f"unsupported server suite: {suite}")
    ensure_clean_tracked_worktree(root)
    run_id = uuid.uuid4().hex
    result: dict[str, object] = {
        "format": "dinorl-server-result-v1",
        "suite": suite,
        "run_id": run_id,
        "provenance": collect_provenance(root),
        "configuration": {
            "warmups": 1,
            "repetitions": 1,
            "seed": 19,
            "rollout_transitions": 2048,
            "epochs": 4,
            "batch_size": 256,
        },
        "warmup": run_warmup(),
        "repetitions": [run_with_measurement()],
    }
    archive = create_archive(
        output_dir=output_dir,
        suite=suite,
        run_id=run_id,
        result=result,
        summary=_summary(result),
    )
    return {
        "archive": str(archive),
        "archive_sha256": _sha256_file(archive),
        "run_id": run_id,
        "status": "passed",
        "suite": suite,
    }
