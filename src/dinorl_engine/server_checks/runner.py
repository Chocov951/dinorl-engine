"""Shared safeguards and provenance capture for server-only gate suites."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import platform
import statistics
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Final

from dinorl_engine.server_checks.manifests import create_archive
from dinorl_engine.server_checks.profiles import (
    STANDARD_PROFILE,
    DependencyProfile,
    runtime_version_matches,
)
from dinorl_engine.server_checks.suites.rl_s0 import run_warmup, run_with_measurement
from dinorl_engine.server_checks.suites.rl_s1 import (
    REPEATED_MEASUREMENTS,
    build_decision,
    candidate_configurations,
)
from dinorl_engine.server_checks.suites.rl_s1 import (
    run_warmup as run_s1_warmup,
)
from dinorl_engine.server_checks.suites.rl_s1 import (
    run_with_measurement as run_s1_measurement,
)
from dinorl_engine.server_checks.suites.rl_s2 import (
    CORPUS_TRANSITIONS,
    ITERATIONS,
    MAX_OVERHEAD_RATIO,
)
from dinorl_engine.server_checks.suites.rl_s2 import (
    REPEATED_MEASUREMENTS as RL_S2_REPEATED_MEASUREMENTS,
)
from dinorl_engine.server_checks.suites.rl_s2 import run_measurement as run_s2_measurement
from dinorl_engine.server_checks.suites.rl_s2 import run_warmup as run_s2_warmup
from dinorl_engine.server_checks.suites.rl_s3 import (
    SELECTED_CONFIGURATION,
)
from dinorl_engine.server_checks.suites.rl_s3 import (
    build_decision as build_s3_decision,
)
from dinorl_engine.server_checks.suites.rl_s3 import (
    run_measurement as run_s3_measurement,
)
from dinorl_engine.server_checks.suites.rl_s4 import (
    ARCHITECTURES,
    UNITS_PER_ARCHITECTURE,
    smoke_passed,
)
from dinorl_engine.server_checks.suites.rl_s4 import run_measurement as run_s4_measurement

__all__ = [
    "DirtyWorktreeError",
    "ServerCheckError",
    "collect_provenance",
    "ensure_clean_tracked_worktree",
    "repository_root",
    "run_suite",
    "verify_import_provenance",
]

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


def _profile_lock_hashes(root: Path, profile: DependencyProfile) -> dict[str, str]:
    """Hash every immutable lock which identifies the selected profile."""

    lock_hashes: dict[str, str] = {}
    for filename in profile.provenance_lock_filenames:
        path = root / filename
        if not path.is_file():
            raise ServerCheckError(f"required lockfile is missing: {filename}")
        lock_hashes[filename] = _sha256_file(path)
    return lock_hashes


def _verify_profile_environment(profile: DependencyProfile) -> None:
    """Enforce the virtualenv feature required by a constrained profile."""

    if not profile.requires_system_site_packages:
        return
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    try:
        lines = configuration.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ServerCheckError(
            "pythonanywhere profile requires a virtualenv with system site packages"
        ) from error
    if not any(line.strip().lower() == "include-system-site-packages = true" for line in lines):
        raise ServerCheckError(
            "pythonanywhere profile requires --system-site-packages when creating the virtualenv"
        )


def _verify_runtime_dependencies(root: Path, profile: DependencyProfile) -> dict[str, str]:
    _verify_profile_environment(profile)
    lock_path = root / profile.runtime_lock_filename
    if not lock_path.is_file():
        raise ServerCheckError(f"{profile.runtime_lock_filename} is missing")
    installed: dict[str, str] = {}
    for package, expected_version in _locked_versions(lock_path).items():
        try:
            installed_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise ServerCheckError(f"locked package is not installed: {package}") from error
        if not runtime_version_matches(installed_version, expected_version):
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
                f"required module is absent from {profile.runtime_lock_filename}: {distribution}"
            )
    importlib.import_module("dinorl_engine.core.engine")
    return installed


def collect_provenance(
    root: Path, profile: DependencyProfile = STANDARD_PROFILE
) -> dict[str, object]:
    """Capture the code, lock, runtime, CPU and OS identity of this gate run."""

    return {
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "lock_sha256": _profile_lock_hashes(root, profile),
        "installed_packages": _verify_runtime_dependencies(root, profile),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def verify_import_provenance(
    root: Path, profile: DependencyProfile, provenance: dict[str, object]
) -> None:
    """Verify an archive against its profile lock without requiring its host runtime.

    A PythonAnywhere archive is imported on a development machine with a newer
    Torch build, so importer validation compares the archive runtime to its
    pinned profile rather than to the local interpreter's installed packages.
    """

    if provenance.get("git_commit") != _git(root, "rev-parse", "HEAD"):
        raise ServerCheckError("server archive provenance mismatch: git_commit")
    if provenance.get("lock_sha256") != _profile_lock_hashes(root, profile):
        raise ServerCheckError("server archive provenance mismatch: lock_sha256")
    installed = provenance.get("installed_packages")
    if not isinstance(installed, dict):
        raise ServerCheckError("server archive provenance installed_packages is invalid")
    expected = _locked_versions(root / profile.runtime_lock_filename)
    if set(installed) != set(expected):
        raise ServerCheckError("server archive provenance mismatch: installed_packages")
    for package, expected_version in expected.items():
        installed_version = installed.get(package)
        if not isinstance(installed_version, str) or not runtime_version_matches(
            installed_version, expected_version
        ):
            raise ServerCheckError(f"server archive provenance mismatch: {package}")


def _summary(result: dict[str, object]) -> str:
    if result["suite"] == "RL-S1":
        return _summary_rl_s1(result)
    if result["suite"] == "RL-S2":
        return _summary_rl_s2(result)
    if result["suite"] == "RL-S3":
        return _summary_rl_s3(result)
    if result["suite"] == "RL-S4":
        return _summary_rl_s4(result)
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
        f"- Dependency profile: `{result['dependency_profile']}`\n"
        f"- Duration: `{duration}` seconds\n"
        f"- Peak RSS: `{memory}` bytes\n"
        "- Status: passed\n"
    )


def _summary_rl_s1(result: dict[str, object]) -> str:
    """Render the server-selected vectorization decision into the result summary."""

    decision = result["decision"]
    if not isinstance(decision, dict):
        raise ServerCheckError("RL-S1 decision has an invalid format")
    return (
        "# RL-S1 server result\n\n"
        f"- Run: `{result['run_id']}`\n"
        f"- Dependency profile: `{result['dependency_profile']}`\n"
        f"- Selected backend: `{decision['backend']}`\n"
        f"- Selected environments: `{decision['n_envs']}`\n"
        f"- Median throughput: `{decision['median_transitions_per_second']}` transitions/s\n"
        "- Criterion: highest median end-to-end throughput among the complete matrix\n"
    )


def _summary_rl_s2(result: dict[str, object]) -> str:
    """Render a server-measured VM gate without hiding a failed threshold."""

    decision = result["decision"]
    if not isinstance(decision, dict):
        raise ServerCheckError("RL-S2 decision has an invalid format")
    status = "passed" if decision["passed"] else "blocked: profile before native extension"
    return (
        "# RL-S2 server result\n\n"
        f"- Run: `{result['run_id']}`\n"
        f"- Dependency profile: `{result['dependency_profile']}`\n"
        f"- Median VM overhead: `{decision['median_overhead_ratio']}`\n"
        f"- Maximum permitted overhead: `{decision['max_overhead_ratio']}`\n"
        "- Exact AST/VM/native parity: passed\n"
        f"- Status: {status}\n"
    )


def _summary_rl_s3(result: dict[str, object]) -> str:
    """Render the server-selected interactive priority mode."""

    decision = result["decision"]
    if not isinstance(decision, dict):
        raise ServerCheckError("RL-S3 decision has an invalid format")
    status = "passed" if result["passed"] else "failed"
    return (
        "# RL-S3 server result\n\n"
        f"- Run: `{result['run_id']}`\n"
        f"- Dependency profile: `{result['dependency_profile']}`\n"
        f"- Selected interactive mode: `{decision['mode']}`\n"
        f"- Interactive latency: `{decision['play_latency_seconds']}` seconds\n"
        "- Selection: minimum play latency, then maximum learning throughput\n"
        f"- Status: {status}\n"
    )


def _summary_rl_s4(result: dict[str, object]) -> str:
    """Render the architecture smoke result without selecting either candidate."""

    measurement = result["measurement"]
    if not isinstance(measurement, dict):
        raise ServerCheckError("RL-S4 measurement has an invalid format")
    candidates = measurement.get("candidates")
    if not isinstance(candidates, list):
        raise ServerCheckError("RL-S4 candidates have an invalid format")
    status = "passed" if result["passed"] else "failed"
    lines = [
        "# RL-S4 server result\n",
        f"- Run: `{result['run_id']}`",
        f"- Dependency profile: `{result['dependency_profile']}`",
        f"- Units per architecture: `{UNITS_PER_ARCHITECTURE}`",
        "- Architecture selection is deferred to RL-S5.",
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ServerCheckError("RL-S4 candidate has an invalid format")
        lines.append(
            "- Candidate "
            f"`{candidate['architecture']}`: "
            f"`{candidate['encoder_parameters']}` encoder parameters, "
            f"`{candidate['inference_seconds_per_observation']}` seconds/observation"
        )
    lines.append(f"- Status: {status}\n")
    return "\n".join(lines)


def _run_rl_s1(*, profile: DependencyProfile, root: Path, output_dir: Path) -> dict[str, object]:
    """Execute the complete measured RL-S1 matrix and archive its decision."""

    seed = 19
    configurations = candidate_configurations(seed=seed)
    candidates: list[dict[str, object]] = []
    for configuration in configurations:
        candidates.append(
            {
                "backend": configuration.backend.value,
                "n_envs": configuration.n_envs,
                "n_steps": configuration.n_steps,
                "warmup": run_s1_warmup(configuration),
                "repetitions": [
                    run_s1_measurement(configuration) for _ in range(REPEATED_MEASUREMENTS)
                ],
            }
        )
    decision = build_decision(candidates)
    run_id = uuid.uuid4().hex
    result: dict[str, object] = {
        "format": "dinorl-server-result-v1",
        "suite": "RL-S1",
        "dependency_profile": profile.name,
        "run_id": run_id,
        "provenance": collect_provenance(root, profile),
        "configuration": {
            "warmups": 1,
            "repetitions": REPEATED_MEASUREMENTS,
            "seed": seed,
            "rollout_transitions": 2048,
            "epochs": 4,
            "batch_size": 256,
            "candidates": [
                {
                    "backend": configuration.backend.value,
                    "n_envs": configuration.n_envs,
                    "n_steps": configuration.n_steps,
                }
                for configuration in configurations
            ],
        },
        "candidates": candidates,
        "decision": decision,
    }
    archive = create_archive(
        output_dir=output_dir,
        suite="RL-S1",
        run_id=run_id,
        result=result,
        summary=_summary(result),
    )
    return {
        "archive": str(archive),
        "archive_sha256": _sha256_file(archive),
        "run_id": run_id,
        "status": "passed",
        "suite": "RL-S1",
        "profile": profile.name,
    }


def _run_rl_s2(*, profile: DependencyProfile, root: Path, output_dir: Path) -> dict[str, object]:
    """Measure one fixed public corpus, retaining failed-gate evidence."""

    warmup = run_s2_warmup()
    repetitions = [run_s2_measurement() for _ in range(RL_S2_REPEATED_MEASUREMENTS)]
    overheads: list[float] = []
    for repetition in repetitions:
        overhead = repetition.get("overhead_ratio")
        exact = repetition.get("exact")
        if isinstance(overhead, bool) or not isinstance(overhead, int | float) or exact is not True:
            raise ServerCheckError("RL-S2 produced an invalid parity measurement")
        overheads.append(float(overhead))
    median_overhead = statistics.median(overheads)
    decision = {
        "max_overhead_ratio": MAX_OVERHEAD_RATIO,
        "median_overhead_ratio": median_overhead,
        "passed": median_overhead <= MAX_OVERHEAD_RATIO,
    }
    run_id = uuid.uuid4().hex
    result: dict[str, object] = {
        "format": "dinorl-server-result-v1",
        "suite": "RL-S2",
        "dependency_profile": profile.name,
        "run_id": run_id,
        "provenance": collect_provenance(root, profile),
        "configuration": {
            "warmups": 1,
            "repetitions": RL_S2_REPEATED_MEASUREMENTS,
            "seed": 19,
            "corpus_transitions": CORPUS_TRANSITIONS,
            "iterations": ITERATIONS,
            "max_overhead_ratio": MAX_OVERHEAD_RATIO,
        },
        "warmup": warmup,
        "repetitions": repetitions,
        "decision": decision,
    }
    archive = create_archive(
        output_dir=output_dir,
        suite="RL-S2",
        run_id=run_id,
        result=result,
        summary=_summary(result),
    )
    return {
        "archive": str(archive),
        "archive_sha256": _sha256_file(archive),
        "run_id": run_id,
        "status": "passed" if decision["passed"] else "failed",
        "suite": "RL-S2",
        "profile": profile.name,
    }


def _run_rl_s3(*, profile: DependencyProfile, root: Path, output_dir: Path) -> dict[str, object]:
    """Run exact recovery and the two interactive-priority alternatives on the server."""

    measurement = run_s3_measurement()
    resume = measurement.get("resume")
    crash = measurement.get("crash")
    candidates = measurement.get("priority_candidates")
    decision = measurement.get("decision")
    if (
        not isinstance(resume, dict)
        or not isinstance(crash, dict)
        or not isinstance(candidates, list)
        or not isinstance(decision, dict)
    ):
        raise ServerCheckError("RL-S3 produced an invalid measurement")
    expected_decision = build_s3_decision(
        [
            {
                "mode": candidate.get("mode"),
                "play_latency_seconds": candidate.get("play_latency_seconds"),
                "training_transitions_per_second": candidate.get("training_transitions_per_second"),
            }
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
    )
    if decision != expected_decision:
        raise ServerCheckError("RL-S3 decision does not match its priority measurements")
    passed = (
        resume.get("exact") is True
        and resume.get("processes_cleaned_up") is True
        and crash
        == {
            "before_rename_preserved": True,
            "post_rename_recovered": True,
            "no_duplicate_debit": True,
        }
        and all(
            isinstance(candidate, dict) and candidate.get("interactive_served") is True
            for candidate in candidates
        )
    )
    run_id = uuid.uuid4().hex
    result: dict[str, object] = {
        "format": "dinorl-server-result-v1",
        "suite": "RL-S3",
        "dependency_profile": profile.name,
        "run_id": run_id,
        "provenance": collect_provenance(root, profile),
        "configuration": {
            "seed": SELECTED_CONFIGURATION.seed,
            "backend": SELECTED_CONFIGURATION.backend.value,
            "n_envs": SELECTED_CONFIGURATION.n_envs,
            "n_steps": SELECTED_CONFIGURATION.n_steps,
            "rollout_transitions": 2048,
            "priority_modes": ["unit_boundary", "separate_worker"],
        },
        "resume": resume,
        "crash": crash,
        "priority_candidates": candidates,
        "decision": decision,
        "passed": passed,
    }
    archive = create_archive(
        output_dir=output_dir,
        suite="RL-S3",
        run_id=run_id,
        result=result,
        summary=_summary(result),
    )
    return {
        "archive": str(archive),
        "archive_sha256": _sha256_file(archive),
        "run_id": run_id,
        "status": "passed" if passed else "failed",
        "suite": "RL-S3",
        "profile": profile.name,
    }


def _run_rl_s4(*, profile: DependencyProfile, root: Path, output_dir: Path) -> dict[str, object]:
    """Run the fixed ten-unit smoke for both frozen architecture candidates."""

    measurement = run_s4_measurement()
    passed = smoke_passed(measurement)
    run_id = uuid.uuid4().hex
    result: dict[str, object] = {
        "format": "dinorl-server-result-v1",
        "suite": "RL-S4",
        "dependency_profile": profile.name,
        "run_id": run_id,
        "provenance": collect_provenance(root, profile),
        "configuration": {
            "seed": SELECTED_CONFIGURATION.seed,
            "backend": SELECTED_CONFIGURATION.backend.value,
            "n_envs": SELECTED_CONFIGURATION.n_envs,
            "n_steps": SELECTED_CONFIGURATION.n_steps,
            "rollout_transitions": 2048,
            "epochs": 4,
            "batch_size": 256,
            "units_per_architecture": UNITS_PER_ARCHITECTURE,
            "architectures": [architecture.value for architecture in ARCHITECTURES],
        },
        "measurement": measurement,
        "passed": passed,
    }
    archive = create_archive(
        output_dir=output_dir,
        suite="RL-S4",
        run_id=run_id,
        result=result,
        summary=_summary(result),
    )
    return {
        "archive": str(archive),
        "archive_sha256": _sha256_file(archive),
        "run_id": run_id,
        "status": "passed" if passed else "failed",
        "suite": "RL-S4",
        "profile": profile.name,
    }


def run_suite(
    *, suite: str, profile: DependencyProfile = STANDARD_PROFILE, root: Path, output_dir: Path
) -> dict[str, object]:
    """Run one supported gate only after verifying a clean tracked worktree."""

    if suite not in {"RL-S0", "RL-S1", "RL-S2", "RL-S3", "RL-S4"}:
        raise ServerCheckError(f"unsupported server suite: {suite}")
    ensure_clean_tracked_worktree(root)
    if suite == "RL-S1":
        return _run_rl_s1(profile=profile, root=root, output_dir=output_dir)
    if suite == "RL-S2":
        return _run_rl_s2(profile=profile, root=root, output_dir=output_dir)
    if suite == "RL-S3":
        return _run_rl_s3(profile=profile, root=root, output_dir=output_dir)
    if suite == "RL-S4":
        return _run_rl_s4(profile=profile, root=root, output_dir=output_dir)
    run_id = uuid.uuid4().hex
    result: dict[str, object] = {
        "format": "dinorl-server-result-v1",
        "suite": suite,
        "dependency_profile": profile.name,
        "run_id": run_id,
        "provenance": collect_provenance(root, profile),
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
        "profile": profile.name,
    }
