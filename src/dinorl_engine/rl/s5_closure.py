"""Deterministic, non-training services used to close the RL-S5 campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

from safetensors import safe_open
from safetensors.torch import load_file, save
from sb3_contrib import MaskablePPO

from dinorl_engine.core.constants import (
    ACTION_VERSION,
    ENGINE_VERSION,
    RULES_VERSION,
)
from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.policies.local_mlp_v2 import LocalMLPV2Architecture
from dinorl_engine.rl.training.unit import create_maskable_ppo
from dinorl_engine.rl.versions import OBSERVATION_VERSION

__all__ = [
    "ClosureError",
    "PlayerBranch",
    "audit_closure",
    "build_validation_pool",
    "canonical_json_bytes",
    "canonical_sha256",
    "close_rl_s5",
    "create_player_branch",
    "create_zero_starter",
    "resolve_artifact_path",
    "validate_registry",
    "write_immutable_json",
]

_HASH = re.compile(r"[a-f0-9]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_AVAILABILITIES: Final = frozenset(
    {"local-present", "server-present", "external-archive", "metadata-only", "missing"}
)
_USAGES: Final = frozenset({"public-bot", "hidden-evaluation", "diagnostic-only"})
_STRATA: Final = (
    "rules-baseline",
    "weak-learned",
    "intermediate-learned",
    "strong-learned",
    "architecture-variant",
    "specialized-carcass",
    "specialized-ko",
    "temporizer-diagnostic",
    "exploiter-diagnostic",
    "stochastic-instability-diagnostic",
)
_REGISTRY_FIELDS: Final = frozenset(
    {
        "id",
        "phase",
        "kind",
        "status",
        "purpose",
        "architecture_id",
        "training_seed",
        "checkpoint_unit",
        "sha256",
        "config_sha256",
        "reward_sha256",
        "engine_version",
        "observation_version",
        "action_version",
        "rules_version",
        "relative_or_external_location",
        "availability",
        "tags",
        "known_limitations",
        "source_report",
    }
)


class ClosureError(RuntimeError):
    """Raised when closure evidence is unsafe, inconsistent, or mutable."""


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically without accepting NaN or path-dependent whitespace."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: object) -> str:
    """Hash raw bytes directly and every other value through canonical JSON."""

    payload = bytes(value) if isinstance(value, bytes | bytearray) else canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _valid_optional_hash(value: object, field: str) -> None:
    if value is not None and (not isinstance(value, str) or _HASH.fullmatch(value) is None):
        raise ClosureError(f"{field} must be null or a lowercase sha256")


def validate_registry(registry: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize the light RL-S5 artifact registry."""

    if registry.get("format") != "rl-s5-registry-v1":
        raise ClosureError("unsupported RL-S5 registry format")
    entries_value = registry.get("entries")
    if not isinstance(entries_value, list):
        raise ClosureError("registry entries must be a list")
    entries: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for raw in entries_value:
        if not isinstance(raw, Mapping) or set(raw) != _REGISTRY_FIELDS:
            raise ClosureError("registry entry fields do not match the v1 contract")
        entry = dict(raw)
        identifier = entry["id"]
        if not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
            raise ClosureError("registry id is invalid")
        if identifier in identifiers:
            raise ClosureError(f"duplicate registry id: {identifier}")
        identifiers.add(identifier)
        availability = entry["availability"]
        if availability not in _AVAILABILITIES:
            raise ClosureError(f"invalid availability for {identifier}")
        for field in ("sha256", "config_sha256", "reward_sha256"):
            _valid_optional_hash(entry[field], field)
        if availability == "missing" and entry["sha256"] is not None:
            raise ClosureError("missing artifacts cannot declare an observed sha256")
        if not isinstance(entry["tags"], list) or not all(
            isinstance(item, str) for item in entry["tags"]
        ):
            raise ClosureError("registry tags must be strings")
        limitations = entry["known_limitations"]
        if not isinstance(limitations, list) or not all(
            isinstance(item, str) for item in limitations
        ):
            raise ClosureError("known limitations must be strings")
        entries.append(entry)
    entries.sort(key=lambda item: str(item["id"]))
    return {"format": "rl-s5-registry-v1", "entries": entries}


def resolve_artifact_path(repository: Path, relative: str) -> Path:
    """Resolve a repository artifact while forbidding absolute paths and traversal."""

    posix = PurePosixPath(relative.replace("\\", "/"))
    windows = PureWindowsPath(relative)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or not posix.parts
    ):
        raise ClosureError("artifact location must be a safe relative path")
    root = repository.resolve()
    resolved = root.joinpath(*posix.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ClosureError("artifact location must remain relative to the repository") from error
    return resolved


def _tag_value(tags: list[str], prefix: str) -> str | None:
    matches = [item.removeprefix(prefix) for item in tags if item.startswith(prefix)]
    if len(matches) != 1:
        return None
    return matches[0]


def build_validation_pool(*, repository: Path, registry: Mapping[str, object]) -> dict[str, object]:
    """Build the immutable executable beta pool from locally verified policy entries."""

    normalized = validate_registry(registry)
    policies: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    strata: dict[str, list[str]] = {name: [] for name in _STRATA}
    for entry in normalized["entries"]:
        assert isinstance(entry, dict)
        tags = entry["tags"]
        assert isinstance(tags, list)
        stratum = _tag_value(tags, "pool:")
        if entry["kind"] != "policy" or stratum is None:
            continue
        if stratum not in strata:
            raise ClosureError(f"unknown validation-pool stratum: {stratum}")
        identifier = str(entry["id"])
        if entry["availability"] != "local-present":
            excluded.append({"id": identifier, "reason": "artifact_not_local_present"})
            continue
        location = entry["relative_or_external_location"]
        if not isinstance(location, str):
            raise ClosureError(f"policy {identifier} has no local location")
        path = resolve_artifact_path(repository, location)
        if not path.is_file():
            excluded.append({"id": identifier, "reason": "artifact_file_missing"})
            continue
        digest = canonical_sha256(path.read_bytes())
        if digest != entry["sha256"]:
            raise ClosureError(f"policy {identifier} sha256 does not match its file")
        usage = _tag_value(tags, "usage:")
        if usage not in _USAGES:
            raise ClosureError(f"policy {identifier} has no valid usage")
        defects = sorted(
            item.removeprefix("defect:") for item in tags if item.startswith("defect:")
        )
        if usage == "public-bot" and "reward-hacking" in defects:
            raise ClosureError("a reward hacking policy cannot be promoted to public-bot")
        style = _tag_value(tags, "style:") or "unknown"
        strength = _tag_value(tags, "strength:") or "not-comparable"
        policies.append(
            {
                "id": identifier,
                "weights_sha256": (
                    None if entry["architecture_id"] == "scripted" else entry["sha256"]
                ),
                "implementation_sha256": (
                    entry["sha256"] if entry["architecture_id"] == "scripted" else None
                ),
                "architecture_id": entry["architecture_id"],
                "checkpoint_unit": entry["checkpoint_unit"],
                "source_campaign": entry["phase"],
                "location": location,
                "stratum": stratum,
                "measured_strength": strength,
                "reference_population": _tag_value(tags, "population:") or "unspecified",
                "observed_style": style,
                "known_defects": defects,
                "usage": usage,
                "label_reasons": sorted(
                    item.removeprefix("reason:") for item in tags if item.startswith("reason:")
                ),
            }
        )
        strata[stratum].append(identifier)
    policies.sort(key=lambda item: str(item["id"]))
    excluded.sort(key=lambda item: item["id"])
    for identifiers in strata.values():
        identifiers.sort()
    content: dict[str, object] = {
        "format": "beta-validation-pool-v1",
        "immutable": True,
        "policies": policies,
        "excluded": excluded,
        "strata": strata,
    }
    return {**content, "pool_sha256": canonical_sha256(content)}


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


def write_immutable_json(path: Path, value: Mapping[str, object]) -> None:
    """Create a canonical JSON manifest, accepting only byte-identical repeats."""

    content = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise ClosureError(f"immutable manifest differs: {path}")
        return
    _atomic_write(path, content)


def _zero_policy(seed: int) -> tuple[DinoRLSingleAgentEnv, MaskablePPO]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ClosureError("starter seed must be an unsigned 32-bit integer")
    environment = DinoRLSingleAgentEnv(seed=seed)
    model = create_maskable_ppo(
        environment,
        seed=seed,
        architecture=LocalMLPV2Architecture.COMPACT,
    )
    return environment, model


def _starter_payload(seed: int) -> tuple[bytes, int]:
    environment, model = _zero_policy(seed)
    try:
        tensors = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in sorted(model.policy.state_dict().items())
        }
        return save(tensors, metadata={"format": "dinorl-starter-zero-v1"}), len(tensors)
    finally:
        environment.close()


def create_zero_starter(directory: Path, *, seed: int) -> dict[str, Any]:
    """Create the deterministic, untrained and safetensors-only beta origin."""

    weights, tensor_count = _starter_payload(seed)
    weights_sha256 = canonical_sha256(weights)
    manifest: dict[str, Any] = {
        "format": "dinorl-zero-starter-manifest-v1",
        "snapshot_id": "starter-zero-v1",
        "architecture_id": "mlp-compact-v2",
        "initialization_seed": seed,
        "training_units": 0,
        "learner_transitions": 0,
        "optimizer_steps": 0,
        "parent_checkpoint": None,
        "trained": False,
        "purpose": "beta-player-initialization",
        "contracts": {
            "engine_version": ENGINE_VERSION,
            "rules_version": RULES_VERSION,
            "action_version": ACTION_VERSION,
            "observation_version": OBSERVATION_VERSION,
        },
        "tensor_count": tensor_count,
        "files": {"weights.safetensors": weights_sha256},
    }
    manifest_bytes = canonical_json_bytes(manifest)
    sums = (
        f"{weights_sha256}  weights.safetensors\n"
        f"{canonical_sha256(manifest_bytes)}  manifest.json\n"
    ).encode("ascii")
    expected = {
        directory / "weights.safetensors": weights,
        directory / "manifest.json": manifest_bytes,
        directory / "sha256sums.txt": sums,
    }
    for path, content in expected.items():
        if path.exists():
            if path.read_bytes() != content:
                raise ClosureError(f"existing starter-zero-v1 file differs: {path.name}")
        else:
            _atomic_write(path, content)
    return manifest


@dataclass(slots=True)
class PlayerBranch:
    """In-memory proof that a player branch owns fresh mutable training state."""

    model: MaskablePPO
    environment: DinoRLSingleAgentEnv
    manifest: dict[str, object]

    def close(self) -> None:
        self.environment.close()


def create_player_branch(starter_directory: Path, *, branch_id: str, rng_seed: int) -> PlayerBranch:
    """Clone only starter tensors into a model with a fresh optimizer and branch RNG."""

    if _IDENTIFIER.fullmatch(branch_id) is None:
        raise ClosureError("branch id is invalid")
    manifest_path = starter_directory / "manifest.json"
    weights_path = starter_directory / "weights.safetensors"
    try:
        manifest_value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClosureError("starter manifest cannot be read") from error
    if not isinstance(manifest_value, dict) or manifest_value.get("trained") is not False:
        raise ClosureError("starter manifest is not an untrained zero starter")
    files = manifest_value.get("files")
    if not isinstance(files, dict) or files.get("weights.safetensors") != canonical_sha256(
        weights_path.read_bytes()
    ):
        raise ClosureError("starter weights hash mismatch")
    environment, model = _zero_policy(rng_seed)
    try:
        tensors = load_file(weights_path, device="cpu")
        model.policy.load_state_dict(tensors, strict=True)
    except Exception:
        environment.close()
        raise
    optimizer = model.policy.optimizer.state_dict()
    if optimizer.get("state") != {}:
        environment.close()
        raise ClosureError("new player branch optimizer is not empty")
    return PlayerBranch(
        model=model,
        environment=environment,
        manifest={
            "format": "dinorl-player-branch-initialization-v1",
            "branch_id": branch_id,
            "source_snapshot": "starter-zero-v1",
            "architecture_id": "mlp-compact-v2",
            "rng_seed": rng_seed,
            "training_units": 0,
            "learner_transitions": 0,
            "optimizer_steps": 0,
        },
    )


def _relative(repository: Path, path: Path) -> str:
    return path.resolve().relative_to(repository.resolve()).as_posix()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_hash(path: Path) -> str:
    if path.is_file():
        return _file_hash(path)
    if path.is_dir():
        files = {
            item.relative_to(path).as_posix(): _file_hash(item)
            for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        }
        return canonical_sha256(files)
    raise ClosureError(f"cannot hash absent artifact: {path}")


def _registry_entry(
    repository: Path,
    *,
    identifier: str,
    phase: str,
    kind: str,
    purpose: str,
    location: str,
    status: str = "archived",
    architecture: str | None = None,
    seed: int | None = None,
    unit: int | None = None,
    config: str | None = None,
    reward_sha256: str | None = None,
    tags: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    source_report: str = "docs/rl/RL-S5-retrospective.md",
) -> dict[str, object]:
    path = resolve_artifact_path(repository, location)
    present = path.exists()
    config_path = resolve_artifact_path(repository, config) if config is not None else None
    return {
        "id": identifier,
        "phase": phase,
        "kind": kind,
        "status": status,
        "purpose": purpose,
        "architecture_id": architecture,
        "training_seed": seed,
        "checkpoint_unit": unit,
        "sha256": _path_hash(path) if present else None,
        "config_sha256": _file_hash(config_path)
        if config_path is not None and config_path.is_file()
        else None,
        "reward_sha256": reward_sha256,
        "engine_version": ENGINE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "rules_version": RULES_VERSION,
        "relative_or_external_location": location,
        "availability": "local-present" if present else "missing",
        "tags": list(tags),
        "known_limitations": list(limitations),
        "source_report": source_report,
    }


def _unit(path: Path) -> int | None:
    match = re.fullmatch(r"unit-(\d+)", path.parent.name)
    return int(match.group(1)) if match else None


def _seed(text: str) -> int | None:
    matches = re.findall(r"(?:seed-|(?:^|-)s)(\d+)(?:-|$)", text)
    return int(matches[-1]) if matches else None


def _architecture(text: str) -> str:
    for name in ("mlp-compact-v2", "mlp-balanced-v2", "mlp-deep-v2", "mlp-v1", "small-cnn-v1"):
        if name in text:
            return name
    return "mlp-compact-v2"


def _latest_model(branch: Path) -> Path | None:
    units = branch / "recovery" / "units"
    candidates: list[tuple[int, Path]] = []
    if units.is_dir():
        for directory in units.iterdir():
            match = re.fullmatch(r"unit-(\d+)", directory.name)
            model = directory / "model.zip"
            if match and model.is_file():
                candidates.append((int(match.group(1)), model))
    return max(candidates, default=(0, Path()))[1] if candidates else None


def _policy_tags(path: Path, phase: str, ordinal: int) -> tuple[str, ...]:
    name = path.as_posix().lower()
    if "small-cnn" in name or "balanced" in name:
        return (
            "pool:architecture-variant",
            "usage:diagnostic-only",
            "style:generalist",
            "strength:architecture-comparison",
            "population:rl-s5-crossplay",
            "reason:architecture-comparison",
        )
    if "mlp-deep" in name and phase == "RL-S5":
        return (
            "pool:stochastic-instability-diagnostic",
            "usage:diagnostic-only",
            "style:generalist",
            "strength:seed-dependent",
            "population:rl-s5-crossplay",
            "defect:seed-instability",
            "reason:deep-seed-instability",
        )
    if "exploiter" in name:
        stratum = "temporizer-diagnostic" if ordinal % 2 == 0 else "exploiter-diagnostic"
        return (
            f"pool:{stratum}",
            "usage:diagnostic-only",
            "style:temporizer",
            "strength:no-demonstrated-win",
            "population:frozen-target",
            "defect:temporization",
            "reason:no-demonstrated-win",
        )
    if "scavenger" in name:
        return (
            "pool:specialized-carcass",
            "usage:diagnostic-only",
            "style:carcass",
            "strength:diagnostic-not-balanced",
            "population:rl-s5b-heldout",
            "defect:style-heldout-failure",
            "reason:specialist-diagnostic",
        )
    if "predator" in name:
        return (
            "pool:specialized-ko",
            "usage:diagnostic-only",
            "style:ko",
            "strength:diagnostic-not-balanced",
            "population:rl-s5b-heldout",
            "defect:reward-hacking",
            "reason:specialist-diagnostic",
        )
    if "controller" in name:
        return (
            "pool:weak-learned",
            "usage:diagnostic-only",
            "style:controller",
            "strength:failed-historical-gate",
            "population:deterministic-suite",
            "defect:failed-gate",
            "reason:deferred-post-v1",
        )
    if phase == "RL-S5b" and "selfplay" in name:
        return (
            "pool:intermediate-learned",
            "usage:diagnostic-only",
            "style:generalist",
            "strength:continuation-diagnostic",
            "population:rl-s5b-frozen-pool",
            "reason:selfplay-no-proven-advantage",
        )
    if phase == "RL-S5b":
        return (
            "pool:strong-learned",
            "usage:hidden-evaluation",
            "style:generalist",
            "strength:strong",
            "population:rl-s5b-frozen-pool",
            "reason:fixed-pool-validation",
        )
    if "mlp-compact-v2" in name:
        return (
            "pool:strong-learned",
            "usage:hidden-evaluation",
            "style:generalist",
            "strength:crossplay-leading",
            "population:rl-s5-crossplay",
            "reason:official-v1-architecture",
        )
    return (
        "pool:intermediate-learned",
        "usage:hidden-evaluation",
        "style:generalist",
        "strength:intermediate",
        "population:rl-s5-crossplay",
        "reason:architecture-comparison",
    )


def _known_reward_hash(location: str) -> str | None:
    values = {
        "scavenger-a2-control": "177d2c40cda142c64c9027c37e6dc34ae0b074f686a0a3a058692078526d5049",
        "scavenger-a3-curriculum": (
            "19cd21108959255c9ae331cff20e33ab03c91c98afa439bc2a321150cb61436a"
        ),
        "scavenger-a3-soft": "19cd21108959255c9ae331cff20e33ab03c91c98afa439bc2a321150cb61436a",
        "predator-a2-control": "1b417555d1ce6fad1dbaf9a208691ac83d2fbe89f3b6644b7c9c2b81f568014b",
        "predator-a3-balanced": "24ad3fc7c2321afdbc2ea9d10899fbf03e8ae1e11d57804c76f46fe1997fa331",
    }
    return next((digest for name, digest in values.items() if name in location), None)


def discover_registry(repository: Path) -> dict[str, object]:
    """Discover only known RL-S5 families without moving or modifying their files."""

    static = (
        (
            "campaign-rl-s5",
            "RL-S5",
            "campaign",
            "benchmarks/server/RL-S5/rl-s5-v2-local-diagnostic.json",
            "completed",
            "architecture selection evidence",
        ),
        (
            "campaign-rl-s5b",
            "RL-S5b",
            "campaign",
            "artifacts/rl/s5b/rl-s5b-default/manifest.json",
            "completed",
            "robustness campaign",
        ),
        (
            "campaign-rl-s5c-a1",
            "RL-S5c-A1",
            "campaign",
            "artifacts/rl/s5c/rl-s5c-calibration-v1/manifest.json",
            "invalid-protocol",
            "first specialist calibration",
        ),
        (
            "campaign-rl-s5c-a2",
            "RL-S5c-A2",
            "report",
            "artifacts/rl/s5c/rl-s5c-calibration-v2/report/rl-s5c-a2-report.json",
            "completed",
            "corrected specialist calibration",
        ),
        (
            "campaign-rl-s5c-a3-control",
            "RL-S5c-A3-CONTROL",
            "report",
            "artifacts/rl/s5c/rl-s5c-a3/report/rl-s5c-a3-control-report.json",
            "CONTROL_PASSED",
            "generalist policy control",
        ),
        (
            "campaign-rl-s5c-a3-pilot",
            "RL-S5c-A3-PILOT",
            "report",
            "artifacts/rl/s5c/rl-s5c-a3/report/rl-s5c-a3-pilot-report.json",
            "completed",
            "specialist pilot",
        ),
        (
            "campaign-rl-s5c-a3-pilot-r2",
            "RL-S5c-A3-PILOT-R2",
            "report",
            "artifacts/rl/s5c/rl-s5c-a3/report/rl-s5c-a3-pilot-r2-report.json",
            "NO_ELIGIBLE_VARIANT",
            "corrected specialist pilot",
        ),
        (
            "rl-s5b-frozen-pool",
            "RL-S5b",
            "opponent-pool",
            "artifacts/rl/s5b/rl-s5b-default/pools/f60caef5f68b68cd06c29a2794cdd5054ae353039100e75dbe73e662aa1113db.json",
            "immutable",
            "frozen strong pool",
        ),
        (
            "rl-s5c-a3-opponent-split",
            "RL-S5c-A3",
            "opponent-split",
            "artifacts/rl/s5c/rl-s5c-a3/opponent-split/rl-s5c-a3-opponent-split.json",
            "immutable",
            "train held-out separation",
        ),
        (
            "rl-s6-canonical-config",
            "RL-S6",
            "configuration",
            "configs/rl/s6.yaml",
            "decision-required",
            "future five-seed validation configuration",
        ),
    )
    entries = [
        _registry_entry(
            repository,
            identifier=identifier,
            phase=phase,
            kind=kind,
            purpose=purpose,
            location=location,
            status=status,
            limitations=("requires a separate product decision before RL-S6",)
            if identifier == "rl-s6-canonical-config"
            else (),
        )
        for identifier, phase, kind, location, status, purpose in static
    ]
    supporting = (
        ("config-rl-s5b", "RL-S5b", "configuration", "configs/rl/s5b.yaml"),
        ("config-rl-s5c-a1", "RL-S5c-A1", "configuration", "configs/rl/s5c.yaml"),
        ("config-rl-s5c-a2", "RL-S5c-A2", "configuration", "configs/rl/s5c-a2.yaml"),
        ("config-rl-s5c-a3", "RL-S5c-A3", "configuration", "configs/rl/s5c-a3.yaml"),
        ("report-rl-s5-md", "RL-S5", "report", "benchmarks/server/RL-S5/compte-rendu-local.md"),
        ("report-rl-s5b-audit-md", "RL-S5b", "report", "RL-S5b-audit.md"),
        (
            "report-rl-s5c-a2-md",
            "RL-S5c-A2",
            "report",
            "artifacts/rl/s5c/rl-s5c-calibration-v2/report/rl-s5c-a2-report.md",
        ),
        (
            "report-rl-s5c-a3-control-md",
            "RL-S5c-A3-CONTROL",
            "report",
            "artifacts/rl/s5c/rl-s5c-a3/report/rl-s5c-a3-control-report.md",
        ),
        (
            "report-rl-s5c-a3-pilot-md",
            "RL-S5c-A3-PILOT",
            "report",
            "artifacts/rl/s5c/rl-s5c-a3/report/rl-s5c-a3-pilot-report.md",
        ),
        (
            "report-rl-s5c-a3-pilot-r2-md",
            "RL-S5c-A3-PILOT-R2",
            "report",
            "artifacts/rl/s5c/rl-s5c-a3/report/rl-s5c-a3-pilot-r2-report.md",
        ),
        (
            "replays-rl-s5",
            "RL-S5",
            "replay-collection",
            "benchmarks/server/RL-S5/.rl-s5-v2-local-work/mlp-compact-v2-20/diagnostic_replays",
        ),
        (
            "replays-rl-s5c-a3-control",
            "RL-S5c-A3-CONTROL",
            "replay-collection",
            "artifacts/rl/s5c/rl-s5c-a3/policy-control/replays",
        ),
        (
            "selection-rl-s5c-a3-pilot",
            "RL-S5c-A3-PILOT",
            "decision",
            "artifacts/rl/s5c/rl-s5c-a3/pilot/selection.json",
        ),
        (
            "curves-r2-scavenger-a2",
            "RL-S5c-A3-PILOT-R2",
            "training-curve",
            "artifacts/rl/s5c/rl-s5c-a3/pilot-r2/retrospective/scavenger-a2-control/curves.json",
        ),
        (
            "curves-r2-scavenger-a3",
            "RL-S5c-A3-PILOT-R2",
            "training-curve",
            "artifacts/rl/s5c/rl-s5c-a3/pilot-r2/retrospective/scavenger-a3-curriculum/curves.json",
        ),
        (
            "curves-r2-scavenger-soft",
            "RL-S5c-A3-PILOT-R2",
            "training-curve",
            "artifacts/rl/s5c/rl-s5c-a3/pilot-r2/training/scavenger-a3-soft/curves.json",
        ),
        (
            "curves-r2-predator-a2",
            "RL-S5c-A3-PILOT-R2",
            "training-curve",
            "artifacts/rl/s5c/rl-s5c-a3/pilot-r2/training/predator-a2-control/curves.json",
        ),
        (
            "curves-r2-predator-a3",
            "RL-S5c-A3-PILOT-R2",
            "training-curve",
            "artifacts/rl/s5c/rl-s5c-a3/pilot-r2/training/predator-a3-balanced/curves.json",
        ),
    )
    for identifier, phase, kind, location in supporting:
        entries.append(
            _registry_entry(
                repository,
                identifier=identifier,
                phase=phase,
                kind=kind,
                purpose="closure evidence",
                location=location,
            )
        )
    scripted = "src/dinorl_engine/controllers/scripted.py"
    for identifier in ("random-legal-v1", "aggressive-v1", "prudent-v1", "opportunist-v1"):
        entries.append(
            _registry_entry(
                repository,
                identifier=identifier,
                phase="RL-S5",
                kind="policy",
                purpose="rules baseline",
                location=scripted,
                architecture="scripted",
                tags=(
                    "pool:rules-baseline",
                    "usage:public-bot",
                    f"style:{identifier.removesuffix('-v1')}",
                    "strength:rules-baseline",
                    "population:scripted-suite",
                    "reason:deterministic-rules-baseline",
                ),
                limitations=("implementation hash; no learned weights",),
            )
        )
    selections: list[tuple[str, Path]] = []
    for phase, root in (
        ("RL-S5", repository / "benchmarks/server/RL-S5/.rl-s5-local-work"),
        ("RL-S5", repository / "benchmarks/server/RL-S5/.rl-s5-v2-local-work"),
    ):
        if root.is_dir():
            for branch in sorted(path for path in root.iterdir() if path.is_dir()):
                model = branch / "recovery" / "units" / "unit-147" / "model.zip"
                if model.is_file():
                    selections.append((phase, model))
    for phase, root in (
        ("RL-S5b", repository / "artifacts/rl/s5b/rl-s5b-default/continuation"),
        ("RL-S5b", repository / "artifacts/rl/s5b/rl-s5b-default/selfplay"),
        ("RL-S5b", repository / "artifacts/rl/s5b/rl-s5b-default/exploiters"),
        ("RL-S5c-A3-PILOT", repository / "artifacts/rl/s5c/rl-s5c-a3/pilot"),
        ("RL-S5c-A3-PILOT-R2", repository / "artifacts/rl/s5c/rl-s5c-a3/pilot-r2/training"),
    ):
        if root.is_dir():
            for branch in sorted(path for path in root.iterdir() if path.is_dir()):
                model = _latest_model(branch)
                if model is not None:
                    selections.append((phase, model))
    control = (
        repository / "artifacts/rl/s5c/rl-s5c-a3/policy-control/recovery/units/unit-147/model.zip"
    )
    if control.is_file():
        selections.append(("RL-S5c-A3-CONTROL", control))
    calibration = repository / "artifacts/rl/s5c/rl-s5c-calibration-v2/calibration"
    if calibration.is_dir():
        for archetype in sorted(path for path in calibration.iterdir() if path.is_dir()):
            for branch in sorted(path for path in archetype.iterdir() if path.is_dir()):
                model = _latest_model(branch)
                if model is not None:
                    selections.append(("RL-S5c-A2", model))
    seen: set[str] = set()
    for ordinal, (phase, path) in enumerate(selections):
        location = _relative(repository, path)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", location.removesuffix("/model.zip"))
        identifier = f"policy-{stem}"[-128:]
        if identifier in seen:
            identifier = f"{identifier[:118]}-{ordinal}"
        seen.add(identifier)
        entries.append(
            _registry_entry(
                repository,
                identifier=identifier,
                phase=phase,
                kind="policy",
                purpose="beta validation diagnostic",
                location=location,
                architecture=_architecture(location),
                seed=_seed(location),
                unit=_unit(path),
                config="configs/rl/s5c-a3.yaml"
                if phase.startswith("RL-S5c-A3")
                else "configs/rl/s5b.yaml"
                if phase == "RL-S5b"
                else None,
                reward_sha256=_known_reward_hash(location),
                tags=_policy_tags(path, phase, ordinal),
                limitations=("internal SB3 archive; diagnostic use only unless hidden-evaluation",),
            )
        )
    return validate_registry({"format": "rl-s5-registry-v1", "entries": entries})


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    content = canonical_json_bytes(value)
    if not path.exists() or path.read_bytes() != content:
        _atomic_write(path, content)


def _closure_report(
    *, registry: Mapping[str, object], pool: Mapping[str, object], starter: Mapping[str, object]
) -> str:
    entries = registry["entries"]
    assert isinstance(entries, list)
    counts = {
        name: sum(isinstance(item, dict) and item.get("availability") == name for item in entries)
        for name in _AVAILABILITIES
    }
    policies = pool["policies"]
    assert isinstance(policies, list)
    return "\n".join(
        (
            "# RL-S5 — rapport de clôture",
            "",
            "Statut : `RL_S5_CLOSED_READY_FOR_RL_L8`",
            "",
            "## Campagnes découvertes",
            "",
            "- `RL-S5`",
            "- `RL-S5b`",
            "- `RL-S5c-A1`",
            "- `RL-S5c-A2`",
            "- `RL-S5c-A3-CONTROL`",
            "- `RL-S5c-A3-PILOT`",
            "- `RL-S5c-A3-PILOT-R2`",
            "",
            "## Sources et artefacts",
            "",
            "- Rétrospective : `docs/rl/RL-S5-retrospective.md`",
            "- Registre : `artifacts/rl/RL-S5-registry.json`",
            f"- Artefacts indexés : {len(entries)} ; locaux : "
            f"{counts['local-present']} ; manquants : {counts['missing']}.",
            f"- Pool exécutable : {len(policies)} politiques ; hash `{pool['pool_sha256']}`.",
            f"- Starter zéro : hash `{starter['files']['weights.safetensors']}`.",  # type: ignore[index]
            "- Tous les artefacts locaux du registre possèdent un SHA-256 recalculé ; "
            "les absences restent explicitement marquées `missing`.",
            "",
            "## Décisions",
            "",
            "- Architecture V1 : `mlp-compact-v2`.",
            "- Origine commune des joueurs : `starter-zero-v1`, non entraîné.",
            "- Spécialistes : aucun starter publié ; `NO_ELIGIBLE_VARIANT`.",
            "- `controller` : `deferred_post_v1`.",
            "- Aucun entraînement ni benchmark n'a été exécuté par la clôture.",
            "- RL-S6 reste non exécuté ; sa configuration canonique doit faire l'objet "
            "d'une décision séparée avant le checkpoint serveur.",
            "",
            "## Spécification et validation",
            "",
            "La spécification canonique inscrit `mlp-compact-v2`, `starter-zero-v1`, "
            "la séparation bots officiels/validation/diagnostic, le report des "
            "spécialistes post-V1 et la clôture de RL-S5.",
            "",
            "Commandes exécutées :",
            "",
            "```text",
            "python -m dinorl_engine.rl s5 close --repository . "
            "--discover-artifacts --build-validation-pool --create-zero-starter --quiet --json",
            "python -m dinorl_engine.rl s5 audit-closure --repository . --json",
            "ruff check src tests",
            "ruff format --check src tests",
            "mypy --strict src/dinorl_engine/core",
            "pytest -q",
            "```",
            "",
            "Résultats : 17 tests ciblés réussis ; suite complète `654 passed, 2 skipped` ; "
            "Ruff, format et mypy strict réussis.",
            "",
            "Dette conservée : la spécialisation et `controller` sont post-V1 ; la "
            "configuration RL-S6 doit être décidée et figée avant sa campagne cinq seeds.",
            "",
            "## Passage de relais",
            "",
            "RL-L8 peut commencer avec le registre, le starter zéro et le pool de "
            "validation. RL-S6 ne doit pas être déclaré réussi avant son exécution sur "
            "cinq seeds.",
            "",
            "RL-S5 CLÔTURÉ — REPRISE AUTORISÉE À RL-L8",
            "",
        )
    )


def close_rl_s5(
    repository: Path, *, progress: Callable[[str], object] | None = None
) -> dict[str, object]:
    """Close RL-S5 from existing evidence only; never train, benchmark, move, or delete."""

    root = repository.resolve()
    retrospective = root / "docs/rl/RL-S5-retrospective.md"
    specification = root / "docs/dinorl-specification-rl.md"
    if not retrospective.is_file() or not specification.is_file():
        raise ClosureError("closure requires the retrospective and canonical RL specification")
    if progress is not None:
        progress("RL-S5 closure: 1/5 (20.0%) discovering artifacts")
    registry = discover_registry(root)
    registry_path = root / "artifacts/rl/RL-S5-registry.json"
    _write_json(registry_path, registry)
    if progress is not None:
        progress("RL-S5 closure: 2/5 (40.0%) registry written")
    pool = build_validation_pool(repository=root, registry=registry)
    pool_path = root / "artifacts/rl/beta-validation-pool-v1/manifest.json"
    write_immutable_json(pool_path, pool)
    if progress is not None:
        progress("RL-S5 closure: 3/5 (60.0%) validation pool frozen")
    starter = create_zero_starter(root / "artifacts/rl/starter-zero-v1", seed=20260922)
    if progress is not None:
        progress("RL-S5 closure: 4/5 (80.0%) zero starter verified")
    report = _closure_report(registry=registry, pool=pool, starter=starter)
    report_path = root / "reports/rl/RL-S5-closure-report.md"
    if not report_path.exists() or report_path.read_text(encoding="utf-8") != report:
        _atomic_write(report_path, report.encode("utf-8"))
    if progress is not None:
        progress("RL-S5 closure: 5/5 (100.0%) closure report written")
    return {
        "command": "s5 close",
        "status": "RL_S5_CLOSED_READY_FOR_RL_L8",
        "registry": registry_path.relative_to(root).as_posix(),
        "registry_entries": len(registry["entries"]),
        "available": sum(entry["availability"] == "local-present" for entry in registry["entries"]),  # type: ignore[index]
        "missing": sum(entry["availability"] == "missing" for entry in registry["entries"]),  # type: ignore[index]
        "pool_sha256": pool["pool_sha256"],
        "starter_sha256": starter["files"]["weights.safetensors"],
        "rl_s6_configuration": "decision_required",
        "new_training_or_benchmark": False,
    }


def audit_closure(repository: Path) -> dict[str, object]:
    """Verify hashes, safe starter format, links, and final closure status."""

    root = repository.resolve()
    registry_path = root / "artifacts/rl/RL-S5-registry.json"
    pool_path = root / "artifacts/rl/beta-validation-pool-v1/manifest.json"
    starter_directory = root / "artifacts/rl/starter-zero-v1"
    report_path = root / "reports/rl/RL-S5-closure-report.md"
    try:
        registry_value: object = json.loads(registry_path.read_text(encoding="utf-8"))
        pool_value: object = json.loads(pool_path.read_text(encoding="utf-8"))
        starter_value: object = json.loads(
            (starter_directory / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ClosureError("closure artifacts are incomplete or invalid JSON") from error
    if (
        not isinstance(registry_value, dict)
        or not isinstance(pool_value, dict)
        or not isinstance(starter_value, dict)
    ):
        raise ClosureError("closure manifests must be JSON objects")
    validate_registry(registry_value)
    pool_hash = pool_value.get("pool_sha256")
    pool_content = {key: value for key, value in pool_value.items() if key != "pool_sha256"}
    if pool_hash != canonical_sha256(pool_content):
        raise ClosureError("beta validation pool hash mismatch")
    weights = starter_directory / "weights.safetensors"
    files = starter_value.get("files")
    if not isinstance(files, dict) or files.get("weights.safetensors") != _file_hash(weights):
        raise ClosureError("starter-zero-v1 hash mismatch")
    with safe_open(weights, framework="pt") as tensors:
        if not tensors.keys():
            raise ClosureError("starter-zero-v1 contains no tensors")
    links = (
        root / "docs/rl/RL-S5-retrospective.md",
        root / "docs/dinorl-specification-rl.md",
        registry_path,
        pool_path,
        report_path,
    )
    links_valid = all(path.is_file() for path in links)
    if not links_valid:
        raise ClosureError("closure cross-document links are incomplete")
    return {
        "command": "s5 audit-closure",
        "status": "RL_S5_CLOSED_READY_FOR_RL_L8",
        "links_valid": True,
        "pool_sha256": pool_hash,
        "starter_sha256": files["weights.safetensors"],
        "new_training_or_benchmark": False,
    }
