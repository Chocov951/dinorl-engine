"""Immutable, no-match RL-S6 V3 aggregation from the completed V2 evidence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from dinorl_engine.rl.s6_evaluation_v3 import publication_gate_v3

__all__ = ["S6V3Error", "recalculate_v3"]


class S6V3Error(RuntimeError):
    """V3 cannot be traced exactly to the immutable V1 and V2 inputs."""


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
        raise S6V3Error(f"cannot read JSON: {path}") from error


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


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise S6V3Error(f"{field} is invalid")
    return value


def _inputs(source: Path) -> tuple[Mapping[str, object], Mapping[str, object], Path]:
    manifest = _mapping(_read(source / "manifest.json"), "V2 manifest")
    campaign = _mapping(_read(source / "campaign-result.json"), "V2 campaign result")
    if manifest.get("format") != "rl-s6-evaluation-v2-manifest-v1":
        raise S6V3Error("V2 manifest format is invalid")
    if campaign.get("format") != "rl-s6-campaign-result-v2":
        raise S6V3Error("V2 campaign result format is invalid")
    v1_path = manifest.get("source_campaign")
    v1_digest = manifest.get("source_campaign_sha256")
    if not isinstance(v1_path, str) or not isinstance(v1_digest, str):
        raise S6V3Error("V2 manifest lacks V1 provenance")
    v1 = Path(v1_path)
    if _sha256(v1 / "campaign-result.json") != v1_digest:
        raise S6V3Error("V1 campaign result hash differs from V2 provenance")
    if (
        campaign.get("source_campaign") != str(v1)
        or campaign.get("source_decision") != "FAILED_RL_S6"
    ):
        raise S6V3Error("V2 campaign provenance is inconsistent")
    return manifest, campaign, v1


def _seed_results(source: Path, campaign: Mapping[str, object]) -> list[dict[str, object]]:
    raw = campaign.get("seeds")
    if not isinstance(raw, list) or len(raw) != 5:
        raise S6V3Error("V2 campaign must contain exactly five seeds")
    results: list[dict[str, object]] = []
    seen: set[int] = set()
    for item in raw:
        seed_result = _mapping(item, "V2 seed result")
        seed = seed_result.get("seed")
        if type(seed) is not int or seed in seen:
            raise S6V3Error("V2 seed identity is invalid")
        seen.add(seed)
        stored = _mapping(_read(source / "seeds" / f"seed-{seed}" / "result.json"), "V2 seed file")
        if dict(stored) != dict(seed_result):
            raise S6V3Error("V2 aggregate and seed result differ")
        gate = _mapping(seed_result.get("gate"), "V2 gate")
        v3_gate = publication_gate_v3(gate)
        record_path = source / "seeds" / f"seed-{seed}" / "records.json"
        results.append(
            {
                "seed": seed,
                "candidate_unit": seed_result.get("candidate_unit"),
                "checkpoint_sha256": seed_result.get("checkpoint_sha256"),
                "v2_gate_sha256": _sha256(source / "seeds" / f"seed-{seed}" / "result.json"),
                "v2_records_sha256": _sha256(record_path),
                "gate": v3_gate,
            }
        )

    def seed_key(item: Mapping[str, object]) -> int:
        value = item.get("seed")
        if type(value) is not int:
            raise S6V3Error("V3 seed identity is invalid")
        return value

    return sorted(results, key=seed_key)


def recalculate_v3(
    *, source_v2_directory: Path, output_directory: Path, dry_run: bool
) -> dict[str, object]:
    """Recalculate V3 from JSON artifacts only; no model or game is constructed."""

    source = source_v2_directory.resolve()
    output = output_directory.resolve()
    manifest_v2, campaign_v2, source_v1 = _inputs(source)
    seeds = _seed_results(source, campaign_v2)
    conforming = sum(
        _mapping(seed["gate"], "V3 seed gate").get("decision") == "pass" for seed in seeds
    )
    manifest = {
        "format": "rl-s6-evaluation-v3-manifest-v1",
        "source_v1_campaign": str(source_v1),
        "source_v1_campaign_sha256": _sha256(source_v1 / "campaign-result.json"),
        "source_v2_directory": str(source),
        "source_v2_manifest_sha256": _sha256(source / "manifest.json"),
        "source_v2_campaign_sha256": _sha256(source / "campaign-result.json"),
        "source_v2_decision": campaign_v2.get("decision"),
        "architecture": manifest_v2.get("architecture"),
        "architecture_dimensions": manifest_v2.get("architecture_dimensions"),
        "checkpoint_hashes": manifest_v2.get("checkpoints"),
        "no_training": True,
        "no_new_matches": True,
    }
    result: dict[str, object] = {
        "format": "rl-s6-campaign-result-v3",
        "source_v1_decision": "FAILED_RL_S6",
        "source_v2_decision": campaign_v2.get("decision"),
        "decision": "RL_S6_V3_PASSED" if conforming >= 4 else "FAILED_RL_S6_V3",
        "conforming_seed_count": conforming,
        "required_conforming_seed_count": 4,
        "seeds": seeds,
        "no_training": True,
        "no_new_matches": True,
    }
    if dry_run:
        return {
            "command": "s6 recalculate-v3",
            "status": "dry_run",
            "manifest": manifest,
            "result": result,
        }
    if output.exists():
        raise S6V3Error(
            "V3 output directory already exists; immutable aggregation cannot overwrite"
        )
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(output / "campaign-result.json", result)
    return result
