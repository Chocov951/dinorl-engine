"""Freeze RL-S5c-A3 inputs without running any learning measurement."""

from __future__ import annotations

from pathlib import Path

from dinorl_engine.rl.contracts import canonical_json_bytes
from dinorl_engine.rl.s5c.a3 import (
    ACTIVE_V1_PROFILES,
    CONTROLLER_STATUS,
    deterministic_opponent_split,
)
from dinorl_engine.rl.s5c.a3_config import A3Config
from dinorl_engine.rl.s5c.provenance import build_campaign_provenance, write_immutable_manifest
from dinorl_engine.rl.s5c.rewards import a3_rewards
from dinorl_engine.rl.s5c.strength import verify_strong_pool


def _write_once(path: Path, value: object) -> None:
    content = canonical_json_bytes(value) + b"\n"
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError(f"immutable A3 artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def run_a3_preflight(config: A3Config, *, repository: Path, allow_dirty: bool) -> dict[str, object]:
    pool = verify_strong_pool(config.rl_s5b_pool, expected_sha256=config.rl_s5b_pool_sha256)
    raw_entries = pool.get("entries")
    assert isinstance(raw_entries, list)
    entries = [
        entry
        for entry in raw_entries
        if isinstance(entry, dict)
        and entry.get("kind") == "checkpoint"
        and entry.get("status") == "available"
        and entry.get("unit") == 147
    ]
    split = deterministic_opponent_split(entries)
    by_id = {str(entry["id"]): entry for entry in entries}
    split_document = {
        **split.to_mapping(),
        "source_pool_sha256": config.rl_s5b_pool_sha256,
        "method": "per-architecture seed-20 training; all other final seeds held-out",
        "training_opponents": [
            {"id": item, "artifact_sha256": by_id[item]["artifact_sha256"]}
            for item in split.training_ids
        ],
        "heldout_opponents": [
            {"id": item, "artifact_sha256": by_id[item]["artifact_sha256"]}
            for item in split.heldout_ids
        ],
    }
    output = config.output_directory
    for relative in (
        "policy-control",
        "opponent-split",
        "pilot/scavenger-a2-control",
        "pilot/scavenger-a3-curriculum",
        "pilot/predator-a2-control",
        "pilot/predator-a3-balanced",
        "confirm",
        "replays",
        "report",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)
    _write_once(output / "opponent-split" / "rl-s5c-a3-opponent-split.json", split_document)
    rewards = a3_rewards()
    provenance = build_campaign_provenance(
        repository=repository,
        config_sha256=config.sha256,
        reward_sha256={name: reward.cache_key for name, reward in rewards.items()},
        pool_sha256=config.rl_s5b_pool_sha256,
        allow_dirty=allow_dirty,
    )
    manifest = {
        "format": "s5c-a3-run-manifest-v1",
        "phase": "RL-S5c-A3",
        "run_id": config.run_id,
        "config": config.mapping,
        "config_sha256": config.sha256,
        "active_profiles": list(ACTIVE_V1_PROFILES),
        "public_labels": {"predator": "Agressif"},
        "controller_status": CONTROLLER_STATUS,
        "reward_sha256": {name: reward.cache_key for name, reward in rewards.items()},
        "opponent_split": split_document,
        "provenance": provenance,
        "next_allowed_phase": "A3-CONTROL",
        "status": "READY",
    }
    write_immutable_manifest(output / "manifest.json", manifest)
    return manifest
