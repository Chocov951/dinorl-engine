"""RL-S5b phase-A pool contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dinorl_engine.rl.s5b.config import S5bConfig, S5bConfigError
from dinorl_engine.rl.s5b.pool import PoolError, freeze_pool, read_verified_pool


def _config(tmp_path: Path) -> S5bConfig:
    checkpoints = []
    for architecture in ("mlp-v1", "mlp-compact-v2"):
        for seed in (19, 20):
            directory = tmp_path / "source" / f"{architecture}-{seed}" / "units" / "unit-147"
            directory.mkdir(parents=True)
            for name in ("manifest.json", "model.zip", "ppo_state.npz", "state.json"):
                (directory / name).write_bytes(f"{architecture}-{seed}-{name}".encode())
            checkpoints.append(
                {
                    "id": f"{architecture}-{seed}",
                    "architecture": architecture,
                    "seed": seed,
                    "directory": str(directory),
                }
            )
    return S5bConfig.from_mapping(
        {
            "format": "dinorl-s5b-config-v1",
            "run_id": "s5b-smoke",
            "output_directory": "artifacts",
            "checkpoints": checkpoints,
            "final_architectures": ["mlp-v1", "mlp-compact-v2"],
            "confrontations": 2,
            "evaluation_seeds": [19, 20],
        },
        base_directory=tmp_path,
    )


def test_frozen_pool_is_canonical_idempotent_and_declares_missing_history(tmp_path: Path) -> None:
    config = _config(tmp_path)

    first = freeze_pool(config)
    second = freeze_pool(config)
    pool = read_verified_pool(first)

    assert first == second
    assert pool["format"] == "s5b-benchmark-pool-v1"
    assert isinstance(pool["config_sha256"], str) and len(pool["config_sha256"]) == 64
    assert isinstance(pool["git_commit"], str) and len(pool["git_commit"]) == 40
    assert (tmp_path / "artifacts" / "manifest.json").is_file()
    entries = pool["entries"]
    assert isinstance(entries, list)
    assert sum(entry["status"] == "missing" for entry in entries if isinstance(entry, dict)) == 12
    assert (
        sum(
            entry["unit"] == 147 and entry["status"] == "available"
            for entry in entries
            if isinstance(entry, dict) and entry["kind"] == "checkpoint"
        )
        == 4
    )


def test_pool_verification_rejects_an_altered_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    pool_path = freeze_pool(config)
    checkpoint = config.checkpoints[0].directory / "model.zip"
    checkpoint.write_bytes(b"altered")

    with pytest.raises(PoolError, match="altered"):
        read_verified_pool(pool_path)


def test_pool_config_rejects_a_finalist_without_a_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(S5bConfigError, match="finalist"):
        S5bConfig.from_mapping(
            {
                "format": "dinorl-s5b-config-v1",
                "run_id": "s5b",
                "output_directory": "artifacts",
                "checkpoints": [
                    {
                        "id": "mlp-19",
                        "architecture": "mlp-v1",
                        "seed": 19,
                        "directory": "checkpoint",
                    }
                ],
                "final_architectures": ["mlp-deep-v2"],
                "confrontations": 1,
                "evaluation_seeds": [19],
            },
            base_directory=tmp_path,
        )


def test_pool_manifest_rejects_a_changed_digest(tmp_path: Path) -> None:
    path = freeze_pool(_config(tmp_path))
    document = json.loads(path.read_text(encoding="utf-8"))
    document["run_id"] = "tampered"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PoolError, match="SHA-256"):
        read_verified_pool(path, verify_artifacts=False)
