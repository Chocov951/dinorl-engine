"""TDD contracts for RL-L8 safe publication and five-seed validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from dinorl_engine.rl.artifacts.snapshots import (
    SnapshotError,
    assess_s6_publication,
    load_published_policy,
    publish_policy_snapshot,
    replay_policy_actions,
)
from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.policies.local_mlp_v2 import LocalMLPV2Architecture
from dinorl_engine.rl.training.unit import create_maskable_ppo

_ROOT = Path(__file__).parents[3]


def _schema(name: str) -> Draft202012Validator:
    return Draft202012Validator(
        json.loads((_ROOT / "schemas" / "rl" / name).read_text(encoding="utf-8"))
    )


def _scores(*, random: float = 0.95, scripted: float = 0.70) -> dict[str, float]:
    return {
        "random-legal-v1": random,
        "aggressive-v1": scripted,
        "prudent-v1": scripted,
        "opportunist-v1": scripted,
    }


def _seed_evidence(seed: int, *, boundary: bool = False) -> dict[str, object]:
    stochastic = _scores(random=0.85 if boundary else 0.90, scripted=0.60 if boundary else 0.65)
    evidence: dict[str, object] = {
        "seed": seed,
        "deterministic_evaluations": [_scores(), _scores()],
        "stochastic_evaluation": stochastic,
    }
    if boundary:
        evidence["stochastic_extension"] = _scores(random=0.90, scripted=0.65)
    return evidence


def test_s6_publication_requires_four_of_five_seed_gates_and_boundary_extension() -> None:
    four_of_five = [_seed_evidence(seed, boundary=seed == 1) for seed in range(5)]
    result = assess_s6_publication(four_of_five)

    assert result["eligible"] is True
    assert result["conforming_seed_count"] == 5
    assert result["required_conforming_seed_count"] == 4
    assert result["seeds"][1]["stochastic_extension_required"] is True
    assert list(_schema("s6-publication-evaluation-v1.schema.json").iter_errors(result)) == []

    three_of_five = [_seed_evidence(seed) for seed in range(5)]
    for seed in three_of_five[:2]:
        seed["stochastic_evaluation"] = _scores(random=0.70)
    rejected = assess_s6_publication(three_of_five)

    assert rejected["eligible"] is False
    assert rejected["conforming_seed_count"] == 3
    with pytest.raises(SnapshotError, match="extension"):
        assess_s6_publication([_seed_evidence(0, boundary=True) | {"stochastic_extension": None}])


def test_published_snapshot_is_safe_immutable_and_replays_identical_actions(tmp_path: Path) -> None:
    environment = DinoRLSingleAgentEnv(seed=19)
    source = create_maskable_ppo(environment, seed=19, architecture=LocalMLPV2Architecture.COMPACT)
    try:
        observation, _ = environment.reset(seed=19)
        masks = [environment.action_masks().copy()]
        observations = [observation]
        first_actions = replay_policy_actions(source, observations, masks)
        evaluation = assess_s6_publication([_seed_evidence(seed) for seed in range(5)])
        destination = tmp_path / "snapshots" / "s6-compact-001"
        manifest = publish_policy_snapshot(
            policy=source,
            destination=destination,
            snapshot_id="s6-compact-001",
            owner_id="server",
            source_checkpoint_id="s6-seed-19-unit-147",
            evaluation=evaluation,
            evaluation_id="s6-evaluation-001",
            created_at="2026-09-22T12:00:00Z",
        )
        loaded_environment, loaded = load_published_policy(destination, seed=19)
        try:
            assert manifest["architecture_id"] == "mlp-compact-v2"
            assert list(_schema("snapshot-manifest-v1.schema.json").iter_errors(manifest)) == []
            assert sorted(path.name for path in destination.iterdir()) == [
                "evaluation.json",
                "manifest.json",
                "sha256sums.txt",
                "weights.safetensors",
            ]
            assert replay_policy_actions(loaded, observations, masks) == first_actions
        finally:
            loaded_environment.close()
        assert (tmp_path / "snapshots" / "registry.json").is_file()
        with pytest.raises(SnapshotError, match="immutable"):
            publish_policy_snapshot(
                policy=source,
                destination=destination,
                snapshot_id="s6-compact-001",
                owner_id="server",
                source_checkpoint_id="s6-seed-19-unit-147",
                evaluation=evaluation,
                evaluation_id="s6-evaluation-001",
                created_at="2026-09-22T12:00:00Z",
            )
    finally:
        environment.close()


def test_loader_refuses_corruption_pickle_and_unknown_architecture(tmp_path: Path) -> None:
    environment = DinoRLSingleAgentEnv(seed=20)
    source = create_maskable_ppo(environment, seed=20, architecture=LocalMLPV2Architecture.COMPACT)
    try:
        destination = tmp_path / "snapshots" / "s6-compact-002"
        publish_policy_snapshot(
            policy=source,
            destination=destination,
            snapshot_id="s6-compact-002",
            owner_id="server",
            source_checkpoint_id="s6-seed-20-unit-147",
            evaluation=assess_s6_publication([_seed_evidence(seed) for seed in range(5)]),
            evaluation_id="s6-evaluation-002",
            created_at="2026-09-22T12:00:00Z",
        )
        (destination / "weights.safetensors").write_bytes(b"corrupt")
        with pytest.raises(SnapshotError, match="hash_mismatch"):
            load_published_policy(destination, seed=20)
        (destination / "pickle.pkl").write_bytes(b"not allowed")
        with pytest.raises(SnapshotError, match="unexpected_file"):
            load_published_policy(destination, seed=20)
        (destination / "pickle.pkl").unlink()
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["architecture_id"] = "unknown-v999"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(SnapshotError, match="unknown_architecture"):
            load_published_policy(destination, seed=20)
    finally:
        environment.close()


def test_loader_refuses_a_standalone_external_snapshot(tmp_path: Path) -> None:
    environment = DinoRLSingleAgentEnv(seed=21)
    source = create_maskable_ppo(environment, seed=21, architecture=LocalMLPV2Architecture.COMPACT)
    try:
        registered = tmp_path / "snapshots" / "s6-compact-003"
        publish_policy_snapshot(
            policy=source,
            destination=registered,
            snapshot_id="s6-compact-003",
            owner_id="server",
            source_checkpoint_id="s6-seed-21-unit-147",
            evaluation=assess_s6_publication([_seed_evidence(seed) for seed in range(5)]),
            evaluation_id="s6-evaluation-003",
            created_at="2026-09-22T12:00:00Z",
        )
        external = tmp_path / "external-copy"
        shutil.copytree(registered, external)
        with pytest.raises(SnapshotError, match="external_import_disabled"):
            load_published_policy(external, seed=21)
    finally:
        environment.close()
