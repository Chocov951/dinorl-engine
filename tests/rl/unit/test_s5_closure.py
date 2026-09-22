"""TDD contracts for the non-training RL-S5 closure ticket."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from safetensors import safe_open

from dinorl_engine.rl.__main__ import main
from dinorl_engine.rl.s5_closure import (
    ClosureError,
    audit_closure,
    build_validation_pool,
    canonical_json_bytes,
    canonical_sha256,
    close_rl_s5,
    create_player_branch,
    create_zero_starter,
    resolve_artifact_path,
    validate_registry,
    write_immutable_json,
)

SHA = "a" * 64


def _entry(
    identifier: str = "policy-1", *, availability: str = "local-present"
) -> dict[str, object]:
    return {
        "id": identifier,
        "phase": "RL-S5",
        "kind": "policy",
        "status": "archived",
        "purpose": "validation",
        "architecture_id": "mlp-compact-v2",
        "training_seed": 19,
        "checkpoint_unit": 147,
        "sha256": SHA if availability != "missing" else None,
        "config_sha256": None,
        "reward_sha256": None,
        "engine_version": "0.1.0",
        "observation_version": "rl-observation-v1",
        "action_version": "1.0.0",
        "rules_version": "1.0.0",
        "relative_or_external_location": "weights/model.zip",
        "availability": availability,
        "tags": ["pool:strong-learned", "usage:hidden-evaluation"],
        "known_limitations": [],
        "source_report": "reports/source.md",
    }


def test_registry_is_canonical_and_rejects_duplicate_ids_and_bad_hashes() -> None:
    registry = {"format": "rl-s5-registry-v1", "entries": [_entry()]}
    validated = validate_registry(registry)

    assert canonical_json_bytes(validated) == canonical_json_bytes(copy.deepcopy(validated))
    with pytest.raises(ClosureError, match="duplicate"):
        validate_registry({"format": "rl-s5-registry-v1", "entries": [_entry(), _entry()]})
    invalid = _entry()
    invalid["sha256"] = "not-a-hash"
    with pytest.raises(ClosureError, match="sha256"):
        validate_registry({"format": "rl-s5-registry-v1", "entries": [invalid]})


def test_missing_artifact_is_preserved_explicitly() -> None:
    missing = _entry("missing-policy", availability="missing")
    missing["relative_or_external_location"] = "external/missing.zip"

    result = validate_registry({"format": "rl-s5-registry-v1", "entries": [missing]})

    assert result["entries"][0]["availability"] == "missing"
    assert result["entries"][0]["sha256"] is None


def test_artifact_resolution_rejects_absolute_and_parent_paths(tmp_path: Path) -> None:
    present = tmp_path / "weights" / "model.zip"
    present.parent.mkdir()
    present.write_bytes(b"weights")

    assert resolve_artifact_path(tmp_path, "weights/model.zip") == present.resolve()
    for unsafe in ("../model.zip", str(present.resolve())):
        with pytest.raises(ClosureError, match="relative"):
            resolve_artifact_path(tmp_path, unsafe)


def test_pool_is_deterministic_stratified_and_excludes_missing_files(tmp_path: Path) -> None:
    weights = tmp_path / "weights" / "model.zip"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    available = _entry()
    available["sha256"] = canonical_sha256(weights.read_bytes())
    missing = _entry("missing-policy", availability="missing")
    missing["relative_or_external_location"] = "weights/missing.zip"

    first = build_validation_pool(
        repository=tmp_path,
        registry={"format": "rl-s5-registry-v1", "entries": [missing, available]},
    )
    second = build_validation_pool(
        repository=tmp_path,
        registry={"format": "rl-s5-registry-v1", "entries": [available, missing]},
    )

    assert first == second
    assert first["pool_sha256"] == second["pool_sha256"]
    assert [item["id"] for item in first["policies"]] == ["policy-1"]
    assert first["excluded"] == [{"id": "missing-policy", "reason": "artifact_not_local_present"}]
    assert first["strata"]["strong-learned"] == ["policy-1"]


@pytest.mark.parametrize("usage", ["public-bot", "hidden-evaluation", "diagnostic-only"])
def test_pool_preserves_usage_separation(tmp_path: Path, usage: str) -> None:
    weights = tmp_path / f"{usage}.zip"
    weights.write_bytes(usage.encode())
    entry = _entry(usage)
    entry["relative_or_external_location"] = weights.name
    entry["sha256"] = canonical_sha256(weights.read_bytes())
    entry["tags"] = ["pool:weak-learned", f"usage:{usage}"]

    pool = build_validation_pool(
        repository=tmp_path,
        registry={"format": "rl-s5-registry-v1", "entries": [entry]},
    )

    assert pool["policies"][0]["usage"] == usage


def test_reward_hacking_policy_cannot_be_promoted_to_public_bot(tmp_path: Path) -> None:
    weights = tmp_path / "hacker.zip"
    weights.write_bytes(b"hacker")
    entry = _entry("hacker")
    entry["relative_or_external_location"] = weights.name
    entry["sha256"] = canonical_sha256(weights.read_bytes())
    entry["tags"] = [
        "pool:specialized-ko",
        "usage:public-bot",
        "defect:reward-hacking",
    ]

    with pytest.raises(ClosureError, match="reward hacking"):
        build_validation_pool(
            repository=tmp_path,
            registry={"format": "rl-s5-registry-v1", "entries": [entry]},
        )


def test_pool_manifest_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    original = {"format": "beta-validation-pool-v1", "pool_sha256": SHA}
    write_immutable_json(path, original)
    write_immutable_json(path, original)

    with pytest.raises(ClosureError, match="immutable"):
        write_immutable_json(path, {**original, "pool_sha256": "b" * 64})


def test_zero_starter_is_safe_deterministic_and_has_zero_counters(tmp_path: Path) -> None:
    first = create_zero_starter(tmp_path / "first", seed=20260922)
    second = create_zero_starter(tmp_path / "second", seed=20260922)

    assert first["architecture_id"] == "mlp-compact-v2"
    assert first["training_units"] == 0
    assert first["learner_transitions"] == 0
    assert first["optimizer_steps"] == 0
    assert first["parent_checkpoint"] is None
    assert first["trained"] is False
    assert "reward" not in json.dumps(first).lower()
    assert first["files"]["weights.safetensors"] == second["files"]["weights.safetensors"]
    assert (tmp_path / "first" / "weights.safetensors").read_bytes() == (
        tmp_path / "second" / "weights.safetensors"
    ).read_bytes()
    with safe_open(tmp_path / "first" / "weights.safetensors", framework="pt") as tensors:
        assert tensors.keys()


def test_player_branches_share_only_weight_values_and_have_independent_rngs_and_optimizers(
    tmp_path: Path,
) -> None:
    create_zero_starter(tmp_path / "starter", seed=20260922)
    first = create_player_branch(tmp_path / "starter", branch_id="alice", rng_seed=101)
    second = create_player_branch(tmp_path / "starter", branch_id="bob", rng_seed=202)
    try:
        first_state = first.model.policy.state_dict()
        second_state = second.model.policy.state_dict()
        assert first.manifest["rng_seed"] != second.manifest["rng_seed"]
        assert first.model.policy.optimizer is not second.model.policy.optimizer
        assert first.model.policy.optimizer.state_dict()["state"] == {}
        assert second.model.policy.optimizer.state_dict()["state"] == {}
        assert all(
            np.array_equal(first_state[key].detach().numpy(), second_state[key].detach().numpy())
            for key in first_state
        )
        first_tensor = next(iter(first_state.values()))
        second_tensor = next(iter(second_state.values()))
        assert first_tensor.data_ptr() != second_tensor.data_ptr()
        observation, _ = first.environment.reset(seed=303)
        first.model.set_random_seed(404)
        second.model.set_random_seed(404)
        first_action, _ = first.model.predict(observation, deterministic=True)
        second_action, _ = second.model.predict(observation, deterministic=True)
        assert np.array_equal(first_action, second_action)
    finally:
        first.close()
        second.close()


def test_zero_starter_creation_is_idempotent(tmp_path: Path) -> None:
    directory = tmp_path / "starter-zero-v1"
    first = create_zero_starter(directory, seed=20260922)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    second = create_zero_starter(directory, seed=20260922)

    assert first == second
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_closure_is_idempotent_and_audits_cross_document_links(tmp_path: Path) -> None:
    retrospective = tmp_path / "docs" / "rl" / "RL-S5-retrospective.md"
    retrospective.parent.mkdir(parents=True)
    retrospective.write_text("# RL-S5 retrospective\n", encoding="utf-8")
    specification = tmp_path / "docs" / "dinorl-specification-rl.md"
    specification.write_text("RL-S5 closed; RL-L8 next.\n", encoding="utf-8")

    first = close_rl_s5(tmp_path)
    paths = [
        tmp_path / "artifacts" / "rl" / "RL-S5-registry.json",
        tmp_path / "artifacts" / "rl" / "beta-validation-pool-v1" / "manifest.json",
        tmp_path / "artifacts" / "rl" / "starter-zero-v1" / "manifest.json",
        tmp_path / "artifacts" / "rl" / "starter-zero-v1" / "weights.safetensors",
        tmp_path / "reports" / "rl" / "RL-S5-closure-report.md",
    ]
    before = {path: path.read_bytes() for path in paths}
    second = close_rl_s5(tmp_path)

    assert first == second
    assert before == {path: path.read_bytes() for path in paths}
    audit = audit_closure(tmp_path)
    assert audit["status"] == "RL_S5_CLOSED_READY_FOR_RL_L8"
    assert audit["new_training_or_benchmark"] is False
    assert audit["links_valid"] is True


def test_s5_closure_cli_exposes_close_audit_and_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    retrospective = tmp_path / "docs" / "rl" / "RL-S5-retrospective.md"
    retrospective.parent.mkdir(parents=True)
    retrospective.write_text("# RL-S5 retrospective\n", encoding="utf-8")
    (tmp_path / "docs" / "dinorl-specification-rl.md").write_text(
        "RL-S5 closed; RL-L8 next.\n", encoding="utf-8"
    )

    assert (
        main(
            [
                "s5",
                "close",
                "--repository",
                str(tmp_path),
                "--discover-artifacts",
                "--build-validation-pool",
                "--create-zero-starter",
                "--quiet",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "RL_S5_CLOSED_READY_FOR_RL_L8"
    for command in ("audit-closure", "closure-status"):
        assert main(["s5", command, "--repository", str(tmp_path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "RL_S5_CLOSED_READY_FOR_RL_L8"


@pytest.mark.parametrize(
    ("schema_name", "artifact"),
    [
        ("rl-s5-registry-v1.schema.json", "artifacts/rl/RL-S5-registry.json"),
        (
            "beta-validation-pool-v1.schema.json",
            "artifacts/rl/beta-validation-pool-v1/manifest.json",
        ),
        (
            "zero-starter-manifest-v1.schema.json",
            "artifacts/rl/starter-zero-v1/manifest.json",
        ),
    ],
)
def test_closure_artifacts_match_versioned_schemas(
    tmp_path: Path, schema_name: str, artifact: str
) -> None:
    retrospective = tmp_path / "docs" / "rl" / "RL-S5-retrospective.md"
    retrospective.parent.mkdir(parents=True)
    retrospective.write_text("# RL-S5 retrospective\n", encoding="utf-8")
    (tmp_path / "docs" / "dinorl-specification-rl.md").write_text(
        "RL-S5 closed; RL-L8 next.\n", encoding="utf-8"
    )
    close_rl_s5(tmp_path)
    root = Path(__file__).parents[3]
    schema = json.loads((root / "schemas" / "rl" / schema_name).read_text(encoding="utf-8"))
    value = json.loads((tmp_path / artifact).read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(value)) == []
