"""Stable identifiers and canonical JSON helpers for RL contracts."""

from dinorl_engine.rl.contracts import canonical_json_bytes, canonical_sha256
from dinorl_engine.rl.versions import (
    CHECKPOINT_VERSION,
    OBSERVATION_VERSION,
    REWARD_CATALOG_VERSION,
    REWARD_DSL_VERSION,
    SNAPSHOT_VERSION,
)


def test_initial_rl_contract_identifiers_are_explicit_and_stable() -> None:
    assert OBSERVATION_VERSION == "rl-observation-v1"
    assert REWARD_DSL_VERSION == "reward-dsl-v1"
    assert REWARD_CATALOG_VERSION == "reward-catalog-v1"
    assert CHECKPOINT_VERSION == "training-checkpoint-v1"
    assert SNAPSHOT_VERSION == "policy-snapshot-v1"


def test_canonical_json_and_hash_are_order_independent_and_reject_non_finite_values() -> None:
    first = {"b": [2, 1], "a": "é"}
    reordered = {"a": "é", "b": [2, 1]}

    assert canonical_json_bytes(first) == b'{"a":"\xc3\xa9","b":[2,1]}'
    assert canonical_json_bytes(first) == canonical_json_bytes(reordered)
    assert canonical_sha256(first) == canonical_sha256(reordered)
