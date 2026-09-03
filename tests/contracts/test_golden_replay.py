"""Golden deterministic replay contract."""

import copy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from dinorl_engine.match.replay import canonical_replay_json, replay_sha256
from dinorl_engine.match.runner import MatchOutcome, run_match

ROOT = Path(__file__).parents[2]
FIXTURE_PATH = ROOT / "fixtures" / "matches" / "golden-v1.json"


def _fixture() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _run(configuration: dict[str, Any]) -> MatchOutcome:
    return run_match(
        map_id=configuration["map_id"],
        seed=configuration["seed"],
        controller_a_id=configuration["controllers"]["A"],
        controller_b_id=configuration["controllers"]["B"],
        include_replay=True,
    )


def test_identical_matches_repeat_and_match_the_golden_hash() -> None:
    configuration = _fixture()

    first = _run(configuration)
    second = _run(configuration)

    assert first.replay is not None
    assert second.replay is not None
    assert first.result == second.result
    assert canonical_replay_json(first.replay) == canonical_replay_json(second.replay)
    assert first.replay_sha256 == second.replay_sha256
    assert first.replay_sha256 == configuration["replay_sha256"]
    with (ROOT / "schemas" / "replay-v1.schema.json").open(encoding="utf-8") as schema_file:
        Draft202012Validator(json.load(schema_file)).validate(first.replay)


def test_golden_hash_detects_a_rules_trace_mutation() -> None:
    configuration = _fixture()
    outcome = _run(configuration)
    assert outcome.replay is not None
    mutated = copy.deepcopy(outcome.replay)
    events: list[dict[str, Any]] = mutated["events"]  # type: ignore[assignment]
    first_action = next(event for event in events if event["type"] == "action_resolved")
    first_action["cost"]["movement"] += 1

    assert replay_sha256(mutated) != configuration["replay_sha256"]
