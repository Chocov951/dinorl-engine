"""Replay v1 recording and canonical serialization tests."""

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.match.replay import (
    ReplayRecorder,
    canonical_replay_json,
    replay_sha256,
)

ROOT = Path(__file__).parents[2]


def _finished_replay() -> dict[str, object]:
    env = DinoRLEnv("arena_mvp_v1", seed=42, replay=True)
    state = env.reset(first_actor=Actor.A)
    state.raptor(Actor.A).position = (1, 1)
    state.raptor(Actor.A).endurance = 1
    state.raptor(Actor.B).position = (1, 2)
    state.raptor(Actor.B).hp = 2
    recorder = ReplayRecorder(
        state,
        controller_a="aggressive-v1",
        controller_b="prudent-v1",
    )

    recorder.record_action(env.step(Action.REST), state)
    recorder.record_action(env.step(Action.END_TURN), state)
    recorder.record_action(env.step(Action.BITE), state)

    return recorder.finish(env.result)


def test_replay_records_ordered_complete_events_and_automatic_effects() -> None:
    replay = _finished_replay()
    events: list[dict[str, Any]] = replay["events"]  # type: ignore[assignment]

    assert [event["seq"] for event in events] == list(range(len(events)))
    assert [event["type"] for event in events] == [
        "match_started",
        "turn_started",
        "action_resolved",
        "turn_started",
        "action_resolved",
        "turn_started",
        "action_resolved",
        "match_ended",
    ]
    assert events[5]["automatic_effects"] == [
        {"type": "endurance_restored", "actor": "A", "amount": 4}
    ]
    assert events[6]["action"] == "BITE"
    assert events[6]["cost"] == {"movement": 1, "endurance": 2}
    assert events[6]["effects"] == [
        {"type": "damage_dealt", "actor": "A", "target": "B", "amount": 2}
    ]
    assert events[-1]["state"]["terminal"] is True
    assert all("state" in event for event in events)


def test_generated_replay_is_valid_against_the_versioned_schema() -> None:
    with (ROOT / "schemas" / "replay-v1.schema.json").open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)

    Draft202012Validator(schema).validate(_finished_replay())


def test_replay_serialization_is_canonical_utf8_and_hash_stable() -> None:
    replay = _finished_replay()
    reordered = dict(reversed(tuple(replay.items())))

    serialized = canonical_replay_json(replay)

    assert serialized == canonical_replay_json(reordered)
    assert serialized == canonical_replay_json(json.loads(serialized))
    assert b" " not in serialized
    assert replay_sha256(replay) == replay_sha256(reordered)
    assert len(replay_sha256(replay)) == 64


def test_replay_result_contains_only_the_public_contract_fields() -> None:
    replay = _finished_replay()

    assert replay["result"] == {
        "winner": "A",
        "reason": "ko",
        "rounds_completed": 1,
        "individual_turns": 3,
    }
