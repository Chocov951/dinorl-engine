"""JSON Schema v1 contract tests."""

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).parents[2]
SCHEMA_DIRECTORY = ROOT / "schemas"
FIXTURE_DIRECTORY = ROOT / "fixtures" / "contracts"
SCHEMA_FILES = (
    "simulate-request-v1.schema.json",
    "simulate-response-v1.schema.json",
    "replay-v1.schema.json",
)

type JsonObject = dict[str, Any]
type Mutation = Callable[[JsonObject], None]


def _read_json(path: Path) -> JsonObject:
    with path.open(encoding="utf-8") as json_file:
        value: JsonObject = json.load(json_file)
    return value


@pytest.fixture(scope="module")
def schemas() -> dict[str, JsonObject]:
    return {name: _read_json(SCHEMA_DIRECTORY / name) for name in SCHEMA_FILES}


@pytest.fixture(scope="module")
def validators(
    schemas: dict[str, JsonObject],
) -> dict[str, Draft202012Validator]:
    registry = Registry()
    for schema in schemas.values():
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return {
        name: Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())
        for name, schema in schemas.items()
    }


def _assert_valid(validator: Draft202012Validator, instance: JsonObject) -> None:
    assert list(validator.iter_errors(instance)) == []


def _assert_invalid(validator: Draft202012Validator, instance: JsonObject) -> None:
    assert list(validator.iter_errors(instance))


def test_schemas_are_valid_draft_2020_12(schemas: dict[str, JsonObject]) -> None:
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    ("schema_name", "fixture_name"),
    [
        ("simulate-request-v1.schema.json", "simulate-request-v1.valid.json"),
        ("simulate-response-v1.schema.json", "simulate-response-v1.valid.json"),
        ("replay-v1.schema.json", "replay-v1.valid.json"),
    ],
)
def test_valid_contract_fixtures_are_accepted(
    validators: dict[str, Draft202012Validator], schema_name: str, fixture_name: str
) -> None:
    _assert_valid(validators[schema_name], _read_json(FIXTURE_DIRECTORY / fixture_name))


def test_response_accepts_and_validates_an_embedded_replay(
    validators: dict[str, Draft202012Validator],
) -> None:
    response = _read_json(FIXTURE_DIRECTORY / "simulate-response-v1.valid.json")
    response["replay"] = _read_json(FIXTURE_DIRECTORY / "replay-v1.valid.json")
    response["replay_sha256"] = "0" * 64

    _assert_valid(validators["simulate-response-v1.schema.json"], response)

    response["replay"]["events"][0]["state"]["raptors"]["A"]["hp"] = 7
    _assert_invalid(validators["simulate-response-v1.schema.json"], response)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(protocol_version="2.0.0"),
        lambda value: value.update(rules_version="2.0.0"),
        lambda value: value.update(map_id="unknown-map"),
        lambda value: value.update(seed=-1),
        lambda value: value["controllers"].pop("B"),
        lambda value: value["controllers"]["A"].pop("id"),
        lambda value: value["controllers"]["A"].update(id="unknown-v1"),
        lambda value: value.update(hp=6),
    ],
)
def test_invalid_simulation_requests_are_rejected(
    validators: dict[str, Draft202012Validator], mutation: Mutation
) -> None:
    request = _read_json(FIXTURE_DIRECTORY / "simulate-request-v1.valid.json")
    mutation(request)
    _assert_invalid(validators["simulate-request-v1.schema.json"], request)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(protocol_version="2.0.0"),
        lambda value: value.update(rules_version="2.0.0"),
        lambda value: value.update(map_id="unknown-map"),
        lambda value: value.update(request_id="not-a-uuid"),
        lambda value: value.update(replay_sha256="not-a-sha256"),
        lambda value: value["result"].update(hp_a=7),
        lambda value: value["result"].update(winner="draw", reason="ko"),
    ],
)
def test_invalid_simulation_responses_are_rejected(
    validators: dict[str, Draft202012Validator], mutation: Mutation
) -> None:
    response = _read_json(FIXTURE_DIRECTORY / "simulate-response-v1.valid.json")
    mutation(response)
    _assert_invalid(validators["simulate-response-v1.schema.json"], response)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(replay_version="2.0.0"),
        lambda value: value.update(action_version="2.0.0"),
        lambda value: value.update(observation_version="2.0.0"),
        lambda value: value["controllers"]["B"].pop("kind"),
        lambda value: value["events"][0].update(extra=True),
        lambda value: value["events"][0].update(type="unknown"),
        lambda value: value["events"][0]["state"]["raptors"]["A"].update(hp=7),
        lambda value: value["events"][0]["state"].update(terminal=True),
        lambda value: value["events"][0]["state"].update(extra=True),
    ],
)
def test_invalid_replays_and_public_states_are_rejected(
    validators: dict[str, Draft202012Validator], mutation: Mutation
) -> None:
    replay = copy.deepcopy(_read_json(FIXTURE_DIRECTORY / "replay-v1.valid.json"))
    mutation(replay)
    _assert_invalid(validators["replay-v1.schema.json"], replay)


def test_replay_accepts_closed_local_debug_descriptors_without_paths(
    validators: dict[str, Draft202012Validator],
) -> None:
    replay = _read_json(FIXTURE_DIRECTORY / "replay-v1.valid.json")
    replay["controllers"] = {
        "A": {"kind": "manual", "id": "manual"},
        "B": {"kind": "sequence", "id": "interrupted-feed-a"},
    }
    _assert_valid(validators["replay-v1.schema.json"], replay)

    for invalid_id in ("../plan", "C:\\plan", "sequence:plan", "plan/name"):
        replay["controllers"]["B"]["id"] = invalid_id
        _assert_invalid(validators["replay-v1.schema.json"], replay)


@pytest.mark.parametrize(
    "descriptor",
    [
        {"kind": "manual", "id": "manual"},
        {"kind": "sequence", "id": "interrupted-feed-a"},
        {"kind": "sequence", "id": "fixtures/debug/plan.json"},
    ],
)
def test_simulation_request_keeps_debug_controllers_and_paths_outside_http(
    validators: dict[str, Draft202012Validator], descriptor: JsonObject
) -> None:
    request = _read_json(FIXTURE_DIRECTORY / "simulate-request-v1.valid.json")
    request["controllers"]["A"] = descriptor

    _assert_invalid(validators["simulate-request-v1.schema.json"], request)
