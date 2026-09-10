"""Contract tests for the synchronous match simulation endpoint."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from flask.testing import FlaskClient
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from dinorl_engine.service.api import create_app

ROOT = Path(__file__).parents[2]
FIXTURE_PATH = ROOT / "fixtures" / "contracts" / "simulate-request-v1.valid.json"
SCHEMA_DIRECTORY = ROOT / "schemas"
TOKEN = "t" * 32
AUTHORIZATION = {"Authorization": f"Bearer {TOKEN}"}
SIMULATE_PATH = "/v1/matches/simulate"

type JsonObject = dict[str, Any]


@pytest.fixture
def client() -> FlaskClient:
    return create_app({"DINORL_ENGINE_TOKEN": TOKEN, "TESTING": True}).test_client()


@pytest.fixture
def valid_request() -> JsonObject:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def response_validator() -> Draft202012Validator:
    schemas: list[JsonObject] = []
    for path in SCHEMA_DIRECTORY.glob("*.json"):
        schemas.append(json.loads(path.read_text(encoding="utf-8")))
    registry = Registry()
    for schema in schemas:
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    response_schema = next(
        schema for schema in schemas if schema["title"] == "DinoRL simulate response v1"
    )
    return Draft202012Validator(
        response_schema,
        registry=registry,
        format_checker=FormatChecker(),
    )


def test_valid_request_returns_a_schema_valid_hashed_replay(
    client: FlaskClient,
    valid_request: JsonObject,
    response_validator: Draft202012Validator,
) -> None:
    response = client.post(SIMULATE_PATH, json=valid_request, headers=AUTHORIZATION)

    assert response.status_code == 200
    assert response.content_type == "application/json"
    assert len(response.data) <= 512 * 1024
    payload = response.get_json()
    assert list(response_validator.iter_errors(payload)) == []
    assert payload["request_id"] == valid_request["request_id"]
    assert payload["replay"] is not None
    canonical_replay = json.dumps(
        payload["replay"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert payload["replay_sha256"] == hashlib.sha256(canonical_replay).hexdigest()


def test_valid_request_can_explicitly_disable_replay(
    client: FlaskClient,
    valid_request: JsonObject,
    response_validator: Draft202012Validator,
) -> None:
    valid_request["include_replay"] = False

    response = client.post(SIMULATE_PATH, json=valid_request, headers=AUTHORIZATION)

    assert response.status_code == 200
    payload = response.get_json()
    assert list(response_validator.iter_errors(payload)) == []
    assert payload["replay"] is None
    assert payload["replay_sha256"] is None


@pytest.mark.parametrize(
    "body,content_type",
    [(b"{", "application/json"), (b"{}", "text/plain")],
)
def test_unreadable_json_or_wrong_mime_is_rejected(
    client: FlaskClient, body: bytes, content_type: str
) -> None:
    response = client.post(
        SIMULATE_PATH,
        data=body,
        content_type=content_type,
        headers=AUTHORIZATION,
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_json"


def test_custom_rule_parameter_is_rejected_by_the_closed_schema(
    client: FlaskClient, valid_request: JsonObject
) -> None:
    valid_request["starting_hp"] = 99

    response = client.post(SIMULATE_PATH, json=valid_request, headers=AUTHORIZATION)

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "schema_validation_failed"


@pytest.mark.parametrize(
    "descriptor",
    [
        {"kind": "scripted", "id": "unknown-v1"},
        {"kind": "manual", "id": "manual"},
        {"kind": "sequence", "id": "fixture-controller"},
        {"kind": "sequence", "id": "fixtures/debug/plan.json"},
    ],
)
def test_unknown_and_local_debug_controllers_are_rejected_over_http(
    client: FlaskClient,
    valid_request: JsonObject,
    descriptor: JsonObject,
) -> None:
    valid_request["controllers"]["A"] = descriptor

    response = client.post(SIMULATE_PATH, json=valid_request, headers=AUTHORIZATION)

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "unknown_controller"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", "2.0.0"),
        ("rules_version", "2.0.0"),
        ("map_id", "other-map"),
    ],
)
def test_unsupported_versions_and_map_return_conflict(
    client: FlaskClient,
    valid_request: JsonObject,
    field: str,
    value: str,
) -> None:
    valid_request[field] = value

    response = client.post(SIMULATE_PATH, json=valid_request, headers=AUTHORIZATION)

    assert response.status_code == 409
    assert response.get_json() == {
        "error": {
            "code": "unsupported_version",
            "message": "Requested protocol, rules version, or map is not supported.",
            "request_id": valid_request["request_id"],
        }
    }


def test_internal_simulation_error_is_not_disclosed(
    client: FlaskClient,
    valid_request: JsonObject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**_arguments: object) -> None:
        raise RuntimeError("sensitive local path C:/private/controller.py")

    monkeypatch.setattr("dinorl_engine.service.api.run_match", fail)

    response = client.post(SIMULATE_PATH, json=valid_request, headers=AUTHORIZATION)

    assert response.status_code == 500
    assert response.get_json()["error"] == {
        "code": "simulation_failed",
        "message": "Simulation could not be completed.",
        "request_id": valid_request["request_id"],
    }
    assert "private" not in response.get_data(as_text=True)


def test_complete_response_over_512_kib_fails_closed(
    client: FlaskClient,
    valid_request: JsonObject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized_outcome = SimpleNamespace(
        replay={"data": "x" * (512 * 1024)},
        replay_sha256="0" * 64,
        summary=lambda: {
            "winner": "A",
            "reason": "ko",
            "score_a": 0,
            "score_b": 0,
            "hp_a": 6,
            "hp_b": 0,
            "rounds_completed": 0,
            "individual_turns": 1,
            "actions": 1,
        },
    )
    monkeypatch.setattr(
        "dinorl_engine.service.api.run_match",
        lambda **_arguments: oversized_outcome,
    )

    response = client.post(SIMULATE_PATH, json=valid_request, headers=AUTHORIZATION)

    assert response.status_code == 500
    assert response.get_json()["error"]["code"] == "simulation_failed"


def test_every_schema_error_body_preserves_a_safe_request_id(
    client: FlaskClient, valid_request: JsonObject
) -> None:
    valid_request.pop("seed")

    response = client.post(SIMULATE_PATH, json=valid_request, headers=AUTHORIZATION)

    assert response.status_code == 422
    assert response.get_json()["error"]["request_id"] == valid_request["request_id"]
