"""Flask application exposing the versioned DinoRL HTTP service."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.core.constants import (
    ENGINE_VERSION,
    MAP_ID,
    PROTOCOL_VERSION,
    REPLAY_VERSION,
    RULES_VERSION,
)
from dinorl_engine.match.replay import canonical_replay_json
from dinorl_engine.match.runner import run_match
from dinorl_engine.service.auth import configure_request_security
from dinorl_engine.service.errors import error_response
from dinorl_engine.service.schemas import request_validator

__all__ = ["create_app"]

MAX_RESPONSE_BYTES: Final = 512 * 1024

type JsonObject = dict[str, Any]


def _safe_request_id(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("request_id")
    return value if isinstance(value, str) and len(value) <= 36 else None


def _requests_unsupported_contract(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    supported = {
        "protocol_version": PROTOCOL_VERSION,
        "rules_version": RULES_VERSION,
        "map_id": MAP_ID,
    }
    return any(
        field in payload and payload[field] != expected for field, expected in supported.items()
    )


def _requests_unknown_controller(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    controllers = payload.get("controllers")
    if not isinstance(controllers, Mapping):
        return False
    for actor in ("A", "B"):
        descriptor = controllers.get(actor)
        if not isinstance(descriptor, Mapping):
            continue
        kind = descriptor.get("kind")
        controller_id = descriptor.get("id")
        if isinstance(kind, str) and kind != "scripted":
            return True
        if (
            kind == "scripted"
            and isinstance(controller_id, str)
            and controller_id not in SCRIPTED_CONTROLLER_IDS
        ):
            return True
    return False


def _parse_request_json() -> object:
    if request.mimetype != "application/json":
        raise BadRequest()
    return request.get_json(cache=False)


def create_app(configuration: Mapping[str, object] | None = None) -> Flask:
    """Build an isolated WSGI application."""

    app = Flask(__name__)
    if configuration is not None:
        app.config.update(configuration)
    configure_request_security(app)
    validator = request_validator()

    @app.get("/health")
    def health() -> tuple[Response, int]:
        return (
            jsonify(
                {
                    "status": "ok",
                    "engine_version": ENGINE_VERSION,
                    "protocol_versions": [PROTOCOL_VERSION],
                    "rules_versions": [RULES_VERSION],
                    "maps": [MAP_ID],
                }
            ),
            200,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error: RequestEntityTooLarge) -> tuple[Response, int]:
        return error_response(
            "request_too_large",
            "Request body exceeds 32 KiB.",
            413,
        )

    @app.post("/v1/matches/simulate")
    def simulate_match() -> tuple[Response, int] | Response:
        try:
            payload = _parse_request_json()
        except BadRequest:
            return error_response(
                "invalid_json",
                "Request body must be valid UTF-8 JSON.",
                400,
            )
        request_id = _safe_request_id(payload)
        if _requests_unsupported_contract(payload):
            return error_response(
                "unsupported_version",
                "Requested protocol, rules version, or map is not supported.",
                409,
                request_id=request_id,
            )
        if _requests_unknown_controller(payload):
            return error_response(
                "unknown_controller",
                "Requested controller is not supported.",
                422,
                request_id=request_id,
            )
        if list(validator.iter_errors(payload)):
            return error_response(
                "schema_validation_failed",
                "Request does not match simulate request v1.",
                422,
                request_id=request_id,
            )

        assert isinstance(payload, dict)
        controllers = payload["controllers"]
        try:
            outcome = run_match(
                map_id=payload["map_id"],
                seed=payload["seed"],
                controller_a_id=controllers["A"]["id"],
                controller_b_id=controllers["B"]["id"],
                include_replay=payload["include_replay"],
            )
            response_document: JsonObject = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": payload["request_id"],
                "engine_version": ENGINE_VERSION,
                "rules_version": RULES_VERSION,
                "map_id": MAP_ID,
                "replay_version": REPLAY_VERSION,
                "result": outcome.summary(),
                "replay_sha256": outcome.replay_sha256,
                "replay": outcome.replay,
            }
            serialized = canonical_replay_json(response_document)
            if len(serialized) > MAX_RESPONSE_BYTES:
                raise RuntimeError("complete response exceeded 512 KiB")
        except Exception:
            app.logger.exception("DinoRL simulation failed request_id=%s", request_id)
            return error_response(
                "simulation_failed",
                "Simulation could not be completed.",
                500,
                request_id=request_id,
            )
        return app.response_class(serialized, status=200, mimetype="application/json")

    return app
