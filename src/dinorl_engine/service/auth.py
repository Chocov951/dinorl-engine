"""Bearer authentication and request-size enforcement for the HTTP service."""

from __future__ import annotations

import os
from hmac import compare_digest
from typing import Final

from flask import Flask, Response, request

from dinorl_engine.service.errors import error_response

__all__ = ["configure_request_security"]

CURRENT_TOKEN_CONFIG: Final = "DINORL_ENGINE_TOKEN"
PREVIOUS_TOKEN_CONFIG: Final = "DINORL_ENGINE_PREVIOUS_TOKEN"
MAX_REQUEST_BYTES: Final = 32 * 1024
SIMULATE_PATH: Final = "/v1/matches/simulate"
_DUMMY_TOKEN: Final = b"\0" * 32


def _validate_token(name: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value.encode("utf-8")) < 32:
        raise ValueError(f"{name} must contain at least 32 bytes")


def _presented_token() -> str | None:
    authorization = request.headers.get("Authorization")
    if authorization is None or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ")
    if not token or any(character.isspace() for character in token):
        return None
    return token


def _is_authorized(app: Flask) -> bool:
    presented = _presented_token()
    presented_bytes = presented.encode("utf-8") if presented is not None else b""
    current = app.config.get(CURRENT_TOKEN_CONFIG)
    previous = app.config.get(PREVIOUS_TOKEN_CONFIG)
    current_bytes = current.encode("utf-8") if isinstance(current, str) else _DUMMY_TOKEN
    previous_bytes = previous.encode("utf-8") if isinstance(previous, str) else _DUMMY_TOKEN
    current_matches = compare_digest(presented_bytes, current_bytes)
    previous_matches = compare_digest(presented_bytes, previous_bytes)
    return bool(
        (isinstance(current, str) and current_matches)
        | (isinstance(previous, str) and previous_matches)
    )


def configure_request_security(app: Flask) -> None:
    """Load token configuration and protect the simulation endpoint."""

    app.config.setdefault(CURRENT_TOKEN_CONFIG, os.environ.get(CURRENT_TOKEN_CONFIG))
    app.config.setdefault(PREVIOUS_TOKEN_CONFIG, os.environ.get(PREVIOUS_TOKEN_CONFIG))
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    _validate_token(CURRENT_TOKEN_CONFIG, app.config.get(CURRENT_TOKEN_CONFIG))
    _validate_token(PREVIOUS_TOKEN_CONFIG, app.config.get(PREVIOUS_TOKEN_CONFIG))

    @app.before_request
    def protect_simulation() -> tuple[Response, int] | None:
        if request.method != "POST" or request.path != SIMULATE_PATH:
            return None
        if request.content_length is not None and request.content_length > MAX_REQUEST_BYTES:
            return error_response(
                "request_too_large",
                "Request body exceeds 32 KiB.",
                413,
            )
        if not _is_authorized(app):
            return error_response(
                "unauthorized",
                "A valid bearer token is required.",
                401,
            )
        return None
