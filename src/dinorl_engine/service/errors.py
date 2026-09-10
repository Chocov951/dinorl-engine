"""Closed public error responses for the HTTP service."""

from __future__ import annotations

from flask import Response, jsonify

__all__ = ["error_response"]


def error_response(
    code: str,
    message: str,
    status: int,
    *,
    request_id: str | None = None,
) -> tuple[Response, int]:
    """Return the sole public API error shape without internal details."""

    return (
        jsonify(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": request_id,
                }
            }
        ),
        status,
    )
