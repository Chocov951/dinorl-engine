"""Load the versioned JSON Schema contracts owned by the engine repository."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from referencing import Registry, Resource

__all__ = ["request_validator"]

type JsonObject = dict[str, Any]

_SCHEMA_DIRECTORY = Path(__file__).resolve().parents[3] / "schemas"
_SCHEMA_NAMES = (
    "simulate-request-v1.schema.json",
    "simulate-response-v1.schema.json",
    "replay-v1.schema.json",
)


def _read_schema(name: str) -> JsonObject:
    value: JsonObject = json.loads((_SCHEMA_DIRECTORY / name).read_text(encoding="utf-8"))
    return value


@lru_cache(maxsize=1)
def request_validator() -> Draft202012Validator:
    """Return the cached request validator with all local references registered."""

    schemas = {name: _read_schema(name) for name in _SCHEMA_NAMES}
    registry = Registry()
    for schema in schemas.values():
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return Draft202012Validator(
        schemas["simulate-request-v1.schema.json"],
        registry=registry,
        format_checker=FormatChecker(),
    )
