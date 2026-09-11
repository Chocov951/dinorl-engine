"""Canonical serialization primitives shared by RL configuration and artifacts."""

import hashlib
import json
from collections.abc import Mapping

__all__ = ["canonical_json_bytes", "canonical_sha256"]


def canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    """Serialize a JSON object as compact, sorted, finite UTF-8 JSON."""

    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(document: Mapping[str, object]) -> str:
    """Return the lowercase SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()
