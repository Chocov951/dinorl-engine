"""Small, self-validating archive format for server gate results."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dinorl_engine.rl.contracts import canonical_json_bytes, canonical_sha256

__all__ = [
    "ArchiveValidationError",
    "ServerArchive",
    "create_archive",
    "read_archive",
]

_ARCHIVE_FORMAT: Final = "dinorl-server-results-v1"
_MEMBER_NAMES: Final = frozenset({"manifest.json", "result.json", "summary.md"})
_MAX_MEMBER_BYTES: Final = 2 * 1024 * 1024
_MAX_ARCHIVE_BYTES: Final = 8 * 1024 * 1024


class ArchiveValidationError(ValueError):
    """Raised when a server archive fails its structural integrity checks."""


@dataclass(frozen=True, slots=True)
class ServerArchive:
    """Validated light-weight contents of one gate archive."""

    manifest: dict[str, object]
    result: dict[str, object]
    summary: str
    sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: dict[str, object]) -> bytes:
    return canonical_json_bytes(value)


def _archive_manifest(
    *, suite: str, run_id: str, result_bytes: bytes, summary_bytes: bytes
) -> dict[str, object]:
    files = {
        "result.json": _sha256_bytes(result_bytes),
        "summary.md": _sha256_bytes(summary_bytes),
    }
    return {
        "format": _ARCHIVE_FORMAT,
        "suite": suite,
        "run_id": run_id,
        "files": files,
        "payload_sha256": canonical_sha256({"files": files, "suite": suite, "run_id": run_id}),
    }


def _tar_member(name: str, content: bytes) -> tuple[tarfile.TarInfo, io.BytesIO]:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mtime = int(time.time())
    return info, io.BytesIO(content)


def create_archive(
    *,
    output_dir: Path,
    suite: str,
    run_id: str,
    result: dict[str, object],
    summary: str,
) -> Path:
    """Write one compressed result archive without embedding model weights."""

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"server-results-{suite}-{run_id}.tar.gz"
    if archive_path.exists():
        raise FileExistsError(f"server archive already exists: {archive_path}")
    result_bytes = _json_bytes(result)
    summary_bytes = summary.encode("utf-8")
    manifest_bytes = _json_bytes(
        _archive_manifest(
            suite=suite,
            run_id=run_id,
            result_bytes=result_bytes,
            summary_bytes=summary_bytes,
        )
    )
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for name, content in (
            ("manifest.json", manifest_bytes),
            ("result.json", result_bytes),
            ("summary.md", summary_bytes),
        ):
            info, data = _tar_member(name, content)
            archive.addfile(info, data)
    return archive_path


def _read_json_member(members: dict[str, bytes], name: str) -> dict[str, object]:
    try:
        value = json.loads(members[name].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveValidationError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ArchiveValidationError(f"{name} must contain a JSON object")
    return value


def read_archive(path: Path) -> ServerArchive:
    """Read an archive while rejecting traversal, extra files, and bad hashes."""

    try:
        if path.stat().st_size > _MAX_ARCHIVE_BYTES:
            raise ArchiveValidationError("archive exceeds the maximum allowed size")
    except OSError as error:
        raise ArchiveValidationError("archive cannot be read") from error
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = {member.name for member in members}
            if names != _MEMBER_NAMES or len(members) != len(_MEMBER_NAMES):
                raise ArchiveValidationError(
                    "archive must contain exactly its three expected members"
                )
            contents: dict[str, bytes] = {}
            for member in members:
                if not member.isfile() or member.size > _MAX_MEMBER_BYTES:
                    raise ArchiveValidationError(f"invalid archive member: {member.name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ArchiveValidationError(f"unable to read archive member: {member.name}")
                contents[member.name] = extracted.read()
    except (OSError, tarfile.TarError) as error:
        raise ArchiveValidationError("archive is not a readable gzip tarball") from error

    manifest = _read_json_member(contents, "manifest.json")
    result = _read_json_member(contents, "result.json")
    summary_bytes = contents["summary.md"]
    try:
        summary = summary_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArchiveValidationError("summary.md is not valid UTF-8") from error
    files = manifest.get("files")
    suite = manifest.get("suite")
    run_id = manifest.get("run_id")
    if (
        set(manifest) != {"format", "suite", "run_id", "files", "payload_sha256"}
        or manifest.get("format") != _ARCHIVE_FORMAT
        or not isinstance(files, dict)
        or not isinstance(suite, str)
        or not isinstance(run_id, str)
    ):
        raise ArchiveValidationError("archive manifest has an invalid format")
    expected_files = {
        "result.json": _sha256_bytes(contents["result.json"]),
        "summary.md": _sha256_bytes(summary_bytes),
    }
    if files != expected_files:
        raise ArchiveValidationError("archive member SHA-256 does not match the manifest")
    if manifest.get("payload_sha256") != canonical_sha256(
        {"files": expected_files, "suite": suite, "run_id": run_id}
    ):
        raise ArchiveValidationError("archive payload SHA-256 does not match the manifest")
    return ServerArchive(
        manifest=manifest,
        result=result,
        summary=summary,
        sha256=_sha256_bytes(path.read_bytes()),
    )
