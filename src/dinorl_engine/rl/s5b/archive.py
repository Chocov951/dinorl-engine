"""Light-weight, importable server evidence for the RL-S5b-A checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from dinorl_engine.server_checks.manifests import create_archive

__all__ = ["ArchiveError", "create_crossplay_archive"]


class ArchiveError(RuntimeError):
    """Raised when a completed phase-A campaign cannot be packaged safely."""


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArchiveError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ArchiveError(f"{description} must be a JSON object")
    return value


def create_crossplay_archive(*, run_directory: Path, output_directory: Path) -> Path:
    """Package only phase-A evidence; checkpoint model files remain on the server."""

    campaign = _read_json(run_directory / "manifest.json", "campaign manifest")
    crossplay = _read_json(run_directory / "crossplay" / "crossplay.json", "cross-play result")
    phases = campaign.get("phases")
    if (
        campaign.get("format") != "s5b-run-manifest-v1"
        or not isinstance(phases, dict)
        or not isinstance(phases.get("cross-evaluate"), dict)
        or phases["cross-evaluate"].get("status") != "completed"
    ):
        raise ArchiveError("campaign has not completed RL-S5b-A cross-play")
    run_id = campaign.get("run_id")
    pool_sha256 = campaign.get("pool_sha256")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(pool_sha256, str)
        or crossplay.get("format") != "s5b-crossplay-v1"
        or crossplay.get("pool_sha256") != pool_sha256
    ):
        raise ArchiveError("campaign and cross-play evidence are incompatible")
    try:
        summary = (run_directory / "crossplay" / "crossplay-report.md").read_text(encoding="utf-8")
    except OSError as error:
        raise ArchiveError("cross-play report is missing") from error
    result: dict[str, object] = {
        "campaign": campaign,
        "crossplay": crossplay,
        "format": "dinorl-s5b-server-result-v1",
        "run_id": run_id,
        "suite": "RL-S5b-A",
    }
    return create_archive(
        output_dir=output_directory,
        suite="RL-S5b-A",
        run_id=run_id,
        result=result,
        summary=summary,
    )
