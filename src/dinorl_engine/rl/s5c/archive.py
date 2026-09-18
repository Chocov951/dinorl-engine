"""Lightweight, importable evidence for the RL-S5c-A calibration checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from dinorl_engine.rl.s5c.rewards import ARCHETYPES
from dinorl_engine.server_checks.manifests import create_archive

__all__ = ["S5cArchiveError", "create_calibration_archive"]


class S5cArchiveError(RuntimeError):
    """Raised when an incomplete calibration is about to be archived."""


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S5cArchiveError(f"cannot read calibration evidence: {path}") from error
    if not isinstance(value, dict):
        raise S5cArchiveError("calibration evidence must be a JSON object")
    return value


def create_calibration_archive(*, run_directory: Path, output_directory: Path) -> Path:
    """Package all three calibration reports without including policy weights."""

    campaign = _read(run_directory / "manifest.json")
    if campaign.get("format") != "s5c-run-manifest-v1":
        raise S5cArchiveError("RL-S5c campaign manifest has an unsupported format")
    run_id = campaign.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise S5cArchiveError("RL-S5c campaign has no run identifier")
    calibration = {
        archetype: _read(run_directory / "calibration" / archetype / "result.json")
        for archetype in ARCHETYPES
    }
    for archetype, result in calibration.items():
        seeds = result.get("seeds")
        if (
            result.get("format") != "s5c-calibration-v1"
            or result.get("archetype") != archetype
            or not isinstance(seeds, list)
            or len(seeds) != 3
        ):
            raise S5cArchiveError("RL-S5c calibration is incomplete")
    result = {
        "calibration": calibration,
        "campaign": campaign,
        "format": "dinorl-s5c-server-result-v1",
        "run_id": run_id,
        "suite": "RL-S5c-A",
    }
    summary = "# RL-S5c-A calibration\n\n- Three calibration seeds per archetype.\n"
    return create_archive(
        output_dir=output_directory,
        suite="RL-S5c-A",
        run_id=run_id,
        result=result,
        summary=summary,
    )
