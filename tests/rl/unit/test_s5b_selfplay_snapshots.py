"""Snapshot manifests keep their physical checkpoint unit and league role."""

from pathlib import Path

from dinorl_engine.rl.s5b.opponents import FrozenOpponentPool
from dinorl_engine.rl.s5b.selfplay import _snapshot


def test_snapshot_manifest_preserves_unit_and_declares_historical_role(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "league" / "unit-197"
    source.mkdir()
    for name in ("model.zip", "ppo_state.npz", "state.json", "manifest.json"):
        (source / name).write_text(name, encoding="utf-8")

    snapshot = _snapshot(source, destination, identifier="compact-19-snapshot-197", unit=197)

    assert snapshot["unit"] == 197
    assert snapshot["s5b_category"] == "historical"


def test_unit_147_snapshot_is_historical_when_explicitly_declared() -> None:
    pool = FrozenOpponentPool(
        {
            "entries": [
                {
                    "id": "snapshot-147",
                    "kind": "checkpoint",
                    "unit": 147,
                    "s5b_category": "historical",
                    "status": "available",
                    "architecture": "selfplay-snapshot",
                    "provenance": "unused",
                }
            ]
        },
        category_weights={"random": 0, "scripted": 0, "final": 0, "historical": 1},
    )
    assert pool.available_categories["historical"] == ["snapshot-147"]
