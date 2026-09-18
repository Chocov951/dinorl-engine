"""Resumable RL-S5b-C self-play branches with a per-seed snapshot league."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.s5b.continuation import (
    _atomic_json,
    _configuration,
    _environment_factory,
    _evaluation,
    _finalists,
    _metadata,
    _parent_state,
    _restore_parent,
)
from dinorl_engine.rl.s5b.opponents import FrozenOpponentPool
from dinorl_engine.rl.s5b.pool import read_verified_pool
from dinorl_engine.rl.training.runner import AtomicPPOUnitRunner, AtomicRecoveryStore

__all__ = ["SelfPlayError", "run_self_play"]

_FORMAT: Final = "s5b-selfplay-v1"
_UNITS: Final = 50
_INTERVAL: Final = 10
_WEIGHTS: Final = {"random": 10, "scripted": 20, "final": 20, "historical": 50}


class SelfPlayError(RuntimeError):
    """Raised when a self-play branch violates its independent snapshot league."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(source: Path, destination: Path, *, identifier: str, unit: int) -> dict[str, object]:
    """Copy a validated immutable policy snapshot; never alias its mutable source."""

    files = ("model.zip", "ppo_state.npz", "state.json", "manifest.json")
    if not all((source / name).is_file() for name in files):
        raise SelfPlayError("self-play snapshot source is incomplete")
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {name: _sha256(source / name) for name in files}
    for name in files:
        target = destination / name
        if target.is_file() and _sha256(target) == hashes[name]:
            continue
        if target.exists():
            raise SelfPlayError("existing self-play snapshot does not match its source")
        shutil.copy2(source / name, target)
    return {
        "architecture": "selfplay-snapshot",
        "id": identifier,
        "kind": "checkpoint",
        # Preserve the real checkpoint unit while selecting the snapshot as a
        # historical opponent.  Older manifests used unit=32 as a category
        # surrogate; they are retained as erroneous historical evidence only.
        "s5b_category": "historical",
        "provenance": str(destination.resolve()),
        # Deliberately historical: the live learner can never become an opponent.
        "seed": None,
        "status": "available",
        "unit": unit,
        "artifacts": hashes,
    }


def _league_pool(
    benchmark: Mapping[str, object], snapshots: list[Mapping[str, object]]
) -> dict[str, object]:
    entries = benchmark.get("entries")
    if not isinstance(entries, list):
        raise SelfPlayError("benchmark pool has invalid entries")
    return {"entries": [*entries, *snapshots]}


def _run_branch(
    *,
    benchmark: Mapping[str, object],
    entry: Mapping[str, object],
    destination: Path,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    parent_directory, parent_state = _parent_state(entry)
    configuration = _configuration(parent_state)
    parent_id = entry.get("id")
    if not isinstance(parent_id, str):
        raise SelfPlayError("self-play parent identifier is invalid")
    branch = destination / "selfplay" / parent_id
    league_directory = branch / "league"
    league_path = branch / "league.json"
    try:
        league = json.loads(league_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        initial = _snapshot(
            parent_directory,
            league_directory / "unit-147",
            identifier=f"{parent_id}-snapshot-147",
            unit=147,
        )
        league = {"format": _FORMAT, "parent": parent_id, "snapshots": [initial]}
        _atomic_json(league_path, league)
    except json.JSONDecodeError as error:
        raise SelfPlayError("self-play league manifest is invalid") from error
    if (
        not isinstance(league, dict)
        or league.get("format") != _FORMAT
        or league.get("parent") != parent_id
        or not isinstance(league.get("snapshots"), list)
    ):
        raise SelfPlayError("self-play league manifest is incompatible")
    snapshots = league["snapshots"]
    if not all(isinstance(item, Mapping) for item in snapshots):
        raise SelfPlayError("self-play league snapshots are invalid")
    opponents = FrozenOpponentPool(_league_pool(benchmark, snapshots), category_weights=_WEIGHTS)
    metadata = _metadata(benchmark, entry, opponents, random_only=False)
    metadata["mode"] = "self-play"
    recovery = AtomicRecoveryStore(branch / "recovery")
    ledger = UnitLedger(branch / "ledger.sqlite3")
    run_id = f"s5b-c-selfplay-{hashlib.sha256(parent_id.encode()).hexdigest()[:16]}"
    ledger.reserve(run_id=run_id, units=_UNITS, idempotency_key=f"{run_id}-reserve")
    latest = recovery.load_latest()
    if latest is None:
        model, environment = _restore_parent(
            parent_directory, parent_state, configuration, opponents
        )
        runner = AtomicPPOUnitRunner(
            run_id=run_id,
            configuration=configuration,
            model=model,
            environment=environment,
            recovery_store=recovery,
            ledger=ledger,
            recovery_metadata=metadata,
        )
        completed = 0
    else:
        runner, _metrics, latest = AtomicPPOUnitRunner.restore_latest(
            run_id=run_id,
            configuration=configuration,
            recovery_store=recovery,
            ledger=ledger,
            environment_factory=_environment_factory(opponents),
            recovery_metadata=metadata,
        )
        completed = latest.sequence
    evaluations_path = branch / "evaluations.json"
    try:
        try:
            evaluations = json.loads(evaluations_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            evaluations = {}
        if not isinstance(evaluations, dict):
            raise SelfPlayError("self-play evaluations are invalid")
        if "147" not in evaluations:
            evaluations["147"] = _evaluation(
                runner.model, _finalists(benchmark), seed=configuration.seed
            )
            _atomic_json(evaluations_path, evaluations)
        for sequence in range(completed + 1, _UNITS + 1):
            result = runner.run_unit(unit_id=f"unit-{147 + sequence}", sequence=sequence)
            if progress is not None:
                progress(f"self-play {parent_id}: {sequence}/{_UNITS} units completed")
            if sequence % _INTERVAL:
                continue
            snapshot = _snapshot(
                result.recovery.directory,
                league_directory / f"unit-{147 + sequence}",
                identifier=f"{parent_id}-snapshot-{147 + sequence}",
                unit=147 + sequence,
            )
            snapshots.append(snapshot)
            _atomic_json(league_path, league)
            opponents.add_historical_snapshot(snapshot)
            evaluations[str(147 + sequence)] = _evaluation(
                runner.model, _finalists(benchmark), seed=configuration.seed + sequence
            )
            _atomic_json(evaluations_path, evaluations)
        summary = {
            "completed_units": _UNITS,
            "format": _FORMAT,
            "label": parent_id,
            "status": "completed",
        }
        _atomic_json(branch / "result.json", summary)
        return summary
    finally:
        runner.environment.close()
        opponents.close()


def run_self_play(
    *,
    pool_path: Path,
    output_directory: Path | None = None,
    units: int = _UNITS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run independent 50-unit self-play experiments for all Compact/Deep finalists."""

    if units != _UNITS:
        raise SelfPlayError("phase C self-play is fixed to 50 units pending result review")
    benchmark = read_verified_pool(pool_path)
    destination = pool_path.parent.parent if output_directory is None else output_directory
    branches = [
        _run_branch(benchmark=benchmark, entry=entry, destination=destination, progress=progress)
        for entry in _finalists(benchmark)
    ]
    result = {
        "branches": branches,
        "format": _FORMAT,
        "pool_sha256": benchmark["pool_sha256"],
        "status": "completed",
    }
    _atomic_json(destination / "selfplay" / "result.json", result)
    return result
