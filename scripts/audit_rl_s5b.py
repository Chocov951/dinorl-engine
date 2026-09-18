"""Read-only RL-S5b artifact audit.

This intentionally consumes only lightweight JSON artifacts.  It never loads
or alters a model checkpoint.  Missing per-game telemetry is reported as such
instead of inferred from aggregate win/draw/loss counters.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _optional_read(path: Path, missing: list[str]) -> dict[str, Any] | None:
    """Read an optional artifact and retain a machine-readable absence reason."""

    if not path.is_file():
        missing.append(str(path))
        return None
    return _read(path)


def _score(value: dict[str, Any]) -> float:
    return float(value["score"])


def _counts(value: dict[str, Any]) -> dict[str, int]:
    return {name: int(value.get(name, 0)) for name in ("games", "wins", "draws", "losses")}


def _first_player_crossplay(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    dimensions: dict[str, dict[str, dict[str, int]]] = {
        "architecture": defaultdict(lambda: defaultdict(int)),
        "checkpoint": defaultdict(lambda: defaultdict(int)),
        "checkpoint_seed": defaultdict(lambda: defaultdict(int)),
        "evaluation_seed": defaultdict(lambda: defaultdict(int)),
        "matchup": defaultdict(lambda: defaultdict(int)),
    }
    for record in records:
        left, right = record["left"], record["right"]
        matchup = f"{left['checkpoint_id']}--{right['checkpoint_id']}"
        for position in record["positions"].values():
            is_left_first = position["left_actor"] == position["first_actor"]
            first = {
                "games": int(position["games"]),
                "wins": int(position["wins"] if is_left_first else position["losses"]),
                "draws": int(position["draws"]),
            }
            first["losses"] = first["games"] - first["wins"] - first["draws"]
            policy = left if is_left_first else right
            keys = {
                "architecture": str(policy["architecture"]),
                "checkpoint": str(policy["checkpoint_id"]),
                "checkpoint_seed": f"{policy['checkpoint_id']}|s{policy['seed']}",
                "evaluation_seed": str(record["evaluation_seed"]),
                "matchup": matchup,
            }
            for name, bucket in (("global", totals["global"]), *dimensions.items()):
                target = bucket if name == "global" else bucket[keys[name]]
                for metric, number in first.items():
                    target[metric] += number

    def finalise(values: dict[str, int]) -> dict[str, Any]:
        games = values["games"]
        return {
            **dict(values),
            "score": (values["wins"] + 0.5 * values["draws"]) / games,
            "advantage_vs_neutral": (values["wins"] + 0.5 * values["draws"]) / games - 0.5,
        }

    return {
        "global": finalise(dict(totals["global"])),
        **{
            name: {key: finalise(dict(value)) for key, value in sorted(bucket.items())}
            for name, bucket in dimensions.items()
        },
    }


def _evaluation_categories(path: Path) -> dict[str, Any]:
    evaluations = _read(path)
    output: dict[str, Any] = {}
    for raw_unit, suite in sorted(evaluations.items(), key=lambda item: int(item[0])):
        bots = suite["deterministic_bots"]["opponents"]
        random = bots["random-legal-v1"]
        scripted = [value for key, value in bots.items() if key != "random-legal-v1"]
        finalists = list(suite["frozen_finalists"].values())

        def aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
            counts = defaultdict(int)
            for value in values:
                for name, number in _counts(value).items():
                    counts[name] += number
            games = counts["games"]
            return {**dict(counts), "score": (counts["wins"] + 0.5 * counts["draws"]) / games}

        output[raw_unit] = {
            "random_legal": {**_counts(random), "score": _score(random)},
            "scripted_bots": aggregate(scripted),
            "frozen_finalists": aggregate(finalists),
            "all": aggregate([random, *scripted, *finalists]),
        }
    return output


def _branches(root: Path, phase: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for evaluations in sorted((root / phase).glob("*/evaluations.json")):
        result[evaluations.parent.name] = _evaluation_categories(evaluations)
    return result


def _exploiter_summary(result: dict[str, Any]) -> dict[str, Any]:
    data = result["results"]
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        by_target[str(item["target_id"])].append(item)
    return {
        target: {
            "runs": len(values),
            "best_exploiter_score": max(
                float(value["official_score"]["score"]) for value in values
            ),
            "total": {
                name: sum(int(value["official_score"][name]) for value in values)
                for name in ("games", "wins", "draws", "losses")
            },
            "by_exploiter_seed": {str(value["seed"]): value["official_score"] for value in values},
        }
        for target, values in sorted(by_target.items())
    }


def _historical_snapshot_manifest_errors(root: Path) -> list[dict[str, Any]]:
    """Identify, but never rewrite, obsolete self-play unit=32 manifests."""

    errors: list[dict[str, Any]] = []
    for path in sorted(root.glob("selfplay/*/league.json")):
        try:
            league = _read(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        snapshots = league.get("snapshots")
        if not isinstance(snapshots, list):
            continue
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            identifier = snapshot.get("id")
            unit = snapshot.get("unit")
            if isinstance(identifier, str) and "-snapshot-" in identifier and unit == 32:
                errors.append(
                    {
                        "manifest": str(path),
                        "snapshot_id": identifier,
                        "recorded_unit": 32,
                        "status": "historical_manifest_error_not_modified",
                    }
                )
    return errors


def build_audit(root: Path) -> dict[str, Any]:
    missing_files: list[str] = []
    if not root.is_dir():
        return {
            "format": "rl-s5b-turn-limit-audit-v1",
            "artifact_root": str(root),
            "status": "artifact_root_not_found",
            "required_next_step": (
                "Import or extract the audit archive so this directory exists, then rerun with "
                "--root pointing at the directory that contains crossplay/, pools/, and any "
                "available phase directories."
            ),
        }
    crossplay_path = root / "crossplay" / "crossplay.json"
    continuation_path = root / "continuation" / "result.json"
    selfplay_path = root / "selfplay" / "result.json"
    exploiters_path = root / "exploiters" / "result.json"
    measured_turn_limit_path = root / "turn-limit-results.json"
    crossplay = _optional_read(crossplay_path, missing_files)
    exploiters = _optional_read(exploiters_path, missing_files)
    measured_turn_limit = _optional_read(measured_turn_limit_path, missing_files)
    pool_paths = sorted((root / "pools").glob("*.json"))
    pool = _optional_read(pool_paths[0], missing_files) if pool_paths else None
    if pool is None and not pool_paths:
        missing_files.append(str(root / "pools" / "<pool-sha256>.json"))
    missing_pool_entries = (
        [
            {"id": entry["id"], "unit": entry.get("unit"), "provenance": entry["provenance"]}
            for entry in pool["entries"]
            if entry.get("status") == "missing"
        ]
        if pool is not None
        else []
    )
    replay_count = sum(1 for _ in root.rglob("*replay*"))
    historical_manifest_errors = _historical_snapshot_manifest_errors(root)
    return {
        "format": "rl-s5b-turn-limit-audit-v1",
        "artifact_root": str(root),
        "status": (
            "complete"
            if measured_turn_limit is not None
            else "complete"
            if not missing_files
            else "partial_artifact_set"
        ),
        "source_files": {
            "crossplay": str(crossplay_path),
            "continuation": str(continuation_path),
            "selfplay": str(selfplay_path),
            "exploiters": str(exploiters_path),
        },
        "crossplay": None
        if crossplay is None
        else {
            "records": int(crossplay["records_completed"]),
            "games": int(crossplay["games_completed"]),
            "first_player": _first_player_crossplay(crossplay["records"]),
        },
        "continuation": _branches(root, "continuation"),
        "selfplay": _branches(root, "selfplay"),
        "exploiters": None if exploiters is None else _exploiter_summary(exploiters),
        "turn_limit": {
            "status": (
                "measured_server_evaluation"
                if measured_turn_limit is not None
                else "not_identifiable_from_crossplay_or_finalist aggregates"
            ),
            "server_evaluation": measured_turn_limit,
            "replay_files_found": replay_count,
            "available_terminal_reason_source": "deterministic_bots.metrics.victory_route only",
            "unavailable": [
                "global and per-crossplay terminal causes",
                "duration histogram and quantiles",
                "state and final five turns at the limit",
                "replay classification",
                "paired 30/40/50/60 counterfactuals",
            ],
        },
        "missing_artifacts": {
            "required_json_files_not_found": missing_files,
            "result(1).json": "not present anywhere in the workspace",
            "replays": "no replay artifact found under the S5b run",
            "historical_checkpoints": missing_pool_entries,
            "selfplay_snapshot_unit_errors": historical_manifest_errors,
            "per_game_crossplay_terminal_telemetry": (
                "not stored; only aggregate W/D/L, rounds and actions"
            ),
            "exploiter_learning_curves": (
                "no evaluation curve artifact; recovery checkpoints are not "
                "performance measurements"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/rl/s5b/rl-s5b-default"))
    parser.add_argument("--output", type=Path, default=Path("turn-limit-audit.json"))
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(build_audit(args.root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
