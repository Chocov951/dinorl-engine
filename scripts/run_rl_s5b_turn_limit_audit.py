"""Server-only paired 30/40/50/60 evaluation for RL-S5b-AUDIT.

This program is deliberately evaluation-only: it loads immutable recovery
directories, hashes model weights before/after use, and writes a JSONL record
for every game.  Do not run it on a development workstation for final figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.protocol import derive_evaluation_seed
from dinorl_engine.rl.evaluation.tournament import run_policy_duel
from dinorl_engine.rl.s5b.crossplay import _load_policy, _sha256_model
from dinorl_engine.rl.s5b.pool import read_verified_pool


def planned_game_count(
    *, task_count: int, evaluation_seed_count: int, confrontations: int, limit_count: int
) -> int:
    """Count actual games, including both player sides and both initiatives."""

    if min(task_count, evaluation_seed_count, confrontations, limit_count) < 0:
        raise ValueError("audit game-count inputs must be non-negative")
    return (
        task_count * evaluation_seed_count * confrontations * len(Actor) * len(Actor) * limit_count
    )


def _checkpoint(identifier: str, directory: Path, *, unit: int, group: str) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "checkpoint",
        "provenance": str(directory.resolve()),
        "status": "available",
        "unit": unit,
        "group": group,
    }


def _catalog(
    root: Path, pool_path: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return unit-197 policies and exploiters paired with their declared target."""

    pool = read_verified_pool(pool_path)
    targets = {
        str(entry["id"]): entry
        for entry in pool["entries"]
        if isinstance(entry, dict)
        and entry.get("kind") == "checkpoint"
        and entry.get("unit") == 147
    }
    unit_197 = [
        _checkpoint(f"{path.parent.parent.parent.name}-unit-197", path, unit=197, group="unit-197")
        for path in sorted(root.glob("continuation/*/recovery/units/unit-197"))
        if all((path / name).is_file() for name in ("model.zip", "ppo_state.npz", "state.json"))
    ]
    exploiters: list[dict[str, object]] = []
    for result_path in sorted(root.glob("exploiters/*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        target_id = result.get("target_id")
        unit = result_path.parent / "recovery" / "units" / "unit-50"
        if (
            isinstance(target_id, str)
            and target_id in targets
            and all(
                (unit / name).is_file() for name in ("model.zip", "ppo_state.npz", "state.json")
            )
        ):
            entry = _checkpoint(result_path.parent.name, unit, unit=50, group="exploiter")
            entry["target_id"] = target_id
            exploiters.append(entry)
    return unit_197, exploiters


def _quantile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def _classify_round_limit(telemetry: dict[str, object]) -> str:
    """Deterministic first-pass classification; preserve the trace for review."""

    turns = telemetry["last_five_turns"]
    assert isinstance(turns, list)
    actions = [action for turn in turns for action in turn["actions"]]
    if "FEED" in actions:
        return "préparation_de_nutrition"
    if actions and set(actions) == {"END_TURN"}:
        return "blocage"
    if any(action.startswith("MOVE_") for action in actions):
        return "poursuite_active_ou_fuite_à_revoir"
    return "stratégie_encore_active_mais_lente"


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    by_limit: dict[str, dict[str, object]] = {}
    for limit in sorted({int(record["limit"]) for record in records}):
        group = [record for record in records if record["limit"] == limit]
        reasons = Counter(str(record["telemetry"]["terminal_reason"]) for record in group)
        durations = [int(record["telemetry"]["rounds"]) for record in group]
        first = [
            record for record in group if record["telemetry"]["policies"]["left"]["role"] == "first"
        ]
        roles = Counter(
            f"{side}_{policy['role']}"
            for record in group
            for side, policy in record["telemetry"]["policies"].items()
        )
        sides = Counter(
            str(policy["actor"])
            for record in group
            for policy in record["telemetry"]["policies"].values()
        )
        by_limit[str(limit)] = {
            "games": len(group),
            "terminal_rate": {name: count / len(group) for name, count in sorted(reasons.items())},
            "duration_histogram": dict(sorted(Counter(durations).items())),
            "duration_quantiles": {
                "p50": _quantile(durations, 0.50),
                "p90": _quantile(durations, 0.90),
                "p95": _quantile(durations, 0.95),
                "p99": _quantile(durations, 0.99),
            },
            "completed_between": {
                "31_40": sum(31 <= value <= 40 for value in durations),
                "41_50": sum(41 <= value <= 50 for value in durations),
                "51_60": sum(51 <= value <= 60 for value in durations),
            },
            "first_role_games": len(first),
            "realized_policy_role_counts": dict(sorted(roles.items())),
            "realized_policy_side_counts": dict(sorted(sides.items())),
        }
    # Same game_id is deliberately independent of cutoff, so these are paired conversions.
    paired: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for record in records:
        paired[str(record["game_id"])][int(record["limit"])] = record["telemetry"]
    conversions = Counter()
    for values in paired.values():
        at_30, at_60 = values.get(30), values.get(60)
        if at_30 and at_60 and at_30["terminal_reason"] == "round_limit":
            conversions[str(at_60["winner"])] += 1
    return {"limits": by_limit, "round_limit_30_to_60": dict(sorted(conversions.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limits", default="30,40,50,60")
    parser.add_argument(
        "--confrontations",
        type=int,
        default=5,
        help="paired confrontations retained per evaluation seed (default: 5)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress live progress written to stderr"
    )
    args = parser.parse_args()
    limits = tuple(int(value) for value in args.limits.split(","))
    if limits != (30, 40, 50, 60):
        raise ValueError("RL-S5b-AUDIT requires exactly 30,40,50,60")
    if args.output.exists():
        raise ValueError("output already exists; keep completed audit evidence immutable")
    pool = read_verified_pool(args.pool)
    pool_confrontations = int(pool["confrontations"])
    if not 1 <= args.confrontations <= pool_confrontations:
        raise ValueError(
            f"confrontations must be between 1 and the frozen-pool maximum ({pool_confrontations})"
        )
    finals = [
        entry
        for entry in pool["entries"]
        if isinstance(entry, dict)
        and entry.get("kind") == "checkpoint"
        and entry.get("unit") == 147
        and entry.get("status") == "available"
    ]
    unit_197, exploiters = _catalog(args.root, args.pool)
    tasks: list[tuple[dict[str, object], dict[str, object], str]] = [
        (left, right, "unit-147-crossplay")
        for left, right in combinations_with_replacement(finals, 2)
    ]
    tasks += [
        (candidate, target, "unit-197-v-final") for candidate in unit_197 for target in finals
    ]
    targets = {str(item["id"]): item for item in finals}
    tasks += [
        (candidate, targets[str(candidate["target_id"])], "exploiter-v-target")
        for candidate in exploiters
    ]
    total_games = planned_game_count(
        task_count=len(tasks),
        evaluation_seed_count=len(pool["evaluation_seeds"]),
        confrontations=args.confrontations,
        limit_count=len(limits),
    )
    completed_games = 0
    started = time.monotonic()

    def progress(*, force: bool = False) -> None:
        if args.quiet or (not force and completed_games % 100 != 0):
            return
        elapsed = time.monotonic() - started
        rate = completed_games / elapsed if elapsed else 0.0
        remaining = (total_games - completed_games) / rate if rate else 0.0
        print(
            f"RL-S5b-AUDIT: {completed_games:,}/{total_games:,} games "
            f"({completed_games / total_games:.1%}) | {rate:.2f} games/s | "
            f"ETA {remaining / 60:.1f} min",
            file=sys.stderr,
            flush=True,
        )

    progress(force=True)
    loaded: dict[str, Any] = {}
    records: list[dict[str, object]] = []
    try:
        for left, right, cohort in tasks:
            left_id, right_id = str(left["id"]), str(right["id"])
            if left_id not in loaded:
                loaded[left_id] = _load_policy(left)
            if right_id not in loaded:
                loaded[right_id] = _load_policy(right)
            before = {
                identifier: _sha256_model(policy.model) for identifier, policy in loaded.items()
            }
            matchup = f"{left_id}--{right_id}"
            for evaluation_seed in pool["evaluation_seeds"]:
                for confrontation in range(args.confrontations):
                    seed = derive_evaluation_seed(
                        int(evaluation_seed),
                        suite="s5b-turn-limit-audit-v1",
                        opponent_id=matchup,
                        index=confrontation,
                    )
                    for left_actor in Actor:
                        for first_actor in Actor:
                            game_id = hashlib.sha256(
                                f"{cohort}|{matchup}|{evaluation_seed}|{confrontation}|{left_actor.name}|{first_actor.name}".encode()
                            ).hexdigest()
                            for limit in limits:
                                outcome = run_policy_duel(
                                    loaded[left_id].model,
                                    loaded[right_id].model,
                                    seed=seed,
                                    left_actor=left_actor,
                                    first_actor=first_actor,
                                    deterministic=False,
                                    matchup_id=matchup,
                                    game_id=game_id,
                                    max_rounds=limit,
                                    left_policy_id=left_id,
                                    right_policy_id=right_id,
                                )
                                assert outcome.telemetry is not None
                                record: dict[str, object] = {
                                    "game_id": game_id,
                                    "cohort": cohort,
                                    "left_id": left_id,
                                    "right_id": right_id,
                                    "evaluation_seed": evaluation_seed,
                                    "engine_seed": seed,
                                    "confrontation": confrontation,
                                    "limit": limit,
                                    "telemetry": dict(outcome.telemetry),
                                }
                                telemetry = record["telemetry"]
                                assert isinstance(telemetry, dict)
                                if (
                                    telemetry["terminal_reason"] == "round_limit"
                                    and int(game_id[:8], 16) % 20 == 0
                                ):
                                    record["round_limit_trace_sample"] = {
                                        "classification": _classify_round_limit(telemetry),
                                        "id": game_id,
                                    }
                                records.append(record)
                                completed_games += 1
                                progress()
            if any(
                before[identifier] != _sha256_model(policy.model)
                for identifier, policy in loaded.items()
            ):
                raise RuntimeError("evaluation mutated an immutable checkpoint")
    finally:
        for policy in loaded.values():
            policy.close()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "games.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    summary = _summary(records)
    summary["configuration"] = {
        "confrontations_per_seed": args.confrontations,
        "evaluation_seeds": pool["evaluation_seeds"],
        "limits": list(limits),
        "pool_sha256": pool["pool_sha256"],
        "total_games": total_games,
    }
    (args.output / "turn-limit-results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    progress(force=True)


if __name__ == "__main__":
    main()
