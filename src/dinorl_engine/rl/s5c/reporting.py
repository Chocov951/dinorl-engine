"""Progress inspection and independent-audit report generation for RL-S5c-A2."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping
from pathlib import Path

from dinorl_engine.rl.s5c.rewards import ARCHETYPES

__all__ = ["campaign_status", "generate_report"]


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read A2 artifact: {path}") from error


def _seed_status(run: Path, archetype: str, seed: int, max_units: int) -> dict[str, object]:
    branch = run / "calibration" / archetype / f"seed-{seed}"
    state_path = branch / "calibration-state.json"
    cycles_path = branch / "cycles.json"
    state_value = _json(state_path) if state_path.is_file() else {}
    cycles_value = _json(cycles_path) if cycles_path.is_file() else []
    state = state_value if isinstance(state_value, Mapping) else {}
    cycles = cycles_value if isinstance(cycles_value, list) else []
    completed = max(
        (int(cycle.get("unit", 0)) for cycle in cycles if isinstance(cycle, Mapping)),
        default=0,
    )
    seconds = [
        float(cycle.get("training_seconds", 0.0)) + float(cycle.get("checkpoint_seconds", 0.0))
        for cycle in cycles
        if isinstance(cycle, Mapping)
    ]
    terminal = str(state.get("status", "not_started")) in {
        "first_stable_gate",
        "failed_budget",
        "failed_style",
        "failed_integrity",
    }
    eta = (
        0.0
        if terminal
        else (statistics.fmean(seconds) * (max_units - completed) if seconds else None)
    )
    latest_checkpoint = branch / "recovery" / "units" / f"unit-{completed}"
    return {
        "archetype": archetype,
        "seed": seed,
        "status": state.get("status", "not_started"),
        "completed_units": completed,
        "max_units": max_units,
        "progress_percent": round(100.0 * completed / max_units, 2),
        "first_stable_gate": state.get("first_stable_gate"),
        "consecutive_passes": state.get("consecutive_passes", 0),
        "latest_checkpoint": str(latest_checkpoint) if completed else None,
        "eta_seconds": eta,
        "error": None,
    }


def campaign_status(run: Path) -> dict[str, object]:
    """Read progress without modifying or loading any model checkpoint."""

    manifest_value = _json(run / "manifest.json")
    if not isinstance(manifest_value, Mapping):
        raise ValueError("A2 manifest is malformed")
    config = manifest_value.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("A2 manifest has no resolved configuration")
    seeds = config.get("calibration_seeds")
    max_units = config.get("max_units")
    if not isinstance(seeds, list) or type(max_units) is not int:
        raise ValueError("A2 manifest has invalid calibration settings")
    branches = [
        _seed_status(run, archetype, int(seed), max_units)
        for archetype in ARCHETYPES
        for seed in seeds
    ]
    completed = sum(
        str(branch["status"])
        in {"first_stable_gate", "failed_budget", "failed_style", "failed_integrity"}
        for branch in branches
    )
    eta_values = [
        float(branch["eta_seconds"]) for branch in branches if branch["eta_seconds"] is not None
    ]
    return {
        "format": "s5c-a2-status-v1",
        "run_id": manifest_value.get("run_id"),
        "branches": branches,
        "completed_branches": completed,
        "total_branches": len(branches),
        "eta_seconds": sum(eta_values) if len(eta_values) == len(branches) else None,
        "status": "completed" if completed == len(branches) else "running",
    }


def _load_optional(path: Path) -> object | None:
    return _json(path) if path.is_file() else None


def _evaluation_curve(root: Path) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    evaluation_root = root / "evaluations"
    if not evaluation_root.is_dir():
        return points
    for directory in evaluation_root.glob("unit-*"):
        try:
            unit = int(directory.name.removeprefix("unit-"))
        except ValueError:
            continue
        summary = _load_optional(directory / "summary.json")
        if isinstance(summary, Mapping):
            points.append(
                {
                    "unit": unit,
                    "competence": summary.get("gate"),
                    "style": summary.get("style_validation"),
                    "aggregates": summary.get("aggregates"),
                    "reward_hacking": summary.get("reward_hacking"),
                }
            )
    return sorted(points, key=lambda point: int(point["unit"]))


def _nested_mapping(value: object, *keys: str) -> Mapping[str, object]:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _comparative_style(branches: list[dict[str, object]]) -> dict[str, object]:
    by_seed: dict[int, dict[str, dict[str, object]]] = {}
    for branch in branches:
        final = branch.get("final_evaluation")
        if isinstance(final, Mapping):
            by_seed.setdefault(int(branch["seed"]), {})[str(branch["archetype"])] = branch
    comparisons: dict[str, object] = {}
    for seed, family in sorted(by_seed.items()):
        if set(family) != set(ARCHETYPES):
            comparisons[str(seed)] = {"passed": False, "reason": "incomplete_family"}
            continue
        style = {
            name: _nested_mapping(branch["final_evaluation"], "style_validation")
            for name, branch in family.items()
        }
        global_style = {
            name: _nested_mapping(
                branch["final_evaluation"], "aggregates", "global", "style_events"
            )
            for name, branch in family.items()
        }
        scavenger_over_predator = _number(style["scavenger"].get("value")) > _number(
            style["predator"].get("value")
        )
        predator_damage = _number(global_style["predator"].get("damage_dealt"))
        scavenger_damage = _number(global_style["scavenger"].get("damage_dealt"))
        controller_useful = _number(global_style["controller"].get("useful_shove"))
        other_useful = max(
            _number(global_style["scavenger"].get("useful_shove")),
            _number(global_style["predator"].get("useful_shove")),
        )
        checks = {
            "scavenger_point_rate_above_predator": scavenger_over_predator,
            "predator_damage_above_scavenger": predator_damage > scavenger_damage,
            "controller_useful_shoves_30_percent_above_others": (
                controller_useful > 1.30 * other_useful
            ),
        }
        comparisons[str(seed)] = {"checks": checks, "passed": all(checks.values())}
    return {"format": "s5c-a2-comparative-style-v1", "by_seed": comparisons}


def generate_report(run: Path) -> dict[str, object]:
    """Combine immutable provenance and all available calibration evidence."""

    status = campaign_status(run)
    manifest = _json(run / "manifest.json")
    branches: list[dict[str, object]] = []
    for branch_value in status["branches"]:  # type: ignore[index]
        if not isinstance(branch_value, Mapping):
            continue
        archetype = str(branch_value["archetype"])
        seed = int(branch_value["seed"])
        root = run / "calibration" / archetype / f"seed-{seed}"
        gate = branch_value.get("first_stable_gate")
        final_unit = gate if isinstance(gate, int) else branch_value["completed_units"]
        summary = _load_optional(root / "evaluations" / f"unit-{final_unit}" / "summary.json")
        branches.append(
            {
                **dict(branch_value),
                "final_evaluation": summary,
                "evaluation_curve": _evaluation_curve(root),
                "strength": _load_optional(root / "strength.json"),
                "diagnostic_replays": str(root / "evaluations"),
            }
        )
    report = {
        "format": "s5c-a2-final-report-v1",
        "manifest": manifest,
        "status": status,
        "branches": branches,
        "comparative_style": _comparative_style(branches),
        "performance_conclusion": (
            "available_for_independent_audit"
            if status["status"] == "completed"
            else "incomplete_server_campaign"
        ),
        "next_phase_started": False,
    }
    report_directory = run / "report"
    report_directory.mkdir(parents=True, exist_ok=True)
    json_path = report_directory / "rl-s5c-a2-report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# RL-S5c-A2 — rapport d’audit",
        "",
        f"- État : `{status['status']}`",
        f"- Branches terminales : {status['completed_branches']}/{status['total_branches']}",
        "- RL-S5c-B démarré : non",
        "",
        "| Archétype | Seed | État | Unités | Gate stable |",
        "|---|---:|---|---:|---:|",
    ]
    for branch in branches:
        lines.append(
            f"| {branch['archetype']} | {branch['seed']} | {branch['status']} | "
            f"{branch['completed_units']} | {branch['first_stable_gate']} |"
        )
    lines.extend(
        [
            "",
            "Les métriques détaillées, replays, hashes et agrégats sont "
            "référencés dans le rapport JSON.",
            "",
        ]
    )
    markdown_path = report_directory / "rl-s5c-a2-report.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "format": report["format"],
        "json": str(json_path),
        "markdown": str(markdown_path),
        "status": "completed",
    }
