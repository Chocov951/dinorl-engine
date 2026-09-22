"""Audit and complete only missing RL-S5c-A3-CONTROL measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.s5c.a3_config import A3Config
from dinorl_engine.rl.s5c.calibration import _atomic_json
from dinorl_engine.rl.s5c.strength import evaluate_against_strong_pool
from dinorl_engine.rl.training.unit import load_maskable_ppo


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crossplay_baseline(
    crossplay: dict[str, object], *, baseline_id: str, opponent_ids: set[str]
) -> float:
    records = crossplay.get("records")
    if not isinstance(records, list):
        raise ValueError("RL-S5b cross-play records are missing")
    scores: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        left, right, score = record.get("left"), record.get("right"), record.get("score")
        if (
            not isinstance(left, dict)
            or not isinstance(right, dict)
            or not isinstance(score, int | float)
        ):
            continue
        left_id, right_id = left.get("checkpoint_id"), right.get("checkpoint_id")
        if left_id == baseline_id and right_id in opponent_ids:
            scores.append(float(score))
        elif right_id == baseline_id and left_id in opponent_ids:
            scores.append(1.0 - float(score))
    if not scores:
        raise ValueError("no comparable RL-S5b cross-play baseline")
    return sum(scores) / len(scores)


def generate_control_report(config: A3Config) -> dict[str, object]:
    run = config.output_directory
    branch = run / "policy-control"
    checkpoint = branch / "recovery" / "units" / "unit-147"
    recovery_manifest = _json(checkpoint / "manifest.json")
    declared_files = recovery_manifest.get("files")
    if not isinstance(declared_files, dict):
        raise ValueError("unit-147 recovery manifest has no file hashes")
    actual_files = {name: _sha256(checkpoint / name) for name in declared_files}
    checkpoint_valid = actual_files == declared_files
    run_manifest = _json(run / "manifest.json")
    config_valid = run_manifest.get("config_sha256") == config.sha256
    split = _json(run / "opponent-split" / "rl-s5c-a3-opponent-split.json")
    heldout_ids_value = split.get("heldout_ids")
    if not isinstance(heldout_ids_value, list) or not all(
        isinstance(item, str) for item in heldout_ids_value
    ):
        raise ValueError("A3 held-out split is invalid")
    heldout_path = branch / "heldout-strength.json"
    previous_report_path = run / "report" / "rl-s5c-a3-control-report.json"
    previous_report = _json(previous_report_path) if previous_report_path.is_file() else {}
    previous_recalculated = previous_report.get("recalculated", [])
    recalculated = (
        [str(item) for item in previous_recalculated]
        if isinstance(previous_recalculated, list)
        else []
    )
    if not heldout_path.is_file():
        state = _json(checkpoint / "state.json")
        vector_value = state.get("configuration")
        if not isinstance(vector_value, dict):
            raise ValueError("unit-147 has no vector configuration")
        vector = VectorEnvironmentConfig(
            backend=VectorBackend(str(vector_value["backend"])),
            n_envs=int(vector_value["n_envs"]),
            seed=int(vector_value["seed"]),
        )
        environment = create_vector_environment(vector)
        try:
            model = load_maskable_ppo(checkpoint / "model.zip", environment)
            heldout = evaluate_against_strong_pool(
                model,
                pool_path=config.rl_s5b_pool,
                expected_sha256=config.rl_s5b_pool_sha256,
                evaluation_seeds=config.evaluation_seeds,
                confrontations=20,
                learner_policy_id="s5c-a3-policy-control-s20-u147-heldout",
                opponent_ids=tuple(heldout_ids_value),
            )
        finally:
            environment.close()
        _atomic_json(heldout_path, heldout)
        recalculated.append("heldout_rl_s5b_evaluation")
    heldout = _json(heldout_path)
    evaluations = json.loads((branch / "evaluations.json").read_text(encoding="utf-8"))
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("A3 control learning curve is missing")
    final_evaluation = _json(branch / "final-evaluation.json")
    strength = _json(branch / "strength.json")
    original_result = _json(branch / "result.json")
    crossplay = _json(run.parent.parent / "s5b" / "rl-s5b-default" / "crossplay" / "crossplay.json")
    baseline_id = "mlp-compact-v2-s20-unit-147-final"
    full_opponents = strength.get("by_opponent")
    if not isinstance(full_opponents, dict):
        raise ValueError("RL-S5b strength opponent aggregate is missing")
    baseline = _crossplay_baseline(
        crossplay, baseline_id=baseline_id, opponent_ids=set(full_opponents)
    )
    heldout_baseline = _crossplay_baseline(
        crossplay, baseline_id=baseline_id, opponent_ids=set(heldout_ids_value)
    )
    strong_score = float(strength["global"]["score"])  # type: ignore[index]
    heldout_score = float(heldout["global"]["score"])  # type: ignore[index]
    stochastic_gate = final_evaluation.get("stochastic_gate")
    stochastic_passed = isinstance(stochastic_gate, dict) and stochastic_gate.get("passed") is True
    historical_unit = original_result.get("historical_stable_gate")
    learning_gain = float(evaluations[-1]["scores"]["random-legal-v1"]) - float(
        evaluations[0]["scores"]["random-legal-v1"]
    )
    comparison_delta = strong_score - baseline
    passed = (
        checkpoint_valid
        and config_valid
        and isinstance(historical_unit, int)
        and learning_gain > 0.20
        and stochastic_passed
        and abs(comparison_delta) <= 0.15
    )
    decision = "CONTROL_PASSED" if passed else "FAILED_POLICY_CONTROL"
    report = {
        "format": "s5c-a3-control-report-v1",
        "decision": decision,
        "checkpoint": {
            "unit": 147,
            "directory": str(checkpoint),
            "files_sha256": actual_files,
            "valid": checkpoint_valid,
        },
        "provenance": {
            "config_sha256": config.sha256,
            "config_valid": config_valid,
            "rl_s5b_pool_sha256": config.rl_s5b_pool_sha256,
            "training_opponents_sha256": split["training_sha256"],
            "heldout_opponents_sha256": split["heldout_sha256"],
        },
        "learning_curve": evaluations,
        "learning_gain_random": learning_gain,
        "historical_stable_gate": historical_unit,
        "final_elementary_evaluation": final_evaluation["deterministic"],
        "stochastic_evaluation": final_evaluation["stochastic"],
        "stochastic_gate": stochastic_gate,
        "rl_s5b_full": strength,
        "rl_s5b_heldout": heldout,
        "rl_s5b_comparison": {
            "baseline_id": baseline_id,
            "full_pool_baseline_score": baseline,
            "full_pool_control_score": strong_score,
            "full_pool_delta": comparison_delta,
            "heldout_baseline_score": heldout_baseline,
            "heldout_control_score": heldout_score,
            "heldout_delta": heldout_score - heldout_baseline,
            "coherent_maximum_absolute_delta": 0.15,
        },
        "recalculated": recalculated,
        "training_restarted": False,
    }
    report_dir = run / "report"
    _atomic_json(report_dir / "rl-s5c-a3-control-report.json", report)
    global_full = strength["global"]  # type: ignore[index]
    global_heldout = heldout["global"]  # type: ignore[index]
    initial_random = float(evaluations[0]["scores"]["random-legal-v1"])
    final_random = float(evaluations[-1]["scores"]["random-legal-v1"])
    deterministic_scores = json.dumps(final_evaluation["deterministic"]["scores"], sort_keys=True)
    full_wdl = f"{global_full['wins']}/{global_full['draws']}/{global_full['losses']}"
    heldout_wdl = f"{global_heldout['wins']}/{global_heldout['draws']}/{global_heldout['losses']}"
    markdown = "\n".join(
        (
            "# RL-S5c-A3-CONTROL — rapport final",
            "",
            f"Décision : **{decision}**",
            "",
            f"- Checkpoint : unité 147, hash modèle `{actual_files['model.zip']}`",
            f"- Progression random : {initial_random:.4f} → {final_random:.4f}",
            f"- Premier gate historique stable : unité {historical_unit}",
            f"- Évaluation déterministe finale : `{deterministic_scores}`",
            f"- Gate stochastique : {'réussi' if stochastic_passed else 'échoué'}",
            f"- Pool RL-S5b complet : score {strong_score:.4f}, IC95 "
            f"{strength['score_ci95']}, V/N/D {full_wdl}, ROUND_LIMIT "
            f"{100 * float(global_full['round_limit_rate']):.2f} %",
            f"- Held-out RL-S5b : score {heldout_score:.4f}, IC95 "
            f"{heldout['score_ci95']}, V/N/D {heldout_wdl}, ROUND_LIMIT "
            f"{100 * float(global_heldout['round_limit_rate']):.2f} %",
            f"- Généraliste RL-S5b correspondant : {baseline:.4f}; "
            f"écart CONTROL {comparison_delta:+.4f} sur les six adversaires communs",
            f"- Baseline held-out comparable : {heldout_baseline:.4f}; "
            f"écart CONTROL {heldout_score - heldout_baseline:+.4f}",
            "- Entraînement relancé : non",
            f"- Mesures recalculées : {', '.join(recalculated) if recalculated else 'aucune'}",
            "",
        )
    )
    (report_dir / "rl-s5c-a3-control-report.md").write_text(markdown, encoding="utf-8")
    return report
