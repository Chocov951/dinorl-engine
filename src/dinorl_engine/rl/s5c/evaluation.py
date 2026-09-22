"""Three-seed gate evaluation and auditable evidence persistence for RL-S5c-A2."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.rl.evaluation.deterministic import deterministic_game_specs
from dinorl_engine.rl.evaluation.gates import deterministic_gate
from dinorl_engine.rl.evaluation.match import run_evaluation_game
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec
from dinorl_engine.rl.evaluation.stochastic import paired_game_specs
from dinorl_engine.rl.rewards.runtime import CompiledReward
from dinorl_engine.rl.s5c.config import S5cConfig
from dinorl_engine.rl.s5c.evidence import (
    aggregate_game_records,
    classify_reward_hacking,
    select_diagnostic_records,
)
from dinorl_engine.rl.s5c.rewards import specialist_rewards

__all__ = [
    "evaluate_gate_checkpoint",
    "read_evaluation_evidence",
    "render_game_record",
    "write_evaluation_evidence",
]


class _GameResult(Protocol):
    record: Mapping[str, object]


type GameRunner = Callable[..., _GameResult]


def _game_specs(config: S5cConfig, evaluation_seed: int) -> tuple[EvaluationGameSpec, ...]:
    games: list[EvaluationGameSpec] = []
    for opponent_id in SCRIPTED_CONTROLLER_IDS:
        games.extend(deterministic_game_specs(seed=evaluation_seed, opponent_id=opponent_id))
    games.extend(
        paired_game_specs(
            seed=evaluation_seed,
            opponent_id=RANDOM_LEGAL_CONTROLLER_ID,
            confrontations=config.random_confrontations,
        )
    )
    return tuple(games)


def evaluate_gate_checkpoint(
    model: object,
    *,
    config: S5cConfig,
    archetype: str,
    training_seed: int,
    unit: int,
    previous_passed: bool = False,
    checkpoint_sha256: str | None = None,
    progress: Callable[[str], None] | None = None,
    game_runner: GameRunner = run_evaluation_game,
    reward_program: CompiledReward | None = None,
    safe_feed_episode_cap: float | None = None,
    evaluation_label: str = "S5c-A2",
) -> dict[str, object]:
    """Run all three configured evaluation seeds and compute one combined gate."""

    reward = specialist_rewards()[archetype] if reward_program is None else reward_program
    records: list[Mapping[str, object]] = []
    scheduled = tuple(
        specification
        for evaluation_seed in config.evaluation_seeds
        for specification in _game_specs(config, evaluation_seed)
    )
    total = len(scheduled)
    interval = max(1, total // 100)
    started = time.perf_counter()
    for completed, specification in enumerate(scheduled, start=1):
        arguments: dict[str, object] = {
            "specification": specification,
            "deterministic": True,
            "diagnostic_replay": False,
            "learner_policy_id": f"{archetype}-s{training_seed}-u{unit}",
            "checkpoint_id": f"unit-{unit}",
            "checkpoint_sha256": checkpoint_sha256,
            "configuration_sha256": config.resolved_sha256,
            "training_seed": training_seed,
            "reward_program": reward,
        }
        if safe_feed_episode_cap is not None:
            arguments["safe_feed_episode_cap"] = safe_feed_episode_cap
        result = game_runner(model, **arguments)
        records.append(dict(result.record))
        if progress is not None and (completed % interval == 0 or completed == total):
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            eta = (total - completed) / rate if rate > 0.0 else 0.0
            progress(
                f"{evaluation_label} evaluation {archetype} seed {training_seed} unit {unit}: "
                f"{completed}/{total} games ({100.0 * completed / total:.1f}%) "
                f"rate={rate:.2f} game/s ETA={eta:.0f}s"
            )
    opponent_records: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        opponent_records.setdefault(str(record["opponent_id"]), []).append(record)
    scores = {
        opponent: sum(float(record["score"]) for record in values) / len(values)
        for opponent, values in opponent_records.items()
    }
    gate = deterministic_gate(scores, previous_passed=previous_passed)
    return {
        "format": "s5c-a2-evaluation-v1",
        "archetype": archetype,
        "training_seed": training_seed,
        "unit": unit,
        "evaluation_seeds": list(config.evaluation_seeds),
        "records": records,
        "scores": scores,
        "gate": gate,
        "aggregates": aggregate_game_records(records),
        "reward_hacking": classify_reward_hacking(records),
    }


def render_game_record(record: Mapping[str, object]) -> str:
    """Render one machine game record as deterministic plain text."""

    lines = [
        f"game_id: {record.get('game_id')}",
        f"checkpoint: {record.get('checkpoint_id')}",
        f"checkpoint_sha256: {record.get('checkpoint_sha256')}",
        f"configuration_sha256: {record.get('configuration_sha256')}",
        f"reward_sha256: {record.get('reward_sha256')}",
        f"learner: {record.get('learner_policy_id')}",
        f"opponent: {record.get('opponent_id')}",
        (
            f"seat: side={record.get('learner_side')} role={record.get('learner_role')} "
            f"first={record.get('first_actor')}"
        ),
        (
            f"result: {record.get('outcome')} via {record.get('victory_mode')} "
            f"rounds={record.get('rounds')} score={record.get('score')}"
        ),
        (
            f"reward: terminal={record.get('terminal_return')} "
            f"auxiliary={record.get('auxiliary_return')}"
        ),
        "actions:",
    ]
    trace = record.get("action_trace")
    if isinstance(trace, list):
        lines.extend(f"  {index + 1}: {action}" for index, action in enumerate(trace))
    return "\n".join(lines) + "\n"


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_evaluation_evidence(directory: Path) -> dict[str, object] | None:
    """Recover a fully computed evaluation even if diagnostic rendering was interrupted."""

    summary_path = directory / "summary.json"
    games_path = directory / "games.jsonl"
    if not summary_path.is_file() or not games_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records = [
            json.loads(line) for line in games_path.read_text(encoding="utf-8").splitlines() if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("evaluation evidence is corrupt") from error
    if (
        not isinstance(summary, dict)
        or not records
        or not all(isinstance(record, dict) for record in records)
    ):
        raise ValueError("evaluation evidence is incomplete")
    return {**summary, "records": records}


def write_evaluation_evidence(
    directory: Path,
    evaluation: Mapping[str, object],
    *,
    diagnostic_replay_sample: int,
) -> None:
    """Persist complete game rows plus categorized machine/text diagnostics."""

    records_value = evaluation.get("records")
    if not isinstance(records_value, list) or not all(
        isinstance(record, Mapping) for record in records_value
    ):
        raise ValueError("evaluation records are missing")
    records = [dict(record) for record in records_value]
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    _atomic_bytes(directory / "games.jsonl", ("\n".join(lines) + "\n").encode("utf-8"))
    summary = {key: value for key, value in evaluation.items() if key != "records"}
    _atomic_bytes(
        directory / "summary.json",
        (
            json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8"),
    )
    selected = select_diagnostic_records(records, ordinary_sample=diagnostic_replay_sample)
    for category, category_records in selected.items():
        for index, record in enumerate(category_records):
            game_id = str(record["game_id"])
            short_id = hashlib.sha256(game_id.encode("utf-8")).hexdigest()[:20]
            base = directory / "diagnostics" / category / f"{index:04d}-{short_id}"
            _atomic_bytes(
                base.with_suffix(".json"),
                (
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8"),
            )
            _atomic_bytes(base.with_suffix(".txt"), render_game_record(record).encode("utf-8"))
