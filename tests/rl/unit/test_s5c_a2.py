"""RL-S5c-A2 reproducibility, state, telemetry, and CLI contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dinorl_engine.rl.__main__ import _build_parser
from dinorl_engine.rl.evaluation.protocol import derive_game_stream_seed
from dinorl_engine.rl.evaluation.stochastic import paired_game_specs
from dinorl_engine.rl.s5c.calibration import CalibrationError, CalibrationSeedState
from dinorl_engine.rl.s5c.config import S5cConfig
from dinorl_engine.rl.s5c.evaluation import (
    evaluate_gate_checkpoint,
    read_evaluation_evidence,
    write_evaluation_evidence,
)
from dinorl_engine.rl.s5c.evidence import (
    aggregate_game_records,
    classify_reward_hacking,
    controller_zero_shove_failure,
    select_diagnostic_records,
    validate_absolute_style,
)
from dinorl_engine.rl.s5c.provenance import (
    ProvenanceError,
    initialization_record,
    verify_worktree_policy,
)
from dinorl_engine.rl.s5c.reporting import campaign_status
from dinorl_engine.rl.s5c.rewards import specialist_reward_terms, specialist_rewards
from dinorl_engine.rl.s5c.strength import EXPECTED_RL_S5B_POOL_SHA256, verify_strong_pool


def _mapping(tmp_path: Path) -> dict[str, object]:
    return {
        "format": "dinorl-s5c-a2-config-v1",
        "phase": "RL-S5c-A2",
        "run_id": "rl-s5c-calibration-v2",
        "output_directory": str(tmp_path / "rl-s5c-calibration-v2"),
        "architecture": "mlp-compact-v2",
        "max_rounds": 30,
        "training_opponent": "random-legal-v1",
        "calibration_seeds": [19, 20, 21],
        "production_seeds": [19, 20, 21, 22, 23],
        "evaluation_seeds": [101, 102, 103],
        "max_units": 147,
        "rl_s5b_pool": str(tmp_path / "pool.json"),
        "rl_s5b_pool_sha256": EXPECTED_RL_S5B_POOL_SHA256,
        "random_confrontations": 2,
        "strong_pool_confrontations": 2,
        "diagnostic_replay_sample": 2,
    }


def test_a2_configuration_is_distinct_and_executes_all_three_evaluation_seeds(
    tmp_path: Path,
) -> None:
    config = S5cConfig.from_mapping(_mapping(tmp_path))

    assert config.phase == "RL-S5c-A2"
    assert config.run_id == "rl-s5c-calibration-v2"
    assert config.output_directory.name == "rl-s5c-calibration-v2"
    assert config.calibration_seeds == (19, 20, 21)
    assert config.evaluation_seeds == (101, 102, 103)
    assert config.rl_s5b_pool_sha256 == EXPECTED_RL_S5B_POOL_SHA256


def test_server_worktree_is_rejected_by_default_and_development_override_is_visible() -> None:
    with pytest.raises(ProvenanceError, match="dirty"):
        verify_worktree_policy(dirty=True, allow_dirty=False)

    assert verify_worktree_policy(dirty=True, allow_dirty=True) == (
        "WARNING: dirty worktree explicitly allowed; full content hash recorded"
    )
    assert verify_worktree_policy(dirty=False, allow_dirty=False) is None


class _Policy:
    def __init__(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.network = torch.nn.Linear(4, 3)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.network.state_dict()


def _fake_model(seed: int) -> SimpleNamespace:
    return SimpleNamespace(policy=_Policy(seed))


def test_initialization_hashes_are_common_by_seed_and_distinct_between_seeds() -> None:
    first = initialization_record(_fake_model(19), seed=19)
    repeated = initialization_record(_fake_model(19), seed=19)
    different = initialization_record(_fake_model(20), seed=20)

    assert first["weights_sha256"] == repeated["weights_sha256"]
    assert first["weights_sha256"] != different["weights_sha256"]
    assert first["optimizer_state_entries"] == 0
    assert first["parent_checkpoint"] is None
    assert first["provenance"] == "fresh-maskable-ppo"


def test_calibration_states_include_failures_and_resume_rules() -> None:
    state = CalibrationSeedState(archetype="predator", seed=19, max_units=147)
    assert state.status == "training"
    assert state.can_resume is True

    state.record_evaluation(unit=5, thresholds_passed=True)
    assert state.status == "first_gate_pass"
    assert state.last_evaluation_unit == 5
    assert state.can_resume is False
    state.record_evaluation(unit=10, thresholds_passed=False)
    assert state.status == "training"

    state.mark_failed_budget(unit=147)
    assert state.status == "failed_budget"
    assert state.can_resume is False
    with pytest.raises(CalibrationError, match="failed_budget"):
        state.record_evaluation(unit=145, thresholds_passed=True)


def test_controller_integrity_failure_requires_three_zero_attempt_seeds_at_units_five_and_ten() -> (
    None
):
    records = {
        19: {5: (0, 4), 10: (0, 7)},
        20: {5: (0, 3), 10: (0, 8)},
        21: {5: (0, 2), 10: (0, 5)},
    }

    assert controller_zero_shove_failure(records) is True
    records[21][10] = (1, 5)
    assert controller_zero_shove_failure(records) is False


def _record(
    game_id: str,
    *,
    outcome: str,
    role: str,
    side: str,
    evaluation_seed: int,
    mode: str,
    auxiliary: float,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "training_seed": 19,
        "evaluation_seed": evaluation_seed,
        "opponent_id": "prudent-v1",
        "learner_role": role,
        "learner_side": side,
        "outcome": outcome,
        "victory_mode": mode,
        "score": {"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome],
        "rounds": 30 if mode == "round_limit" else 8,
        "terminal_return": {"win": 1.0, "draw": 0.0, "loss": -1.0}[outcome],
        "auxiliary_return": auxiliary,
        "auxiliary_by_rule": {"damage": auxiliary},
        "style_events": {"shove_attempted": 1 if game_id == "win" else 0},
        "action_distribution": {"shove": 1 if game_id == "win" else 0},
        "events": [],
    }


def test_evaluation_aggregates_cover_seed_role_side_outcome_and_victory_mode() -> None:
    records = [
        _record(
            "win",
            outcome="win",
            role="first",
            side="A",
            evaluation_seed=101,
            mode="ko",
            auxiliary=0.2,
        ),
        _record(
            "draw",
            outcome="draw",
            role="second",
            side="B",
            evaluation_seed=102,
            mode="round_limit",
            auxiliary=0.1,
        ),
        _record(
            "loss",
            outcome="loss",
            role="first",
            side="B",
            evaluation_seed=103,
            mode="carcass_score",
            auxiliary=0.8,
        ),
    ]

    aggregates = aggregate_game_records(records)

    assert aggregates["global"]["wins"] == 1
    assert set(aggregates["by_evaluation_seed"]) == {"101", "102", "103"}
    assert set(aggregates["by_role"]) == {"first", "second"}
    assert set(aggregates["by_side"]) == {"A", "B"}
    assert set(aggregates["by_outcome"]) == {"win", "draw", "loss"}
    assert set(aggregates["by_victory_mode"]) == {"ko", "round_limit", "carcass_score"}


def test_reward_hacking_and_replay_selection_cover_required_extremes() -> None:
    records = [
        _record(
            "win",
            outcome="win",
            role="first",
            side="A",
            evaluation_seed=101,
            mode="ko",
            auxiliary=0.2,
        ),
        _record(
            "loss",
            outcome="loss",
            role="second",
            side="B",
            evaluation_seed=101,
            mode="carcass_score",
            auxiliary=3.0,
        ),
        _record(
            "round-limit",
            outcome="draw",
            role="second",
            side="A",
            evaluation_seed=101,
            mode="round_limit",
            auxiliary=0.0,
        ),
    ]

    report = classify_reward_hacking(records)
    selected = select_diagnostic_records(records, ordinary_sample=1)

    assert "loss_total_exceeds_win" in report["warnings"]
    assert "round_limit_above_10_percent" in report["warnings"]
    assert selected["highest_auxiliary"][0]["game_id"] == "loss"
    assert selected["loss_high_auxiliary"][0]["game_id"] == "loss"
    assert selected["round_limit"][0]["game_id"] == "round-limit"


def test_paired_games_have_persistent_ids_same_map_seed_and_only_invert_first_player() -> None:
    games = paired_game_specs(seed=101, opponent_id="random-legal-v1", confrontations=2)

    for first, second in zip(games[::2], games[1::2], strict=True):
        assert first.game_id and second.game_id and first.game_id != second.game_id
        assert first.seed == second.seed
        assert first.learner_actor == second.learner_actor
        assert first.first_actor != second.first_actor
        assert derive_game_stream_seed(first.game_id, policy_id="learner") == (
            derive_game_stream_seed(first.game_id, policy_id="learner")
        )


def test_strong_pool_rejects_any_digest_other_than_the_frozen_s5b_hash(tmp_path: Path) -> None:
    path = tmp_path / "pool.json"
    path.write_text('{"pool_sha256":"wrong"}', encoding="utf-8")

    with pytest.raises(ValueError, match="frozen RL-S5b"):
        verify_strong_pool(path, expected_sha256=EXPECTED_RL_S5B_POOL_SHA256)


def test_controller_reward_revision_is_event_only_and_scavenger_is_unchanged() -> None:
    rewards = specialist_rewards()

    assert "return 0.12" in rewards["controller"].source
    assert "shove_moved_target" in rewards["controller"].source
    assert "shove_attempted" not in rewards["controller"].source
    assert "0.15 * carcass_points_gained" in rewards["scavenger"].source
    assert "0.10 * damage_dealt" in rewards["predator"].source
    assert "0.03 * damage_dealt" in rewards["predator"].source


def test_cli_parses_a2_preflight_resumable_calibration_status_and_report() -> None:
    parser = _build_parser()
    commands = (
        ["s5c", "preflight", "--config", "configs/rl/s5c-a2.yaml", "--allow-dirty"],
        [
            "s5c",
            "calibrate",
            "--config",
            "configs/rl/s5c-a2.yaml",
            "--run-id",
            "rl-s5c-calibration-v2",
            "--resume",
        ],
        ["s5c", "status", "--run", "artifacts/rl/s5c/rl-s5c-calibration-v2"],
        ["s5c", "report", "--run", "artifacts/rl/s5c/rl-s5c-calibration-v2"],
    )

    assert [parser.parse_args(command).s5c_command for command in commands] == [
        "preflight",
        "calibrate",
        "status",
        "report",
    ]


def test_resolved_configuration_hash_changes_with_any_protocol_input(tmp_path: Path) -> None:
    config = S5cConfig.from_mapping(_mapping(tmp_path))
    original = config.resolved_sha256
    changed = _mapping(tmp_path)
    changed["diagnostic_replay_sample"] = 3

    assert len(original) == 64
    assert original == hashlib.sha256(config.canonical_bytes).hexdigest()
    assert S5cConfig.from_mapping(changed).resolved_sha256 != original


def test_gate_evaluation_really_runs_every_configured_evaluation_seed(tmp_path: Path) -> None:
    config = S5cConfig.from_mapping(_mapping(tmp_path))
    seen: list[int] = []
    progress: list[str] = []

    def fake_game(*_: object, specification: object, **__: object) -> SimpleNamespace:
        evaluation_seed = specification.evaluation_seed  # type: ignore[attr-defined]
        seen.append(evaluation_seed)
        opponent = specification.opponent_id  # type: ignore[attr-defined]
        game_id = specification.game_id  # type: ignore[attr-defined]
        return SimpleNamespace(
            record={
                **_record(
                    game_id,
                    outcome="win",
                    role="first",
                    side="A",
                    evaluation_seed=evaluation_seed,
                    mode="ko",
                    auxiliary=0.0,
                ),
                "opponent_id": opponent,
            }
        )

    result = evaluate_gate_checkpoint(
        object(),
        config=config,
        archetype="predator",
        training_seed=19,
        unit=5,
        progress=progress.append,
        game_runner=fake_game,
    )

    assert set(seen) == {101, 102, 103}
    assert result["evaluation_seeds"] == [101, 102, 103]
    assert result["records"]
    assert result["gate"]["thresholds_passed"] is True
    assert "48/48 games" in progress[-1]
    assert "ETA=" in progress[-1]


def test_evaluation_evidence_writes_machine_and_text_replays(tmp_path: Path) -> None:
    record = _record(
        "diagnostic/game-1",
        outcome="loss",
        role="second",
        side="B",
        evaluation_seed=101,
        mode="round_limit",
        auxiliary=2.0,
    )
    record.update(
        {
            "checkpoint_id": "unit-5",
            "learner_policy_id": "predator-s19-u5",
            "first_actor": "A",
            "action_trace": ["A:MOVE_NORTH", "B:END_TURN"],
        }
    )
    directory = tmp_path / "evaluation"

    write_evaluation_evidence(
        directory,
        {"format": "s5c-a2-evaluation-v1", "records": [record]},
        diagnostic_replay_sample=1,
    )

    assert (directory / "games.jsonl").is_file()
    replay_json = next((directory / "diagnostics" / "round_limit").glob("*.json"))
    replay_text = replay_json.with_suffix(".txt")
    assert json.loads(replay_json.read_text(encoding="utf-8"))["game_id"] == "diagnostic/game-1"
    assert "checkpoint: unit-5" in replay_text.read_text(encoding="utf-8")
    recovered = read_evaluation_evidence(directory)
    assert recovered is not None
    assert recovered["records"][0]["game_id"] == "diagnostic/game-1"


def test_replay_filenames_are_bounded_independently_of_game_id(tmp_path: Path) -> None:
    record = _record(
        "counterfactual/" + "very-long-identity/" * 40,
        outcome="loss",
        role="second",
        side="B",
        evaluation_seed=101,
        mode="round_limit",
        auxiliary=2.0,
    )
    record.update(
        {
            "checkpoint_id": "unit-5",
            "learner_policy_id": "scavenger-s19-u5",
            "first_actor": "A",
            "action_trace": [],
        }
    )

    write_evaluation_evidence(
        tmp_path / "evaluation",
        {"format": "s5c-a2-evaluation-v1", "records": [record]},
        diagnostic_replay_sample=1,
    )

    replay = next((tmp_path / "evaluation" / "diagnostics" / "round_limit").glob("*.json"))
    assert len(replay.name) <= 32
    assert json.loads(replay.read_text(encoding="utf-8"))["game_id"] == record["game_id"]


def test_specialist_reward_sidecar_exposes_each_auxiliary_rule() -> None:
    reward = specialist_rewards()["predator"]
    terms = specialist_reward_terms(
        reward,
        {
            "events": {
                "damage_dealt": {"SELF": 2.0, "OPPONENT": 1.0},
            }
        },
    )

    assert terms == {"damage_dealt": 0.2, "damage_received": -0.03}
    assert sum(terms.values()) == pytest.approx(0.17)


def test_specialist_reward_programs_are_compiled_once_and_reused() -> None:
    first = specialist_rewards()
    second = specialist_rewards()

    assert first is second


def test_competence_style_and_strength_are_distinct_validations() -> None:
    point_win = _record(
        "point-win",
        outcome="win",
        role="first",
        side="A",
        evaluation_seed=101,
        mode="carcass_score",
        auxiliary=0.2,
    )
    ko_win = _record(
        "ko-win",
        outcome="win",
        role="second",
        side="B",
        evaluation_seed=101,
        mode="ko",
        auxiliary=0.2,
    )

    scavenger = validate_absolute_style("scavenger", [point_win, point_win, ko_win])
    predator = validate_absolute_style("predator", [point_win, ko_win, ko_win])

    assert scavenger["passed"] is True
    assert predator["passed"] is True
    assert scavenger["metric"] == "point_win_rate"
    assert predator["metric"] == "ko_win_rate"


def test_status_reports_all_nine_branches_and_realized_role_counters(tmp_path: Path) -> None:
    run = tmp_path / "rl-s5c-calibration-v2"
    config = S5cConfig.from_mapping(_mapping(tmp_path))
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"run_id": config.run_id, "config": config.to_mapping()}),
        encoding="utf-8",
    )
    branch = run / "calibration" / "scavenger" / "seed-19"
    branch.mkdir(parents=True)
    (branch / "cycles.json").write_text(
        json.dumps(
            [
                {
                    "unit": 1,
                    "training_seconds": 2.0,
                    "checkpoint_seconds": 1.0,
                    "role_counts": {
                        "learner_first": 10,
                        "learner_second": 9,
                        "learner_a": 8,
                        "learner_b": 11,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    (branch / "calibration-state.json").write_text(
        json.dumps({"status": "training", "consecutive_passes": 0}),
        encoding="utf-8",
    )

    result = campaign_status(run)

    assert result["total_branches"] == 9
    first = result["branches"][0]
    assert first["completed_units"] == 1
    assert first["eta_seconds"] == pytest.approx(438.0)
