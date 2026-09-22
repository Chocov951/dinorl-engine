"""RL-S5c-A3 protocol contracts, written before the implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dinorl_engine.rl.s5c.a3 import (
    ACTIVE_V1_PROFILES,
    A3ProtocolError,
    CompositeEvidence,
    CurriculumPhase,
    EpisodeAuxiliaryBudget,
    composite_gate,
    curriculum_weights,
    deterministic_opponent_split,
    rule_dominates,
    select_variant,
)
from dinorl_engine.rl.s5c.a3_curriculum import (
    CurriculumSchedule,
    curriculum_category,
)
from dinorl_engine.rl.s5c.a3_pilot import PilotVariantState


def test_v1_exposes_only_scavenger_and_predator_with_stable_internal_id() -> None:
    assert ACTIVE_V1_PROFILES == ("scavenger", "predator")
    assert "controller" not in ACTIVE_V1_PROFILES


def test_opponent_split_is_deterministic_disjoint_and_hashed() -> None:
    entries = [
        {"id": f"{arch}-s{seed}", "architecture": arch, "seed": seed}
        for arch in ("compact", "deep")
        for seed in (19, 20, 21)
    ]
    first = deterministic_opponent_split(entries)
    second = deterministic_opponent_split(list(reversed(entries)))
    assert first == second
    assert set(first.training_ids).isdisjoint(first.heldout_ids)
    assert set(first.training_ids) | set(first.heldout_ids) == {
        str(entry["id"]) for entry in entries
    }
    assert len(first.training_sha256) == len(first.heldout_sha256) == 64
    assert {entry.split("-s")[0] for entry in first.training_ids} == {"compact", "deep"}
    assert {entry.split("-s")[0] for entry in first.heldout_ids} == {"compact", "deep"}


def test_curriculum_is_random_only_then_90_10_then_80_20() -> None:
    assert curriculum_weights(CurriculumPhase.BOOTSTRAP, robustification_unit=0) == {
        "random": 1.0,
        "training_pool": 0.0,
    }
    assert curriculum_weights(CurriculumPhase.ROBUSTIFY, robustification_unit=1) == {
        "random": 0.9,
        "training_pool": 0.1,
    }
    assert curriculum_weights(CurriculumPhase.ROBUSTIFY, robustification_unit=20) == {
        "random": 0.9,
        "training_pool": 0.1,
    }
    assert curriculum_weights(CurriculumPhase.ROBUSTIFY, robustification_unit=21) == {
        "random": 0.8,
        "training_pool": 0.2,
    }


def _evidence(*, heldout: float = 0.4, style: bool = True) -> CompositeEvidence:
    return CompositeEvidence(
        random_score=0.91,
        deterministic_scores=(0.6, 0.7, 0.8),
        absolute_style_passed=style,
        comparative_style_passed=style,
        heldout_score=heldout,
        reward_hacking_blocking=False,
        round_limit_strategy=False,
        stochastic_collapse=False,
        replay_identity_visible=True,
    )


def test_composite_gate_requires_every_component_and_two_consecutive_passes() -> None:
    first = composite_gate(_evidence(), previous_passed=False)
    second = composite_gate(_evidence(), previous_passed=True)
    assert first["historical_gate_passed"] is True
    assert first["passed"] is False
    assert second["passed"] is True
    assert composite_gate(_evidence(heldout=0.249), previous_passed=True)["passed"] is False
    assert composite_gate(_evidence(heldout=0.501), previous_passed=True)["passed"] is False
    assert composite_gate(_evidence(style=False), previous_passed=True)["passed"] is False


def test_historical_gate_alone_never_stops_a3() -> None:
    result = composite_gate(_evidence(heldout=0.2), previous_passed=True)
    assert result["historical_gate_passed"] is True
    assert result["passed"] is False


def test_variant_selection_is_mechanical_and_rejects_ineligible_candidates() -> None:
    candidates = [
        {
            "id": "late",
            "eligible": True,
            "gate_unit": 80,
            "heldout_score": 0.40,
            "opponent_variance": 0.01,
            "round_limit_rate": 0.01,
            "style_margin": 0.4,
        },
        {
            "id": "early-far",
            "eligible": True,
            "gate_unit": 60,
            "heldout_score": 0.35,
            "opponent_variance": 0.01,
            "round_limit_rate": 0.01,
            "style_margin": 0.4,
        },
        {
            "id": "early-close",
            "eligible": True,
            "gate_unit": 60,
            "heldout_score": 0.39,
            "opponent_variance": 0.02,
            "round_limit_rate": 0.02,
            "style_margin": 0.2,
        },
    ]
    assert select_variant(candidates)["id"] == "early-close"
    with pytest.raises(A3ProtocolError, match="NO_ELIGIBLE_VARIANT"):
        select_variant([{**candidates[0], "eligible": False}])


def test_phase_lock_refuses_pilot_without_passed_policy_control(tmp_path: Path) -> None:
    from dinorl_engine.rl.s5c.a3 import require_prior_checkpoint

    with pytest.raises(A3ProtocolError, match="FAILED_POLICY_CONTROL"):
        require_prior_checkpoint(tmp_path, phase="pilot")


def test_safe_feed_is_once_per_opportunity_and_capped_per_episode() -> None:
    budget = EpisodeAuxiliaryBudget(safe_feed_cap=0.30)
    assert budget.safe_feed("carcass-1/availability-1", 0.03) == 0.03
    assert budget.safe_feed("carcass-1/availability-1", 0.03) == 0.0
    paid = sum(budget.safe_feed(f"opportunity-{index}", 0.03) for index in range(20))
    assert paid == pytest.approx(0.27)
    assert budget.safe_feed_total == pytest.approx(0.30)


def test_safe_feed_budget_has_exact_recovery_state() -> None:
    budget = EpisodeAuxiliaryBudget(safe_feed_cap=0.30)
    budget.safe_feed("carcass-1/availability-1", 0.03)
    restored = EpisodeAuxiliaryBudget.from_mapping(budget.to_mapping())
    assert restored.safe_feed("carcass-1/availability-1", 0.03) == 0.0
    assert restored.safe_feed("carcass-1/availability-2", 0.03) == 0.03
    assert restored.safe_feed_total == pytest.approx(0.06)


def test_single_rule_domination_is_detected_from_detailed_auxiliary_return() -> None:
    assert rule_dominates({"damage": 9.0, "interrupt": 1.0}, maximum_share=0.75)
    assert not rule_dominates({"damage": 6.0, "interrupt": 4.0}, maximum_share=0.75)


def test_crossplay_baseline_uses_the_same_opponents_and_inverts_right_scores() -> None:
    from dinorl_engine.rl.s5c.a3_control_report import _crossplay_baseline

    crossplay = {
        "records": [
            {
                "left": {"checkpoint_id": "baseline"},
                "right": {"checkpoint_id": "opponent-a"},
                "score": 0.6,
            },
            {
                "left": {"checkpoint_id": "opponent-b"},
                "right": {"checkpoint_id": "baseline"},
                "score": 0.3,
            },
            {
                "left": {"checkpoint_id": "baseline"},
                "right": {"checkpoint_id": "excluded"},
                "score": 0.0,
            },
        ]
    }
    assert _crossplay_baseline(
        crossplay, baseline_id="baseline", opponent_ids={"opponent-a", "opponent-b"}
    ) == pytest.approx(0.65)


def test_curriculum_category_is_deterministic_and_never_selects_heldout() -> None:
    schedule = CurriculumSchedule(
        phase=CurriculumPhase.ROBUSTIFY,
        robustification_unit=1,
        training_ids=("train-a", "train-b"),
        heldout_ids=("heldout",),
    )
    draws = [curriculum_category(seed, schedule=schedule) for seed in range(10_000)]
    assert draws == [curriculum_category(seed, schedule=schedule) for seed in range(10_000)]
    assert set(draws) == {"random", "training_pool"}
    training_rate = draws.count("training_pool") / len(draws)
    assert training_rate == pytest.approx(0.10, abs=0.015)
    assert "heldout" not in schedule.training_ids


def test_bootstrap_curriculum_is_exclusively_random_for_every_episode() -> None:
    schedule = CurriculumSchedule(
        phase=CurriculumPhase.BOOTSTRAP,
        robustification_unit=0,
        training_ids=("train-a",),
        heldout_ids=("heldout",),
    )
    assert {curriculum_category(seed, schedule=schedule) for seed in range(1_000)} == {"random"}


def test_curriculum_rejects_any_train_heldout_overlap() -> None:
    with pytest.raises(A3ProtocolError, match="overlap"):
        CurriculumSchedule(
            phase=CurriculumPhase.BOOTSTRAP,
            robustification_unit=0,
            training_ids=("same",),
            heldout_ids=("same",),
        )


def test_pilot_curriculum_bootstrap_needs_two_random_passes_and_never_historical_stop() -> None:
    state = PilotVariantState.new("predator-a3-balanced")
    state.record_elementary(unit=5, random_score=0.91)
    assert state.phase == "bootstrap"
    state.record_elementary(unit=10, random_score=0.89)
    assert state.phase == "bootstrap"
    state.record_elementary(unit=15, random_score=0.90)
    state.record_elementary(unit=20, random_score=0.92)
    assert state.phase == "robustify"
    assert state.bootstrap_completed_unit == 20
    assert state.status == "training"


def test_pilot_stops_only_on_second_consecutive_composite_pass() -> None:
    state = PilotVariantState.new("scavenger-a2-control")
    state.record_composite(unit=50, thresholds_passed=True)
    assert state.status == "training"
    state.record_composite(unit=55, thresholds_passed=True)
    assert state.status == "composite_stable_gate"
    assert state.gate_unit == 55
