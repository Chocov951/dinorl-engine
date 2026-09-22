"""Regression contracts for the corrected RL-S5c-A3 PILOT-R2 protocol."""

from __future__ import annotations

import json

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.events import ActionCost, ActionTransition, ConsumptionStartedEffect
from dinorl_engine.rl.s5c.a3 import EpisodeAuxiliaryBudget, bounded_safe_feed_adjustment
from dinorl_engine.rl.s5c.a3_r2 import (
    R2CompositeEvidence,
    composite_gate_r2,
    not_evaluated_metrics,
    r2_curriculum_weights,
    reward_semantics_sha256,
    reward_warning_is_blocking,
)
from dinorl_engine.rl.s5c.rewards import a3_rewards


def test_r2_curriculum_is_random_then_95_5_then_90_10() -> None:
    assert r2_curriculum_weights("bootstrap", robustification_unit=0) == {
        "random": 1.0,
        "training_pool": 0.0,
    }
    assert r2_curriculum_weights("robustify", robustification_unit=1) == {
        "random": 0.95,
        "training_pool": 0.05,
    }
    assert r2_curriculum_weights("robustify", robustification_unit=20) == {
        "random": 0.95,
        "training_pool": 0.05,
    }
    assert r2_curriculum_weights("robustify", robustification_unit=21) == {
        "random": 0.90,
        "training_pool": 0.10,
    }


def test_non_evaluated_metrics_are_null_and_explicit() -> None:
    assert not_evaluated_metrics() == {
        "evaluation_status": "not_evaluated",
        "gate_unit": None,
        "heldout_score": None,
        "opponent_variance": None,
        "round_limit_rate": None,
        "style_margin": None,
    }
    assert "1000197" not in json.dumps(not_evaluated_metrics())
    assert "1000100" not in json.dumps(not_evaluated_metrics())


def test_reward_semantic_hash_includes_bounded_runtime_semantics() -> None:
    rewards = a3_rewards()
    control = reward_semantics_sha256(
        rewards["scavenger-a2-control"], safe_feed_cap=None, once_per_opportunity=False
    )
    bounded = reward_semantics_sha256(
        rewards["scavenger-a3-curriculum"],
        safe_feed_cap=0.30,
        once_per_opportunity=True,
    )
    assert control != bounded
    assert control == reward_semantics_sha256(
        rewards["scavenger-a2-control"], safe_feed_cap=None, once_per_opportunity=False
    )


def test_repeatable_warning_blocks_only_when_the_reward_is_exploitable() -> None:
    assert reward_warning_is_blocking(
        "repeatable_action_reward_risk", bounded=False, exploitable=True
    )
    assert not reward_warning_is_blocking(
        "repeatable_action_reward_risk", bounded=True, exploitable=False
    )
    assert reward_warning_is_blocking("loss_total_exceeds_win", bounded=True, exploitable=False)


def test_r2_composite_requires_heldout_style_not_only_weak_suite_style() -> None:
    evidence = R2CompositeEvidence(
        historical_gate_passed=True,
        heldout_score=0.40,
        heldout_style_passed=False,
        reward_hacking_blocking=False,
        round_limit_strategy=False,
        stochastic_collapse=False,
    )
    assert composite_gate_r2(evidence, previous_passed=True)["passed"] is False
    assert (
        composite_gate_r2(
            R2CompositeEvidence(
                historical_gate_passed=True,
                heldout_score=0.40,
                heldout_style_passed=True,
                reward_hacking_blocking=False,
                round_limit_strategy=False,
                stochastic_collapse=False,
            ),
            previous_passed=True,
        )["passed"]
        is True
    )


def test_safe_feed_runtime_adjustment_really_caps_the_episode_total() -> None:
    budget = EpisodeAuxiliaryBudget(safe_feed_cap=0.30)
    generations: dict[str, int] = {}
    actual_rewards = []
    for index in range(20):
        transition = ActionTransition(
            action=Action.FEED,
            actor=Actor.A,
            cost=ActionCost(movement=0, endurance=0),
            effects=(ConsumptionStartedEffect(actor=Actor.A, carcass_id=f"carcass-{index}"),),
            turn_ended=False,
        )
        adjustment = bounded_safe_feed_adjustment(
            transition=transition,
            reward_transition={
                "before": {
                    "self": {"position": (0, 0)},
                    "opponent": {"position": (4, 0)},
                }
            },
            learner_actor=Actor.A,
            budget=budget,
            generations=generations,
        )
        actual_rewards.append(0.03 + adjustment)

    assert sum(actual_rewards) == pytest.approx(0.30)
    assert actual_rewards[10:] == pytest.approx([0.0] * 10)
