"""Public canonical Reward DSL transition construction."""

from __future__ import annotations

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.events import ActionCost, ActionTransition, DamageDealtEffect
from dinorl_engine.rl.rewards.reference import (
    REFERENCE_REWARD_SOURCE,
    reference_reward,
    reference_reward_public,
)
from dinorl_engine.rl.rewards.runtime import compile_reward
from dinorl_engine.rl.rewards.transition import public_reward_transition


def test_public_transition_exposes_only_the_learner_oriented_action_and_events() -> None:
    environment = DinoRLEnv("arena_mvp_v1", seed=19)
    environment.reset(first_actor=Actor.A)
    before = environment.snapshot_public()
    transition = ActionTransition(
        action=Action.BITE,
        actor=Actor.A,
        cost=ActionCost(movement=1, endurance=2),
        effects=(DamageDealtEffect(actor=Actor.A, target=Actor.B, amount=2),),
        turn_ended=False,
    )

    public = public_reward_transition(
        before=before,
        after=before,
        transition=transition,
        learner_actor=Actor.A,
        first_actor=Actor.A,
        result=None,
    )

    assert public["action"]["actor"] == "SELF"
    assert public["action"]["kind"] == "BITE"
    assert public["events"]["damage_dealt"] == {"SELF": 2.0, "OPPONENT": 0.0}
    assert public["outcome"] == "ONGOING"
    assert public["before"]["active_actor"] == "SELF"
    assert public["before"]["initiative"] == "SELF"
    assert "movement_points" not in public["before"]["opponent"]
    assert public["map"]["walls"]


def test_reference_dsl_is_exactly_equivalent_to_the_native_reward() -> None:
    environment = DinoRLEnv("arena_mvp_v1", seed=19)
    environment.reset(first_actor=Actor.A)
    before = environment.snapshot_public()
    action = next(
        Action(index) for index, allowed in enumerate(environment.legal_actions()) if allowed
    )
    legal_actions = environment.legal_actions()
    transition = environment.step(action)
    result = environment.result if environment.is_terminal else None
    public = public_reward_transition(
        before=before,
        after=environment.snapshot_public(),
        transition=transition,
        learner_actor=Actor.A,
        first_actor=Actor.A,
        result=result,
        legal_actions=legal_actions,
    )
    compiled = compile_reward(REFERENCE_REWARD_SOURCE)

    assert compiled.evaluate_reference(public) == reference_reward(
        transition, learner_actor=Actor.A, result=result
    )
    assert reference_reward_public(public) == reference_reward(
        transition, learner_actor=Actor.A, result=result
    )
    assert compiled.evaluate_vm(public) == compiled.evaluate_reference(public)
