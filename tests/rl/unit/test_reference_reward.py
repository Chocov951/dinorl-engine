"""Native reference reward used before the programmable Reward DSL."""

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor, EndReason
from dinorl_engine.core.engine import GameResult
from dinorl_engine.core.events import (
    ActionCost,
    ActionTransition,
    CarcassPointAwardedEffect,
    DamageDealtEffect,
)
from dinorl_engine.rl.rewards.reference import reference_reward


def _transition(*effects: object) -> ActionTransition:
    return ActionTransition(
        action=Action.BITE,
        actor=Actor.A,
        cost=ActionCost(movement=1, endurance=2),
        effects=effects,  # type: ignore[arg-type]
        turn_ended=False,
    )


@pytest.mark.parametrize(
    ("winner", "expected"),
    [(Actor.A, 1.0), (Actor.B, -1.0), ("draw", 0.0)],
)
def test_reference_reward_adds_the_terminal_outcome_from_the_learner_perspective(
    winner: Actor | str, expected: float
) -> None:
    result = GameResult(
        winner=winner,  # type: ignore[arg-type]
        reason=EndReason.KO,
        score_a=0,
        score_b=0,
        hp_a=6,
        hp_b=0,
        rounds_completed=1,
        individual_turns=2,
        actions=3,
    )

    assert reference_reward(_transition(), learner_actor=Actor.A, result=result) == expected


def test_reference_reward_accumulates_damage_and_carcass_events_including_automatic_effects() -> (
    None
):
    transition = ActionTransition(
        action=Action.END_TURN,
        actor=Actor.A,
        cost=ActionCost(movement=0, endurance=0),
        effects=(DamageDealtEffect(actor=Actor.A, target=Actor.B, amount=2),),
        turn_ended=True,
        automatic_effects=(
            CarcassPointAwardedEffect(actor=Actor.B, carcass_id="carcass_left_b", amount=1),
            DamageDealtEffect(actor=Actor.B, target=Actor.A, amount=1),
        ),
    )

    assert reference_reward(transition, learner_actor=Actor.A, result=None) == pytest.approx(-0.05)
