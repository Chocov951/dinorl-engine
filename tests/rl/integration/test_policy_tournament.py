"""Real-engine policy-versus-policy tournament execution."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.evaluation.tournament import run_policy_duel
from dinorl_engine.rl.policies.local_mlp_v2 import LocalMLPV2Architecture
from dinorl_engine.rl.training.unit import create_maskable_ppo


def test_two_distinct_policy_models_complete_a_position_balanced_duel() -> None:
    left_environment = DinoRLSingleAgentEnv(seed=19)
    right_environment = DinoRLSingleAgentEnv(seed=20)
    try:
        left = create_maskable_ppo(
            left_environment,
            seed=19,
            architecture=LocalMLPV2Architecture.COMPACT,
        )
        right = create_maskable_ppo(
            right_environment,
            seed=20,
            architecture=LocalMLPV2Architecture.BALANCED,
        )

        outcome = run_policy_duel(
            left,
            right,
            seed=21,
            left_actor=Actor.B,
            first_actor=Actor.A,
            deterministic=True,
            matchup_id="compact-vs-balanced",
        )

        assert outcome.wins + outcome.draws + outcome.losses == 1
        assert 1 <= outcome.rounds <= 30
        assert outcome.actions > 0
    finally:
        left_environment.close()
        right_environment.close()
