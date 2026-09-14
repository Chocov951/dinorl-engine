"""RL-L7 policy evaluation against the real deterministic game engine."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.evaluation.match import run_evaluation_game
from dinorl_engine.rl.evaluation.policy import MaskablePolicyController
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec
from dinorl_engine.rl.training.unit import create_maskable_ppo


def test_policy_controller_returns_an_engine_legal_action_from_actor_b_perspective() -> None:
    environment = DinoRLSingleAgentEnv(seed=19)
    model = create_maskable_ppo(environment, seed=19)
    try:
        environment.reset(options={"learner_actor": "B", "first_actor": "B"})
        controller = MaskablePolicyController(
            model,
            learner_actor=Actor.B,
            first_actor=Actor.B,
            deterministic=True,
            stochastic_seed=19,
        )
        action = controller.choose_action(
            environment.engine.snapshot_public(), environment.engine.legal_actions()
        )

        assert environment.engine.legal_actions()[action]
    finally:
        environment.close()


def test_evaluation_game_collects_only_aggregate_metrics_and_an_optional_replay() -> None:
    environment = DinoRLSingleAgentEnv(seed=19)
    model = create_maskable_ppo(environment, seed=19)
    try:
        result = run_evaluation_game(
            model,
            EvaluationGameSpec(
                seed=19,
                opponent_id="aggressive-v1",
                learner_actor=Actor.A,
                first_actor=Actor.A,
            ),
            deterministic=True,
            diagnostic_replay=True,
        )

        assert result.outcomes.games == 1
        assert result.metrics["action_distribution"]
        assert result.metrics["average_legal_actions"] >= 1.0
        assert result.replay is not None
    finally:
        environment.close()
