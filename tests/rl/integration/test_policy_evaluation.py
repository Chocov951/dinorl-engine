"""RL-L7 policy evaluation against the real deterministic game engine."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.evaluation.match import run_evaluation_game
from dinorl_engine.rl.evaluation.policy import MaskablePolicyController
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec
from dinorl_engine.rl.s5c.rewards import specialist_rewards
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


def test_game_id_reproduces_actions_and_report_options_do_not_change_the_rng() -> None:
    environment = DinoRLSingleAgentEnv(seed=23)
    model = create_maskable_ppo(environment, seed=23)
    specification = EvaluationGameSpec(
        seed=101,
        opponent_id="random-legal-v1",
        learner_actor=Actor.A,
        first_actor=Actor.B,
        game_id="a2/reproducible-game-1",
        evaluation_seed=101,
    )
    try:
        compact = run_evaluation_game(
            model,
            specification,
            deterministic=False,
            diagnostic_replay=False,
            learner_policy_id="controller-s23-u5",
            checkpoint_id="unit-5",
            training_seed=23,
            reward_program=specialist_rewards()["controller"],
        )
        diagnostic = run_evaluation_game(
            model,
            specification,
            deterministic=False,
            diagnostic_replay=True,
            learner_policy_id="controller-s23-u5",
            checkpoint_id="unit-5",
            training_seed=23,
            reward_program=specialist_rewards()["controller"],
        )

        assert compact.record["game_id"] == specification.game_id
        assert compact.record["action_trace"] == diagnostic.record["action_trace"]
        assert compact.record["outcome"] in {"win", "draw", "loss"}
        assert compact.record["learner_role"] == "second"
        assert compact.record["learner_side"] == "A"
        assert isinstance(compact.record["events"], list)
        assert isinstance(compact.record["terminal_return"], float)
        assert isinstance(compact.record["auxiliary_return"], float)
        assert isinstance(compact.record["auxiliary_by_rule"], dict)
        assert diagnostic.replay is not None
        assert diagnostic.replay["game_id"] == specification.game_id
        assert diagnostic.replay["checkpoint_id"] == "unit-5"
    finally:
        environment.close()
