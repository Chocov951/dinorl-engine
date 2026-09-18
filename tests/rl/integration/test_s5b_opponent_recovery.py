"""Exact interrupted-game recovery with an immutable RL-S5b opponent pool."""

from __future__ import annotations

from dinorl_engine.core.actions import Action
from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.s5b.opponents import FrozenOpponentPool


def _random_only_pool() -> FrozenOpponentPool:
    return FrozenOpponentPool(
        {
            "entries": [
                {
                    "id": "random-legal-v1",
                    "kind": "controller",
                    "status": "available",
                    "unit": None,
                }
            ]
        },
        random_only=True,
    )


def test_s5b_random_opponent_is_recovered_without_changing_the_game() -> None:
    original = DinoRLSingleAgentEnv(seed=19, training_opponent_pool=_random_only_pool())
    original.reset()
    action = Action(int(original.action_masks().nonzero()[0][0]))
    original.step(int(action))
    recovery = original.export_recovery_state()

    recovered = DinoRLSingleAgentEnv(seed=0, training_opponent_pool=_random_only_pool())
    recovered.restore_recovery_state(recovery)
    assert recovered.snapshot_recovery_state() == original.snapshot_recovery_state()

    next_action = int(original.action_masks().nonzero()[0][0])
    original.step(next_action)
    recovered.step(next_action)
    assert recovered.snapshot_recovery_state() == original.snapshot_recovery_state()


def test_phase_a_open_episode_can_start_a_phase_b_branch_without_a_reset() -> None:
    phase_a = DinoRLSingleAgentEnv(seed=19)
    phase_a.reset()
    phase_a_recovery = phase_a.export_recovery_state()

    phase_b = DinoRLSingleAgentEnv(seed=0, training_opponent_pool=_random_only_pool())
    phase_b.restore_recovery_state(phase_a_recovery)
    assert phase_b.export_recovery_state() == phase_a_recovery
