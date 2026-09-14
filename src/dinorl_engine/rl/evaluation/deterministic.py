"""The four frozen seat/initiative games of a deterministic evaluation."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec, derive_evaluation_seed

__all__ = ["deterministic_game_specs"]


def deterministic_game_specs(*, seed: int, opponent_id: str) -> tuple[EvaluationGameSpec, ...]:
    """Return each learner seat and initiative combination exactly once."""

    return tuple(
        EvaluationGameSpec(
            seed=derive_evaluation_seed(
                seed, suite="deterministic", opponent_id=opponent_id, index=index
            ),
            opponent_id=opponent_id,
            learner_actor=learner_actor,
            first_actor=first_actor,
        )
        for index, (learner_actor, first_actor) in enumerate(
            ((Actor.A, Actor.A), (Actor.A, Actor.B), (Actor.B, Actor.A), (Actor.B, Actor.B))
        )
    )
