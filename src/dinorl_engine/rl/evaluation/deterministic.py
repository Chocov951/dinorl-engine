"""The four frozen seat/initiative games of a deterministic evaluation."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec, derive_game_stream_seed

__all__ = ["deterministic_game_specs"]


def deterministic_game_specs(*, seed: int, opponent_id: str) -> tuple[EvaluationGameSpec, ...]:
    """Return each learner seat and initiative combination exactly once."""

    return tuple(
        EvaluationGameSpec(
            seed=derive_game_stream_seed(
                f"deterministic/{seed}/{opponent_id}/{index}", policy_id="engine"
            ),
            opponent_id=opponent_id,
            learner_actor=learner_actor,
            first_actor=first_actor,
            game_id=f"deterministic/{seed}/{opponent_id}/{index}",
            evaluation_seed=seed,
        )
        for index, (learner_actor, first_actor) in enumerate(
            ((Actor.A, Actor.A), (Actor.A, Actor.B), (Actor.B, Actor.A), (Actor.B, Actor.B))
        )
    )
