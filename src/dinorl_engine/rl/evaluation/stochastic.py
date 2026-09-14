"""Paired opponent/initiative fixtures for publication-style evaluation."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec, derive_evaluation_seed

__all__ = ["paired_game_specs"]


def paired_game_specs(
    *, seed: int, opponent_id: str, confrontations: int
) -> tuple[EvaluationGameSpec, ...]:
    """Return paired games: one base seed and both possible first actors."""

    if type(confrontations) is not int or confrontations <= 0:
        raise ValueError("confrontations must be a positive integer")
    games: list[EvaluationGameSpec] = []
    for index in range(confrontations):
        game_seed = derive_evaluation_seed(
            seed, suite="publication", opponent_id=opponent_id, index=index
        )
        games.extend(
            (
                EvaluationGameSpec(game_seed, opponent_id, Actor.A, Actor.A),
                EvaluationGameSpec(game_seed, opponent_id, Actor.A, Actor.B),
            )
        )
    return tuple(games)
