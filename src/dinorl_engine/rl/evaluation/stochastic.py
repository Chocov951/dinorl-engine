"""Paired opponent/initiative fixtures for publication-style evaluation."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.protocol import EvaluationGameSpec, derive_game_stream_seed

__all__ = ["paired_game_specs"]


def paired_game_specs(
    *, seed: int, opponent_id: str, confrontations: int
) -> tuple[EvaluationGameSpec, ...]:
    """Return paired games: one base seed and both possible first actors."""

    if type(confrontations) is not int or confrontations <= 0:
        raise ValueError("confrontations must be a positive integer")
    games: list[EvaluationGameSpec] = []
    for index in range(confrontations):
        pair_id = f"publication/{seed}/{opponent_id}/{index}"
        game_seed = derive_game_stream_seed(pair_id, policy_id="engine")
        games.extend(
            (
                EvaluationGameSpec(
                    game_seed,
                    opponent_id,
                    Actor.A,
                    Actor.A,
                    f"{pair_id}/first-A",
                    seed,
                    pair_id,
                ),
                EvaluationGameSpec(
                    game_seed,
                    opponent_id,
                    Actor.A,
                    Actor.B,
                    f"{pair_id}/first-B",
                    seed,
                    pair_id,
                ),
            )
        )
    return tuple(games)
