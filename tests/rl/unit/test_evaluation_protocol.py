"""RL-L7 frozen deterministic and paired-evaluation protocol contracts."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.deterministic import deterministic_game_specs
from dinorl_engine.rl.evaluation.stochastic import paired_game_specs


def test_deterministic_suite_covers_each_learner_seat_and_initiative_once() -> None:
    games = deterministic_game_specs(seed=19, opponent_id="aggressive-v1")

    assert {(game.learner_actor, game.first_actor) for game in games} == {
        (Actor.A, Actor.A),
        (Actor.A, Actor.B),
        (Actor.B, Actor.A),
        (Actor.B, Actor.B),
    }
    assert all(game.opponent_id == "aggressive-v1" for game in games)


def test_stochastic_confrontations_are_paired_by_seed_and_opposite_initiative() -> None:
    games = paired_game_specs(seed=19, opponent_id="random-legal-v1", confrontations=3)

    assert len(games) == 6
    for first, second in zip(games[::2], games[1::2], strict=True):
        assert first.seed == second.seed
        assert first.learner_actor is second.learner_actor is Actor.A
        assert {first.first_actor, second.first_actor} == {Actor.A, Actor.B}
