"""RL-L7 frozen deterministic and paired-evaluation protocol contracts."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.deterministic import deterministic_game_specs
from dinorl_engine.rl.evaluation.protocol import derive_evaluation_seed, derive_game_stream_seed
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


def test_seed_derivation_accepts_a_prior_engine_seed_for_a_child_rng_stream() -> None:
    game_seed = deterministic_game_specs(seed=19, opponent_id="aggressive-v1")[0].seed

    child_seed = derive_evaluation_seed(
        game_seed, suite="policy-sampling", opponent_id="aggressive-v1", index=0
    )

    assert 0 <= child_seed < 2**63


def test_game_streams_are_keyed_by_game_identity_not_reporting_options() -> None:
    """Counterfactual evaluations must retain their sampled-action prefix."""

    first = derive_game_stream_seed("game-0001", policy_id="left")
    repeated = derive_game_stream_seed("game-0001", policy_id="left")
    opposite_policy = derive_game_stream_seed("game-0001", policy_id="right")
    other_game = derive_game_stream_seed("game-0002", policy_id="left")

    assert first == repeated
    assert first != opposite_policy
    assert first != other_game
