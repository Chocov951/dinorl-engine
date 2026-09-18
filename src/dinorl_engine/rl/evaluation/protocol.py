"""Versioned game identities shared by deterministic and stochastic suites."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from dinorl_engine.core.constants import Actor

__all__ = ["EvaluationGameSpec", "derive_evaluation_seed", "derive_game_stream_seed"]


@dataclass(frozen=True, slots=True)
class EvaluationGameSpec:
    """One fixed game identity, independent of policy weights or host timing."""

    seed: int
    opponent_id: str
    learner_actor: Actor
    first_actor: Actor


def derive_evaluation_seed(seed: int, *, suite: str, opponent_id: str, index: int) -> int:
    """Derive a stable engine seed without Python's process-randomized hash."""

    # Top-level suite seeds are unsigned 32-bit values, while the stable game
    # identities derived below occupy the non-negative signed 63-bit range.
    # A game seed can legitimately be used again to derive an isolated policy
    # sampling stream for that same game.
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a non-negative signed 63-bit integer")
    if not suite or not opponent_id or type(index) is not int or index < 0:
        raise ValueError("evaluation seed inputs are invalid")
    payload = f"dinorl/v1/evaluation/{seed}/{suite}/{opponent_id}/{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def derive_game_stream_seed(game_id: str, *, policy_id: str) -> int:
    """Derive one sampled-policy stream from an immutable game identity.

    A counterfactual may change its round cap or reporting options, but it must
    use the same ``game_id``.  Consequently every action sampled before the
    two rule paths diverge comes from the same isolated Torch stream.
    """

    if not isinstance(game_id, str) or not game_id:
        raise ValueError("game_id must be a non-empty string")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy_id must be a non-empty string")
    payload = f"dinorl/v1/evaluation-game/{game_id}/{policy_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)
