"""Versioned game identities shared by deterministic and stochastic suites."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from dinorl_engine.core.constants import Actor

__all__ = ["EvaluationGameSpec", "derive_evaluation_seed"]


@dataclass(frozen=True, slots=True)
class EvaluationGameSpec:
    """One fixed game identity, independent of policy weights or host timing."""

    seed: int
    opponent_id: str
    learner_actor: Actor
    first_actor: Actor


def derive_evaluation_seed(seed: int, *, suite: str, opponent_id: str, index: int) -> int:
    """Derive a stable engine seed without Python's process-randomized hash."""

    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not suite or not opponent_id or type(index) is not int or index < 0:
        raise ValueError("evaluation seed inputs are invalid")
    payload = f"dinorl/v1/evaluation/{seed}/{suite}/{opponent_id}/{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)
