"""RL-S2: exact Reward DSL AST/VM parity and native-cost measurement."""

from __future__ import annotations

import time

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import DinoRLEnv, GameResult
from dinorl_engine.core.events import ActionTransition
from dinorl_engine.core.state import PublicSnapshot
from dinorl_engine.rl.rewards.reference import (
    REFERENCE_REWARD_SOURCE,
    reference_reward,
    reference_reward_public,
)
from dinorl_engine.rl.rewards.runtime import compile_reward
from dinorl_engine.rl.rewards.transition import public_reward_transition

__all__ = [
    "CORPUS_TRANSITIONS",
    "ITERATIONS",
    "MAX_OVERHEAD_RATIO",
    "REPEATED_MEASUREMENTS",
    "run_measurement",
    "run_warmup",
]

CORPUS_TRANSITIONS = 128
ITERATIONS = 64
MAX_OVERHEAD_RATIO = 0.10
REPEATED_MEASUREMENTS = 5

type CorpusEntry = tuple[
    PublicSnapshot,
    PublicSnapshot,
    ActionTransition,
    GameResult | None,
    Actor,
    tuple[bool, ...],
]


def _corpus() -> tuple[CorpusEntry, ...]:
    environment = DinoRLEnv("arena_mvp_v1", seed=19)
    environment.reset(first_actor=Actor.A)
    transitions: list[CorpusEntry] = []
    for step in range(CORPUS_TRANSITIONS):
        before = environment.snapshot_public()
        legal = environment.legal_actions()
        choices = tuple(Action(index) for index, allowed in enumerate(legal) if allowed)
        action = choices[step % len(choices)]
        transition = environment.step(action)
        result = environment.result if environment.is_terminal else None
        transitions.append(
            (
                before,
                environment.snapshot_public(),
                transition,
                result,
                environment.state.first_actor,
                legal,
            )
        )
        if environment.is_terminal:
            environment.reset(first_actor=Actor.A)
    return tuple(transitions)


def _measure() -> dict[str, object]:
    compiled = compile_reward(REFERENCE_REWARD_SOURCE)
    corpus = _corpus()
    public = [
        public_reward_transition(
            before=before,
            after=after,
            transition=transition,
            learner_actor=Actor.A,
            first_actor=first_actor,
            result=result,
            legal_actions=legal_actions,
        )
        for before, after, transition, result, first_actor, legal_actions in corpus
    ]
    native_values = [
        reference_reward(transition, learner_actor=Actor.A, result=result)
        for _before, _after, transition, result, _first_actor, _legal_actions in corpus
    ]
    precalculated_values = [reference_reward_public(item) for item in public]
    ast_values = [compiled.evaluate_reference(item) for item in public]
    vm_values = [compiled.evaluate_vm(item) for item in public]
    if (
        native_values != precalculated_values
        or precalculated_values != ast_values
        or ast_values != vm_values
    ):
        raise RuntimeError(
            "Reward DSL reference program is not exactly equivalent to native reward"
        )
    ast_started = time.perf_counter()
    for _ in range(ITERATIONS):
        for item in public:
            compiled.evaluate_reference(item)
    ast_seconds = time.perf_counter() - ast_started
    native_started = time.perf_counter()
    for _ in range(ITERATIONS):
        for item in public:
            reference_reward_public(item)
    native_seconds = time.perf_counter() - native_started
    vm_started = time.perf_counter()
    for _ in range(ITERATIONS):
        for item in public:
            compiled.evaluate_vm(item)
    vm_seconds = time.perf_counter() - vm_started
    overhead = (vm_seconds / native_seconds - 1.0) if native_seconds > 0 else float("inf")
    return {
        "ast_seconds": ast_seconds,
        "native_seconds": native_seconds,
        "vm_seconds": vm_seconds,
        "overhead_ratio": overhead,
        "transitions": len(corpus),
        "exact": True,
    }


def run_warmup() -> dict[str, object]:
    """Run excluded parity and cache warm-up."""

    return _measure()


def run_measurement() -> dict[str, object]:
    """Run one measured RL-S2 repetition."""

    return _measure()
