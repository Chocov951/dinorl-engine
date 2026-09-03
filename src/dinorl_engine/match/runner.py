"""Bounded execution of complete deterministic scripted matches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from dinorl_engine.controllers.protocol import Controller, step_with_controller
from dinorl_engine.controllers.scripted import create_scripted_controller
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import DinoRLEnv, GameResult
from dinorl_engine.match.replay import (
    ReplayRecorder,
    canonical_replay_json,
    replay_sha256,
)

__all__ = ["MatchExecutionError", "MatchOutcome", "run_match"]

MAX_ACTIONS_PER_TURN: Final = 4
MAX_MATCH_ACTIONS: Final = 240
MAX_REPLAY_BYTES: Final = 480 * 1024


class MatchExecutionError(RuntimeError):
    """A defensive execution bound or replay limit was exceeded."""


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """Complete local match output, with optional replay data."""

    result: GameResult
    replay: Mapping[str, object] | None
    replay_sha256: str | None

    def summary(self) -> dict[str, object]:
        """Return the schema-shaped public result summary."""

        result = self.result
        return {
            "winner": (result.winner.name if isinstance(result.winner, Actor) else result.winner),
            "reason": result.reason.name.lower(),
            "score_a": result.score_a,
            "score_b": result.score_b,
            "hp_a": result.hp_a,
            "hp_b": result.hp_b,
            "rounds_completed": result.rounds_completed,
            "individual_turns": result.individual_turns,
            "actions": result.actions,
        }


def _controller_for_actor(
    actor: Actor,
    controller_a: Controller,
    controller_b: Controller,
) -> Controller:
    return controller_a if actor is Actor.A else controller_b


def run_match(
    *,
    map_id: str,
    seed: int,
    controller_a_id: str,
    controller_b_id: str,
    include_replay: bool,
) -> MatchOutcome:
    """Run one match to a rules terminal state under strict defensive bounds."""

    controller_a = create_scripted_controller(controller_a_id)
    controller_b = create_scripted_controller(controller_b_id)
    env = DinoRLEnv(map_id=map_id, seed=seed, replay=include_replay)
    state = env.reset()
    recorder = (
        ReplayRecorder(
            state,
            controller_a=controller_a_id,
            controller_b=controller_b_id,
        )
        if include_replay
        else None
    )

    counted_turn = state.turn
    actions_this_turn = 0
    total_actions = 0
    while not env.is_terminal:
        if state.turn != counted_turn:
            counted_turn = state.turn
            actions_this_turn = 0
        actions_this_turn += 1
        if actions_this_turn > MAX_ACTIONS_PER_TURN:
            raise MatchExecutionError("controller exceeded four actions in one turn")
        total_actions += 1
        if total_actions > MAX_MATCH_ACTIONS:
            raise MatchExecutionError("match exceeded 240 actions")
        controller = _controller_for_actor(state.active_actor, controller_a, controller_b)
        transition = step_with_controller(env, controller)
        if recorder is not None:
            recorder.record_action(transition, state)

    result = env.result
    if result.actions > MAX_MATCH_ACTIONS:
        raise MatchExecutionError("match exceeded 240 actions")
    if recorder is None:
        return MatchOutcome(result=result, replay=None, replay_sha256=None)

    replay = recorder.finish(result)
    serialized = canonical_replay_json(replay)
    if len(serialized) > MAX_REPLAY_BYTES:
        raise MatchExecutionError("replay exceeded 480 KiB")
    return MatchOutcome(
        result=result,
        replay=replay,
        replay_sha256=replay_sha256(replay),
    )
