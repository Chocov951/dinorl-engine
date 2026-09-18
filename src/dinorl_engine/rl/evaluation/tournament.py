"""Position-balanced policy-versus-policy matches for local architecture diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sb3_contrib import MaskablePPO

from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.match.runner import MAX_ACTIONS_PER_TURN, MatchExecutionError
from dinorl_engine.rl.evaluation.policy import MaskablePolicyController
from dinorl_engine.rl.evaluation.protocol import derive_game_stream_seed

__all__ = ["PolicyDuelOutcome", "run_policy_duel"]


@dataclass(frozen=True, slots=True)
class PolicyDuelOutcome:
    """One result from the named left policy's perspective."""

    wins: int
    draws: int
    losses: int
    rounds: int
    actions: int
    telemetry: Mapping[str, object] | None = None


def run_policy_duel(
    left_model: MaskablePPO,
    right_model: MaskablePPO,
    *,
    seed: int,
    left_actor: Actor,
    first_actor: Actor,
    deterministic: bool,
    matchup_id: str,
    game_id: str | None = None,
    max_rounds: int = 30,
    left_policy_id: str = "left",
    right_policy_id: str = "right",
) -> PolicyDuelOutcome:
    """Play one real-engine duel while keeping both policies canonically oriented."""

    if not matchup_id:
        raise ValueError("matchup_id must not be empty")
    identity = game_id or f"{matchup_id}/{seed}/{left_actor.name}/{first_actor.name}"
    right_actor = Actor.B if left_actor is Actor.A else Actor.A
    environment = DinoRLEnv(map_id=MAP_ID, seed=seed, replay=False, max_rounds=max_rounds)
    environment.reset(first_actor=first_actor)
    controllers = {
        left_actor: MaskablePolicyController(
            left_model,
            learner_actor=left_actor,
            first_actor=first_actor,
            deterministic=deterministic,
            max_rounds=max_rounds,
            stochastic_seed=derive_game_stream_seed(identity, policy_id=left_policy_id),
        ),
        right_actor: MaskablePolicyController(
            right_model,
            learner_actor=right_actor,
            first_actor=first_actor,
            deterministic=deterministic,
            max_rounds=max_rounds,
            stochastic_seed=derive_game_stream_seed(identity, policy_id=right_policy_id),
        ),
    }
    counted_turn = environment.state.turn
    actions_this_turn = 0
    total_actions = 0
    turns: list[dict[str, object]] = []
    current_actions: list[str] = []
    current_actor = environment.state.active_actor
    while not environment.is_terminal:
        if environment.state.turn != counted_turn:
            turns.append(
                {"turn": counted_turn, "actor": current_actor.name, "actions": current_actions}
            )
            counted_turn = environment.state.turn
            actions_this_turn = 0
            current_actions = []
            current_actor = environment.state.active_actor
        actions_this_turn += 1
        total_actions += 1
        maximum_actions = MAX_ACTIONS_PER_TURN * 2 * max_rounds
        if actions_this_turn > MAX_ACTIONS_PER_TURN or total_actions > maximum_actions:
            raise MatchExecutionError("policy duel exceeded its engine safety bound")
        active = environment.state.active_actor
        controller = controllers[active]
        action = controller.choose_action(
            environment.snapshot_public(), environment.legal_actions()
        )
        current_actions.append(action.name)
        environment.step(action)
    turns.append({"turn": counted_turn, "actor": current_actor.name, "actions": current_actions})
    winner = environment.result.winner
    state = environment.state

    def _raptor(actor: Actor) -> dict[str, object]:
        raptor = state.raptor(actor)
        return {
            "hp": raptor.hp,
            "endurance": raptor.endurance,
            "points": raptor.carcass_score,
            "position": list(raptor.position),
            "nutrition_pending": raptor.consumption_pending,
        }

    def _role(policy_actor: Actor) -> str:
        return "first" if policy_actor is first_actor else "second"

    telemetry: dict[str, object] = {
        "terminal_reason": environment.result.reason.name.lower(),
        "winner": winner if winner == "draw" else winner.name,
        "rounds": environment.result.rounds_completed,
        "player_turns": {
            "A": sum(1 for turn in turns if turn["actor"] == "A"),
            "B": sum(1 for turn in turns if turn["actor"] == "B"),
        },
        "final": {"A": _raptor(Actor.A), "B": _raptor(Actor.B)},
        "last_five_turns": turns[-5:],
        "policies": {
            "left": {"id": left_policy_id, "actor": left_actor.name, "role": _role(left_actor)},
            "right": {"id": right_policy_id, "actor": right_actor.name, "role": _role(right_actor)},
        },
        "max_rounds": max_rounds,
        "seed": seed,
    }
    return PolicyDuelOutcome(
        wins=int(winner is left_actor),
        draws=int(winner == "draw"),
        losses=int(winner is right_actor),
        rounds=environment.result.rounds_completed,
        actions=environment.result.actions,
        telemetry=telemetry,
    )
