"""Masked Gymnasium environment that gives one learner control of one raptor."""

from __future__ import annotations

import hashlib
from random import Random
from typing import Final

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from dinorl_engine.controllers.protocol import Controller
from dinorl_engine.controllers.scripted import (
    SCRIPTED_CONTROLLER_IDS,
    ControllerId,
    create_scripted_controller,
)
from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.events import ActionTransition
from dinorl_engine.rl.env.canonical import canonical_to_engine_action, engine_to_canonical_action
from dinorl_engine.rl.env.observation import (
    FEATURE_COUNT,
    GRID_CHANNELS,
    GRID_COLUMNS,
    GRID_ROWS,
    Observation,
    build_observation,
)
from dinorl_engine.rl.rewards.reference import reference_reward

__all__ = ["DinoRLSingleAgentEnv", "IllegalActionEscapeError"]

_MAX_SEED: Final = 2**32 - 1
_ENGINE_SEED_MODULUS: Final = 2**63
_OPTION_KEYS: Final = frozenset({"learner_actor", "first_actor", "opponent_id"})


class IllegalActionEscapeError(RuntimeError):
    """A policy selected an action that its current legal-action mask forbade."""


def _validate_seed(seed: object) -> int:
    if type(seed) is not int or not 0 <= seed <= _MAX_SEED:
        raise ValueError("seed must be an unsigned 32-bit integer")
    return seed


def _actor_option(value: object, field_name: str) -> Actor:
    if isinstance(value, Actor):
        return value
    if value == "A":
        return Actor.A
    if value == "B":
        return Actor.B
    raise ValueError(f"{field_name} must be Actor.A, Actor.B, 'A', or 'B'")


class DinoRLSingleAgentEnv(gym.Env[Observation, int]):
    """Expose only learner decision states while internally playing the opponent turn."""

    metadata = {"render_modes": []}

    def __init__(self, *, seed: int = 0, environment_index: int = 0) -> None:
        super().__init__()
        self._base_seed = _validate_seed(seed)
        if type(environment_index) is not int or environment_index < 0:
            raise ValueError("environment_index must be a non-negative integer")
        self._environment_index = environment_index
        self._episode_index = 0
        self._engine: DinoRLEnv | None = None
        self._learner_actor: Actor | None = None
        self._first_actor: Actor | None = None
        self._opponent_id: ControllerId | None = None
        self._opponent: Controller | None = None
        self._learner_transitions = 0
        self._engine_actions = 0
        self._total_learner_transitions = 0
        self._total_engine_actions = 0
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(GRID_CHANNELS, GRID_ROWS, GRID_COLUMNS),
                    dtype=np.float32,
                ),
                "features": gym.spaces.Box(0.0, 1.0, shape=(FEATURE_COUNT,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Discrete(len(Action))

    @property
    def engine(self) -> DinoRLEnv:
        """Return the current engine after a successful :meth:`reset`."""

        if self._engine is None:
            raise RuntimeError("reset() must be called before accessing the engine")
        return self._engine

    @property
    def learner_transitions(self) -> int:
        """Return valid learner ``step()`` calls in the current episode."""

        return self._learner_transitions

    @property
    def engine_actions(self) -> int:
        """Return all atomic engine actions in the current episode."""

        return self._engine_actions

    @property
    def total_learner_transitions(self) -> int:
        """Return all learner transitions collected by this environment instance."""

        return self._total_learner_transitions

    @property
    def total_engine_actions(self) -> int:
        """Return all atomic engine actions performed by this environment instance."""

        return self._total_engine_actions

    @property
    def active_actor(self) -> Actor:
        """Return the engine actor currently holding the turn."""

        return self.engine.state.active_actor

    def _derive_seed(self, domain: str, episode_index: int) -> int:
        payload = (
            f"dinorl/v1/{self._base_seed}/{self._environment_index}/{episode_index}/{domain}"
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % _ENGINE_SEED_MODULUS

    def _choose_actor(self, domain: str, episode_index: int) -> Actor:
        return Actor(self._derive_seed(domain, episode_index) % len(Actor))

    def _choose_opponent_id(self, episode_index: int) -> ControllerId:
        return Random(self._derive_seed("opponent", episode_index)).choice(SCRIPTED_CONTROLLER_IDS)

    def _parse_options(
        self, options: dict[str, object] | None
    ) -> tuple[Actor | None, Actor | None, ControllerId | None]:
        if options is None:
            return None, None, None
        if not isinstance(options, dict) or set(options) - _OPTION_KEYS:
            raise ValueError(
                "reset options may only contain learner_actor, first_actor, and opponent_id"
            )
        learner_actor = (
            _actor_option(options["learner_actor"], "learner_actor")
            if "learner_actor" in options
            else None
        )
        first_actor = (
            _actor_option(options["first_actor"], "first_actor")
            if "first_actor" in options
            else None
        )
        opponent_id_value = options.get("opponent_id")
        if opponent_id_value is not None and opponent_id_value not in SCRIPTED_CONTROLLER_IDS:
            raise ValueError("opponent_id must be one of the three versioned scripted controllers")
        return learner_actor, first_actor, opponent_id_value

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[Observation, dict[str, object]]:
        """Start a deterministic episode and auto-play an opponent opening when needed."""

        if seed is not None:
            self._base_seed = _validate_seed(seed)
            self._episode_index = 0
        super().reset(seed=seed)
        learner_override, first_override, opponent_override = self._parse_options(options)
        episode_index = self._episode_index
        self._episode_index += 1
        self._learner_actor = (
            learner_override
            if learner_override is not None
            else self._choose_actor("learner", episode_index)
        )
        self._first_actor = (
            first_override
            if first_override is not None
            else self._choose_actor("initiative", episode_index)
        )
        self._opponent_id = (
            opponent_override
            if opponent_override is not None
            else self._choose_opponent_id(episode_index)
        )
        self._engine = DinoRLEnv(
            MAP_ID,
            seed=self._derive_seed("engine", episode_index),
            replay=False,
        )
        self._engine.reset(first_actor=self._first_actor)
        self._opponent = create_scripted_controller(self._opponent_id)
        self._learner_transitions = 0
        self._engine_actions = 0
        self._play_opponent_turn(collect_reward=False)
        return self._observation(), self._info()

    def _opponent_controller(self) -> Controller:
        if self._opponent is None:
            raise RuntimeError("reset() must be called before accessing the opponent controller")
        return self._opponent

    def _learner(self) -> Actor:
        if self._learner_actor is None:
            raise RuntimeError("reset() must be called before accessing the learner actor")
        return self._learner_actor

    def _first(self) -> Actor:
        if self._first_actor is None:
            raise RuntimeError("reset() must be called before accessing the first actor")
        return self._first_actor

    def _apply_engine_action(self, action: Action) -> ActionTransition:
        """Apply an engine action and account for it in both action counters."""

        transition = self.engine.step(action)
        self._engine_actions += 1
        self._total_engine_actions += 1
        return transition

    def _transition_reward(self, transition: ActionTransition) -> float:
        """Score one public engine transition from the learner perspective."""

        result = self.engine.result if self.engine.is_terminal else None
        return reference_reward(transition, learner_actor=self._learner(), result=result)

    def _play_opponent_turn(self, *, collect_reward: bool) -> float:
        """Auto-play until learner control returns, optionally accumulating reward."""

        reward = 0.0
        while not self.engine.is_terminal and self.engine.state.active_actor is not self._learner():
            legal_actions = self.engine.legal_actions()
            action = self._opponent_controller().choose_action(
                self.engine.snapshot_public(), legal_actions
            )
            if not isinstance(action, Action) or not legal_actions[action]:
                raise RuntimeError("opponent controller selected an illegal action")
            transition = self._apply_engine_action(action)
            if collect_reward:
                reward += self._transition_reward(transition)
        return reward

    def action_masks(self) -> NDArray[np.bool_]:
        """Return the nine-action legal mask in the learner's canonical orientation."""

        engine_mask = self.engine.legal_actions()
        mask = np.zeros(len(Action), dtype=np.bool_)
        for index, allowed in enumerate(engine_mask):
            canonical_action = engine_to_canonical_action(Action(index), self._learner())
            mask[int(canonical_action)] = allowed
        return mask

    def _observation(self) -> Observation:
        return build_observation(
            self.engine.snapshot_public(), learner_actor=self._learner(), first_actor=self._first()
        )

    def _result_info(self) -> dict[str, str] | None:
        if not self.engine.is_terminal:
            return None
        result = self.engine.result
        winner = result.winner.name if isinstance(result.winner, Actor) else result.winner
        return {"winner": winner, "reason": result.reason.name.lower()}

    def _info(self) -> dict[str, object]:
        opponent_id = self._opponent_id
        if opponent_id is None:
            raise RuntimeError("reset() must be called before creating info")
        return {
            "learner_actor": self._learner().name,
            "first_actor": self._first().name,
            "opponent_id": opponent_id,
            "learner_transitions": self._learner_transitions,
            "engine_actions": self._engine_actions,
            "result": self._result_info(),
        }

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, object]]:
        """Apply one legal learner action and then auto-play the opponent as required."""

        if self.engine.is_terminal:
            raise RuntimeError("reset() must be called after a terminal episode")
        if self.engine.state.active_actor is not self._learner():
            raise RuntimeError("the learner is not active")
        if isinstance(action, bool) or not isinstance(action, int | np.integer):
            raise IllegalActionEscapeError("action must be an integer in the current action mask")
        action_index = int(action)
        if not 0 <= action_index < len(Action) or not self.action_masks()[action_index]:
            raise IllegalActionEscapeError(
                "action is not legal according to the current action mask"
            )
        engine_action = canonical_to_engine_action(Action(action_index), self._learner())
        transition = self._apply_engine_action(engine_action)
        self._learner_transitions += 1
        self._total_learner_transitions += 1
        reward = self._transition_reward(transition)
        reward += self._play_opponent_turn(collect_reward=True)
        return self._observation(), reward, self.engine.is_terminal, False, self._info()
