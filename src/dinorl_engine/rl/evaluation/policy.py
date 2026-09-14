"""Controller adapter for evaluating a masked PPO policy in real engine matches."""

from __future__ import annotations

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.core.state import PublicSnapshot
from dinorl_engine.rl.env.canonical import canonical_to_engine_action, engine_to_canonical_action
from dinorl_engine.rl.env.observation import build_observation

__all__ = ["MaskablePolicyController"]


class MaskablePolicyController:
    """Present one PPO policy as an engine controller from one learner perspective."""

    def __init__(
        self,
        model: MaskablePPO,
        *,
        learner_actor: Actor,
        first_actor: Actor,
        deterministic: bool,
        stochastic_seed: int,
    ) -> None:
        if not isinstance(learner_actor, Actor) or not isinstance(first_actor, Actor):
            raise ValueError("learner_actor and first_actor must be Actors")
        if type(stochastic_seed) is not int or not 0 <= stochastic_seed < 2**63:
            raise ValueError("stochastic_seed must be a non-negative signed 63-bit integer")
        self._model = model
        self._learner_actor = learner_actor
        self._first_actor = first_actor
        self._deterministic = deterministic
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stochastic_seed)
        self._random_state = generator.get_state()

    def _canonical_mask(self, legal_actions: tuple[bool, ...]) -> np.ndarray:
        if len(legal_actions) != len(Action) or any(
            type(legal) is not bool for legal in legal_actions
        ):
            raise ValueError("legal_actions must be a boolean Action mask")
        mask = np.zeros(len(Action), dtype=np.bool_)
        for index, legal in enumerate(legal_actions):
            canonical = engine_to_canonical_action(Action(index), self._learner_actor)
            mask[int(canonical)] = legal
        if not bool(mask.any()):
            raise ValueError("policy cannot choose an action from an empty mask")
        return mask

    def choose_action(self, state: PublicSnapshot, legal_actions: tuple[bool, ...]) -> Action:
        """Choose one legal engine action after masked deterministic or sampled inference."""

        observation = build_observation(
            state, learner_actor=self._learner_actor, first_actor=self._first_actor
        )
        mask = self._canonical_mask(legal_actions)
        if self._deterministic:
            prediction, _state = self._model.predict(
                observation, action_masks=mask, deterministic=True
            )
        else:
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(self._random_state)
                prediction, _state = self._model.predict(
                    observation, action_masks=mask, deterministic=False
                )
                self._random_state = torch.random.get_rng_state()
        canonical_index = int(np.asarray(prediction).item())
        if not 0 <= canonical_index < len(Action):
            raise RuntimeError("PPO prediction is outside the canonical action space")
        action = canonical_to_engine_action(Action(canonical_index), self._learner_actor)
        if not legal_actions[int(action)]:
            raise RuntimeError("masked PPO prediction selected an illegal engine action")
        return action
