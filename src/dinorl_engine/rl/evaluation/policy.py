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
        max_rounds: int = 30,
    ) -> None:
        if not isinstance(learner_actor, Actor) or not isinstance(first_actor, Actor):
            raise ValueError("learner_actor and first_actor must be Actors")
        if type(stochastic_seed) is not int or not 0 <= stochastic_seed < 2**63:
            raise ValueError("stochastic_seed must be a non-negative signed 63-bit integer")
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        self._model = model
        self._learner_actor = learner_actor
        self._first_actor = first_actor
        self._deterministic = deterministic
        self._max_rounds = max_rounds
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
            state,
            learner_actor=self._learner_actor,
            first_actor=self._first_actor,
            max_rounds=self._max_rounds,
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

    def export_recovery_state(self) -> dict[str, object]:
        """Export sampled-policy state without serialising or mutating its weights."""

        return {
            "format": "masked-policy-controller-recovery-v1",
            "random_state": self._random_state.cpu().tolist(),
        }

    def restore_recovery_state(self, state: object) -> None:
        """Restore the private sampled-policy stream used inside an unfinished game."""

        if (
            not isinstance(state, dict)
            or set(state) != {"format", "random_state"}
            or state.get("format") != "masked-policy-controller-recovery-v1"
            or not isinstance(state.get("random_state"), list)
            or not all(type(value) is int and 0 <= value <= 255 for value in state["random_state"])
        ):
            raise ValueError("masked-policy controller recovery state is invalid")
        restored = torch.tensor(state["random_state"], dtype=torch.uint8)
        if restored.shape != self._random_state.shape:
            raise ValueError("masked-policy controller recovery state has an invalid length")
        self._random_state = restored
