"""Uniform diagnostic controller over a supplied legal-action mask."""

from __future__ import annotations

from random import Random
from typing import Final

from dinorl_engine.core.actions import Action
from dinorl_engine.core.state import PublicSnapshot

__all__ = ["RANDOM_LEGAL_CONTROLLER_ID", "RandomLegalController"]

RANDOM_LEGAL_CONTROLLER_ID: Final = "random-legal-v1"


class RandomLegalController:
    """Choose uniformly from legal actions using an isolated random stream."""

    controller_id: Final = RANDOM_LEGAL_CONTROLLER_ID

    def __init__(self, *, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        self._random = Random(seed)

    def choose_action(self, state: PublicSnapshot, legal_actions: tuple[bool, ...]) -> Action:
        """Return a uniformly sampled legal action from a complete V1 mask."""

        del state
        if len(legal_actions) != len(Action) or any(
            type(legal) is not bool for legal in legal_actions
        ):
            raise ValueError("legal_actions must be a boolean mask of length 9")
        choices = [Action(index) for index, legal in enumerate(legal_actions) if legal]
        if not choices:
            raise ValueError("legal_actions must contain at least one action")
        return self._random.choice(choices)

    def export_recovery_state(self) -> dict[str, object]:
        """Expose the isolated stream needed by an exact interrupted RL run."""

        version, internal, gaussian = self._random.getstate()
        return {
            "format": "random-legal-recovery-v1",
            "gaussian": gaussian,
            "internal": list(internal),
            "version": version,
        }

    def restore_recovery_state(self, state: object) -> None:
        """Restore a state emitted by :meth:`export_recovery_state`."""

        if (
            not isinstance(state, dict)
            or set(state) != {"format", "gaussian", "internal", "version"}
            or state.get("format") != "random-legal-recovery-v1"
            or type(state.get("version")) is not int
            or not isinstance(state.get("internal"), list)
            or not all(type(value) is int for value in state["internal"])
            or (
                state.get("gaussian") is not None
                and (
                    isinstance(state.get("gaussian"), bool)
                    or not isinstance(state.get("gaussian"), float)
                )
            )
        ):
            raise ValueError("random-legal recovery state is invalid")
        try:
            self._random.setstate((state["version"], tuple(state["internal"]), state["gaussian"]))
        except ValueError as error:
            raise ValueError("random-legal recovery state cannot be restored") from error
