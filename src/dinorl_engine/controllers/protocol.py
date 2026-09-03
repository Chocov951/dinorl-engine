"""Minimal replaceable controller protocol for local matches."""

from typing import Protocol, runtime_checkable

from dinorl_engine.core.actions import Action
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.events import ActionTransition
from dinorl_engine.core.state import PublicSnapshot

__all__ = ["Controller", "ControllerDecisionError", "step_with_controller"]


class ControllerDecisionError(ValueError):
    """A controller returned a value outside the versioned action enum."""


@runtime_checkable
class Controller(Protocol):
    """Choose one atomic action from public state and the engine legal mask."""

    def choose_action(
        self,
        state: PublicSnapshot,
        legal_actions: tuple[bool, ...],
    ) -> Action:
        """Return one action for the currently active actor."""
        ...


def step_with_controller(env: DinoRLEnv, controller: Controller) -> ActionTransition:
    """Ask a compatible controller, then let the engine validate and apply its action."""

    if not isinstance(controller, Controller):
        raise TypeError("controller does not implement the Controller protocol")
    action = controller.choose_action(env.snapshot_public(), env.legal_actions())
    if not isinstance(action, Action):
        raise ControllerDecisionError("controller must return an Action enum member")
    return env.step(action)
