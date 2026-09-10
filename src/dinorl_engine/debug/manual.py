"""Injectable local manual controller."""

from __future__ import annotations

from collections.abc import Callable

from dinorl_engine.core.actions import Action
from dinorl_engine.core.state import PublicSnapshot
from dinorl_engine.debug.text_renderer import render_state

__all__ = ["ManualCancelledError", "ManualController"]

type Reader = Callable[[], str]
type Writer = Callable[[str], object]
type Renderer = Callable[[PublicSnapshot], str]


class ManualCancelledError(RuntimeError):
    """The local operator cancelled an interactive match."""


class ManualController:
    """Choose legal actions through injected terminal-like I/O."""

    def __init__(
        self,
        *,
        reader: Reader,
        writer: Writer,
        renderer: Renderer = render_state,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._renderer = renderer

    def choose_action(
        self,
        state: PublicSnapshot,
        legal_actions: tuple[bool, ...],
    ) -> Action:
        """Prompt until the operator selects one of the supplied legal actions."""

        legal = tuple(
            action
            for action in Action
            if len(legal_actions) == len(Action) and legal_actions[action]
        )
        if not legal:
            raise ValueError("manual controller received no legal action")
        menu = "Legal actions:\n" + "".join(
            f"{index}. {action.name}\n" for index, action in enumerate(legal, start=1)
        )
        self._writer(self._renderer(state))
        self._writer(menu + "\nAction > ")
        while True:
            try:
                answer = self._reader().strip()
            except (EOFError, KeyboardInterrupt) as error:
                raise ManualCancelledError("manual controller cancelled") from error
            if answer.isdecimal():
                index = int(answer) - 1
                if 0 <= index < len(legal):
                    return legal[index]
            else:
                for action in legal:
                    if answer == action.name:
                        return action
            self._writer(f"Invalid action: {answer}\n")
            self._writer("Action > ")
