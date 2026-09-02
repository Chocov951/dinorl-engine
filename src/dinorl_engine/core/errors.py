"""Exceptions raised by the pure DinoRL engine core."""

__all__ = ["DinoRLError", "IllegalActionError"]


class DinoRLError(Exception):
    """Base class for errors deliberately exposed by the engine core."""


class IllegalActionError(DinoRLError):
    """Raised when an action is not legal in the current game state."""
