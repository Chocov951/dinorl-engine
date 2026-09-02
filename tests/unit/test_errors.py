"""Core exception hierarchy tests."""

from dinorl_engine.core.errors import DinoRLError, IllegalActionError


def test_illegal_action_error_is_a_public_engine_error() -> None:
    error = IllegalActionError("action unavailable")

    assert isinstance(error, DinoRLError)
    assert isinstance(error, Exception)
    assert str(error) == "action unavailable"
