"""Package-level smoke tests."""

import dinorl_engine


def test_package_is_importable() -> None:
    assert dinorl_engine.__doc__ == "DinoRL game engine package."
