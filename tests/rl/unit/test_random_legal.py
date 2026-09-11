"""Uniform, independent random legal controller."""

from collections import Counter

import pytest

from dinorl_engine.controllers.random_legal import RandomLegalController
from dinorl_engine.core.actions import Action


def test_random_legal_controller_is_uniform_on_a_synthetic_mask() -> None:
    controller = RandomLegalController(seed=123)
    mask = (True, False, True, False, False, False, False, False, True)
    counts: Counter[Action] = Counter(controller.choose_action({}, mask) for _ in range(30_000))

    assert set(counts) == {Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.END_TURN}
    for count in counts.values():
        assert abs(count / 30_000 - 1 / 3) < 0.02


@pytest.mark.parametrize("seed", [-1, 2**64, True])
def test_random_legal_controller_requires_an_unsigned_64_bit_seed(seed: object) -> None:
    with pytest.raises(ValueError):
        RandomLegalController(seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mask",
    [
        (),
        (False,) * len(Action),
        (True,) * (len(Action) - 1),
        (True, False, False, False, False, False, False, False, 1),
    ],
)
def test_random_legal_controller_rejects_malformed_or_empty_masks(mask: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        RandomLegalController(seed=1).choose_action({}, mask)  # type: ignore[arg-type]
