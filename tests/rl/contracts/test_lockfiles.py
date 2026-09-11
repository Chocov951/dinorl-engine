"""The reproducible runtime lock includes the declared RL stack."""

from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_runtime_lock_pins_rl_dependencies() -> None:
    requirements = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    for dependency in (
        "gymnasium==",
        "numpy==",
        "torch==",
        "stable-baselines3==",
        "sb3-contrib==",
        "safetensors==",
    ):
        assert dependency in requirements
