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


def test_pythonanywhere_lock_pins_the_reused_system_rl_stack() -> None:
    requirements = (ROOT / "requirements-pythonanywhere.lock").read_text(encoding="utf-8")
    for dependency in (
        "flask==3.0.3",
        "gymnasium==1.0.0",
        "numpy==2.1.0",
        "safetensors==0.4.5",
        "torch==2.3.1",
        "stable-baselines3==2.5.0",
        "sb3-contrib==2.5.0",
    ):
        assert dependency in requirements
