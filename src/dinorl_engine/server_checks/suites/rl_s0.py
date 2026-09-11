"""RL-S0: compatibility gate for the initial CPU MaskablePPO stack."""

from __future__ import annotations

import importlib
import math
import platform
import tempfile
import time
from pathlib import Path

from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.training.unit import (
    create_maskable_ppo,
    load_maskable_ppo,
    save_maskable_ppo,
    train_one_unit,
)

__all__ = ["run_warmup", "run_with_measurement"]

_SEED = 19


def _assert_legal_prediction(env: DinoRLSingleAgentEnv, model: object) -> None:
    observation, _ = env.reset(seed=_SEED)
    if not env.observation_space.contains(observation):
        raise RuntimeError("wrapper emitted an observation outside its Gymnasium space")
    mask = env.action_masks()
    if mask.shape != (9,) or not bool(mask.any()):
        raise RuntimeError("wrapper emitted an invalid non-terminal action mask")
    action, _ = model.predict(observation, action_masks=mask, deterministic=True)  # type: ignore[attr-defined]
    if not bool(mask[int(action)]):
        raise RuntimeError("masked policy predicted an illegal action")


def _peak_memory_bytes() -> int | None:
    """Return process peak RSS when the server platform exposes it."""

    try:
        resource_module = importlib.import_module("resource")
    except ModuleNotFoundError:  # Windows does not expose POSIX resource usage.
        return None
    getrusage = getattr(resource_module, "getrusage", None)
    usage_self = getattr(resource_module, "RUSAGE_SELF", None)
    if not callable(getrusage) or usage_self is None:
        return None
    peak = getattr(getrusage(usage_self), "ru_maxrss", None)
    if not isinstance(peak, int) or peak <= 0:
        return None
    # Linux reports KiB; macOS reports bytes. RL-S0 currently targets a Linux server.
    return peak if platform.system() == "Darwin" else peak * 1024


def run_warmup() -> dict[str, object]:
    """Warm the model and wrapper once, separately from the measured PPO unit."""

    started = time.perf_counter()
    env = DinoRLSingleAgentEnv(seed=_SEED)
    model = create_maskable_ppo(env, seed=_SEED)
    _assert_legal_prediction(env, model)
    return {"duration_seconds": time.perf_counter() - started, "status": "passed"}


def run_with_measurement() -> dict[str, object]:
    """Execute exactly one full RL-L2 unit and validate its save/load round trip."""

    started = time.perf_counter()
    env = DinoRLSingleAgentEnv(seed=_SEED)
    model = create_maskable_ppo(env, seed=_SEED)
    _assert_legal_prediction(env, model)
    unit = train_one_unit(model, env)
    for key, value in unit.metrics.diagnostics.items():
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite PPO diagnostic after training: {key}")
    with tempfile.TemporaryDirectory(prefix="dinorl-rl-s0-") as temporary_directory:
        checkpoint = Path(temporary_directory) / "minimal-ppo.zip"
        save_maskable_ppo(model, checkpoint)
        if not checkpoint.is_file():
            raise RuntimeError("minimal PPO save did not produce a model file")
        loaded = load_maskable_ppo(checkpoint, env)
        _assert_legal_prediction(env, loaded)
    return {
        "duration_seconds": time.perf_counter() - started,
        "peak_memory_bytes": _peak_memory_bytes(),
        "save_load_round_trip": True,
        "metrics": {
            "learner_transitions": unit.metrics.learner_transitions,
            "engine_actions": unit.metrics.engine_actions,
            "epochs": unit.metrics.epochs,
            "optimizer_steps": unit.metrics.optimizer_steps,
            "diagnostics": unit.metrics.diagnostics,
        },
    }
