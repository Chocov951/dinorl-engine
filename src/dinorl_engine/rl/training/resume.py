"""Serialization helpers for deterministic process-level recovery state."""

from __future__ import annotations

import base64
import random
from collections.abc import Mapping
from typing import cast

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv

__all__ = [
    "capture_process_rng_state",
    "export_vector_recovery_state",
    "restore_process_rng_state",
    "restore_vector_recovery_state",
]


def _as_json_tree(value: object) -> object:
    if isinstance(value, tuple | list):
        return [_as_json_tree(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError("RNG state contains a non-JSON value")


def _as_tuple_tree(value: object) -> object:
    if isinstance(value, list):
        return tuple(_as_tuple_tree(item) for item in value)
    return value


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value: object, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be base64 text")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except ValueError as error:
        raise ValueError(f"{field_name} is not valid base64") from error


def capture_process_rng_state() -> dict[str, object]:
    """Capture every available process RNG stream used by the CPU-only PPO runner."""

    numpy_state = np.random.get_state(legacy=True)
    cuda_states = (
        [_encode_bytes(state.cpu().numpy().tobytes()) for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    return {
        "format": "dinorl-process-rng-v1",
        "python": _as_json_tree(random.getstate()),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": _encode_bytes(numpy_state[1].tobytes()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": _encode_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
        "torch_cuda": cuda_states,
    }


def restore_process_rng_state(state: object) -> None:
    """Restore a previously captured process RNG state after model loading."""

    if not isinstance(state, Mapping) or set(state) != {
        "format",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("process RNG recovery state has unexpected fields")
    if state["format"] != "dinorl-process-rng-v1":
        raise ValueError("process RNG recovery state has an invalid format")
    python_state = _as_tuple_tree(state["python"])
    numpy_state = state["numpy"]
    cuda_states = state["torch_cuda"]
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "algorithm",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("NumPy RNG recovery state has unexpected fields")
    if (
        numpy_state["algorithm"] != "MT19937"
        or type(numpy_state["position"]) is not int
        or type(numpy_state["has_gauss"]) is not int
        or isinstance(numpy_state["cached_gaussian"], bool)
        or not isinstance(numpy_state["cached_gaussian"], int | float)
        or not isinstance(cuda_states, list)
    ):
        raise ValueError("process RNG recovery state has invalid values")
    python_tuple = cast(tuple[object, ...], python_state)
    keys = np.frombuffer(_decode_bytes(numpy_state["keys"], "numpy.keys"), dtype=np.uint32).copy()
    if keys.shape != (624,):
        raise ValueError("NumPy RNG recovery state has invalid key length")
    cpu_state = torch.tensor(
        list(_decode_bytes(state["torch_cpu"], "torch_cpu")), dtype=torch.uint8
    )
    try:
        random.setstate(cast(tuple[int, tuple[int, ...], None], python_tuple))
        np.random.set_state(
            (
                "MT19937",
                keys,
                numpy_state["position"],
                numpy_state["has_gauss"],
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(cpu_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [
                    torch.tensor(list(_decode_bytes(item, "torch_cuda")), dtype=torch.uint8)
                    for item in cuda_states
                ]
            )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("process RNG recovery state cannot be restored") from error


def _dummy_environments(environment: VecEnv) -> list[DinoRLSingleAgentEnv]:
    """Return the selected RL-S1 dummy workers without accepting an opaque wrapper."""

    if not isinstance(environment, DummyVecEnv):
        raise ValueError("exact RL-L5 recovery supports the selected DummyVecEnv backend only")
    environments = environment.envs
    if not all(isinstance(item, DinoRLSingleAgentEnv) for item in environments):
        raise ValueError("DummyVecEnv contains an unsupported environment")
    return cast(list[DinoRLSingleAgentEnv], environments)


def export_vector_recovery_state(environment: VecEnv) -> list[dict[str, object]]:
    """Capture all selected sub-environment game, opponent and RNG states."""

    return [item.export_recovery_state() for item in _dummy_environments(environment)]


def restore_vector_recovery_state(environment: VecEnv, states: object) -> None:
    """Restore every selected sub-environment without a trajectory-altering reset."""

    environments = _dummy_environments(environment)
    if not isinstance(states, list) or len(states) != len(environments):
        raise ValueError("vector recovery state has an unexpected environment count")
    for item, state in zip(environments, states, strict=True):
        item.restore_recovery_state(state)
