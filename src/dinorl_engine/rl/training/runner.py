"""Atomic checkpoint storage for completed PPO training units."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from dinorl_engine.rl.env.vectorization import (
    VectorBackend,
    VectorEnvironmentConfig,
    create_vector_environment,
)
from dinorl_engine.rl.orchestration.ledger import UnitLedger
from dinorl_engine.rl.training.resume import (
    capture_process_rng_state,
    export_vector_recovery_state,
    restore_process_rng_state,
    restore_vector_recovery_state,
)
from dinorl_engine.rl.training.unit import (
    PPOUnitMetrics,
    load_maskable_ppo,
    train_one_unit,
)

__all__ = [
    "AtomicRecoveryStore",
    "AtomicPPOUnitResult",
    "AtomicPPOUnitRunner",
    "CrashPoint",
    "RecoveryRecord",
    "SimulatedCrash",
]

_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_FORMAT = "dinorl-recovery-v1"


class CrashPoint(StrEnum):
    """Deterministic crash injection points in the atomic write protocol."""

    SERIALIZED = "serialized"
    FSYNCED = "fsynced"
    HASHED = "hashed"
    MANIFEST_WRITTEN = "manifest_written"
    RENAMED = "renamed"


class SimulatedCrash(RuntimeError):
    """Raised only by an explicit test crash injection."""


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    """One verified immutable recovery directory."""

    unit_id: str
    sequence: int
    state: dict[str, object]
    directory: Path
    files: dict[str, str]


@dataclass(frozen=True, slots=True)
class AtomicPPOUnitResult:
    """A trained, atomically durable and idempotently debited PPO unit."""

    metrics: PPOUnitMetrics
    recovery: RecoveryRecord
    debited: bool
    training_seconds: float
    checkpoint_seconds: float


def _fsync_file(path: Path) -> None:
    # Windows only accepts FlushFileBuffers on a descriptor opened for writing.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry where the current platform supports it."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


class AtomicRecoveryStore:
    """Store immutable completed states; incomplete staging directories are never recovered."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._units = root / "units"
        self._root.mkdir(parents=True, exist_ok=True)
        self._units.mkdir(exist_ok=True)

    @staticmethod
    def _validate_identity(unit_id: str, sequence: int) -> None:
        if _OPAQUE_ID.fullmatch(unit_id) is None:
            raise ValueError("unit_id must be an opaque identifier")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise ValueError("sequence must be a positive integer")

    @staticmethod
    def _crash_if(point: CrashPoint, selected: CrashPoint | None) -> None:
        if point is selected:
            raise SimulatedCrash(f"simulated crash at {point.value}")

    def commit_json_state(
        self,
        *,
        unit_id: str,
        sequence: int,
        state: Mapping[str, object],
        crash_at: CrashPoint | None = None,
        write_files: Callable[[Path], tuple[str, ...]] | None = None,
    ) -> RecoveryRecord:
        """Serialize, validate and atomically publish one JSON-compatible recovery state."""

        self._validate_identity(unit_id, sequence)
        destination = self._units / unit_id
        if destination.exists():
            existing = self._read_record(destination)
            if existing is None or existing.sequence != sequence:
                raise RuntimeError(
                    "recovery unit identifier already exists with different contents"
                )
            return existing
        staging = self._root / f".{unit_id}.{uuid.uuid4().hex}.staging"
        staging.mkdir()
        state_path = staging / "state.json"
        state_path.write_bytes(_canonical_json(state))
        extra_files = () if write_files is None else write_files(staging)
        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or name in {"state.json", "manifest.json"}
            or not (staging / name).is_file()
            for name in extra_files
        ):
            raise ValueError("recovery writer produced an invalid file name")
        self._crash_if(CrashPoint.SERIALIZED, crash_at)
        for path in (state_path, *(staging / name for name in extra_files)):
            _fsync_file(path)
        _fsync_directory(staging)
        self._crash_if(CrashPoint.FSYNCED, crash_at)
        hashes = {name: _sha256(staging / name) for name in ("state.json", *extra_files)}
        self._crash_if(CrashPoint.HASHED, crash_at)
        manifest = {
            "format": _FORMAT,
            "unit_id": unit_id,
            "sequence": sequence,
            "files": hashes,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        _fsync_file(manifest_path)
        _fsync_directory(staging)
        self._crash_if(CrashPoint.MANIFEST_WRITTEN, crash_at)
        os.replace(staging, destination)
        _fsync_directory(self._units)
        self._crash_if(CrashPoint.RENAMED, crash_at)
        record = self._read_record(destination)
        if record is None:
            raise RuntimeError("atomic recovery state failed validation after rename")
        return record

    def _read_record(self, directory: Path) -> RecoveryRecord | None:
        try:
            manifest_value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            state_value = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(manifest_value, dict) or not isinstance(state_value, dict):
            return None
        if set(manifest_value) != {"format", "unit_id", "sequence", "files"}:
            return None
        unit_id = manifest_value.get("unit_id")
        sequence = manifest_value.get("sequence")
        hashes = manifest_value.get("files")
        if (
            manifest_value.get("format") != _FORMAT
            or not isinstance(unit_id, str)
            or _OPAQUE_ID.fullmatch(unit_id) is None
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= 0
            or not isinstance(hashes, dict)
            or directory.name != unit_id
        ):
            return None
        if not hashes or "state.json" not in hashes:
            return None
        checked_hashes: dict[str, str] = {}
        for filename, expected_hash in hashes.items():
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)
            ):
                return None
            path = directory / filename
            if not path.is_file() or _sha256(path) != expected_hash:
                return None
            checked_hashes[filename] = expected_hash
        return RecoveryRecord(
            unit_id=unit_id,
            sequence=sequence,
            state=state_value,
            directory=directory,
            files=checked_hashes,
        )

    def load_latest(self) -> RecoveryRecord | None:
        """Return the highest verified sequence, ignoring interrupted staging writes."""

        records = [
            record
            for directory in self._units.iterdir()
            if directory.is_dir() and (record := self._read_record(directory)) is not None
        ]
        if not records:
            return None
        return max(records, key=lambda record: (record.sequence, record.unit_id))


def _metrics_mapping(metrics: PPOUnitMetrics) -> dict[str, object]:
    return {
        "learner_transitions": metrics.learner_transitions,
        "engine_actions": metrics.engine_actions,
        "epochs": metrics.epochs,
        "optimizer_steps": metrics.optimizer_steps,
        "diagnostics": metrics.diagnostics,
    }


def _metrics_from_mapping(value: object) -> PPOUnitMetrics:
    if not isinstance(value, dict) or set(value) != {
        "learner_transitions",
        "engine_actions",
        "epochs",
        "optimizer_steps",
        "diagnostics",
    }:
        raise ValueError("PPO recovery metrics have unexpected fields")
    integer_names = ("learner_transitions", "engine_actions", "epochs", "optimizer_steps")
    if any(type(value[name]) is not int or value[name] < 0 for name in integer_names):
        raise ValueError("PPO recovery metrics contain invalid counters")
    diagnostics = value["diagnostics"]
    if not isinstance(diagnostics, dict) or any(
        not isinstance(key, str) or isinstance(number, bool) or not isinstance(number, int | float)
        for key, number in diagnostics.items()
    ):
        raise ValueError("PPO recovery diagnostics are invalid")
    return PPOUnitMetrics(
        learner_transitions=value["learner_transitions"],
        engine_actions=value["engine_actions"],
        epochs=value["epochs"],
        optimizer_steps=value["optimizer_steps"],
        diagnostics={key: float(number) for key, number in diagnostics.items()},
    )


def _save_ppo_resume_arrays(model: MaskablePPO, directory: Path) -> tuple[str, ...]:
    """Write non-persistent SB3 rollout markers alongside its normal model archive."""

    observation = model._last_obs
    episode_starts = model._last_episode_starts
    if (
        not isinstance(observation, dict)
        or set(observation) != {"grid", "features"}
        or not isinstance(observation["grid"], np.ndarray)
        or not isinstance(observation["features"], np.ndarray)
        or not isinstance(episode_starts, np.ndarray)
    ):
        raise RuntimeError("PPO has no complete rollout markers to checkpoint")
    model_path = directory / "model.zip"
    model.save(str(model_path))
    np.savez_compressed(
        directory / "ppo_state.npz",
        last_grid=observation["grid"],
        last_features=observation["features"],
        last_episode_starts=episode_starts,
        num_timesteps=np.asarray([model.num_timesteps], dtype=np.int64),
    )
    return "model.zip", "ppo_state.npz"


def _restore_ppo_resume_arrays(model: MaskablePPO, path: Path) -> None:
    """Restore SB3's private collection markers required before the next rollout."""

    try:
        with np.load(path, allow_pickle=False) as arrays:
            required = {"last_grid", "last_features", "last_episode_starts", "num_timesteps"}
            if set(arrays.files) != required:
                raise ValueError("PPO resume archive has unexpected arrays")
            grid = arrays["last_grid"].copy()
            features = arrays["last_features"].copy()
            episode_starts = arrays["last_episode_starts"].copy()
            num_timesteps = arrays["num_timesteps"].copy()
    except (OSError, ValueError) as error:
        raise ValueError("PPO resume archive cannot be read") from error
    if num_timesteps.shape != (1,) or not np.issubdtype(num_timesteps.dtype, np.integer):
        raise ValueError("PPO resume archive has an invalid timestep counter")
    model._last_obs = {"grid": grid, "features": features}
    model._last_episode_starts = episode_starts
    model._last_original_obs = None
    model.num_timesteps = int(num_timesteps[0])


class AtomicPPOUnitRunner:
    """Persist the selected dummy-vector PPO state before performing one ledger debit."""

    def __init__(
        self,
        *,
        run_id: str,
        configuration: VectorEnvironmentConfig,
        model: MaskablePPO,
        environment: VecEnv,
        recovery_store: AtomicRecoveryStore,
        ledger: UnitLedger,
    ) -> None:
        if configuration.backend is not VectorBackend.DUMMY:
            raise ValueError("RL-L5 exact recovery requires the selected DummyVecEnv backend")
        if not isinstance(environment, DummyVecEnv) or environment.num_envs != configuration.n_envs:
            raise ValueError("environment does not match the selected recovery configuration")
        self._run_id = run_id
        self._configuration = configuration
        self._model = model
        self._environment = environment
        self._store = recovery_store
        self._ledger = ledger

    def run_unit(
        self,
        *,
        unit_id: str,
        sequence: int,
        crash_at: CrashPoint | None = None,
        callback: BaseCallback | None = None,
    ) -> AtomicPPOUnitResult:
        """Train one unit, atomically validate it, then debit its reserved budget once."""

        training_started = time.perf_counter()
        result = train_one_unit(self._model, self._environment, callback=callback)
        training_seconds = time.perf_counter() - training_started
        state = {
            "format": "dinorl-ppo-unit-recovery-v1",
            "run_id": self._run_id,
            "configuration": {
                "backend": self._configuration.backend.value,
                "n_envs": self._configuration.n_envs,
                "seed": self._configuration.seed,
            },
            "metrics": _metrics_mapping(result.metrics),
            "environments": export_vector_recovery_state(self._environment),
            "process_rng": capture_process_rng_state(),
        }
        checkpoint_started = time.perf_counter()
        recovery = self._store.commit_json_state(
            unit_id=unit_id,
            sequence=sequence,
            state=state,
            crash_at=crash_at,
            write_files=lambda directory: _save_ppo_resume_arrays(self._model, directory),
        )
        debited = self._ledger.debit_validated_unit(
            run_id=self._run_id,
            unit_id=unit_id,
            idempotency_key=f"{self._run_id}-debit-{unit_id}",
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        return AtomicPPOUnitResult(
            metrics=result.metrics,
            recovery=recovery,
            debited=debited,
            training_seconds=training_seconds,
            checkpoint_seconds=checkpoint_seconds,
        )

    @classmethod
    def restore_latest(
        cls,
        *,
        run_id: str,
        configuration: VectorEnvironmentConfig,
        recovery_store: AtomicRecoveryStore,
        ledger: UnitLedger,
    ) -> tuple[AtomicPPOUnitRunner, PPOUnitMetrics, RecoveryRecord]:
        """Recreate the PPO/vector pair and reconcile a renamed-but-undebited unit."""

        record = recovery_store.load_latest()
        if record is None:
            raise ValueError("no verified recovery state exists")
        state = record.state
        if (
            state.get("format") != "dinorl-ppo-unit-recovery-v1"
            or state.get("run_id") != run_id
            or state.get("configuration")
            != {
                "backend": configuration.backend.value,
                "n_envs": configuration.n_envs,
                "seed": configuration.seed,
            }
            or not isinstance(state.get("environments"), list)
        ):
            raise ValueError("recovery state does not match this PPO run")
        environment = create_vector_environment(configuration)
        restore_vector_recovery_state(environment, state["environments"])
        model = load_maskable_ppo(record.directory / "model.zip", environment)
        _restore_ppo_resume_arrays(model, record.directory / "ppo_state.npz")
        restore_process_rng_state(state.get("process_rng"))
        runner = cls(
            run_id=run_id,
            configuration=configuration,
            model=model,
            environment=environment,
            recovery_store=recovery_store,
            ledger=ledger,
        )
        ledger.debit_validated_unit(
            run_id=run_id,
            unit_id=record.unit_id,
            idempotency_key=f"{run_id}-debit-{record.unit_id}",
        )
        return runner, _metrics_from_mapping(state.get("metrics")), record

    @property
    def model(self) -> MaskablePPO:
        """Return the currently live recovered model."""

        return self._model

    @property
    def environment(self) -> VecEnv:
        """Return the currently live recovered vector environment."""

        return self._environment
