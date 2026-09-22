"""Deterministic, auditable opponent curriculum for RL-S5c-A3."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from dinorl_engine.controllers.protocol import Controller
from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.s5b.opponents import FrozenOpponentPool, OpponentPoolError
from dinorl_engine.rl.s5c.a3 import (
    A3ProtocolError,
    CurriculumPhase,
    curriculum_weights,
)
from dinorl_engine.rl.s5c.a3_r2 import r2_curriculum_weights


@dataclass(frozen=True, slots=True)
class CurriculumSchedule:
    """Frozen inputs governing one curriculum interval."""

    phase: CurriculumPhase
    robustification_unit: int
    training_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if set(self.training_ids) & set(self.heldout_ids):
            raise A3ProtocolError("training and held-out opponent pools overlap")
        if not self.training_ids:
            raise A3ProtocolError("training opponent pool is empty")
        curriculum_weights(self.phase, robustification_unit=self.robustification_unit)


def curriculum_category(selection_seed: int, *, schedule: CurriculumSchedule) -> str:
    """Choose a curriculum category from a stable domain-separated RNG stream."""

    if type(selection_seed) is not int or not 0 <= selection_seed < 2**63:
        raise A3ProtocolError("curriculum selection seed is invalid")
    weights = curriculum_weights(schedule.phase, robustification_unit=schedule.robustification_unit)
    digest = hashlib.sha256(f"dinorl/s5c-a3/curriculum/{selection_seed}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return "training_pool" if draw >= weights["random"] else "random"


def _training_pool_mapping(
    pool: Mapping[str, object], *, training_ids: tuple[str, ...]
) -> dict[str, object]:
    entries = pool.get("entries")
    if not isinstance(entries, list):
        raise A3ProtocolError("frozen opponent pool has no entries")
    selected = [
        dict(entry)
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("id") in training_ids
    ]
    found = {entry.get("id") for entry in selected}
    missing = set(training_ids) - found
    if missing:
        raise A3ProtocolError(f"training opponents are missing: {sorted(missing)}")
    return {**pool, "entries": selected}


class A3CurriculumOpponentPool:
    """Route each episode to random or the immutable A3 training subset."""

    def __init__(
        self,
        pool: Mapping[str, object],
        *,
        schedule: CurriculumSchedule,
    ) -> None:
        self._schedule = schedule
        self._random = FrozenOpponentPool(pool, random_only=True)
        self._training = FrozenOpponentPool(
            _training_pool_mapping(pool, training_ids=schedule.training_ids),
            category_weights={"random": 0, "scripted": 0, "final": 1, "historical": 1},
        )
        self._training_ids = frozenset(schedule.training_ids)
        self._counts = {"random": 0, "training_pool": 0}

    def set_schedule(self, schedule: CurriculumSchedule) -> None:
        if schedule.training_ids != self._schedule.training_ids:
            raise A3ProtocolError("training opponent subset cannot change during a run")
        if schedule.heldout_ids != self._schedule.heldout_ids:
            raise A3ProtocolError("held-out opponent subset cannot change during a run")
        self._schedule = schedule

    @property
    def schedule(self) -> CurriculumSchedule:
        return self._schedule

    @property
    def actual_counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def effective_weights(self) -> dict[str, float]:
        return curriculum_weights(
            self._schedule.phase,
            robustification_unit=self._schedule.robustification_unit,
        )

    def select(
        self, *, selection_seed: int, learner_actor: Actor, first_actor: Actor
    ) -> tuple[str, Controller]:
        category = curriculum_category(selection_seed, schedule=self._schedule)
        self._counts[category] += 1
        selected = self._training if category == "training_pool" else self._random
        return selected.select(
            selection_seed=selection_seed,
            learner_actor=learner_actor,
            first_actor=first_actor,
        )

    def restore(
        self,
        *,
        opponent_id: str,
        selection_seed: int,
        learner_actor: Actor,
        first_actor: Actor,
        controller_state: object,
    ) -> Controller:
        # Route by the persisted opponent id, not by the current curriculum phase.
        # This keeps an in-flight episode exactly recoverable across a phase switch.
        selected = self._training if opponent_id in self._training_ids else self._random
        try:
            return selected.restore(
                opponent_id=opponent_id,
                selection_seed=selection_seed,
                learner_actor=learner_actor,
                first_actor=first_actor,
                controller_state=controller_state,
            )
        except OpponentPoolError as error:
            raise A3ProtocolError("curriculum recovery opponent is invalid") from error

    def close(self) -> None:
        self._random.close()
        self._training.close()


@dataclass(frozen=True, slots=True)
class R2CurriculumSchedule:
    """Corrected PILOT-R2 schedule with a softer training-pool mixture."""

    phase: str
    robustification_unit: int
    training_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if set(self.training_ids) & set(self.heldout_ids):
            raise A3ProtocolError("training and held-out opponent pools overlap")
        if not self.training_ids:
            raise A3ProtocolError("training opponent pool is empty")
        r2_curriculum_weights(self.phase, robustification_unit=self.robustification_unit)


def r2_curriculum_category(selection_seed: int, *, schedule: R2CurriculumSchedule) -> str:
    if type(selection_seed) is not int or not 0 <= selection_seed < 2**63:
        raise A3ProtocolError("curriculum selection seed is invalid")
    weights = r2_curriculum_weights(
        schedule.phase, robustification_unit=schedule.robustification_unit
    )
    digest = hashlib.sha256(f"dinorl/s5c-a3-r2/curriculum/{selection_seed}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return "training_pool" if draw >= weights["random"] else "random"


class A3R2CurriculumOpponentPool:
    """Route R2 episodes without ever exposing the held-out subset to training."""

    def __init__(self, pool: Mapping[str, object], *, schedule: R2CurriculumSchedule) -> None:
        self._schedule = schedule
        self._random = FrozenOpponentPool(pool, random_only=True)
        self._training = FrozenOpponentPool(
            _training_pool_mapping(pool, training_ids=schedule.training_ids),
            category_weights={"random": 0, "scripted": 0, "final": 1, "historical": 1},
        )
        self._training_ids = frozenset(schedule.training_ids)
        self._counts = {"random": 0, "training_pool": 0}

    def set_schedule(self, schedule: R2CurriculumSchedule) -> None:
        if schedule.training_ids != self._schedule.training_ids:
            raise A3ProtocolError("training opponent subset cannot change during a run")
        if schedule.heldout_ids != self._schedule.heldout_ids:
            raise A3ProtocolError("held-out opponent subset cannot change during a run")
        self._schedule = schedule

    @property
    def schedule(self) -> R2CurriculumSchedule:
        return self._schedule

    @property
    def actual_counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def effective_weights(self) -> dict[str, float]:
        return r2_curriculum_weights(
            self._schedule.phase,
            robustification_unit=self._schedule.robustification_unit,
        )

    def select(
        self, *, selection_seed: int, learner_actor: Actor, first_actor: Actor
    ) -> tuple[str, Controller]:
        category = r2_curriculum_category(selection_seed, schedule=self._schedule)
        self._counts[category] += 1
        selected = self._training if category == "training_pool" else self._random
        return selected.select(
            selection_seed=selection_seed,
            learner_actor=learner_actor,
            first_actor=first_actor,
        )

    def restore(
        self,
        *,
        opponent_id: str,
        selection_seed: int,
        learner_actor: Actor,
        first_actor: Actor,
        controller_state: object,
    ) -> Controller:
        selected = self._training if opponent_id in self._training_ids else self._random
        try:
            return selected.restore(
                opponent_id=opponent_id,
                selection_seed=selection_seed,
                learner_actor=learner_actor,
                first_actor=first_actor,
                controller_state=controller_state,
            )
        except OpponentPoolError as error:
            raise A3ProtocolError("R2 curriculum recovery opponent is invalid") from error

    def close(self) -> None:
        self._random.close()
        self._training.close()
