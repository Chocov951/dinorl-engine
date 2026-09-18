"""Immutable, reproducible opponents for RL-S5b continuation training."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from random import Random
from typing import Final

from dinorl_engine.controllers.protocol import Controller
from dinorl_engine.controllers.random_legal import RANDOM_LEGAL_CONTROLLER_ID, RandomLegalController
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS, create_scripted_controller
from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.policy import MaskablePolicyController
from dinorl_engine.rl.s5b.crossplay import LoadedPolicy, _load_policy

__all__ = ["FrozenOpponentPool", "OpponentPoolError"]

_CATEGORY_WEIGHTS: Final = {
    "random": 10,
    "scripted": 20,
    "final": 50,
    "historical": 20,
}


class OpponentPoolError(RuntimeError):
    """Raised when the declared frozen opponent pool is incompatible."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    identifier: str
    category: str
    entry: Mapping[str, object] | None


def _seed(selection_seed: int, *, domain: str, identifier: str) -> int:
    payload = f"dinorl/s5b/{selection_seed}/{domain}/{identifier}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2**63


class FrozenOpponentPool:
    """Use only the pool's versioned controllers and frozen checkpoint artifacts.

    The pool never references a learner model.  A policy candidate is loaded once
    from its verified, phase-A checkpoint and only a per-game inference stream is
    mutable.  Missing historical checkpoints are deliberately excluded and the
    effective category weights are exposed for the phase-B report.
    """

    def __init__(
        self,
        pool: Mapping[str, object],
        *,
        random_only: bool = False,
        category_weights: Mapping[str, int] | None = None,
    ) -> None:
        entries = pool.get("entries")
        if not isinstance(entries, list):
            raise OpponentPoolError("frozen pool has no entries")
        candidates: dict[str, list[_Candidate]] = {name: [] for name in _CATEGORY_WEIGHTS}
        self._loaded: dict[str, LoadedPolicy] = {}
        for raw in entries:
            if not isinstance(raw, Mapping) or raw.get("status") != "available":
                continue
            identifier = raw.get("id")
            kind = raw.get("kind")
            unit = raw.get("unit")
            if not isinstance(identifier, str):
                continue
            if kind == "controller":
                category = "random" if identifier == RANDOM_LEGAL_CONTROLLER_ID else "scripted"
                if category == "scripted" and identifier not in SCRIPTED_CONTROLLER_IDS:
                    continue
                candidates[category].append(_Candidate(identifier, category, None))
            elif kind == "checkpoint":
                declared_category = raw.get("s5b_category")
                if declared_category is not None and declared_category != "historical":
                    raise OpponentPoolError("checkpoint s5b_category is invalid")
                category = (
                    "historical"
                    if declared_category == "historical"
                    else "final"
                    if unit == 147
                    else "historical"
                )
                candidates[category].append(_Candidate(identifier, category, raw))
        weights = dict(_CATEGORY_WEIGHTS if category_weights is None else category_weights)
        if set(weights) != set(_CATEGORY_WEIGHTS) or any(
            type(weight) is not int or weight < 0 for weight in weights.values()
        ):
            raise OpponentPoolError("opponent category weights are invalid")
        if random_only:
            candidates = {name: [] for name in _CATEGORY_WEIGHTS}
            candidates["random"] = [_Candidate(RANDOM_LEGAL_CONTROLLER_ID, "random", None)]
        if not any(candidates.values()):
            raise OpponentPoolError("frozen pool has no available training opponent")
        self._candidates = {name: tuple(value) for name, value in candidates.items()}
        self._weights = weights
        self._refresh_effective_weights()

    def _refresh_effective_weights(self) -> None:
        available_weight = sum(
            self._weights[name] for name, value in self._candidates.items() if value
        )
        if available_weight <= 0:
            raise OpponentPoolError("available opponent categories have no positive weight")
        self._effective_weights = {
            name: (self._weights[name] / available_weight if value else 0.0)
            for name, value in self._candidates.items()
        }

    def add_historical_snapshot(self, entry: Mapping[str, object]) -> None:
        """Add an already persisted learner snapshot at a declared league boundary."""

        identifier = entry.get("id")
        if (
            not isinstance(identifier, str)
            or entry.get("kind") != "checkpoint"
            or entry.get("status") != "available"
            or not isinstance(entry.get("provenance"), str)
            or identifier in {item.identifier for item in self._candidates["historical"]}
        ):
            raise OpponentPoolError("self-play snapshot entry is invalid")
        self._candidates["historical"] = (
            *self._candidates["historical"],
            _Candidate(identifier, "historical", entry),
        )
        self._refresh_effective_weights()

    @property
    def effective_weights(self) -> dict[str, float]:
        """Return the normalized weights after declared missing categories are removed."""

        return dict(self._effective_weights)

    @property
    def available_categories(self) -> dict[str, list[str]]:
        """Return candidate IDs without exposing mutable policy objects."""

        return {
            name: [candidate.identifier for candidate in values]
            for name, values in self._candidates.items()
        }

    def _candidate(self, selection_seed: int) -> _Candidate:
        if type(selection_seed) is not int or not 0 <= selection_seed < 2**63:
            raise OpponentPoolError("opponent selection seed is invalid")
        random = Random(selection_seed)
        position = random.random()
        cumulative = 0.0
        category = ""
        for name in _CATEGORY_WEIGHTS:
            cumulative += self._effective_weights[name]
            if position < cumulative:
                category = name
                break
        if not category:
            category = next(name for name, value in self._candidates.items() if value)
        return random.choice(self._candidates[category])

    def _controller(
        self,
        candidate: _Candidate,
        *,
        selection_seed: int,
        learner_actor: Actor,
        first_actor: Actor,
    ) -> Controller:
        if candidate.identifier == RANDOM_LEGAL_CONTROLLER_ID:
            return RandomLegalController(
                seed=_seed(selection_seed, domain="random", identifier=candidate.identifier)
            )
        if candidate.identifier in SCRIPTED_CONTROLLER_IDS:
            return create_scripted_controller(candidate.identifier)
        loaded = self._loaded.get(candidate.identifier)
        if loaded is None:
            if candidate.entry is None:
                raise OpponentPoolError("checkpoint opponent has no pool entry")
            loaded = _load_policy(candidate.entry)
            self._loaded[candidate.identifier] = loaded
        return MaskablePolicyController(
            loaded.model,
            learner_actor=Actor.B if learner_actor is Actor.A else Actor.A,
            first_actor=first_actor,
            deterministic=False,
            stochastic_seed=_seed(selection_seed, domain="policy", identifier=candidate.identifier),
        )

    def select(
        self, *, selection_seed: int, learner_actor: Actor, first_actor: Actor
    ) -> tuple[str, Controller]:
        """Select and construct exactly one deterministic frozen opponent."""

        candidate = self._candidate(selection_seed)
        return candidate.identifier, self._controller(
            candidate,
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
        """Recreate and restore precisely the opponent selected for an episode."""

        candidate = self._candidate(selection_seed)
        if candidate.identifier != opponent_id:
            raise OpponentPoolError("S5b recovery opponent does not match its frozen selection")
        controller = self._controller(
            candidate,
            selection_seed=selection_seed,
            learner_actor=learner_actor,
            first_actor=first_actor,
        )
        restore = getattr(controller, "restore_recovery_state", None)
        if controller_state is not None:
            if not callable(restore):
                raise OpponentPoolError("S5b opponent cannot restore its recovery state")
            restore(controller_state)
        return controller

    def close(self) -> None:
        """Release the temporary inference environments owned by loaded snapshots."""

        for loaded in self._loaded.values():
            loaded.close()
        self._loaded.clear()
