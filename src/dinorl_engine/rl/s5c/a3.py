"""Pure protocol contracts for the two-profile RL-S5c-A3 campaign."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from dinorl_engine.core.constants import Actor
from dinorl_engine.core.events import (
    ActionTransition,
    CentralReactivatedEffect,
    ConsumptionStartedEffect,
)

ACTIVE_V1_PROFILES: Final = ("scavenger", "predator")
PUBLIC_PROFILE_LABELS: Final = {"scavenger": "Scavenger", "predator": "Agressif"}
CONTROLLER_STATUS: Final = "deferred_post_v1"


class A3ProtocolError(RuntimeError):
    """Raised when an A3 phase would violate a frozen protocol decision."""


class CurriculumPhase(StrEnum):
    BOOTSTRAP = "bootstrap"
    ROBUSTIFY = "robustify"


@dataclass(frozen=True, slots=True)
class OpponentSplit:
    training_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]
    training_sha256: str
    heldout_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "format": "s5c-a3-opponent-split-v1",
            "training_ids": list(self.training_ids),
            "heldout_ids": list(self.heldout_ids),
            "training_sha256": self.training_sha256,
            "heldout_sha256": self.heldout_sha256,
        }


def _hash_ids(values: tuple[str, ...]) -> str:
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def deterministic_opponent_split(entries: Sequence[Mapping[str, object]]) -> OpponentSplit:
    """Split each architecture deterministically, before observing any A3 result."""

    normalized: list[tuple[str, str, int]] = []
    for entry in entries:
        identifier, architecture, seed = (
            entry.get("id"),
            entry.get("architecture"),
            entry.get("seed"),
        )
        if (
            not isinstance(identifier, str)
            or not isinstance(architecture, str)
            or type(seed) is not int
        ):
            raise A3ProtocolError("opponent split entry is invalid")
        normalized.append((architecture, identifier, seed))
    if len({identifier for _, identifier, _ in normalized}) != len(normalized):
        raise A3ProtocolError("opponent split identifiers must be unique")
    training: list[str] = []
    heldout: list[str] = []
    for architecture in sorted({item[0] for item in normalized}):
        group = sorted(
            (item for item in normalized if item[0] == architecture), key=lambda x: (x[2], x[1])
        )
        if len(group) < 2:
            raise A3ProtocolError("each architecture needs train and held-out opponents")
        # The common pilot seed 20 is assigned to training when present; all other
        # seeds stay held out. This rule is fixed independently of A3 outcomes.
        chosen = next((item for item in group if item[2] == 20), group[len(group) // 2])
        training.append(chosen[1])
        heldout.extend(item[1] for item in group if item != chosen)
    training_ids, heldout_ids = tuple(sorted(training)), tuple(sorted(heldout))
    if set(training_ids) & set(heldout_ids):
        raise A3ProtocolError("training and held-out pools overlap")
    return OpponentSplit(training_ids, heldout_ids, _hash_ids(training_ids), _hash_ids(heldout_ids))


def curriculum_weights(phase: CurriculumPhase, *, robustification_unit: int) -> dict[str, float]:
    if type(robustification_unit) is not int or robustification_unit < 0:
        raise A3ProtocolError("robustification_unit must be non-negative")
    if phase is CurriculumPhase.BOOTSTRAP:
        return {"random": 1.0, "training_pool": 0.0}
    pool = 0.1 if robustification_unit <= 20 else 0.2
    return {"random": 1.0 - pool, "training_pool": pool}


@dataclass(frozen=True, slots=True)
class CompositeEvidence:
    random_score: float
    deterministic_scores: tuple[float, float, float]
    absolute_style_passed: bool
    comparative_style_passed: bool
    heldout_score: float
    reward_hacking_blocking: bool
    round_limit_strategy: bool
    stochastic_collapse: bool
    replay_identity_visible: bool


@dataclass(slots=True)
class EpisodeAuxiliaryBudget:
    """Bound repeatable A3 rewards using stable resource-opportunity identifiers."""

    safe_feed_cap: float
    safe_feed_total: float = 0.0
    _paid_opportunities: set[str] | None = None

    def __post_init__(self) -> None:
        if self.safe_feed_cap <= 0.0:
            raise A3ProtocolError("safe-feed cap must be positive")
        self._paid_opportunities = set()

    def safe_feed(self, opportunity_id: str, amount: float) -> float:
        assert self._paid_opportunities is not None
        if opportunity_id in self._paid_opportunities or amount <= 0.0:
            return 0.0
        self._paid_opportunities.add(opportunity_id)
        paid = min(amount, max(0.0, self.safe_feed_cap - self.safe_feed_total))
        self.safe_feed_total += paid
        return paid

    def to_mapping(self) -> dict[str, object]:
        assert self._paid_opportunities is not None
        return {
            "safe_feed_cap": self.safe_feed_cap,
            "safe_feed_total": self.safe_feed_total,
            "paid_opportunities": sorted(self._paid_opportunities),
        }

    @classmethod
    def from_mapping(cls, value: object) -> EpisodeAuxiliaryBudget:
        if not isinstance(value, Mapping):
            raise A3ProtocolError("safe-feed recovery budget is invalid")
        cap = value.get("safe_feed_cap")
        total = value.get("safe_feed_total")
        paid = value.get("paid_opportunities")
        if (
            not isinstance(cap, int | float)
            or not isinstance(total, int | float)
            or not isinstance(paid, list)
            or not all(isinstance(item, str) for item in paid)
            or not 0.0 <= float(total) <= float(cap)
        ):
            raise A3ProtocolError("safe-feed recovery budget is invalid")
        result = cls(safe_feed_cap=float(cap), safe_feed_total=float(total))
        result._paid_opportunities = set(paid)
        return result


def bounded_safe_feed_adjustment(
    *,
    transition: ActionTransition,
    reward_transition: Mapping[str, object],
    learner_actor: Actor,
    budget: EpisodeAuxiliaryBudget,
    generations: dict[str, int],
    nominal_amount: float = 0.03,
) -> float:
    """Return the delta to the A2 DSL safe-feed term and advance resource generations."""

    adjustment = 0.0
    started = next(
        (
            effect
            for effect in transition.effects
            if isinstance(effect, ConsumptionStartedEffect) and effect.actor is learner_actor
        ),
        None,
    )
    if started is not None:
        before = reward_transition.get("before")
        own = before.get("self") if isinstance(before, Mapping) else None
        opponent = before.get("opponent") if isinstance(before, Mapping) else None
        own_position = own.get("position") if isinstance(own, Mapping) else None
        opponent_position = opponent.get("position") if isinstance(opponent, Mapping) else None
        if (
            isinstance(own_position, tuple)
            and isinstance(opponent_position, tuple)
            and len(own_position) == len(opponent_position) == 2
        ):
            distance = abs(int(own_position[0]) - int(opponent_position[0])) + abs(
                int(own_position[1]) - int(opponent_position[1])
            )
            if distance > 3:
                generation = generations.get(started.carcass_id, 0)
                paid = budget.safe_feed(
                    f"{started.carcass_id}/availability-{generation}", nominal_amount
                )
                adjustment = paid - nominal_amount
    for effect in (*transition.effects, *transition.automatic_effects):
        if isinstance(effect, CentralReactivatedEffect):
            generations[effect.carcass_id] = generations.get(effect.carcass_id, 0) + 1
    return adjustment


def rule_dominates(terms: Mapping[str, float], *, maximum_share: float) -> bool:
    absolute = [abs(float(value)) for value in terms.values()]
    total = sum(absolute)
    return bool(total and max(absolute) / total > maximum_share)


def composite_gate(evidence: CompositeEvidence, *, previous_passed: bool) -> dict[str, object]:
    historical = (
        evidence.random_score >= 0.90
        and sum(evidence.deterministic_scores) / 3 > 0.55
        and all(score > 0.40 for score in evidence.deterministic_scores)
    )
    style = evidence.absolute_style_passed and evidence.comparative_style_passed
    strength = 0.25 <= evidence.heldout_score <= 0.50
    robust = (
        not any(
            (
                evidence.reward_hacking_blocking,
                evidence.round_limit_strategy,
                evidence.stochastic_collapse,
            )
        )
        and evidence.replay_identity_visible
    )
    thresholds = historical and style and strength and robust
    return {
        "historical_gate_passed": historical,
        "style_passed": style,
        "strength_passed": strength,
        "robustness_passed": robust,
        "thresholds_passed": thresholds,
        "previous_passed": previous_passed,
        "passed": thresholds and previous_passed,
    }


def select_variant(candidates: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    eligible = [candidate for candidate in candidates if candidate.get("eligible") is True]
    if not eligible:
        raise A3ProtocolError("NO_ELIGIBLE_VARIANT")
    return min(
        eligible,
        key=lambda item: (
            int(item["gate_unit"]),
            abs(float(item["heldout_score"]) - 0.40),
            float(item["opponent_variance"]),
            float(item["round_limit_rate"]),
            -float(item["style_margin"]),
            str(item["id"]),
        ),
    )


def require_prior_checkpoint(run_directory: Path, *, phase: str) -> Mapping[str, object]:
    prerequisite = {"pilot": "policy-control/result.json", "confirm": "pilot/selection.json"}.get(
        phase
    )
    if prerequisite is None:
        raise A3ProtocolError("unknown A3 phase")
    path = run_directory / prerequisite
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        decision = "FAILED_POLICY_CONTROL" if phase == "pilot" else "NO_ELIGIBLE_VARIANT"
        raise A3ProtocolError(decision) from error
    expected = "PASSED_POLICY_CONTROL" if phase == "pilot" else "ELIGIBLE_VARIANTS_SELECTED"
    if not isinstance(value, dict) or value.get("decision") != expected:
        raise A3ProtocolError(
            "FAILED_POLICY_CONTROL" if phase == "pilot" else "NO_ELIGIBLE_VARIANT"
        )
    return value
