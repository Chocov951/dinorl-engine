"""Corrected pure contracts for the RL-S5c-A3 PILOT-R2 audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dinorl_engine.rl.rewards.runtime import CompiledReward


def r2_curriculum_weights(phase: str, *, robustification_unit: int) -> dict[str, float]:
    """Return the corrected random/training-pool mixture for PILOT-R2."""

    if phase not in {"bootstrap", "robustify"}:
        raise ValueError("unknown R2 curriculum phase")
    if type(robustification_unit) is not int or robustification_unit < 0:
        raise ValueError("robustification_unit must be non-negative")
    if phase == "bootstrap":
        return {"random": 1.0, "training_pool": 0.0}
    training = 0.05 if robustification_unit <= 20 else 0.10
    return {"random": 1.0 - training, "training_pool": training}


def not_evaluated_metrics() -> dict[str, object]:
    """Represent missing evidence without numeric sentinels or invented scores."""

    return {
        "evaluation_status": "not_evaluated",
        "gate_unit": None,
        "heldout_score": None,
        "opponent_variance": None,
        "round_limit_rate": None,
        "style_margin": None,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reward_semantics_descriptor(
    reward: CompiledReward,
    *,
    safe_feed_cap: float | None,
    once_per_opportunity: bool,
) -> dict[str, object]:
    """Describe DSL and orchestration semantics that jointly define a reward."""

    package = Path(__file__).resolve().parents[1]
    runtime_files = (
        package / "rewards" / "runtime.py",
        package / "rewards" / "transition.py",
        package / "env" / "single_agent.py",
        Path(__file__).with_name("a3.py"),
    )
    return {
        "format": "s5c-a3-reward-semantics-v2",
        "dsl_cache_key": reward.cache_key,
        "safe_feed": {
            "episode_cap": safe_feed_cap,
            "once_per_opportunity": once_per_opportunity,
            "opportunity_identity": "carcass_id/availability_generation",
            "generation_event": "central_reactivated",
            "counter_scope": "episode",
        },
        "runtime_component_sha256": {
            path.relative_to(package).as_posix(): _file_sha256(path) for path in runtime_files
        },
    }


def reward_semantics_sha256(
    reward: CompiledReward,
    *,
    safe_feed_cap: float | None,
    once_per_opportunity: bool,
) -> str:
    descriptor = reward_semantics_descriptor(
        reward,
        safe_feed_cap=safe_feed_cap,
        once_per_opportunity=once_per_opportunity,
    )
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def reward_warning_is_blocking(warning: str, *, bounded: bool, exploitable: bool) -> bool:
    """Only downgrade the heuristic repeatability warning when bounded and harmless."""

    if warning == "round_limit_above_10_percent":
        return False
    if warning == "repeatable_action_reward_risk":
        return not bounded or exploitable
    return True


@dataclass(frozen=True, slots=True)
class R2CompositeEvidence:
    historical_gate_passed: bool
    heldout_score: float
    heldout_style_passed: bool
    reward_hacking_blocking: bool
    round_limit_strategy: bool
    stochastic_collapse: bool


def composite_gate_r2(evidence: R2CompositeEvidence, *, previous_passed: bool) -> dict[str, object]:
    """Apply the corrected gate, whose style evidence comes from held-out games."""

    strength = 0.25 <= evidence.heldout_score <= 0.50
    robustness = not (
        evidence.reward_hacking_blocking
        or evidence.round_limit_strategy
        or evidence.stochastic_collapse
    )
    thresholds = (
        evidence.historical_gate_passed
        and strength
        and evidence.heldout_style_passed
        and robustness
    )
    return {
        "historical_gate_passed": evidence.historical_gate_passed,
        "strength_passed": strength,
        "heldout_style_passed": evidence.heldout_style_passed,
        "robustness_passed": robustness,
        "thresholds_passed": thresholds,
        "previous_passed": previous_passed,
        "passed": thresholds and previous_passed,
        "evidence": asdict(evidence),
    }
