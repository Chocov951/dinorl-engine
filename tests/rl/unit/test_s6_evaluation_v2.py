"""TDD contracts for the non-retroactive paired RL-S6 v2 gate."""

from __future__ import annotations

from pathlib import Path

from dinorl_engine.rl.s6_evaluation_v2 import (
    PAIRED_BOOTSTRAP_REPLICATES,
    PAIRED_CONFRONTATIONS,
    paired_gate_v2,
    paired_specs_v2,
)
from dinorl_engine.rl.s6_evaluation_v2_config import S6EvaluationV2Config


def _records(*, deterministic: float, stochastic: float) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for opponent in ("aggressive-v1", "prudent-v1", "opportunist-v1", "random-legal-v1"):
        for repetition in range(40):
            for position in range(4):
                records.append(
                    {
                        "opponent_id": opponent,
                        "pair_id": f"{opponent}/{repetition}",
                        "position": position,
                        "deterministic_score": deterministic,
                        "stochastic_score": stochastic,
                    }
                )
    return records


def test_v2_uses_identical_four_position_game_identities_for_both_modes() -> None:
    specifications = paired_specs_v2(seed=101, opponent_id="prudent-v1", confrontations=2)

    assert len(specifications) == 8
    assert len({item.seed for item in specifications[:4]}) == 1
    assert {(item.learner_actor.name, item.first_actor.name) for item in specifications[:4]} == {
        ("A", "A"),
        ("A", "B"),
        ("B", "A"),
        ("B", "B"),
    }
    assert PAIRED_CONFRONTATIONS == 200
    assert PAIRED_BOOTSTRAP_REPLICATES == 10_000


def test_v2_accepts_only_when_paired_confidence_bounds_clear_all_thresholds() -> None:
    result = paired_gate_v2(_records(deterministic=0.90, stochastic=0.88), bootstrap_seed=7)

    assert result["decision"] == "pass"
    assert result["extension_required"] is False


def test_v2_extends_an_inconclusive_drop_and_rejects_a_clear_collapse() -> None:
    borderline = _records(deterministic=0.90, stochastic=0.90)
    for record in borderline:
        if record["opponent_id"] == "prudent-v1":
            repetition = int(str(record["pair_id"]).rsplit("/", 1)[1])
            record["stochastic_score"] = 0.70 if repetition % 2 else 0.80
    inconclusive = paired_gate_v2(borderline, bootstrap_seed=7)
    collapse = paired_gate_v2(_records(deterministic=1.00, stochastic=0.50), bootstrap_seed=7)

    assert inconclusive["decision"] == "inconclusive"
    assert inconclusive["extension_required"] is True
    assert collapse["decision"] == "fail"
    assert collapse["failed_criteria"] == [
        "random_score",
        "main_score",
        "main_drop",
        "individual_drops",
    ]


def test_v2_keeps_the_individual_drop_as_a_blocking_criterion() -> None:
    records = _records(deterministic=0.90, stochastic=0.90)
    for record in records:
        if record["opponent_id"] == "prudent-v1":
            record["stochastic_score"] = 0.70

    result = paired_gate_v2(records, bootstrap_seed=7)

    assert result["decision"] == "fail"
    assert result["failed_criteria"] == ["individual_drops"]


def test_v2_protocol_requires_a_large_predeclared_paired_volume(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        '{"format":"dinorl-s6-evaluation-v2-config-v1","confrontations_per_opponent":200,'
        '"bootstrap_replicates":10000,"bootstrap_seed":20260924}',
        encoding="utf-8",
    )

    assert S6EvaluationV2Config.load(protocol).confrontations_per_opponent == 200
