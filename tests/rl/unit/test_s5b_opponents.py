"""Unit contracts for the immutable continuation opponent selector."""

from __future__ import annotations

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.s5b.opponents import FrozenOpponentPool


def test_missing_history_is_explicitly_renormalized() -> None:
    pool = FrozenOpponentPool(
        {
            "entries": [
                {
                    "id": "random-legal-v1",
                    "kind": "controller",
                    "status": "available",
                    "unit": None,
                },
                {
                    "id": "aggressive-v1",
                    "kind": "controller",
                    "status": "available",
                    "unit": None,
                },
                {
                    "id": "compact-final",
                    "kind": "checkpoint",
                    "status": "available",
                    "unit": 147,
                },
            ]
        }
    )
    assert pool.effective_weights == {
        "random": 0.125,
        "scripted": 0.25,
        "final": 0.625,
        "historical": 0.0,
    }
    random_only = FrozenOpponentPool(
        {
            "entries": [
                {"id": "random-legal-v1", "kind": "controller", "status": "available", "unit": None}
            ]
        },
        random_only=True,
    )
    identifier, controller = random_only.select(
        selection_seed=42, learner_actor=Actor.A, first_actor=Actor.A
    )
    assert identifier == "random-legal-v1"
    assert controller is not None


def test_snapshot_is_added_only_as_a_historical_opponent() -> None:
    pool = FrozenOpponentPool(
        {
            "entries": [
                {
                    "id": "random-legal-v1",
                    "kind": "controller",
                    "status": "available",
                    "unit": None,
                }
            ]
        },
        category_weights={"random": 10, "scripted": 0, "final": 0, "historical": 50},
    )
    pool.add_historical_snapshot(
        {
            "id": "learner-snapshot-157",
            "kind": "checkpoint",
            "provenance": "C:/immutable/snapshot",
            "status": "available",
        }
    )
    assert pool.available_categories["historical"] == ["learner-snapshot-157"]
    assert pool.effective_weights == {
        "random": 1 / 6,
        "scripted": 0.0,
        "final": 0.0,
        "historical": 5 / 6,
    }
