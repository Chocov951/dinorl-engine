"""Accounting contracts for the server-only S5b turn-limit audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_full_s5b_audit_counts_43200_actual_games() -> None:
    path = Path("scripts/run_rl_s5b_turn_limit_audit.py")
    specification = importlib.util.spec_from_file_location("s5b_turn_limit_audit", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    # 78 unit-147 matrix tasks + 84 unit-197/final tasks + 18 exploiters.
    assert (
        module.planned_game_count(
            task_count=180, evaluation_seed_count=3, confrontations=5, limit_count=4
        )
        == 43_200
    )
