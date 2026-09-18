"""RL-S5b paired cross-play contracts, using no expensive PPO training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation.tournament import PolicyDuelOutcome
from dinorl_engine.rl.s5b.archive import create_crossplay_archive
from dinorl_engine.rl.s5b.config import S5bConfig
from dinorl_engine.rl.s5b.crossplay import LoadedPolicy, cross_evaluate
from dinorl_engine.rl.s5b.pool import freeze_pool
from dinorl_engine.server_checks.importer import import_archive


def _pool(tmp_path: Path) -> Path:
    checkpoints = []
    for architecture, seed in (("mlp-v1", 19), ("mlp-compact-v2", 20)):
        directory = tmp_path / "checkpoints" / architecture / "units" / "unit-147"
        directory.mkdir(parents=True)
        for name in ("manifest.json", "model.zip", "ppo_state.npz", "state.json"):
            (directory / name).write_bytes(f"{architecture}-{name}".encode())
        checkpoints.append(
            {
                "id": f"{architecture}-{seed}",
                "architecture": architecture,
                "seed": seed,
                "directory": str(directory),
            }
        )
    config = S5bConfig.from_mapping(
        {
            "format": "dinorl-s5b-config-v1",
            "run_id": "crossplay-smoke",
            "output_directory": "artifacts",
            "checkpoints": checkpoints,
            "final_architectures": ["mlp-v1", "mlp-compact-v2"],
            "confrontations": 2,
            "evaluation_seeds": [19, 20],
        },
        base_directory=tmp_path,
    )
    return freeze_pool(config)


def test_crossplay_covers_all_pairs_positions_and_resumes(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    calls: list[tuple[int, Actor, Actor]] = []

    def loader(entry: Mapping[str, object]) -> LoadedPolicy:
        return LoadedPolicy(model=entry["id"], close=lambda: None)

    def duel(
        left: object,
        right: object,
        *,
        seed: int,
        left_actor: Actor,
        first_actor: Actor,
        deterministic: bool,
        matchup_id: str,
    ) -> PolicyDuelOutcome:
        assert deterministic is False
        assert isinstance(left, str) and isinstance(right, str) and matchup_id
        calls.append((seed, left_actor, first_actor))
        return PolicyDuelOutcome(
            wins=int(left_actor is first_actor),
            draws=0,
            losses=int(left_actor is not first_actor),
            rounds=4,
            actions=12,
        )

    first = cross_evaluate(
        pool_path=pool,
        output_directory=tmp_path / "artifacts",
        model_loader=loader,
        duel=duel,
    )
    completed_calls = len(calls)
    second = cross_evaluate(
        pool_path=pool,
        output_directory=tmp_path / "artifacts",
        model_loader=loader,
        duel=duel,
    )

    # 2 checkpoints: 3 unordered pairs including the diagonal, 2 seeds,
    # 2 confrontations and the four required seat/initiative configurations.
    assert completed_calls == 3 * 2 * 2 * 4
    assert len(calls) == completed_calls
    assert first["records_completed"] == second["records_completed"] == 6
    assert first["games_completed"] == 48
    assert (tmp_path / "artifacts" / "crossplay" / "crossplay.json").is_file()
    assert (tmp_path / "artifacts" / "crossplay" / "crossplay.csv").is_file()
    assert (tmp_path / "artifacts" / "crossplay" / "crossplay-report.md").is_file()
    manifest = json.loads((tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phases"]["cross-evaluate"]["status"] == "completed"
    archive = create_crossplay_archive(
        run_directory=tmp_path / "artifacts", output_directory=tmp_path / "exports"
    )
    imported = import_archive(archive_path=archive, root=tmp_path / "local")
    destination = Path(str(imported["destination"]))
    assert imported["suite"] == "RL-S5b-A"
    assert (destination / "crossplay.json").is_file()
    assert (destination / "crossplay-report.md").is_file()


def test_crossplay_preserves_paired_seed_across_first_player_inversion(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    seen: dict[int, set[tuple[Actor, Actor]]] = {}

    def loader(entry: Mapping[str, object]) -> LoadedPolicy:
        return LoadedPolicy(model=entry["id"], close=lambda: None)

    def duel(
        left: object,
        right: object,
        *,
        seed: int,
        left_actor: Actor,
        first_actor: Actor,
        deterministic: bool,
        matchup_id: str,
    ) -> PolicyDuelOutcome:
        seen.setdefault(seed, set()).add((left_actor, first_actor))
        return PolicyDuelOutcome(wins=0, draws=1, losses=0, rounds=1, actions=1)

    cross_evaluate(
        pool_path=pool,
        output_directory=tmp_path / "artifacts",
        model_loader=loader,
        duel=duel,
    )

    assert all(positions == {(a, b) for a in Actor for b in Actor} for positions in seen.values())
