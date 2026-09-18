"""Small real-model smoke test for the RL-S5b phase-A matrix."""

from __future__ import annotations

import json
from pathlib import Path

from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.policies.local_mlp_v2 import LocalMLPV2Architecture
from dinorl_engine.rl.s5b.config import S5bConfig
from dinorl_engine.rl.s5b.crossplay import cross_evaluate
from dinorl_engine.rl.s5b.pool import freeze_pool
from dinorl_engine.rl.training.unit import create_maskable_ppo, save_maskable_ppo


def test_real_models_complete_the_smallest_resumable_crossplay_matrix(tmp_path: Path) -> None:
    checkpoints: list[dict[str, object]] = []
    for architecture, seed in (
        (LocalMLPV2Architecture.COMPACT, 19),
        (LocalMLPV2Architecture.BALANCED, 20),
    ):
        directory = tmp_path / "models" / architecture.value / "units" / "unit-147"
        directory.mkdir(parents=True)
        environment = DinoRLSingleAgentEnv(seed=seed)
        try:
            model = create_maskable_ppo(environment, seed=seed, architecture=architecture)
            save_maskable_ppo(model, directory / "model.zip")
        finally:
            environment.close()
        (directory / "state.json").write_text(
            json.dumps({"configuration": {"backend": "dummy", "n_envs": 2, "seed": seed}}),
            encoding="utf-8",
        )
        (directory / "ppo_state.npz").write_bytes(b"not-used-by-crossplay")
        (directory / "manifest.json").write_text("{}", encoding="utf-8")
        checkpoints.append(
            {
                "id": f"{architecture.value}-{seed}",
                "architecture": architecture.value,
                "seed": seed,
                "directory": str(directory),
            }
        )
    config = S5bConfig.from_mapping(
        {
            "format": "dinorl-s5b-config-v1",
            "run_id": "s5b-real-smoke",
            "output_directory": "artifacts",
            "checkpoints": checkpoints,
            "final_architectures": ["mlp-compact-v2", "mlp-balanced-v2"],
            "confrontations": 1,
            "evaluation_seeds": [21],
        },
        base_directory=tmp_path,
    )

    result = cross_evaluate(pool_path=freeze_pool(config), output_directory=tmp_path / "artifacts")

    assert result["records_completed"] == 3
    assert result["games_completed"] == 12
    records = result["records"]
    assert isinstance(records, list)
    assert all(
        isinstance(record, dict)
        and set(record["score_by_role"]) == {"first", "second"}
        and set(record["positions"]) == {"A-A", "A-B", "B-A", "B-B"}
        for record in records
    )
