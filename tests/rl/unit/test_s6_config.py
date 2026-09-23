"""RL-S6 frozen configuration contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dinorl_engine.rl.s6_config import S6Config, S6ConfigError

ROOT = Path(__file__).parents[3]


def test_s6_config_freezes_the_common_beta_path() -> None:
    config = S6Config.load(ROOT / "configs" / "rl" / "s6.yaml")

    assert config.architecture == "mlp-compact-v2"
    assert config.architecture_dimensions == (663, 64, 64)
    assert config.training_seeds == (19, 20, 21, 22, 23)
    assert config.max_units == 147
    assert config.training_opponents == ("aggressive-v1", "prudent-v1", "opportunist-v1")


def test_s6_config_rejects_a_specialist_or_old_mlp(tmp_path: Path) -> None:
    source = ROOT / "configs" / "rl" / "s6.yaml"
    value = json.loads(source.read_text(encoding="utf-8"))
    value["architecture_dimensions"] = [663, 128, 64]
    path = tmp_path / "s6.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(S6ConfigError, match="mlp-compact"):
        S6Config.load(path)
