from pathlib import Path

import pytest

from dinorl_engine.rl.s7_config import S7Config, S7ConfigError


def test_s7_requires_three_seeds_and_explicit_measurement_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "s7.json"
    path.write_text(
        '{"format":"dinorl-s7-config-v1","seeds":[19,21,23],"max_units":200,"visible_score":0.6,"gate_score":0.8,"plateau_delta":0.02}',
        encoding="utf-8",
    )
    assert S7Config.load(path).max_units == 200
    path.write_text(
        '{"format":"dinorl-s7-config-v1","seeds":[19],"max_units":200,"visible_score":0.6,"gate_score":0.8,"plateau_delta":0.02}',
        encoding="utf-8",
    )
    with pytest.raises(S7ConfigError):
        S7Config.load(path)


def test_s7_canonical_protocol_matches_the_approved_server_parameters() -> None:
    config = S7Config.load(Path("configs/rl/s7.json"))
    assert config.seeds == (20, 21, 23)
    assert config.max_units == 300
    assert (config.visible_score, config.gate_score, config.plateau_delta) == (0.40, 0.55, 0.01)
