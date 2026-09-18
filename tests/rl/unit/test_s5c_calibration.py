"""RL-S5c-A local contracts; they do not measure learned strength."""

from __future__ import annotations

import pytest

from dinorl_engine.rl.s5c.archive import create_calibration_archive
from dinorl_engine.rl.s5c.calibration import CalibrationError, CalibrationSeedState
from dinorl_engine.rl.s5c.config import S5cConfig
from dinorl_engine.rl.s5c.rewards import ARCHETYPES, specialist_rewards
from dinorl_engine.server_checks.importer import import_archive


def _config() -> S5cConfig:
    return S5cConfig.from_mapping(
        {
            "format": "dinorl-s5c-config-v1",
            "run_id": "s5c-calibration",
            "output_directory": "artifacts/rl/s5c/s5c-calibration",
            "architecture": "mlp-compact-v2",
            "max_rounds": 30,
            "training_opponent": "random-legal-v1",
            "calibration_seeds": [19, 20, 21],
            "production_seeds": [19, 20, 21, 22, 23],
            "evaluation_seeds": [101, 102, 103],
            "max_units": 147,
        }
    )


def test_calibration_config_freezes_the_s5b_decisions() -> None:
    config = _config()

    assert config.architecture == "mlp-compact-v2"
    assert config.max_rounds == 30
    assert config.training_opponent == "random-legal-v1"
    assert config.calibration_seeds == (19, 20, 21)
    assert config.production_seeds == (19, 20, 21, 22, 23)


@pytest.mark.parametrize("field,value", [("architecture", "mlp-deep-v2"), ("max_rounds", 60)])
def test_calibration_rejects_any_unfrozen_s5b_prerequisite(field: str, value: object) -> None:
    raw = _config().to_mapping()
    raw[field] = value

    with pytest.raises(ValueError):
        S5cConfig.from_mapping(raw)


def test_all_specialist_rewards_are_event_based_and_independently_hashed() -> None:
    rewards = specialist_rewards()

    assert set(rewards) == set(ARCHETYPES)
    assert len({reward.cache_key for reward in rewards.values()}) == 3
    assert all("terminal_score" in reward.source for reward in rewards.values())
    assert "manhattan" not in rewards["predator"].source
    assert "shove_moved_target" in rewards["controller"].source


def test_calibration_stops_at_the_second_consecutive_gate_and_cannot_resume() -> None:
    state = CalibrationSeedState(archetype="scavenger", seed=19, max_units=147)

    state.record_evaluation(unit=5, thresholds_passed=True)
    assert state.first_stable_gate is None
    state.record_evaluation(unit=10, thresholds_passed=True)

    assert state.first_stable_gate == 10
    assert state.status == "first_stable_gate"
    with pytest.raises(CalibrationError, match="first_stable_gate"):
        state.record_evaluation(unit=15, thresholds_passed=True)


def test_complete_calibration_archive_is_importable_without_policy_weights(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    run_id = "s5c-calibration"
    (run / "manifest.json").write_text(
        '{"format":"s5c-run-manifest-v1","run_id":"s5c-calibration"}', encoding="utf-8"
    )
    for archetype in ARCHETYPES:
        directory = run / "calibration" / archetype
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(
            "{"
            + f'"archetype":"{archetype}","format":"s5c-calibration-v1","seeds":[{{}},{{}},{{}}]'
            + "}",
            encoding="utf-8",
        )

    archive = create_calibration_archive(run_directory=run, output_directory=tmp_path / "exports")
    imported = import_archive(archive_path=archive, root=tmp_path / "local")

    assert imported["suite"] == "RL-S5c-A"
    assert imported["run_id"] == run_id
