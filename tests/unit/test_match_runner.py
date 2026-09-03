"""Complete bounded match execution and CLI tests."""

import json
from itertools import product
from pathlib import Path
from typing import Any

from dinorl_engine.cli import main
from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.match.runner import run_match

ROOT = Path(__file__).parents[2]


def _smoke_configuration() -> dict[str, Any]:
    with (ROOT / "fixtures" / "matches" / "smoke-v1.json").open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def test_runner_executes_a_complete_match_with_an_optional_replay() -> None:
    configuration = _smoke_configuration()

    outcome = run_match(
        map_id=configuration["map_id"],
        seed=configuration["seed"],
        controller_a_id=configuration["controllers"]["A"],
        controller_b_id=configuration["controllers"]["B"],
        include_replay=configuration["include_replay"],
    )

    assert outcome.result.actions <= 240
    assert outcome.result.individual_turns <= 60
    assert outcome.replay is not None
    assert outcome.replay_sha256 is not None
    assert outcome.replay["result"]["winner"] == outcome.summary()["winner"]
    assert len(outcome.replay["events"]) <= 302


def test_runner_without_replay_does_not_build_replay_output() -> None:
    outcome = run_match(
        map_id="arena_mvp_v1",
        seed=3,
        controller_a_id="opportunist-v1",
        controller_b_id="prudent-v1",
        include_replay=False,
    )

    assert outcome.replay is None
    assert outcome.replay_sha256 is None


def test_every_ordered_bot_pair_terminates_on_the_100_seed_corpus() -> None:
    for controller_a_id, controller_b_id in product(SCRIPTED_CONTROLLER_IDS, repeat=2):
        for seed in range(100):
            outcome = run_match(
                map_id="arena_mvp_v1",
                seed=seed,
                controller_a_id=controller_a_id,
                controller_b_id=controller_b_id,
                include_replay=False,
            )
            assert outcome.result.actions <= 240
            assert outcome.result.individual_turns <= 60


def test_cli_prints_a_json_summary_and_replay(capsys: Any) -> None:
    exit_code = main(
        [
            "--seed",
            "42",
            "--controller-a",
            "aggressive-v1",
            "--controller-b",
            "prudent-v1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["result"]["actions"] >= 1
    assert payload["replay"]["seed"] == 42
    assert len(payload["replay_sha256"]) == 64
