"""RL CLI declarations are usable before their implementations land."""

import json
import sys

import pytest

from dinorl_engine.rl.__main__ import main


@pytest.mark.parametrize(
    "arguments,command",
    [],
)
def test_declared_commands_return_explicit_not_implemented_json(
    arguments: list[str], command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(arguments) == 3

    assert json.loads(capsys.readouterr().out) == {
        "command": command,
        "status": "not_implemented",
    }


def test_human_output_and_default_arguments_are_explicit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["dinorl-rl", "s5b", "continue", "--checkpoint", "job-1"])

    assert main() == 3

    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err
        == "s5b continue: not_implemented (requires_completion_of_the_prior_server_checkpoint)\n"
    )


def test_l9_cli_submits_then_reads_and_cancels_a_training_job(
    tmp_path,
    capsys: pytest.CaptureFixture[str],  # type: ignore[no-untyped-def]
) -> None:
    config = tmp_path / "experiment.json"
    config.write_text('{"experiment_id":"experiment-a","budget_units":2}', encoding="utf-8")
    database = tmp_path / "l9.sqlite3"

    assert main(["train", "--config", str(config), "--database", str(database), "--json"]) == 0
    job_id = json.loads(capsys.readouterr().out)["job_id"]
    assert main(["status", "--job", job_id, "--database", str(database), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["reserved_units"] == 2
    assert main(["cancel", "--job", job_id, "--database", str(database), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["released_units"] == 2


def test_l9_cli_refuses_publication_without_a_valid_eligibility_report(
    tmp_path,
    capsys: pytest.CaptureFixture[str],  # type: ignore[no-untyped-def]
) -> None:
    database = tmp_path / "l9.sqlite3"
    report = tmp_path / "eligibility.json"
    report.write_text('{"eligible":false}', encoding="utf-8")

    assert (
        main(
            [
                "publish",
                "--checkpoint",
                "checkpoint-a",
                "--eligibility",
                str(report),
                "--database",
                str(database),
                "--json",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_s5b_staged_command_exposes_a_machine_readable_checkpoint_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["s5b", "continue", "--checkpoint", "candidate-19", "--units", "50", "--json"]) == 3

    assert json.loads(capsys.readouterr().out) == {
        "command": "s5b continue",
        "reason": "requires_completion_of_the_prior_server_checkpoint",
        "status": "not_implemented",
    }
