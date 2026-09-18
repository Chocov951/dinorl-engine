"""RL CLI declarations are usable before their implementations land."""

import json
import sys

import pytest

from dinorl_engine.rl.__main__ import main


@pytest.mark.parametrize(
    "arguments,command",
    [
        (["train", "--config", "experiment.json", "--json"], "train"),
        (["resume", "--checkpoint", "checkpoint-1", "--units", "20", "--json"], "resume"),
        (
            ["evaluate", "--checkpoint", "checkpoint-1", "--suite", "deterministic", "--json"],
            "evaluate",
        ),
        (["publish", "--checkpoint", "checkpoint-1", "--json"], "publish"),
        (["inspect", "--run", "run-1", "--json"], "inspect"),
        (["status", "--job", "job-1", "--follow", "--json"], "status"),
        (["cancel", "--job", "job-1", "--json"], "cancel"),
    ],
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
    monkeypatch.setattr(sys, "argv", ["dinorl-rl", "status", "--job", "job-1"])

    assert main() == 3

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "status: not_implemented\n"


def test_s5b_staged_command_exposes_a_machine_readable_checkpoint_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["s5b", "continue", "--checkpoint", "candidate-19", "--units", "50", "--json"]) == 3

    assert json.loads(capsys.readouterr().out) == {
        "command": "s5b continue",
        "reason": "requires_completion_of_the_prior_server_checkpoint",
        "status": "not_implemented",
    }
