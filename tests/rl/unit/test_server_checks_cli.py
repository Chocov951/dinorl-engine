"""Server-check CLI behaviour."""

import json
from unittest.mock import patch

import dinorl_engine.server_checks.__main__  # noqa: F401
from dinorl_engine.server_checks.cli import main
from dinorl_engine.server_checks.runner import DirtyWorktreeError


def test_import_command_reports_a_missing_archive_as_json() -> None:
    output: list[str] = []
    error_output: list[str] = []

    assert (
        main(
            ["import", "server-results-RL-S0-example.tar.gz", "--json"],
            output=output.append,
            error_output=error_output.append,
        )
        == 1
    )

    assert error_output == []
    assert json.loads("".join(output)) == {
        "command": "import",
        "reason": "server archive does not exist: server-results-RL-S0-example.tar.gz",
        "status": "failed",
    }


def test_import_command_reports_a_missing_archive_for_humans() -> None:
    output: list[str] = []
    error_output: list[str] = []

    assert (
        main(
            ["import", "server-results-RL-S0-example.tar.gz"],
            output=output.append,
            error_output=error_output.append,
        )
        == 1
    )

    assert output == []
    assert error_output == [
        "import: failed (server archive does not exist: server-results-RL-S0-example.tar.gz)\n"
    ]


def test_run_command_refuses_a_worktree_with_tracked_changes() -> None:
    output: list[str] = []
    error_output: list[str] = []

    with patch(
        "dinorl_engine.server_checks.cli.run_suite",
        side_effect=DirtyWorktreeError("tracked_worktree_dirty"),
    ):
        assert (
            main(
                ["run", "--suite", "RL-S0", "--json"],
                output=output.append,
                error_output=error_output.append,
            )
            == 1
        )

    assert error_output == []
    assert json.loads("".join(output)) == {
        "command": "run",
        "profile": "standard",
        "reason": "tracked_worktree_dirty",
        "status": "rejected",
        "suite": "RL-S0",
    }


def test_pythonanywhere_profile_is_selectable_for_a_server_gate() -> None:
    output: list[str] = []

    with patch(
        "dinorl_engine.server_checks.cli.run_suite",
        side_effect=DirtyWorktreeError("tracked_worktree_dirty"),
    ):
        assert (
            main(
                ["run", "--suite", "RL-S0", "--profile", "pythonanywhere", "--json"],
                output=output.append,
            )
            == 1
        )

    assert json.loads("".join(output))["profile"] == "pythonanywhere"


def test_rl_s1_is_selectable_for_a_server_gate() -> None:
    output: list[str] = []

    with patch(
        "dinorl_engine.server_checks.cli.run_suite",
        side_effect=DirtyWorktreeError("tracked_worktree_dirty"),
    ):
        assert main(["run", "--suite", "RL-S1", "--json"], output=output.append) == 1

    assert json.loads("".join(output))["suite"] == "RL-S1"
