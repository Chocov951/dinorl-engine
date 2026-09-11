"""Server-check CLI behaviour."""

import json

import dinorl_engine.server_checks.__main__  # noqa: F401
from dinorl_engine.server_checks.cli import main


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
        "reason": "tracked_worktree_dirty",
        "status": "rejected",
        "suite": "RL-S0",
    }
