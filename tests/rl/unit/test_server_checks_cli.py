"""Server-check CLI behaviour."""

import json
from unittest.mock import patch

import dinorl_engine.server_checks.__main__  # noqa: F401
from dinorl_engine.server_checks.cli import main
from dinorl_engine.server_checks.runner import DirtyWorktreeError
from dinorl_engine.server_checks.suites.rl_s5 import S5Progress


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


def test_rl_s2_is_selectable_for_a_server_gate() -> None:
    output: list[str] = []

    with patch(
        "dinorl_engine.server_checks.cli.run_suite",
        side_effect=DirtyWorktreeError("tracked_worktree_dirty"),
    ):
        assert main(["run", "--suite", "RL-S2", "--json"], output=output.append) == 1

    assert json.loads("".join(output))["suite"] == "RL-S2"


def test_rl_s3_is_selectable_for_a_server_gate() -> None:
    output: list[str] = []

    with patch(
        "dinorl_engine.server_checks.cli.run_suite",
        side_effect=DirtyWorktreeError("tracked_worktree_dirty"),
    ):
        assert main(["run", "--suite", "RL-S3", "--json"], output=output.append) == 1

    assert json.loads("".join(output))["suite"] == "RL-S3"


def test_rl_s4_is_selectable_for_a_server_gate() -> None:
    output: list[str] = []

    with patch(
        "dinorl_engine.server_checks.cli.run_suite",
        side_effect=DirtyWorktreeError("tracked_worktree_dirty"),
    ):
        assert main(["run", "--suite", "RL-S4", "--json"], output=output.append) == 1

    assert json.loads("".join(output))["suite"] == "RL-S4"


def test_rl_s5_progress_is_rendered_to_stderr_without_corrupting_json() -> None:
    output: list[str] = []
    error_output: list[str] = []

    def run_with_progress(**kwargs: object) -> dict[str, object]:
        progress = kwargs["progress"]
        assert callable(progress)
        progress(
            S5Progress(
                architecture="mlp-v1",
                seed=19,
                completed_units=12,
                total_units=147,
                completed_runs=0,
                total_runs=6,
                phase="training",
            )
        )
        return {"archive": "result.tar.gz", "status": "passed"}

    with patch("dinorl_engine.server_checks.cli.run_suite", side_effect=run_with_progress):
        assert (
            main(
                ["run", "--suite", "RL-S5", "--progress", "--json"],
                output=output.append,
                error_output=error_output.append,
            )
            == 0
        )

    assert json.loads("".join(output))["status"] == "passed"
    assert "[--------------------]  12/882" in "".join(error_output)
    assert "mlp-v1 seed 19" in "".join(error_output)


def test_local_rl_s5_command_is_explicitly_diagnostic_only() -> None:
    output: list[str] = []

    with patch(
        "dinorl_engine.server_checks.cli.run_local_rl_s5_diagnostic",
        return_value={
            "classification": "diagnostic_only_not_server_evidence",
            "report": "rl-s5-local-diagnostic.json",
            "status": "completed",
        },
    ):
        assert main(["local-rl-s5", "--json"], output=output.append) == 0

    assert json.loads("".join(output))["classification"] == "diagnostic_only_not_server_evidence"


def test_local_rl_s5_v2_command_runs_the_pairwise_mlp_tournament() -> None:
    output: list[str] = []

    with patch(
        "dinorl_engine.server_checks.cli.run_local_rl_s5_v2_diagnostic",
        return_value={
            "classification": "diagnostic_only_not_server_evidence",
            "report": "rl-s5-v2-local-diagnostic.json",
            "status": "completed",
        },
    ) as run_v2:
        assert (
            main(
                [
                    "local-rl-s5-v2",
                    "--output-dir",
                    "results",
                    "--baseline-work-dir",
                    "baseline-work",
                    "--json",
                ],
                output=output.append,
            )
            == 0
        )

    assert json.loads("".join(output))["status"] == "completed"
    assert run_v2.call_args.kwargs["baseline_work_directory"].name == "baseline-work"
