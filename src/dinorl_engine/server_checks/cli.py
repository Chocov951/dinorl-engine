"""Declared command-line interface for server-check imports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from dinorl_engine.server_checks.importer import import_archive
from dinorl_engine.server_checks.profiles import get_profile, profile_names
from dinorl_engine.server_checks.runner import (
    DirtyWorktreeError,
    ServerCheckError,
    repository_root,
    run_local_rl_s5_diagnostic,
    run_local_rl_s5_v2_diagnostic,
    run_suite,
)
from dinorl_engine.server_checks.suites.rl_s5 import S5Progress

__all__ = ["main"]


type Writer = Callable[[str], object]


def _format_s5_progress(progress: S5Progress) -> str:
    """Render a bounded, grep-friendly visual progress line."""

    completed = progress.completed_runs * progress.total_units + progress.completed_units
    total = progress.total_runs * progress.total_units
    filled = min(20, (20 * completed) // total)
    bar = "#" * filled + "-" * (20 - filled)
    return (
        f"RL-S5 [{bar}] {completed:>3}/{total} "
        f"{progress.architecture} seed {progress.seed}: "
        f"{progress.completed_units}/{progress.total_units} ({progress.phase})\n"
    )


def _s5_progress_callback(writer: Writer) -> Callable[[S5Progress], None]:
    def emit(progress: S5Progress) -> None:
        writer(_format_s5_progress(progress))

    return emit


def main(
    arguments: Sequence[str] | None = None,
    *,
    output: Writer | None = None,
    error_output: Writer | None = None,
) -> int:
    """Declare the future server-results importer without pretending to run it."""

    parser = argparse.ArgumentParser(
        prog="python -m dinorl_engine.server_checks",
        epilog="Exit codes: 0 success; 1 failed or rejected; 2 invalid arguments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    import_results = commands.add_parser("import", help="import a server gate archive")
    import_results.add_argument("archive", type=Path)
    import_results.add_argument("--json", action="store_true")
    run = commands.add_parser("run", help="run a server gate suite")
    run.add_argument(
        "--suite", required=True, choices=("RL-S0", "RL-S1", "RL-S2", "RL-S3", "RL-S4", "RL-S5")
    )
    run.add_argument("--profile", default="standard", choices=profile_names())
    run.add_argument("--output-dir", type=Path, default=Path.cwd())
    run.add_argument(
        "--progress",
        action="store_true",
        help="show RL-S5 unit-level progress (stderr when used with --json)",
    )
    run.add_argument("--json", action="store_true")
    local_s5 = commands.add_parser(
        "local-rl-s5", help="run a non-importable local RL-S5 architecture diagnostic"
    )
    local_s5.add_argument("--output-dir", type=Path, default=Path.cwd())
    local_s5.add_argument("--progress", action="store_true")
    local_s5.add_argument("--json", action="store_true")
    local_s5_v2 = commands.add_parser(
        "local-rl-s5-v2", help="compare three local MLP V2 candidates and mlp-v1"
    )
    local_s5_v2.add_argument("--output-dir", type=Path, default=Path.cwd())
    local_s5_v2.add_argument("--baseline-work-dir", type=Path)
    local_s5_v2.add_argument("--baseline-report", type=Path)
    local_s5_v2.add_argument("--progress", action="store_true")
    local_s5_v2.add_argument("--json", action="store_true")
    options = parser.parse_args(sys.argv[1:] if arguments is None else arguments)
    write = sys.stdout.write if output is None else output
    write_error = sys.stderr.write if error_output is None else error_output
    if options.command == "run":
        progress: Callable[[S5Progress], None] | None = None
        if options.suite == "RL-S5" and options.progress:
            progress_writer = write_error if options.json else write
            progress = _s5_progress_callback(progress_writer)
        try:
            result = run_suite(
                suite=options.suite,
                profile=get_profile(options.profile),
                root=repository_root(),
                output_dir=options.output_dir,
                progress=progress,
            )
        except DirtyWorktreeError:
            result = {
                "command": "run",
                "profile": options.profile,
                "reason": "tracked_worktree_dirty",
                "status": "rejected",
                "suite": options.suite,
            }
            if options.json:
                write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            else:
                write_error("run: rejected (tracked_worktree_dirty)\n")
            return 1
        except ServerCheckError as error:
            result = {
                "command": "run",
                "profile": options.profile,
                "reason": str(error),
                "status": "failed",
                "suite": options.suite,
            }
            if options.json:
                write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            else:
                write_error(f"run: failed ({error})\n")
            return 1
        if options.json:
            write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        else:
            write(f"run: {result['status']} ({result['archive']})\n")
        return 0 if result["status"] == "passed" else 1

    if options.command == "local-rl-s5":
        progress = None
        if options.progress:
            progress_writer = write_error if options.json else write
            progress = _s5_progress_callback(progress_writer)
        try:
            result = run_local_rl_s5_diagnostic(
                output_dir=options.output_dir,
                progress=progress,
            )
        except (OSError, RuntimeError, ValueError) as error:
            result = {
                "command": "local-rl-s5",
                "reason": str(error),
                "status": "failed",
            }
            if options.json:
                write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            else:
                write_error(f"local-rl-s5: failed ({error})\n")
            return 1
        if options.json:
            write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        else:
            write(f"local-rl-s5: {result['status']} ({result['report']})\n")
        return 0

    if options.command == "local-rl-s5-v2":
        progress = None
        if options.progress:
            progress_writer = write_error if options.json else write
            progress = _s5_progress_callback(progress_writer)
        baseline_work_directory = (
            options.output_dir / ".rl-s5-local-work"
            if options.baseline_work_dir is None
            else options.baseline_work_dir
        )
        try:
            result = run_local_rl_s5_v2_diagnostic(
                output_dir=options.output_dir,
                baseline_work_directory=baseline_work_directory,
                baseline_report_path=options.baseline_report,
                progress=progress,
            )
        except (OSError, RuntimeError, ValueError) as error:
            result = {
                "command": "local-rl-s5-v2",
                "reason": str(error),
                "status": "failed",
            }
            if options.json:
                write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            else:
                write_error(f"local-rl-s5-v2: failed ({error})\n")
            return 1
        if options.json:
            write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        else:
            write(f"local-rl-s5-v2: {result['status']} ({result['report']})\n")
        return 0

    try:
        result = import_archive(archive_path=options.archive, root=repository_root())
    except ServerCheckError as error:
        result = {
            "command": "import",
            "reason": str(error),
            "status": "failed",
        }
        if options.json:
            write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        else:
            write_error(f"import: failed ({error})\n")
        return 1
    if options.json:
        write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    else:
        write(f"import: imported ({result['destination']})\n")
    return 0
