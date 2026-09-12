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
    run_suite,
)

__all__ = ["main"]


type Writer = Callable[[str], object]


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
    run.add_argument("--suite", required=True, choices=("RL-S0", "RL-S1"))
    run.add_argument("--profile", default="standard", choices=profile_names())
    run.add_argument("--output-dir", type=Path, default=Path.cwd())
    run.add_argument("--json", action="store_true")
    options = parser.parse_args(sys.argv[1:] if arguments is None else arguments)
    write = sys.stdout.write if output is None else output
    write_error = sys.stderr.write if error_output is None else error_output
    if options.command == "run":
        try:
            result = run_suite(
                suite=options.suite,
                profile=get_profile(options.profile),
                root=repository_root(),
                output_dir=options.output_dir,
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
            write(f"run: passed ({result['archive']})\n")
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
