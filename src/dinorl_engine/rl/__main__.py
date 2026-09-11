"""Declared RL command-line interface.

Exit status 0 means a completed command, 2 is reserved for invalid command-line
arguments by ``argparse``, and 3 means that the declared command is not yet
implemented.  RL-L0 intentionally returns only the latter status.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

__all__ = ["main"]


type Writer = Callable[[str], object]


def _add_json_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="write a machine-readable result to stdout"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dinorl_engine.rl",
        description="DinoRL training and policy management commands",
        epilog="Exit codes: 0 success; 2 invalid arguments; 3 not implemented.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="start an experiment")
    train.add_argument("--config", type=Path, required=True)
    _add_json_option(train)

    resume = commands.add_parser("resume", help="resume from a checkpoint")
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--units", type=int, required=True)
    _add_json_option(resume)

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--suite", choices=("deterministic", "publication"), required=True)
    _add_json_option(evaluate)

    publish = commands.add_parser("publish", help="publish a checkpoint")
    publish.add_argument("--checkpoint", required=True)
    _add_json_option(publish)

    inspect = commands.add_parser("inspect", help="inspect an execution")
    inspect.add_argument("--run", required=True)
    _add_json_option(inspect)

    status = commands.add_parser("status", help="show job status")
    status.add_argument("--job", required=True)
    status.add_argument("--follow", action="store_true")
    _add_json_option(status)

    cancel = commands.add_parser("cancel", help="cancel a job")
    cancel.add_argument("--job", required=True)
    _add_json_option(cancel)
    return parser


def main(
    arguments: Sequence[str] | None = None,
    *,
    output: Writer | None = None,
    error_output: Writer | None = None,
) -> int:
    """Run a declared command and explicitly report its current availability."""

    options = _build_parser().parse_args(sys.argv[1:] if arguments is None else arguments)
    write = sys.stdout.write if output is None else output
    write_error = sys.stderr.write if error_output is None else error_output
    result = {"command": options.command, "status": "not_implemented"}
    if options.json:
        write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    else:
        write_error(f"{options.command}: not_implemented\n")
    return 3


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(main())
