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

    s5b = commands.add_parser("s5b", help="run the staged RL-S5b robustness protocol")
    s5b_commands = s5b.add_subparsers(dest="s5b_command", required=True)
    freeze_pool = s5b_commands.add_parser("freeze-pool", help="freeze an immutable benchmark pool")
    freeze_pool.add_argument("--config", type=Path, required=True)
    _add_json_option(freeze_pool)
    crossplay = s5b_commands.add_parser(
        "cross-evaluate", help="complete or resume the paired checkpoint matrix"
    )
    crossplay.add_argument("--pool", type=Path, required=True)
    crossplay.add_argument("--output-dir", type=Path)
    _add_json_option(crossplay)
    report = s5b_commands.add_parser("report", help="inspect an existing RL-S5b result directory")
    report.add_argument("--run", type=Path, required=True)
    _add_json_option(report)
    archive = s5b_commands.add_parser("archive", help="package completed RL-S5b-A server evidence")
    archive.add_argument("--run", type=Path, required=True)
    archive.add_argument("--output-dir", type=Path, required=True)
    _add_json_option(archive)
    continuation = s5b_commands.add_parser(
        "continue", help="run or resume the controlled RL-S5b-B continuation"
    )
    continuation.add_argument("--pool", type=Path)
    # Kept solely to give pre-RL-S5b-B callers the previous explicit gate.
    continuation.add_argument("--checkpoint")
    continuation.add_argument("--output-dir", type=Path)
    continuation.add_argument("--units", type=int, default=50)
    continuation.add_argument("--progress", action="store_true")
    _add_json_option(continuation)
    self_play = s5b_commands.add_parser(
        "self-play", help="run the RL-S5b-C snapshot-league experiment"
    )
    self_play.add_argument("--pool", type=Path)
    self_play.add_argument("--checkpoint")
    self_play.add_argument("--output-dir", type=Path)
    self_play.add_argument("--units", type=int, default=50)
    self_play.add_argument("--progress", action="store_true")
    _add_json_option(self_play)
    exploiters = s5b_commands.add_parser(
        "train-exploiters", help="train phase-C exploiters against frozen targets"
    )
    exploiters.add_argument("--pool", type=Path)
    exploiters.add_argument("--target")
    exploiters.add_argument("--output-dir", type=Path)
    exploiters.add_argument("--units", type=int, default=50)
    exploiters.add_argument("--progress", action="store_true")
    _add_json_option(exploiters)
    s5c = commands.add_parser("s5c", help="run the RL-S5c specialist-family protocol")
    s5c_commands = s5c.add_subparsers(dest="s5c_command", required=True)
    calibration = s5c_commands.add_parser(
        "calibrate", help="run one zero-start specialist reward calibration"
    )
    calibration.add_argument(
        "--archetype", choices=("scavenger", "predator", "controller"), required=True
    )
    calibration.add_argument("--config", type=Path, required=True)
    calibration.add_argument("--progress", action="store_true")
    _add_json_option(calibration)
    s5c_archive = s5c_commands.add_parser(
        "archive-calibration", help="package complete RL-S5c-A server evidence"
    )
    s5c_archive.add_argument("--run", type=Path, required=True)
    s5c_archive.add_argument("--output-dir", type=Path, required=True)
    _add_json_option(s5c_archive)
    return parser


def _s5b_result(options: argparse.Namespace) -> dict[str, object]:
    """Run only the phase allowed before the first server checkpoint."""

    from dinorl_engine.rl.s5b.archive import create_crossplay_archive
    from dinorl_engine.rl.s5b.config import load_config
    from dinorl_engine.rl.s5b.continuation import continue_phase_b
    from dinorl_engine.rl.s5b.crossplay import cross_evaluate
    from dinorl_engine.rl.s5b.exploiters import train_exploiters
    from dinorl_engine.rl.s5b.pool import freeze_pool, read_verified_pool
    from dinorl_engine.rl.s5b.selfplay import run_self_play

    if options.s5b_command == "freeze-pool":
        config = load_config(options.config)
        path = freeze_pool(config)
        pool = read_verified_pool(path)
        return {
            "command": "s5b freeze-pool",
            "pool": str(path),
            "pool_sha256": pool["pool_sha256"],
            "status": "completed",
        }
    if options.s5b_command == "cross-evaluate":
        output_directory = (
            options.pool.parent.parent if options.output_dir is None else options.output_dir
        )
        result = cross_evaluate(pool_path=options.pool, output_directory=output_directory)
        return {
            "command": "s5b cross-evaluate",
            "games_completed": result["games_completed"],
            "pool_sha256": result["pool_sha256"],
            "records_completed": result["records_completed"],
            "report": str(output_directory / "crossplay" / "crossplay-report.md"),
            "status": "completed",
        }
    if options.s5b_command == "report":
        report_path = options.run / "crossplay" / "crossplay.json"
        try:
            result = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read RL-S5b cross-play result: {report_path}") from error
        if not isinstance(result, dict) or result.get("format") != "s5b-crossplay-v1":
            raise ValueError("RL-S5b cross-play result has an unsupported format")
        return {
            "command": "s5b report",
            "pool_sha256": result.get("pool_sha256"),
            "report": str(options.run / "crossplay" / "crossplay-report.md"),
            "status": "completed",
        }
    if options.s5b_command == "archive":
        path = create_crossplay_archive(
            run_directory=options.run,
            output_directory=options.output_dir,
        )
        return {"archive": str(path), "command": "s5b archive", "status": "completed"}
    if options.s5b_command == "continue":
        if options.pool is None:
            return {
                "command": "s5b continue",
                "reason": "requires_completion_of_the_prior_server_checkpoint",
                "status": "not_implemented",
            }
        progress = (lambda message: print(message, file=sys.stderr)) if options.progress else None
        result = continue_phase_b(
            pool_path=options.pool,
            output_directory=options.output_dir,
            units=options.units,
            target_id=options.target,
            progress=progress,
        )
        return {
            "branches": len(result["branches"]),
            "command": "s5b continue",
            "pool_sha256": result["pool_sha256"],
            "status": "completed",
        }
    if options.s5b_command == "self-play":
        if options.pool is None:
            return {
                "command": "s5b self-play",
                "reason": "requires_completion_of_the_prior_server_checkpoint",
                "status": "not_implemented",
            }
        progress = (lambda message: print(message, file=sys.stderr)) if options.progress else None
        result = run_self_play(
            pool_path=options.pool,
            output_directory=options.output_dir,
            units=options.units,
            progress=progress,
        )
        return {
            "branches": len(result["branches"]),
            "command": "s5b self-play",
            "pool_sha256": result["pool_sha256"],
            "status": "completed",
        }
    if options.s5b_command == "train-exploiters":
        if options.pool is None:
            return {
                "command": "s5b train-exploiters",
                "reason": "requires_completion_of_the_prior_server_checkpoint",
                "status": "not_implemented",
            }
        progress = (lambda message: print(message, file=sys.stderr)) if options.progress else None
        result = train_exploiters(
            pool_path=options.pool,
            output_directory=options.output_dir,
            units=options.units,
            progress=progress,
        )
        return {
            "command": "s5b train-exploiters",
            "pool_sha256": result["pool_sha256"],
            "runs": len(result["results"]),
            "status": "completed",
        }
    return {
        "command": f"s5b {options.s5b_command}",
        "reason": "requires_completion_of_the_prior_server_checkpoint",
        "status": "not_implemented",
    }


def _s5c_result(options: argparse.Namespace) -> dict[str, object]:
    """Run only the calibration phase permitted before RL-S5c-A review."""

    from dinorl_engine.rl.s5c.archive import create_calibration_archive
    from dinorl_engine.rl.s5c.calibration import calibrate
    from dinorl_engine.rl.s5c.config import load_config

    if options.s5c_command == "calibrate":
        config = load_config(options.config)
        progress = (lambda message: print(message, file=sys.stderr)) if options.progress else None
        result = calibrate(config, archetype=options.archetype, progress=progress)
        return {
            "archetype": result["archetype"],
            "command": "s5c calibrate",
            "output_directory": str(config.output_directory),
            "status": "completed",
        }
    if options.s5c_command == "archive-calibration":
        archive = create_calibration_archive(
            run_directory=options.run, output_directory=options.output_dir
        )
        return {
            "archive": str(archive),
            "command": "s5c archive-calibration",
            "status": "completed",
        }
    return {
        "command": f"s5c {options.s5c_command}",
        "reason": "requires_completion_of_the_prior_server_checkpoint",
        "status": "not_implemented",
    }


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
    if options.command == "s5b":
        try:
            result = _s5b_result(options)
        except (OSError, RuntimeError, ValueError) as error:
            result = {
                "command": f"s5b {options.s5b_command}",
                "reason": str(error),
                "status": "failed",
            }
        if options.json:
            write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        elif result["status"] == "completed":
            write(f"{result['command']}: completed\n")
        else:
            write_error(f"{result['command']}: {result['status']} ({result.get('reason', '')})\n")
        return (
            0
            if result["status"] == "completed"
            else 3
            if result["status"] == "not_implemented"
            else 1
        )
    if options.command == "s5c":
        try:
            result = _s5c_result(options)
        except (OSError, RuntimeError, ValueError) as error:
            result = {
                "command": f"s5c {options.s5c_command}",
                "reason": str(error),
                "status": "failed",
            }
        if options.json:
            write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        elif result["status"] == "completed":
            write(f"{result['command']}: completed\n")
        else:
            write_error(f"{result['command']}: {result['status']} ({result.get('reason', '')})\n")
        return (
            0
            if result["status"] == "completed"
            else 3
            if result["status"] == "not_implemented"
            else 1
        )
    result = {"command": options.command, "status": "not_implemented"}
    if options.json:
        write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    else:
        write_error(f"{options.command}: not_implemented\n")
    return 3


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(main())
