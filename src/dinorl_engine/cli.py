"""Command-line interface for matches and local debugging tools.

Exit codes are 0 for success, 1 for replay or local I/O failures, and 2 for
invalid command-line arguments (the standard ``argparse`` behavior).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

from dinorl_engine.controllers.protocol import Controller
from dinorl_engine.controllers.scripted import (
    SCRIPTED_CONTROLLER_IDS,
    create_scripted_controller,
)
from dinorl_engine.controllers.sequence import (
    SequenceExhaustedError,
    SequenceIllegalActionError,
    load_sequence_controller,
)
from dinorl_engine.core.constants import MAP_ID, Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.state import PublicSnapshot
from dinorl_engine.debug.manual import ManualCancelledError, ManualController
from dinorl_engine.debug.text_renderer import render_state
from dinorl_engine.match.replay import ReplayControllerDescriptor, canonical_replay_json
from dinorl_engine.match.runner import MatchOutcome, run_controller_match, run_match

__all__ = ["main"]


type Writer = Callable[[str], object]
type Reader = Callable[[], str]
type Sleeper = Callable[[float], object]

_ROOT = Path(__file__).resolve().parents[2]
_REPLAY_SCHEMA_PATH = _ROOT / "schemas" / "replay-v1.schema.json"
_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _match_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one deterministic DinoRL match")
    parser.add_argument("--map-id", choices=(MAP_ID,), default=MAP_ID)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--controller-a", choices=SCRIPTED_CONTROLLER_IDS, required=True)
    parser.add_argument("--controller-b", choices=SCRIPTED_CONTROLLER_IDS, required=True)
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="emit only the result summary",
    )
    return parser


def _inspect_map_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="dinorl inspect-map",
        description="Render the initial arena_mvp_v1 state",
    )


def _replay_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dinorl replay",
        description="Validate and render a replay v1 JSON file",
        epilog="Exit codes: 0 success; 1 invalid replay or I/O; 2 invalid arguments.",
    )
    parser.add_argument("path", type=Path)
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--step", action="store_true", help="wait for Enter between events")
    timing.add_argument("--speed", type=float, choices=_SPEEDS, default=1.0)
    parser.add_argument("--event", type=int, help="render only the event with this index")
    return parser


def _play_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dinorl play",
        description="Play one local match with scripted or manual controllers",
        epilog="Exit codes: 0 success; 1 cancelled or invalid controller; 2 invalid arguments.",
    )
    parser.add_argument("--map-id", choices=(MAP_ID,), default=MAP_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller-a", required=True)
    parser.add_argument("--controller-b", required=True)
    parser.add_argument("--replay-out", type=Path)
    return parser


def _read_valid_replay(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    schema: object = json.loads(_REPLAY_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("replay root must be an object")
    if not isinstance(schema, dict):
        raise ValueError("replay schema root must be an object")
    Draft202012Validator(schema).validate(payload)
    replay = cast(dict[str, object], payload)
    events = replay.get("events")
    if not isinstance(events, list):
        raise ValueError("replay events must be an array")
    if [event.get("seq") if isinstance(event, Mapping) else None for event in events] != list(
        range(len(events))
    ):
        raise ValueError("replay event sequence must be continuous")
    return replay


def _replay_events(replay: Mapping[str, object]) -> list[Mapping[str, object]]:
    values = replay.get("events")
    if not isinstance(values, list) or not all(isinstance(value, Mapping) for value in values):
        raise ValueError("replay events must be objects")
    return [cast(Mapping[str, object], value) for value in values]


def _render_event(event: Mapping[str, object]) -> str:
    state = event.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("replay event state must be an object")
    return render_state(cast(PublicSnapshot, state), event=event)


def _run_replay(
    replay: Mapping[str, object],
    *,
    event_index: int | None,
    step: bool,
    speed: float,
    output: Writer,
    reader: Reader,
    sleeper: Sleeper,
) -> None:
    events = _replay_events(replay)
    if event_index is not None:
        if not 0 <= event_index < len(events):
            raise ValueError(f"event index {event_index} is out of range")
        output(_render_event(events[event_index]))
        return
    for index, event in enumerate(events):
        output(_render_event(event))
        if index == len(events) - 1:
            continue
        if step:
            reader()
        else:
            sleeper(1.0 / speed)


def _write_outcome(outcome: MatchOutcome, output: Writer) -> None:
    output(
        json.dumps(
            {
                "result": outcome.summary(),
                "replay_sha256": outcome.replay_sha256,
                "replay": outcome.replay,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _run_legacy_match(arguments: Sequence[str], output: Writer) -> int:
    options = _match_parser().parse_args(arguments)
    outcome = run_match(
        map_id=options.map_id,
        seed=options.seed,
        controller_a_id=options.controller_a,
        controller_b_id=options.controller_b,
        include_replay=not options.no_replay,
    )
    _write_outcome(outcome, output)
    return 0


def _local_controller(
    specification: str, reader: Reader, writer: Writer
) -> tuple[Controller, ReplayControllerDescriptor]:
    if specification == "manual":
        return (
            ManualController(reader=reader, writer=writer),
            ReplayControllerDescriptor("manual", "manual"),
        )
    if specification.startswith("sequence:"):
        path_value = specification.removeprefix("sequence:")
        if not path_value:
            raise ValueError("sequence controller path is empty")
        controller = load_sequence_controller(Path(path_value))
        return (
            controller,
            ReplayControllerDescriptor("sequence", controller.controller_id),
        )
    return (
        create_scripted_controller(specification),
        ReplayControllerDescriptor("scripted", specification),
    )


def _run_play(
    arguments: Sequence[str],
    *,
    output: Writer,
    error_output: Writer,
    reader: Reader,
) -> int:
    options = _play_parser().parse_args(arguments)
    try:
        controller_a, descriptor_a = _local_controller(options.controller_a, reader, output)
        controller_b, descriptor_b = _local_controller(options.controller_b, reader, output)
        outcome = run_controller_match(
            map_id=options.map_id,
            seed=options.seed,
            controller_a=controller_a,
            controller_b=controller_b,
            controller_a_descriptor=descriptor_a,
            controller_b_descriptor=descriptor_b,
            include_replay=options.replay_out is not None,
        )
        if options.replay_out is not None:
            if outcome.replay is None:
                raise AssertionError("replay output requested without replay data")
            options.replay_out.write_bytes(canonical_replay_json(outcome.replay))
    except ManualCancelledError:
        error_output("Match cancelled by manual controller\n")
        return 1
    except (SequenceIllegalActionError, SequenceExhaustedError) as error:
        error_output(f"Sequence error:\n{error}\n")
        return 1
    except OSError as error:
        error_output(f"Play I/O error: {error}\n")
        return 1
    except ValueError as error:
        error_output(f"Play error: {error}\n")
        return 1
    _write_outcome(outcome, output)
    return 0


def main(
    arguments: Sequence[str] | None = None,
    *,
    output: Writer | None = None,
    error_output: Writer | None = None,
    reader: Reader | None = None,
    sleeper: Sleeper | None = None,
) -> int:
    """Execute one CLI command with injectable local I/O and timing."""

    cli_arguments = list(sys.argv[1:] if arguments is None else arguments)
    write = sys.stdout.write if output is None else output
    write_error = sys.stderr.write if error_output is None else error_output
    read = input if reader is None else reader
    sleep = time.sleep if sleeper is None else sleeper

    if cli_arguments and cli_arguments[0] == "inspect-map":
        _inspect_map_parser().parse_args(cli_arguments[1:])
        env = DinoRLEnv(MAP_ID, seed=0)
        env.reset(first_actor=Actor.A)
        write(render_state(env.snapshot_public()))
        return 0
    if cli_arguments and cli_arguments[0] == "replay":
        options = _replay_parser().parse_args(cli_arguments[1:])
        try:
            replay = _read_valid_replay(options.path)
            _run_replay(
                replay,
                event_index=options.event,
                step=options.step,
                speed=options.speed,
                output=write,
                reader=read,
                sleeper=sleep,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as error:
            write_error(f"Replay error: {error}\n")
            return 1
        except (EOFError, KeyboardInterrupt):
            write_error("Replay cancelled\n")
            return 1
        return 0
    if cli_arguments and cli_arguments[0] == "play":
        return _run_play(
            cli_arguments[1:],
            output=write,
            error_output=write_error,
            reader=read,
        )
    return _run_legacy_match(cli_arguments, write)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
