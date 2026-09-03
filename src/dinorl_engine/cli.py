"""Command-line interface for deterministic local DinoRL matches."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from dinorl_engine.controllers.scripted import SCRIPTED_CONTROLLER_IDS
from dinorl_engine.core.constants import MAP_ID
from dinorl_engine.match.runner import run_match

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
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


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute the CLI and print one compact JSON document."""

    options = _parser().parse_args(arguments)
    outcome = run_match(
        map_id=options.map_id,
        seed=options.seed,
        controller_a_id=options.controller_a,
        controller_b_id=options.controller_b,
        include_replay=not options.no_replay,
    )
    print(
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
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
