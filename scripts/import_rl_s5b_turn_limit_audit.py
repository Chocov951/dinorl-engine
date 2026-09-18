"""Import the light-weight, evaluation-only RL-S5b-AUDIT result directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    required = (args.source / "games.jsonl", args.source / "turn-limit-results.json")
    if not all(path.is_file() for path in required):
        raise ValueError("source is not a completed RL-S5b turn-limit audit")
    if args.destination.exists():
        raise ValueError("destination already exists; retain it as immutable evidence")
    shutil.copytree(args.source, args.destination)


if __name__ == "__main__":
    main()
