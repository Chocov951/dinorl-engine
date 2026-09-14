"""Append-only light-weight JSONL reports; transitions remain in memory only."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Final

__all__ = ["RunReportWriter"]

_IDENTIFIER: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _json_line(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


class RunReportWriter:
    """Persist one aggregate record per unit/evaluation, never one per transition."""

    def __init__(self, root: Path, *, run_id: str, architecture: str, seed: int) -> None:
        if _IDENTIFIER.fullmatch(run_id) is None or _IDENTIFIER.fullmatch(architecture) is None:
            raise ValueError("run_id and architecture must be opaque identifiers")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an unsigned 32-bit integer")
        self._root = root
        self._run_id = run_id
        self._architecture = architecture
        self._seed = seed
        root.mkdir(parents=True, exist_ok=True)
        self._cycle_units = self._recorded_units("cycles.jsonl")
        self._evaluation_units = self._recorded_units("eval.jsonl")

    def _recorded_units(self, name: str) -> set[int]:
        path = self._root / name
        if not path.is_file():
            return set()
        units: set[int] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict) or type(value.get("unit")) is not int:
                raise ValueError(f"{name} contains an invalid record")
            units.add(value["unit"])
        return units

    def has_cycle(self, unit: int) -> bool:
        """Return whether a durable aggregate exists for one completed unit."""

        return unit in self._cycle_units

    def has_evaluation(self, unit: int) -> bool:
        """Return whether an evaluation aggregate is durable for one checkpoint."""

        return unit in self._evaluation_units

    def evaluation_records(self) -> list[dict[str, object]]:
        """Return durable evaluation aggregates to recover gate history after a restart."""

        path = self._root / "eval.jsonl"
        if not path.is_file():
            return []
        records: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("eval.jsonl contains an invalid record")
            records.append(value)
        return records

    def _append(self, name: str, value: dict[str, object]) -> None:
        with (self._root / name).open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_json_line(value))

    def record_cycle(self, record: dict[str, object]) -> None:
        """Append one completed PPO-unit aggregate."""

        unit = record.get("unit")
        if type(unit) is not int or unit <= 0:
            raise ValueError("cycle record must identify a positive unit")
        if unit in self._cycle_units:
            return
        self._append("cycles.jsonl", record)
        self._cycle_units.add(unit)

    def record_evaluation(self, record: dict[str, object]) -> None:
        """Append one completed evaluation aggregate."""

        unit = record.get("unit")
        if type(unit) is not int or unit < 0:
            raise ValueError("evaluation record must identify a non-negative unit")
        if unit in self._evaluation_units:
            return
        self._append("eval.jsonl", record)
        self._evaluation_units.add(unit)

    def write_final_report(self, *, final_score: float, inference_seconds: float) -> Path:
        """Write the final, schema-shaped aggregate after all unit records are durable."""

        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            for value in (final_score, inference_seconds)
        ):
            raise ValueError("final report values must be finite")
        if not 0.0 <= final_score <= 1.0 or inference_seconds < 0.0:
            raise ValueError("final report values are outside their valid range")
        report = {
            "schema_version": "1.0.0",
            "run_id": self._run_id,
            "architecture": self._architecture,
            "seed": self._seed,
            "cycles": len(self._cycle_units),
            "evaluations": len(self._evaluation_units),
            "final_score": final_score,
            "inference_seconds": inference_seconds,
        }
        path = self._root / "final_report.json"
        path.write_text(_json_line(report), encoding="utf-8", newline="\n")
        return path
