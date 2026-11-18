"""Pure progress/ETA projection for CLI and future API consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

__all__ = ["ProgressV1", "project_progress"]


@dataclass(frozen=True, slots=True)
class ProgressV1:
    job_id: str
    run_id: str | None
    status: str
    phase: str | None
    reserved_units: int
    validated_units: int
    remaining_units: int
    heartbeat_at: str | None
    eta_seconds: float | None
    eta_confidence: str

    def mapping(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "status": self.status,
            "phase": self.phase,
            "reserved_units": self.reserved_units,
            "validated_units": self.validated_units,
            "remaining_units": self.remaining_units,
            "heartbeat_at": self.heartbeat_at,
            "eta_seconds": self.eta_seconds,
            "eta_confidence": self.eta_confidence,
        }


def project_progress(status: dict[str, object], *, recent_unit_seconds: float | None) -> ProgressV1:
    """Project ETA from measured unit cost, otherwise state that it is provisional."""

    required = ("job_id", "run_id", "status", "phase", "heartbeat_at", "remaining_units")
    if any(field not in status for field in required):
        raise ValueError("orchestration status lacks progress fields")
    reserved = status.get("reserved_units")
    validated = status.get("validated_units")
    remaining = status["remaining_units"]
    if not all(type(value) is int for value in (reserved, validated, remaining)):
        raise ValueError("orchestration progress units are invalid")
    reserved_units = cast(int, reserved)
    validated_units = cast(int, validated)
    remaining_units = cast(int, remaining)
    eta = None
    confidence = "provisional"
    if type(recent_unit_seconds) is float and recent_unit_seconds > 0:
        eta = float(remaining_units) * recent_unit_seconds
        confidence = "measured"
    return ProgressV1(
        job_id=str(status["job_id"]),
        run_id=None if status["run_id"] is None else str(status["run_id"]),
        status=str(status["status"]),
        phase=None if status["phase"] is None else str(status["phase"]),
        reserved_units=reserved_units,
        validated_units=validated_units,
        remaining_units=remaining_units,
        heartbeat_at=None if status["heartbeat_at"] is None else str(status["heartbeat_at"]),
        eta_seconds=eta,
        eta_confidence=confidence,
    )
