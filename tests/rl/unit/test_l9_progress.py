"""Progress and ETA remain pure and injectable for RL-L9."""

from __future__ import annotations

from dinorl_engine.rl.orchestration.progress import project_progress


def test_progress_is_provisional_without_timing_and_measured_with_a_unit_cost() -> None:
    status: dict[str, object] = {
        "job_id": "job-a",
        "run_id": "run-a",
        "status": "running",
        "phase": "collecting",
        "heartbeat_at": "2026-09-25T00:00:00Z",
        "reserved_units": 5,
        "validated_units": 2,
        "remaining_units": 3,
    }

    assert project_progress(status, recent_unit_seconds=None).eta_confidence == "provisional"
    measured = project_progress(status, recent_unit_seconds=12.5)
    assert measured.eta_seconds == 37.5
    assert measured.eta_confidence == "measured"
