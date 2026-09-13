"""Contracts for the two RL-L5 interactive-priority prototypes."""

from __future__ import annotations

from threading import Event

from dinorl_engine.rl.orchestration.priority import (
    InteractivePriorityMode,
    InteractivePriorityPrototype,
)


def test_boundary_mode_serves_an_interactive_game_before_the_next_training_unit() -> None:
    events: list[str] = []
    scheduler = InteractivePriorityPrototype(InteractivePriorityMode.UNIT_BOUNDARY)
    scheduler.submit_training(lambda: events.append("train-1"))
    scheduler.submit_training(lambda: events.append("train-2"))
    scheduler.submit_interactive(lambda: events.append("interactive"))

    scheduler.run_next()
    scheduler.run_next()
    scheduler.run_next()

    assert events == ["interactive", "train-1", "train-2"]


def test_concurrent_mode_serves_the_interactive_game_while_training_is_active() -> None:
    training_started = Event()
    interactive_served = Event()
    scheduler = InteractivePriorityPrototype(InteractivePriorityMode.SEPARATE_WORKER)

    def training() -> None:
        training_started.set()
        assert interactive_served.wait(timeout=2)

    try:
        scheduler.submit_training(training)
        scheduler.submit_interactive(interactive_served.set)
        scheduler.run_next()
        scheduler.wait_for_interactive()
    finally:
        scheduler.close()

    assert interactive_served.is_set()
    assert not scheduler.worker_alive
