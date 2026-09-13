"""Deliberately small prototypes for the two RL-L5 interactive-priority modes."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum

__all__ = ["InteractivePriorityMode", "InteractivePriorityPrototype"]


class InteractivePriorityMode(StrEnum):
    """The two alternatives measured by the RL-S3 server gate."""

    UNIT_BOUNDARY = "unit_boundary"
    SEPARATE_WORKER = "separate_worker"


class InteractivePriorityPrototype:
    """Run P0 interactive work before training or on its dedicated worker."""

    def __init__(self, mode: InteractivePriorityMode) -> None:
        self._mode = mode
        self._training: deque[Callable[[], None]] = deque()
        self._interactive: deque[Callable[[], None]] = deque()
        self._closed = False
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="dinorl-interactive")
            if mode is InteractivePriorityMode.SEPARATE_WORKER
            else None
        )
        self._futures: list[Future[None]] = []

    def submit_training(self, task: Callable[[], None]) -> None:
        """Queue one indivisible PPO unit (P3)."""

        if self._closed:
            raise RuntimeError("priority prototype is closed")
        self._training.append(task)

    def submit_interactive(self, task: Callable[[], None]) -> None:
        """Submit a P0 game without allowing it to wait behind a later unit."""

        if self._closed:
            raise RuntimeError("priority prototype is closed")
        if self._executor is None:
            self._interactive.append(task)
            return
        self._futures.append(self._executor.submit(task))

    def run_next(self) -> bool:
        """Serve one boundary task, or run the next training unit when no P0 is waiting."""

        if self._closed:
            raise RuntimeError("priority prototype is closed")
        if self._interactive:
            self._interactive.popleft()()
            return True
        if self._training:
            self._training.popleft()()
            return True
        return False

    def wait_for_interactive(self) -> None:
        """Propagate every interactive task failure and wait for their completion."""

        for future in self._futures:
            future.result()

    @property
    def worker_alive(self) -> bool:
        """Report the separate prototype worker state without leaking its executor."""

        if self._executor is None:
            return False
        return any(thread.is_alive() for thread in self._executor._threads)

    def close(self) -> None:
        """Wait for interactive work and release its worker deterministically."""

        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
