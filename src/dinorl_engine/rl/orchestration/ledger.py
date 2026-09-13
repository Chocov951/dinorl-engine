"""Small SQLite unit ledger with reservation and debit idempotence."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["LedgerError", "UnitLedger"]


class LedgerError(RuntimeError):
    """Raised when a ledger operation would violate a budget invariant."""


class UnitLedger:
    """Own the local V1 reservation and one-time debit records for a run."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        with self._session() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                    run_id TEXT PRIMARY KEY,
                    units INTEGER NOT NULL CHECK(units > 0),
                    idempotency_key TEXT NOT NULL UNIQUE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS unit_ledger (
                    run_id TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    PRIMARY KEY (run_id, unit_id),
                    FOREIGN KEY (run_id) REFERENCES reservations(run_id)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    @contextmanager
    def _session(self) -> Generator[sqlite3.Connection, None, None]:
        """Commit or roll back a short transaction and always release the file handle."""

        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _opaque(value: str, field_name: str) -> None:
        if not value or any(character.isspace() for character in value):
            raise ValueError(f"{field_name} must be a non-empty opaque identifier")

    def reserve(self, *, run_id: str, units: int, idempotency_key: str) -> bool:
        """Reserve a complete run budget once, returning whether it was newly reserved."""

        self._opaque(run_id, "run_id")
        self._opaque(idempotency_key, "idempotency_key")
        if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
            raise ValueError("units must be a positive integer")
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT run_id, units FROM reservations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing != (run_id, units):
                    raise LedgerError("reservation idempotency key conflicts with an existing run")
                return False
            run = connection.execute(
                "SELECT units, idempotency_key FROM reservations WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is not None:
                raise LedgerError("run already has a reservation with a different idempotency key")
            connection.execute(
                "INSERT INTO reservations(run_id, units, idempotency_key) VALUES (?, ?, ?)",
                (run_id, units, idempotency_key),
            )
        return True

    def debit_validated_unit(self, *, run_id: str, unit_id: str, idempotency_key: str) -> bool:
        """Debit one atomically validated unit at most once."""

        self._opaque(run_id, "run_id")
        self._opaque(unit_id, "unit_id")
        self._opaque(idempotency_key, "idempotency_key")
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            reservation = connection.execute(
                "SELECT units FROM reservations WHERE run_id = ?", (run_id,)
            ).fetchone()
            if reservation is None:
                raise LedgerError("cannot debit a run without a reservation")
            by_key = connection.execute(
                "SELECT run_id, unit_id FROM unit_ledger WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if by_key is not None:
                if by_key != (run_id, unit_id):
                    raise LedgerError("debit idempotency key conflicts with another unit")
                return False
            duplicate = connection.execute(
                "SELECT 1 FROM unit_ledger WHERE run_id = ? AND unit_id = ?", (run_id, unit_id)
            ).fetchone()
            if duplicate is not None:
                return False
            used = connection.execute(
                "SELECT COUNT(*) FROM unit_ledger WHERE run_id = ?", (run_id,)
            ).fetchone()
            if used is None or type(used[0]) is not int or type(reservation[0]) is not int:
                raise LedgerError("ledger contains an invalid debit count")
            if used[0] >= reservation[0]:
                raise LedgerError("reserved budget is exhausted")
            connection.execute(
                "INSERT INTO unit_ledger(run_id, unit_id, idempotency_key) VALUES (?, ?, ?)",
                (run_id, unit_id, idempotency_key),
            )
        return True

    def debited_units(self, run_id: str) -> int:
        """Return the number of validated units already debited for ``run_id``."""

        self._opaque(run_id, "run_id")
        with self._session() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM unit_ledger WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None or type(row[0]) is not int:
            raise LedgerError("ledger contains an invalid debit count")
        return row[0]

    def remaining_units(self, run_id: str) -> int:
        """Return reserved units which have not yet been debited or released."""

        self._opaque(run_id, "run_id")
        with self._session() as connection:
            row = connection.execute(
                "SELECT units FROM reservations WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None or type(row[0]) is not int:
            raise LedgerError("run has no reservation")
        return row[0] - self.debited_units(run_id)
