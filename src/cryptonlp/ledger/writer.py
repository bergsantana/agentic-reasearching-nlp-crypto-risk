"""Appending to the pre-registration ledger.

The database assigns the sequence number, the time (its own clock), the canonical text and the
hash; this module only sends the record's own fields and turns the ledger's error codes into
exceptions. Use an autocommit connection, as the collectors do: each append is then its own
transaction, and the deadline is checked again when it commits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from cryptonlp.db import TEST_DB_SUFFIX, DatabaseGuardError
from cryptonlp.schemas.forecasts import Forecast
from cryptonlp.schemas.ledger import (
    Document,
    ExperimentPackage,
    Hypothesis,
    Intervention,
    Score,
    Verdict,
    Violation,
    ViolationKind,
)

LedgerRecord = (
    Hypothesis
    | Document
    | ExperimentPackage
    | Forecast
    | Score
    | Verdict
    | Intervention
    | Violation
)

TABLES: dict[type[BaseModel], str] = {
    Hypothesis: "hypotheses",
    Document: "documents",
    ExperimentPackage: "experiment_packages",
    Forecast: "predictions",
    Score: "scores",
    Verdict: "verdicts",
    Intervention: "interventions",
    Violation: "violations",
}


class LedgerError(Exception):
    """The ledger refused a write. ``sqlstate`` is the code raised in db/init/003_ledger.sql."""

    sqlstate = ""


class LatePredictionError(LedgerError):
    sqlstate = "LG001"


class AppendOnlyError(LedgerError):
    sqlstate = "LG002"


class PhaseMismatchError(LedgerError):
    sqlstate = "LG003"


class EarlyScoreError(LedgerError):
    sqlstate = "LG004"


class IsolationError(LedgerError):
    sqlstate = "LG005"


class InvalidEntryError(LedgerError):
    sqlstate = "LG006"


class SimulatedClockError(LedgerError):
    sqlstate = "LG007"


_ERRORS: dict[str, type[LedgerError]] = {
    cls.sqlstate: cls
    for cls in (
        LatePredictionError,
        AppendOnlyError,
        PhaseMismatchError,
        EarlyScoreError,
        IsolationError,
        InvalidEntryError,
        SimulatedClockError,
    )
}


@dataclass(frozen=True)
class Entry:
    """Where a record landed in the chain."""

    table: str
    seq: int
    hash: str
    recorded_at: datetime


def translate(exc: psycopg.Error) -> Exception:
    """The matching LedgerError for a ledger error code, or the original error."""
    cls = _ERRORS.get(exc.sqlstate or "")
    if cls is None:
        return exc
    return cls(exc.diag.message_primary or str(exc))


def _columns(record: BaseModel) -> dict[str, Any]:
    as_json = record.model_dump(mode="json")
    values: dict[str, Any] = {}
    for name in type(record).model_fields:
        value = getattr(record, name)
        if isinstance(value, dict | list | tuple):
            values[name] = Jsonb(as_json[name])
        elif isinstance(value, Enum):
            values[name] = value.value
        else:
            values[name] = value
    return values


def append(
    conn: psycopg.Connection, record: LedgerRecord, *, package_seq: int | None = None
) -> Entry:
    """Insert one record. ``package_seq`` links a forecast to its experiment package."""
    table = TABLES[type(record)]
    values = _columns(record)
    if isinstance(record, Forecast):
        values["package_seq"] = package_seq
    elif package_seq is not None:
        raise TypeError("only forecasts take package_seq")
    query = sql.SQL(
        "INSERT INTO ledger.{} ({}) VALUES ({}) RETURNING seq, hash, recorded_at"
    ).format(
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, values)),
        sql.SQL(", ").join(map(sql.Placeholder, values)),
    )
    try:
        with conn.transaction():
            row = conn.execute(query, values).fetchone()
    except psycopg.Error as exc:
        err = translate(exc)
        if err is exc:
            raise
        raise err from exc
    assert row is not None
    return Entry(table, row[0], row[1], row[2])


def record_prediction(
    conn: psycopg.Connection,
    forecast: Forecast,
    *,
    package_seq: int | None = None,
    reported_by: str = "forecast_loop",
) -> Entry:
    """Append a forecast. If it is late, the rejection itself goes into the ledger, then raises.

    Late predictions rejected by the database clock are a process measure (Quadro 9), so they
    are kept, as violations, with the forecast that was refused.
    """
    try:
        return append(conn, forecast, package_seq=package_seq)
    except LatePredictionError as exc:
        append(
            conn,
            Violation(
                kind=ViolationKind.LATE_PREDICTION,
                reported_by=reported_by,
                detail={
                    "error": str(exc),
                    "forecast": forecast.model_dump(mode="json"),
                    "package_seq": package_seq,
                },
                phase=forecast.phase,
                run_id=forecast.run_id,
            ),
        )
        raise


def head(conn: psycopg.Connection) -> Entry | None:
    """The newest entry, or None for an empty ledger."""
    row = conn.execute(
        "SELECT c.relname, e.seq, e.hash, e.recorded_at FROM ledger.entries e "
        "JOIN pg_class c ON c.oid = e.tableoid ORDER BY e.seq DESC LIMIT 1"
    ).fetchone()
    return None if row is None else Entry(row[0], row[1], row[2], row[3])


def set_simulated_clock(conn: psycopg.Connection, at: datetime | None) -> None:
    """Run this session's ledger on a simulated clock (test database only); None clears it."""
    if not conn.info.dbname.endswith(TEST_DB_SUFFIX):
        raise DatabaseGuardError("the simulated clock may only run against the test database")
    if at is not None and at.tzinfo is None:
        raise ValueError("the simulated time must be timezone-aware")
    conn.execute(
        "SELECT set_config('cryptonlp.sim_now', %s, false)",
        ("" if at is None else at.isoformat(),),
    )
