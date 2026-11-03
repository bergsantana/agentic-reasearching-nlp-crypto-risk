"""Job runs and collector state, the two things the monitoring dashboard reads."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

log = logging.getLogger(__name__)


@dataclass
class JobRun:
    job: str
    run_id: str
    job_run_id: int
    stats: dict[str, Any] = field(default_factory=dict)


@contextmanager
def track_job(
    connect: Callable[[], psycopg.Connection], job: str, run_id: str, *, exclusive: bool = False
) -> Iterator[JobRun]:
    """Record a job run in ``ops.job_runs``: running, then succeeded or failed with the error.

    Uses its own autocommit connection, so the record survives a rollback in the job itself.
    ``exclusive`` jobs only ever run once at a time, so an older run still marked running
    ended without reporting (killed, or the machine went down) and is marked failed.
    """
    with connect() as conn:
        conn.autocommit = True
        if exclusive:
            conn.execute(
                "UPDATE ops.job_runs SET status = 'failed', finished_at = now(), "
                "error = 'ended without reporting (process killed or machine down)' "
                "WHERE job = %s AND status = 'running'",
                (job,),
            )
        row = conn.execute(
            "INSERT INTO ops.job_runs (job, run_id, status) VALUES (%s, %s, 'running') "
            "RETURNING job_run_id",
            (job, run_id),
        ).fetchone()
        assert row is not None
        run = JobRun(job, run_id, row[0])
        log.info("job %s started, run ID %s", job, run_id)
        try:
            yield run
        except BaseException as exc:
            conn.execute(
                "UPDATE ops.job_runs SET status = 'failed', finished_at = now(), error = %s, "
                "stats = %s WHERE job_run_id = %s",
                (
                    "".join(traceback.format_exception(exc)),
                    Jsonb(run.stats),
                    run.job_run_id,
                ),
            )
            log.exception("job %s failed", job)
            raise
        conn.execute(
            "UPDATE ops.job_runs SET status = 'succeeded', finished_at = now(), stats = %s "
            "WHERE job_run_id = %s",
            (Jsonb(run.stats), run.job_run_id),
        )
        log.info("job %s succeeded: %s", job, run.stats)


@dataclass(frozen=True)
class CollectorState:
    collector: str
    cursor: str | None
    last_event_at: datetime | None


def get_state(conn: psycopg.Connection, collector: str) -> CollectorState | None:
    row = conn.execute(
        "SELECT cursor, last_event_at FROM ops.collector_state WHERE collector = %s",
        (collector,),
    ).fetchone()
    return None if row is None else CollectorState(collector, row[0], row[1])


def save_state(
    conn: psycopg.Connection,
    collector: str,
    source: str,
    run_id: str,
    *,
    cursor: str | None = None,
    last_event_at: datetime | None = None,
) -> None:
    """Upsert a collector's state. A None cursor or time keeps the stored value."""
    conn.execute(
        """
        INSERT INTO ops.collector_state (collector, source, cursor, last_event_at, run_id)
        VALUES (%(collector)s, %(source)s, %(cursor)s, %(last_event_at)s, %(run_id)s)
        ON CONFLICT (collector) DO UPDATE SET
            cursor = coalesce(excluded.cursor, ops.collector_state.cursor),
            last_event_at = coalesce(excluded.last_event_at, ops.collector_state.last_event_at),
            heartbeat_at = now(),
            run_id = excluded.run_id
        """,
        {
            "collector": collector,
            "source": source,
            "cursor": cursor,
            "last_event_at": last_event_at,
            "run_id": run_id,
        },
    )


def record_gap(
    conn: psycopg.Connection, collector: str, start: datetime, end: datetime, run_id: str
) -> None:
    log.warning("%s: no data between %s and %s", collector, start.isoformat(), end.isoformat())
    conn.execute(
        "INSERT INTO ops.collector_gaps (collector, gap_start, gap_end, run_id) "
        "VALUES (%s, %s, %s, %s)",
        (collector, start, end, run_id),
    )
