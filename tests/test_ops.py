from __future__ import annotations

from collections.abc import Callable

import psycopg
import pytest

from cryptonlp import ops

pytestmark = pytest.mark.db


def test_track_job_success(
    db: psycopg.Connection, connect: Callable[[], psycopg.Connection], run_id: str
) -> None:
    with ops.track_job(connect, "demo", run_id) as job:
        job.stats["n"] = 3
    row = db.execute("SELECT status, finished_at IS NOT NULL, stats FROM ops.job_runs").fetchone()
    assert row == ("succeeded", True, {"n": 3})


def test_track_job_failure_is_recorded_and_reraised(
    db: psycopg.Connection, connect: Callable[[], psycopg.Connection], run_id: str
) -> None:
    with pytest.raises(ZeroDivisionError), ops.track_job(connect, "demo", run_id):
        _ = 1 / 0
    status, error = db.execute("SELECT status, error FROM ops.job_runs").fetchone()  # type: ignore[misc]
    assert status == "failed" and "ZeroDivisionError" in error
    view = db.execute("SELECT job, status FROM monitoring.job_runs").fetchall()
    assert view == [("demo", "failed")]


def test_save_state_keeps_values_not_given(db: psycopg.Connection, run_id: str) -> None:
    ops.save_state(db, "x", "bluesky", run_id, cursor="10")
    ops.save_state(db, "x", "bluesky", run_id)
    state = ops.get_state(db, "x")
    assert state is not None and state.cursor == "10"


def test_exclusive_job_closes_stale_running_rows(
    db: psycopg.Connection, connect: Callable[[], psycopg.Connection], run_id: str
) -> None:
    db.execute(
        "INSERT INTO ops.job_runs (job, run_id, status) VALUES ('stream', %s, 'running')", (run_id,)
    )
    with ops.track_job(connect, "stream", run_id, exclusive=True):
        pass
    rows = db.execute("SELECT status, error FROM ops.job_runs ORDER BY job_run_id").fetchall()
    assert rows[0][0] == "failed" and "without reporting" in rows[0][1]
    assert rows[1] == ("succeeded", None)
