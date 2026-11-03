from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import psycopg
import pytest

from cryptonlp.collectors.common import WatchFilter
from cryptonlp.config import CollectorsConfig, load_config
from cryptonlp.db import apply_init_sql, connect_test
from cryptonlp.rawstore import RawStore

RUN_ID = "20261103T120000Z-0123456789ab"
FIXTURES = Path(__file__).parent / "fixtures"
STEP1_TABLES = (
    "raw.deletions, raw.posts, raw.files, ops.collector_state, ops.collector_gaps, ops.job_runs"
)


@pytest.fixture
def run_id() -> str:
    return RUN_ID


def _connect_or_skip() -> psycopg.Connection:
    try:
        return connect_test()
    except psycopg.OperationalError as exc:
        if os.environ.get("CI", "").lower() == "true":
            raise
        pytest.skip(f"test database not reachable (docker compose up -d postgres-test): {exc}")


@pytest.fixture(scope="session")
def test_db() -> Iterator[psycopg.Connection]:
    """Connection to the test database. Skipped locally if it's down; a hard failure in CI."""
    conn = _connect_or_skip()
    apply_init_sql(conn)
    yield conn
    conn.close()


@pytest.fixture
def connect(test_db: psycopg.Connection) -> Callable[[], psycopg.Connection]:
    """Opens autocommit connections to the test database, as the collectors use."""

    def _connect() -> psycopg.Connection:
        conn = connect_test()
        conn.autocommit = True
        return conn

    return _connect


@pytest.fixture
def db(connect: Callable[[], psycopg.Connection]) -> Iterator[psycopg.Connection]:
    """An autocommit test-database connection, with the step 1 tables emptied first."""
    conn = connect()
    conn.execute(f"TRUNCATE {STEP1_TABLES} RESTART IDENTITY CASCADE")
    yield conn
    conn.close()


@pytest.fixture
def store(tmp_path: Path) -> RawStore:
    return RawStore(tmp_path / "raw")


@pytest.fixture(scope="session")
def config() -> CollectorsConfig:
    return load_config()


@pytest.fixture
def watch(config: CollectorsConfig) -> WatchFilter:
    return WatchFilter(config.watch)
