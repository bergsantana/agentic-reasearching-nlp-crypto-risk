"""Database connections, with guards that keep tests and the real ledger apart.

- ``connect_test`` only opens a database whose name ends in ``_test``.
- ``connect_main`` refuses to run under pytest or CI, so a test can't reach real data by mistake.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from cryptonlp.settings import Settings

TEST_DB_SUFFIX = "_test"
INIT_SQL_DIR = Path(__file__).resolve().parents[2] / "db" / "init"


class DatabaseGuardError(RuntimeError):
    """Raised when code would connect to the wrong database."""


def database_name(url: str) -> str:
    name = conninfo_to_dict(url).get("dbname")
    if not name:
        raise DatabaseGuardError(f"connection string names no database: {url!r}")
    return str(name)


def require_test_database(url: str) -> str:
    name = database_name(url)
    if not name.endswith(TEST_DB_SUFFIX):
        raise DatabaseGuardError(
            f"refusing to use database {name!r} for tests: its name must end in {TEST_DB_SUFFIX!r}"
        )
    return url


def _in_test_or_ci() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("CI", "").lower() == "true"


def connect(url: str) -> psycopg.Connection:
    # Every session runs in UTC, so now() and timestamps never depend on the host's zone.
    return psycopg.connect(url, options="-c timezone=UTC")


def connect_main(settings: Settings | None = None) -> psycopg.Connection:
    if _in_test_or_ci():
        raise DatabaseGuardError("tests and CI may only use the test database (connect_test)")
    url = (settings or Settings()).database_url
    if database_name(url).endswith(TEST_DB_SUFFIX):
        raise DatabaseGuardError("the main database URL points at a test database")
    return connect(url)


def connect_test(settings: Settings | None = None) -> psycopg.Connection:
    return connect(require_test_database((settings or Settings()).test_database_url))


def apply_init_sql(conn: psycopg.Connection, sql_dir: Path = INIT_SQL_DIR) -> list[str]:
    """Apply the idempotent bootstrap scripts in ``db/init`` (Compose runs them on first start)."""
    applied = []
    for path in sorted(sql_dir.glob("*.sql")):
        conn.execute(path.read_bytes())  # trusted files from the repo
        applied.append(path.name)
    conn.commit()
    return applied


def set_grafana_password(conn: psycopg.Connection, password: str) -> None:
    """Grafana logs in as ``grafana_reader``, which can only read the ``monitoring`` views."""
    conn.execute(
        sql.SQL("ALTER ROLE grafana_reader WITH LOGIN PASSWORD {}").format(sql.Literal(password))
    )
    conn.commit()
