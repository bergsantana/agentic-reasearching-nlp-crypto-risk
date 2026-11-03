from __future__ import annotations

import psycopg
import pytest

from cryptonlp.db import DatabaseGuardError, connect_main, require_test_database
from cryptonlp.settings import Settings


def test_guard_accepts_test_database() -> None:
    url = "postgresql://u:p@localhost:5433/cryptonlp_test"
    assert require_test_database(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://u:p@localhost:5432/cryptonlp",
        "postgresql://u:p@localhost:5432/cryptonlp_test_copy",
        "host=localhost dbname=cryptonlp",
    ],
)
def test_guard_rejects_other_databases(url: str) -> None:
    with pytest.raises(DatabaseGuardError):
        require_test_database(url)


def test_main_database_blocked_under_pytest() -> None:
    with pytest.raises(DatabaseGuardError, match="only use the test database"):
        connect_main(Settings(database_url="postgresql://u:p@localhost:5432/cryptonlp"))


@pytest.mark.db
def test_test_database_is_ready(test_db: psycopg.Connection) -> None:
    row = test_db.execute(
        "select current_database(), current_setting('TimeZone'), "
        "(select extversion from pg_extension where extname = 'vector')"
    ).fetchone()
    assert row is not None
    name, tz, vector_version = row
    assert name.endswith("_test")
    assert tz == "UTC"
    assert vector_version is not None


@pytest.mark.db
def test_pgvector_works(test_db: psycopg.Connection) -> None:
    row = test_db.execute("select '[1,2,3]'::vector <-> '[1,2,4]'::vector").fetchone()
    test_db.rollback()
    assert row is not None and row[0] == pytest.approx(1.0)
