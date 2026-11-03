from __future__ import annotations

import stat
import subprocess
from datetime import timedelta
from pathlib import Path

import psycopg
import pyarrow.parquet as pq
import pytest

from cryptonlp.rawstore import CollectedPost, RawStore, read_metadata, sha256_file
from cryptonlp.schemas import RawPost, Source
from tests.factories import T0

pytestmark = pytest.mark.db


def post(n: int, run_id: str, minutes: float = 0, source: Source = Source.BLUESKY) -> CollectedPost:
    return CollectedPost(
        post=RawPost(
            source=source,
            uri=f"at://did:plc:a/app.bsky.feed.post/{n}",
            author_id="did:plc:a",
            text=f"bitcoin post {n}",
            created_at=T0,
            received_at=T0 + timedelta(minutes=minutes),
            collector_run_id=run_id,
        ),
        payload={"n": n},
        matched=("bitcoin",),
    )


def test_store_posts_writes_hashed_read_only_file(
    db: psycopg.Connection, store: RawStore, run_id: str
) -> None:
    result = store.store_posts(db, [post(1, run_id), post(2, run_id)], run_id, {"collector": "t"})
    assert (result.received, result.written, len(result.files)) == (2, 2, 1)
    written = result.files[0]
    path = store.root / written.path
    assert written.path.startswith("bluesky_posts/date=2026-11-03/")
    assert sha256_file(path) == written.sha256
    assert not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    # The sidecar works with plain sha256sum, without the database.
    check = subprocess.run(
        ["sha256sum", "-c", path.name + ".sha256"], cwd=path.parent, capture_output=True
    )
    assert check.returncode == 0, check.stdout

    table = pq.read_table(path)
    assert table.column("uri").to_pylist() == [
        p.post.uri for p in (post(1, run_id), post(2, run_id))
    ]
    assert table.column("matched").to_pylist() == [["bitcoin"], ["bitcoin"]]
    meta = read_metadata(path)
    assert meta["dataset"] == "bluesky_posts" and meta["collector"] == "t"

    row = db.execute("SELECT count(*), min(file_id) FROM raw.posts").fetchone()
    assert row == (2, 1)
    assert db.execute("SELECT sha256 FROM raw.files").fetchone() == (written.sha256,)


def test_duplicates_are_dropped_in_batch_and_across_batches(
    db: psycopg.Connection, store: RawStore, run_id: str
) -> None:
    store.store_posts(db, [post(1, run_id)], run_id)
    result = store.store_posts(db, [post(1, run_id), post(1, run_id), post(2, run_id)], run_id)
    assert (result.duplicates, result.written) == (2, 1)
    assert db.execute("SELECT count(*) FROM raw.posts").fetchone() == (2,)


def test_posts_split_by_reception_day(db: psycopg.Connection, store: RawStore, run_id: str) -> None:
    late = post(1, run_id, minutes=60 * 9 + 59)  # 23:59 UTC
    next_day = post(2, run_id, minutes=60 * 10 + 1)  # 00:01 UTC the next day
    result = store.store_posts(db, [late, next_day], run_id)
    assert sorted(f.path.split("/")[1] for f in result.files) == [
        "date=2026-11-03",
        "date=2026-11-04",
    ]


def test_verify_catches_tampering_and_missing_files(
    db: psycopg.Connection, store: RawStore, run_id: str
) -> None:
    a, b = (store.store_posts(db, [post(n, run_id)], run_id).files[0] for n in (1, 2))
    assert store.verify(db).ok
    path_a = store.root / a.path
    path_a.chmod(0o644)
    path_a.write_bytes(path_a.read_bytes() + b"x")
    (store.root / b.path).chmod(0o644)
    (store.root / b.path).unlink()
    report = store.verify(db)
    assert report.mismatched == [a.path] and report.missing == [b.path]
    assert not report.ok


def test_reindex_registers_orphan_files(
    db: psycopg.Connection, store: RawStore, run_id: str, tmp_path: Path
) -> None:
    written = store.store_posts(db, [post(1, run_id), post(2, run_id)], run_id).files[0]
    # The database lost the rows (as if the commit had failed after the file was written).
    db.execute("TRUNCATE raw.posts, raw.files RESTART IDENTITY CASCADE")
    assert store.verify(db).unregistered == [written.path]
    assert store.reindex(db) == [written.path]
    assert db.execute("SELECT count(*) FROM raw.posts").fetchone() == (2,)
    assert store.reindex(db) == []


def test_purge_writes_successor_and_keeps_old_hash(
    db: psycopg.Connection, store: RawStore, run_id: str
) -> None:
    old = store.store_posts(db, [post(1, run_id), post(2, run_id)], run_id).files[0]
    deleted_uri = post(1, run_id).post.uri
    db.execute(
        "INSERT INTO raw.deletions (source, uri, method, run_id) "
        "VALUES ('bluesky', %s, 'stream', %s)",
        (deleted_uri, run_id),
    )
    stats = store.purge_deleted(db, run_id)
    assert stats == {"files_rewritten": 1, "posts_purged": 1}

    assert not (store.root / old.path).exists()
    rows = db.execute(
        "SELECT file_id, sha256, removed_at IS NOT NULL, supersedes FROM raw.files ORDER BY 1"
    ).fetchall()
    assert rows[0][1] == old.sha256 and rows[0][2]  # the old version keeps its hash
    assert rows[1][3] == rows[0][0] and not rows[1][2]  # the successor points back to it
    new_row = db.execute("SELECT path FROM raw.files WHERE file_id = 2").fetchone()
    assert new_row is not None
    new_path = store.root / new_row[0]
    assert pq.read_table(new_path).column("uri").to_pylist() == [post(2, run_id).post.uri]
    assert read_metadata(new_path)["supersedes_sha256"] == old.sha256

    index: dict[str, int] = dict(db.execute("SELECT uri, file_id FROM raw.posts").fetchall())
    assert index == {deleted_uri: 1, post(2, run_id).post.uri: 2}
    assert db.execute("SELECT purged_at IS NOT NULL FROM raw.deletions").fetchone() == (True,)
    assert store.verify(db).ok
    assert store.purge_deleted(db, run_id) == {"files_rewritten": 0, "posts_purged": 0}


def test_purge_of_a_fully_deleted_file(
    db: psycopg.Connection, store: RawStore, run_id: str
) -> None:
    old = store.store_posts(db, [post(1, run_id)], run_id).files[0]
    db.execute(
        "INSERT INTO raw.deletions (source, uri, method, run_id) "
        "VALUES ('bluesky', %s, 'sweep', %s)",
        (post(1, run_id).post.uri, run_id),
    )
    store.purge_deleted(db, run_id)
    assert not (store.root / old.path).exists()
    assert db.execute("SELECT count(*) FROM raw.files WHERE removed_at IS NULL").fetchone() == (0,)


def test_blobs_are_checked_and_never_overwritten(store: RawStore) -> None:
    import hashlib

    data = b"zip bytes"
    digest = hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError, match="checksum"):
        store.write_blob("binance_vision/x.zip", data, "0" * 64)
    first = store.write_blob("binance_vision/x.zip", data, digest)
    assert store.write_blob("binance_vision/x.zip", data, digest) == first  # crash recovery
    with pytest.raises(FileExistsError):
        store.write_blob("binance_vision/x.zip", b"other", None)
