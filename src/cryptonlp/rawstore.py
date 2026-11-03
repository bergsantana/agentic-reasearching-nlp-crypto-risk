"""The raw store: Parquet files partitioned by source and day, each with a SHA-256 sidecar.

Rules:

- A file is written once (to a temporary name, fsynced, then renamed) and made read-only.
  It is never edited. Its hash goes into ``raw.files`` and a ``<file>.sha256`` sidecar in
  ``sha256sum`` format, so ``sha256sum -c`` works without the database.
- Honoring a deletion at the source writes a *successor* file without the deleted posts,
  records which file it supersedes, and only then removes the old file. The old row keeps its
  hash, so every version a ledger package cites can still be traced.
- Posts already in the index are dropped before writing, so the at-least-once delivery of the
  sources never puts the same post into two files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from cryptonlp.schemas import RawPost

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
_META_PREFIX = "cryptonlp."
_UTC_US = pa.timestamp("us", tz="UTC")

POSTS_DATASET = {"bluesky": "bluesky_posts", "mastodon": "mastodon_posts"}
POST_DATASETS = frozenset(POSTS_DATASET.values())

POST_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string(), nullable=False),
        pa.field("uri", pa.string(), nullable=False),
        pa.field("author_id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("created_at", _UTC_US, nullable=False),
        pa.field("received_at", _UTC_US, nullable=False),
        pa.field("langs", pa.list_(pa.string()), nullable=False),
        pa.field("origin", pa.string(), nullable=False),
        pa.field("collector_run_id", pa.string(), nullable=False),
        pa.field("matched", pa.list_(pa.string()), nullable=False),  # watch terms or tags
        pa.field("instance", pa.string()),  # Mastodon instance it was read from
        pa.field("native_id", pa.string()),  # that instance's status id
        pa.field("payload", pa.string(), nullable=False),  # the event exactly as received (JSON)
    ]
)


@dataclass(frozen=True)
class CollectedPost:
    """A post plus what the raw store keeps around it."""

    post: RawPost
    payload: Mapping[str, Any]
    matched: tuple[str, ...] = ()
    instance: str | None = None
    native_id: str | None = None

    def row(self) -> dict[str, Any]:
        return {
            **self.post.model_dump(mode="python"),
            "langs": list(self.post.langs),
            "matched": list(self.matched),
            "instance": self.instance,
            "native_id": self.native_id,
            "payload": json.dumps(self.payload, ensure_ascii=False, separators=(",", ":")),
        }


@dataclass(frozen=True)
class WrittenFile:
    path: str  # relative to the store root, with forward slashes
    sha256: str
    bytes: int
    rows: int


@dataclass
class PostWriteResult:
    received: int = 0
    duplicates: int = 0
    written: int = 0
    files: list[WrittenFile] = field(default_factory=list)


@dataclass
class VerifyReport:
    checked: int = 0
    mismatched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unregistered: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.mismatched or self.missing)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_write(path: Path, data: bytes) -> None:
    with path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    # --- files -------------------------------------------------------------------------------

    def partition_dir(
        self, dataset: str, day: date, partition: Mapping[str, str] | None = None
    ) -> Path:
        parts = [f"{k}={v}" for k, v in (partition or {}).items()]
        return self.root.joinpath(dataset, *parts, f"date={day.isoformat()}")

    def _finish(self, tmp: Path, final: Path, rows: int) -> WrittenFile:
        digest = sha256_file(tmp)
        size = tmp.stat().st_size
        os.replace(tmp, final)
        final.chmod(0o444)
        sidecar = final.with_name(final.name + ".sha256")
        _fsync_write(sidecar, f"{digest}  {final.name}\n".encode())
        sidecar.chmod(0o444)
        return WrittenFile(final.relative_to(self.root).as_posix(), digest, size, rows)

    def write_table(
        self,
        directory: Path,
        table: pa.Table,
        metadata: Mapping[str, str],
        prefix: str = "part",
    ) -> WrittenFile:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {f"{_META_PREFIX}{k}": v for k, v in metadata.items()}
        meta[f"{_META_PREFIX}schema_version"] = SCHEMA_VERSION
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
        final = directory / f"{prefix}-{_stamp()}-{uuid4().hex[:8]}.parquet"
        tmp = final.with_name(final.name + ".tmp")
        pq.write_table(table, tmp, compression="zstd")
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        return self._finish(tmp, final, table.num_rows)

    def write_blob(self, rel_path: str, data: bytes, expected_sha256: str | None) -> WrittenFile:
        """Store a downloaded file as is (e.g. a Binance Vision zip), checking its hash."""
        final = self.root / rel_path
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"checksum mismatch for {rel_path}: {digest} != {expected_sha256}")
        if final.exists():
            # Left by a run that crashed before registering it: reuse it if it is identical.
            if sha256_file(final) != digest:
                raise FileExistsError(f"raw files are never overwritten: {rel_path}")
            return WrittenFile(final.relative_to(self.root).as_posix(), digest, len(data), 0)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(final.name + ".tmp")
        _fsync_write(tmp, data)
        return self._finish(tmp, final, rows=0)

    def register(
        self,
        conn: psycopg.Connection,
        written: WrittenFile,
        *,
        dataset: str,
        day: date,
        run_id: str,
        received: Sequence[datetime] = (),
        source_url: str | None = None,
        supersedes: int | None = None,
    ) -> int:
        row = conn.execute(
            """
            INSERT INTO raw.files (dataset, partition_date, path, sha256, bytes, rows,
                                   min_received_at, max_received_at, source_url, run_id,
                                   supersedes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING file_id
            """,
            (
                dataset,
                day,
                written.path,
                written.sha256,
                written.bytes,
                written.rows,
                min(received, default=None),
                max(received, default=None),
                source_url,
                run_id,
                supersedes,
            ),
        ).fetchone()
        assert row is not None
        return int(row[0])

    # --- posts -------------------------------------------------------------------------------

    def store_posts(
        self,
        conn: psycopg.Connection,
        posts: Iterable[CollectedPost],
        run_id: str,
        metadata: Mapping[str, str] | None = None,
    ) -> PostWriteResult:
        """Write new posts to Parquet (one file per source and day) and index them.

        Runs in one transaction. If the database fails after a file is written, the file stays
        and ``reindex`` registers it later; it is never lost.
        """
        result = PostWriteResult()
        unique: dict[tuple[str, str], CollectedPost] = {}
        for p in posts:
            result.received += 1
            unique.setdefault((p.post.source.value, p.post.uri), p)
        result.duplicates = result.received - len(unique)
        if not unique:
            return result

        with conn.transaction():
            known = _known_uris(conn, list(unique))
            result.duplicates += len(known)
            groups: dict[tuple[str, date], list[CollectedPost]] = defaultdict(list)
            for key, p in unique.items():
                if key not in known:
                    groups[(key[0], _utc(p.post.received_at).date())].append(p)

            for (source, day), group in sorted(groups.items()):
                group.sort(key=lambda p: p.post.received_at)
                dataset = POSTS_DATASET[source]
                table = pa.Table.from_pylist([p.row() for p in group], schema=POST_SCHEMA)
                meta = {
                    "dataset": dataset,
                    "partition_date": day.isoformat(),
                    "run_id": run_id,
                    **(metadata or {}),
                }
                written = self.write_table(self.partition_dir(dataset, day), table, meta)
                file_id = self.register(
                    conn,
                    written,
                    dataset=dataset,
                    day=day,
                    run_id=run_id,
                    received=[p.post.received_at for p in group],
                )
                _index_posts(conn, table, file_id)
                result.written += len(group)
                result.files.append(written)
        return result

    # --- integrity -----------------------------------------------------------------------------

    def verify(self, conn: psycopg.Connection) -> VerifyReport:
        """Recompute every live file's hash and compare it with the manifest."""
        report = VerifyReport()
        registered = set()
        for rel, digest in conn.execute(
            "SELECT path, sha256 FROM raw.files WHERE removed_at IS NULL ORDER BY file_id"
        ):
            registered.add(rel)
            path = self.root / rel
            report.checked += 1
            if not path.exists():
                report.missing.append(rel)
            elif sha256_file(path) != digest:
                report.mismatched.append(rel)
        known_paths = {r[0] for r in conn.execute("SELECT path FROM raw.files")}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and not path.name.endswith((".sha256", ".tmp")):
                rel = path.relative_to(self.root).as_posix()
                if rel not in known_paths:
                    report.unregistered.append(rel)
        return report

    def reindex(self, conn: psycopg.Connection) -> list[str]:
        """Register Parquet files that were written but never reached the manifest."""
        known = {r[0] for r in conn.execute("SELECT path FROM raw.files")}
        added = []
        for path in sorted(self.root.rglob("*.parquet")):
            rel = path.relative_to(self.root).as_posix()
            if rel in known:
                continue
            sidecar = path.with_name(path.name + ".sha256")
            digest = sha256_file(path)
            if not sidecar.exists() or sidecar.read_text().split()[0] != digest:
                log.error("not registering %s: missing or wrong sidecar hash", rel)
                continue
            meta = read_metadata(path)
            table = pq.read_table(path)
            has_times = "received_at" in table.column_names
            received = table.column("received_at").to_pylist() if has_times else []
            with conn.transaction():
                file_id = self.register(
                    conn,
                    WrittenFile(rel, digest, path.stat().st_size, table.num_rows),
                    dataset=meta["dataset"],
                    day=date.fromisoformat(meta["partition_date"]),
                    run_id=meta["run_id"],
                    received=received,
                )
                if meta["dataset"] in POST_DATASETS:
                    _index_posts(conn, table, file_id)
            added.append(rel)
        return added

    # --- deletions -----------------------------------------------------------------------------

    def purge_deleted(self, conn: psycopg.Connection, run_id: str) -> dict[str, int]:
        """Rewrite every file holding a post deleted at the source, without those posts."""
        stats = {"files_rewritten": 0, "posts_purged": 0}
        files = conn.execute(
            """
            SELECT DISTINCT f.file_id, f.path, f.dataset, f.partition_date
            FROM raw.deletions d
            JOIN raw.posts p USING (source, uri)
            JOIN raw.files f ON f.file_id = p.file_id
            WHERE d.purged_at IS NULL
            ORDER BY f.file_id
            """
        ).fetchall()
        for file_id, rel, dataset, day in files:
            old = self.root / rel
            deleted = {
                r[0]
                for r in conn.execute(
                    "SELECT p.uri FROM raw.posts p JOIN raw.deletions d USING (source, uri) "
                    "WHERE p.file_id = %s",
                    (file_id,),
                )
            }
            table = pq.read_table(old)
            meta = read_metadata(old)
            mask = [uri not in deleted for uri in table.column("uri").to_pylist()]
            kept = table.filter(pa.array(mask, pa.bool_()))
            gone = sorted(deleted)
            with conn.transaction():
                new_id: int | None = None
                if kept.num_rows:
                    written = self.write_table(
                        old.parent,
                        kept,
                        {**meta, "run_id": run_id, "supersedes_sha256": sha256_file(old)},
                    )
                    new_id = self.register(
                        conn,
                        written,
                        dataset=dataset,
                        day=day,
                        run_id=run_id,
                        received=kept.column("received_at").to_pylist(),
                        supersedes=file_id,
                    )
                    # Deleted posts keep pointing at the removed file they were last in.
                    conn.execute(
                        "UPDATE raw.posts SET file_id = %s WHERE file_id = %s AND uri <> ALL(%s)",
                        (new_id, file_id, gone),
                    )
                reason = (
                    f"deletions at source purged; successor file_id {new_id}"
                    if new_id
                    else "all posts deleted at source"
                )
                conn.execute(
                    "UPDATE raw.files SET removed_at = now(), removed_reason = %s "
                    "WHERE file_id = %s",
                    (reason, file_id),
                )
                conn.execute(
                    "UPDATE raw.deletions d SET purged_at = now() FROM raw.posts p "
                    "WHERE p.source = d.source AND p.uri = d.uri AND p.file_id = %s "
                    "AND d.uri = ANY(%s)",
                    (file_id, gone),
                )
            self._unlink(old)
            stats["files_rewritten"] += 1
            stats["posts_purged"] += len(gone)
        # Old files left behind by a crash between the commit and the unlink.
        for (rel,) in conn.execute("SELECT path FROM raw.files WHERE removed_at IS NOT NULL"):
            if (self.root / rel).exists():
                self._unlink(self.root / rel)
        return stats

    @staticmethod
    def _unlink(path: Path) -> None:
        for p in (path, path.with_name(path.name + ".sha256")):
            if p.exists():
                p.chmod(0o644)
                p.unlink()


def read_metadata(path: Path) -> dict[str, str]:
    raw = pq.read_schema(path).metadata or {}
    return {
        k.decode().removeprefix(_META_PREFIX): v.decode()
        for k, v in raw.items()
        if k.decode().startswith(_META_PREFIX)
    }


def _known_uris(conn: psycopg.Connection, keys: list[tuple[str, str]]) -> set[tuple[str, str]]:
    rows = conn.execute(
        "SELECT source, uri FROM raw.posts WHERE (source, uri) IN "
        "(SELECT * FROM unnest(%s::text[], %s::text[]))",
        ([k[0] for k in keys], [k[1] for k in keys]),
    ).fetchall()
    return {(r[0], r[1]) for r in rows}


def _index_posts(conn: psycopg.Connection, table: pa.Table, file_id: int) -> None:
    cols = table.select(
        ["source", "uri", "instance", "native_id", "created_at", "received_at", "origin"]
    ).to_pydict()
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO raw.posts (source, uri, instance, native_id, created_at, received_at,
                                   origin, file_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, uri) DO NOTHING
            """,
            [
                (*row, file_id)
                for row in zip(
                    cols["source"],
                    cols["uri"],
                    cols["instance"],
                    cols["native_id"],
                    cols["created_at"],
                    cols["received_at"],
                    cols["origin"],
                    strict=True,
                )
            ],
        )
