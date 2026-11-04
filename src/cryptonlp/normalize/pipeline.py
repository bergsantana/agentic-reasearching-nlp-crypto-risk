"""The normalization job: every raw post not yet normalized, oldest first, into clean.posts.

Each batch:

1. picks pending posts (in raw.posts, not in clean.posts, not deleted at the source);
2. reads their rows from the raw Parquet files;
3. identifies the language, extracts entities, applies the spam rules, hashes the author;
4. marks exact duplicates (any time) and near duplicates (English, within the window), only
   ever against posts received no later than the post itself;
5. writes the rows and the new MinHash signatures in one transaction.

Posts are never dropped here: filters set flags, and ``in_study`` combines them.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import psycopg
import pyarrow.parquet as pq

from cryptonlp.normalize.config import NormalizeConfig
from cryptonlp.normalize.dedupe import NearDuplicateIndex, from_bytes, signature, to_bytes
from cryptonlp.normalize.entities import AssetMatcher, addresses, cashtags, hashtags
from cryptonlp.normalize.language import identify
from cryptonlp.normalize.pseudonym import PseudonymKey
from cryptonlp.normalize.spam import SpamFilter
from cryptonlp.normalize.text import URL, strip_nul, text_sha256
from cryptonlp.rawstore import RawStore

log = logging.getLogger(__name__)

# Only one normalization at a time; a second one skips instead of racing the first.
LOCK_ID = 7352460150482437202
RAW_COLUMNS = ["source", "uri", "author_id", "text", "created_at", "received_at", "langs", "origin"]

PENDING_SQL = """
SELECT p.source, p.uri, p.file_id, f.path
FROM raw.posts p
JOIN raw.files f ON f.file_id = p.file_id
LEFT JOIN clean.posts c USING (source, uri)
LEFT JOIN raw.deletions d USING (source, uri)
WHERE c.uri IS NULL AND d.uri IS NULL
ORDER BY p.received_at, p.source, p.uri
LIMIT %s
"""

INSERT_SQL = """
INSERT INTO clean.posts (
    source, uri, author_hash, text, created_at, received_at, origin,
    lang, lang_method, lang_confidence, langs_declared,
    assets, cashtags, hashtags, addresses, has_url, spam_rules, is_spam,
    text_sha256, duplicate_of, duplicate_kind, near_duplicate_score,
    raw_file_id, config_sha256, key_id, normalize_run_id
) VALUES (
    %(source)s, %(uri)s, %(author_hash)s, %(text)s, %(created_at)s, %(received_at)s, %(origin)s,
    %(lang)s, %(lang_method)s, %(lang_confidence)s, %(langs_declared)s,
    %(assets)s, %(cashtags)s, %(hashtags)s, %(addresses)s, %(has_url)s, %(spam_rules)s,
    %(is_spam)s, %(text_sha256)s, %(duplicate_of)s, %(duplicate_kind)s,
    %(near_duplicate_score)s, %(raw_file_id)s, %(config_sha256)s, %(key_id)s, %(run_id)s
)
"""


@dataclass
class NormalizeStats:
    batches: int = 0
    normalized: int = 0
    english: int = 0
    spam: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    in_study: int = 0
    files_read: int = 0
    skipped_locked: bool = False

    def add(self, other: NormalizeStats) -> None:
        for name, value in asdict(other).items():
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, name, getattr(self, name) + value)


@dataclass(frozen=True)
class _Original:
    uri: str
    received_at: datetime
    root: str


class Normalizer:
    def __init__(
        self, config: NormalizeConfig, key: PseudonymKey, store: RawStore, run_id: str
    ) -> None:
        self.config = config
        self.key = key
        self.store = store
        self.run_id = run_id
        self.assets = AssetMatcher(config.entities)
        self.spam = SpamFilter(config.spam)

    # --- driving ---------------------------------------------------------------------------

    def run(
        self, conn: psycopg.Connection, batch_size: int = 5000, max_batches: int | None = None
    ) -> NormalizeStats:
        """Normalize everything pending, in batches. ``conn`` must be autocommit."""
        total = NormalizeStats()
        row = conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_ID,)).fetchone()
        if not (row and row[0]):
            log.warning("another normalization is running; skipping")
            total.skipped_locked = True
            return total
        try:
            while max_batches is None or total.batches < max_batches:
                stats = self.run_batch(conn, batch_size)
                if not stats.normalized:
                    break
                total.add(stats)
                total.batches += 1
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))
        return total

    def run_batch(self, conn: psycopg.Connection, batch_size: int) -> NormalizeStats:
        stats = NormalizeStats()
        pending = conn.execute(PENDING_SQL, (batch_size,)).fetchall()
        if not pending:
            return stats
        posts = self._read(pending, stats)
        rows, signatures = self.process(conn, posts)
        with conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(INSERT_SQL, rows)
                cur.executemany(
                    "INSERT INTO clean.minhash (source, uri, received_at, signature) "
                    "VALUES (%s, %s, %s, %s)",
                    signatures,
                )
            # A deletion recorded while this batch ran: honor it before committing.
            conn.execute(
                "DELETE FROM clean.posts c USING raw.deletions d "
                "WHERE c.source = d.source AND c.uri = d.uri AND c.normalize_run_id = %s",
                (self.run_id,),
            )
            # Signatures older than the window are no longer needed (batches move forward).
            oldest = min(p["received_at"] for p in posts)
            conn.execute(
                "DELETE FROM clean.minhash WHERE received_at < %s",
                (oldest - timedelta(days=self.config.dedupe.window_days),),
            )
        stats.normalized = len(rows)
        stats.english = sum(r["lang"] == "en" for r in rows)
        stats.spam = sum(r["is_spam"] for r in rows)
        stats.exact_duplicates = sum(r["duplicate_kind"] == "exact" for r in rows)
        stats.near_duplicates = sum(r["duplicate_kind"] == "near" for r in rows)
        stats.in_study = sum(
            r["lang"] == "en"
            and not r["is_spam"]
            and r["duplicate_of"] is None
            and r["origin"] == "live"
            for r in rows
        )
        return stats

    def _read(self, pending: list[tuple[Any, ...]], stats: NormalizeStats) -> list[dict[str, Any]]:
        wanted: dict[tuple[int, str], set[str]] = defaultdict(set)
        for _source, uri, file_id, path in pending:
            wanted[(file_id, path)].add(uri)
        posts = []
        for (file_id, path), uris in wanted.items():
            table = pq.read_table(self.store.root / path, columns=RAW_COLUMNS)
            stats.files_read += 1
            for row in table.to_pylist():
                if row["uri"] in uris:
                    row["raw_file_id"] = file_id
                    # Every rule below sees the same text that is stored.
                    row["text"] = strip_nul(row["text"])
                    row["langs"] = [strip_nul(t) for t in row["langs"] or ()]
                    posts.append(row)
        if len(posts) != len(pending):
            raise RuntimeError(
                f"{len(pending) - len(posts)} pending posts were not found in their raw files"
            )
        posts.sort(key=lambda p: (p["received_at"], p["source"], p["uri"]))
        return posts

    # --- the rules -------------------------------------------------------------------------

    def process(
        self, conn: psycopg.Connection, posts: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[tuple[str, str, datetime, bytes]]]:
        """Rows for clean.posts and new signatures, for posts sorted by received_at."""
        dd = self.config.dedupe
        langs = identify(
            [p["text"] for p in posts], [p["langs"] or () for p in posts], self.config.language
        )
        hashes = [text_sha256(p["text"]) for p in posts]
        exact = self._earliest_by_hash(conn, set(hashes))
        index = self._load_window(
            conn,
            min(p["received_at"] for p in posts) - timedelta(days=dd.window_days),
            max(p["received_at"] for p in posts),
        )

        rows = []
        new_sigs = []
        for post, lang, digest in zip(posts, langs, hashes, strict=True):
            text = post["text"]
            received = post["received_at"]
            tags = cashtags(text)
            hashtag_list = hashtags(text)
            has_url = bool(URL.search(text))
            rules = self.spam.rules_matched(
                text, has_url=has_url, n_cashtags=len(tags), n_hashtags=len(hashtag_list)
            )
            duplicate_of = kind = score = None

            original = exact.get(digest)
            if original is not None and original.received_at <= received:
                duplicate_of, kind = original.root, "exact"
            elif original is None:
                exact[digest] = _Original(post["uri"], received, post["uri"])

            if kind is None and lang.lang == "en":
                sig = signature(text, dd)
                if sig is not None:
                    match = index.query(sig, received)
                    if match is not None and match.root != post["uri"]:
                        duplicate_of, kind, score = match.root, "near", match.score
                    index.add(post["uri"], received, sig, duplicate_of)
                    new_sigs.append((post["source"], post["uri"], received, to_bytes(sig)))
            if kind == "near" and exact.get(digest) and exact[digest].uri == post["uri"]:
                # First of its exact text, but itself a near copy: later exact copies lead
                # back to the same original.
                exact[digest] = _Original(post["uri"], received, duplicate_of or post["uri"])

            rows.append(
                {
                    "source": post["source"],
                    "uri": post["uri"],
                    "author_hash": self.key.author_hash(post["source"], post["author_id"]),
                    "text": text,
                    "created_at": post["created_at"],
                    "received_at": received,
                    "origin": post["origin"],
                    "lang": lang.lang,
                    "lang_method": lang.method,
                    "lang_confidence": lang.confidence,
                    "langs_declared": list(post["langs"] or ()),
                    "assets": list(self.assets.assets(text)),
                    "cashtags": list(tags),
                    "hashtags": list(hashtag_list),
                    "addresses": list(addresses(text)),
                    "has_url": has_url,
                    "spam_rules": list(rules),
                    "is_spam": bool(rules),
                    "text_sha256": digest,
                    "duplicate_of": duplicate_of,
                    "duplicate_kind": kind,
                    "near_duplicate_score": score,
                    "raw_file_id": post["raw_file_id"],
                    "config_sha256": self.config.sha256,
                    "key_id": self.key.key_id,
                    "run_id": self.run_id,
                }
            )
        return rows, new_sigs

    @staticmethod
    def _earliest_by_hash(conn: psycopg.Connection, hashes: set[str]) -> dict[str, _Original]:
        rows = conn.execute(
            "SELECT DISTINCT ON (text_sha256) text_sha256, uri, received_at, duplicate_of "
            "FROM clean.posts WHERE text_sha256 = ANY(%s) "
            "ORDER BY text_sha256, received_at, uri",
            (list(hashes),),
        ).fetchall()
        return {r[0]: _Original(r[1], r[2], r[3] or r[1]) for r in rows}

    def _load_window(
        self, conn: psycopg.Connection, start: datetime, end: datetime
    ) -> NearDuplicateIndex:
        index = NearDuplicateIndex(self.config.dedupe)
        rows = conn.execute(
            "SELECT m.uri, m.received_at, m.signature, c.duplicate_of "
            "FROM clean.minhash m JOIN clean.posts c USING (source, uri) "
            "WHERE m.received_at >= %s AND m.received_at <= %s ORDER BY m.received_at",
            (start, end),
        ).fetchall()
        for uri, received, sig, root in rows:
            index.add(uri, received, from_bytes(sig), root)
        return index
