"""The replay day: freeze one recorded UTC day, and feed a frozen day back through the collectors.

Recording happens in the collectors (``--record``), into ``data/replay/<day>/`` (git-ignored).
``freeze`` copies one complete day into ``tests/fixtures/replay/<day>/`` with a manifest of
message counts and hashes. ``replay`` runs it through the same code as live data, with
``origin = replay``, against the test database only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from datetime import UTC, date, timedelta
from pathlib import Path
from typing import Any

import psycopg

from cryptonlp.collectors import binance, bluesky, mastodon, trends
from cryptonlp.collectors.common import Message, WatchFilter, read_recording, recording_files
from cryptonlp.config import CollectorsConfig
from cryptonlp.db import database_name, require_test_database
from cryptonlp.rawstore import RawStore
from cryptonlp.schemas import Origin

log = logging.getLogger(__name__)

STREAMS = (bluesky.STREAM, mastodon.STREAM, trends.STREAM, binance.STREAM)
MANIFEST = "manifest.json"
# A frozen day must start and end this close to midnight UTC.
COVERAGE_SLACK = timedelta(minutes=15)


def _write_gzip(path: Path, lines: list[str]) -> str:
    """Deterministic gzip (no timestamp in the header), so the same day gives the same hash."""
    data = gzip.compress("".join(line + "\n" for line in lines).encode(), mtime=0)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def freeze(source: Path, day: date, dest: Path, *, require_full_day: bool = True) -> dict[str, Any]:
    day_dir = source / day.isoformat()
    if not day_dir.is_dir():
        raise FileNotFoundError(f"no recording for {day} in {source}")
    start = day.isoformat()
    manifest: dict[str, Any] = {"day": start, "streams": {}}
    dest.mkdir(parents=True, exist_ok=True)
    for stream in STREAMS:
        path = day_dir / f"{stream}.jsonl.gz"
        if not path.exists():
            if require_full_day:
                raise FileNotFoundError(f"stream {stream} was not recorded on {day}")
            continue
        msgs = sorted(
            (m for m in read_recording(path) if m.received_at.astimezone(UTC).date() == day),
            key=lambda m: m.received_at,
        )
        if not msgs:
            raise ValueError(f"stream {stream} has no messages on {day}")
        first, last = msgs[0].received_at, msgs[-1].received_at
        midnight = msgs[0].received_at.replace(hour=0, minute=0, second=0, microsecond=0)
        full = first - midnight <= COVERAGE_SLACK and midnight + timedelta(days=1) - last <= (
            COVERAGE_SLACK + timedelta(hours=1)  # hourly jobs leave up to an hour at the end
        )
        if require_full_day and not full:
            raise ValueError(f"stream {stream} covers only {first} to {last}, not the whole day")
        digest = _write_gzip(dest / f"{stream}.jsonl.gz", [m.to_json() for m in msgs])
        manifest["streams"][stream] = {
            "messages": len(msgs),
            "first": first.isoformat(),
            "last": last.isoformat(),
            "sha256": digest,
        }
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _messages(directory: Path, stream: str) -> list[Message]:
    return [m for path in recording_files(directory, stream) for m in read_recording(path)]


async def replay(
    conn: psycopg.Connection,
    store: RawStore,
    config: CollectorsConfig,
    directory: Path,
    run_id: str,
) -> dict[str, Any]:
    """Feed a recorded or frozen day through every collector. Test database only."""
    url = conn.info.dsn
    require_test_database(url)
    log.info("replaying %s into %s", directory, database_name(url))
    watch = WatchFilter(config.watch)
    stats: dict[str, Any] = {}

    bsky = bluesky.BlueskyCollector(
        conn, store, config.bluesky, watch, run_id, origin=Origin.REPLAY
    )
    await bsky.run(bluesky.replay_messages(recording_files(directory, bluesky.STREAM)))
    stats["bluesky"] = bsky.stats

    masto = mastodon.MastodonCollector(
        conn, store, config.mastodon, watch, run_id, origin=Origin.REPLAY
    )
    with conn.transaction():
        result = masto.store_pages(_messages(directory, mastodon.STREAM))
    stats["mastodon"] = {"written": result.written, "duplicates": result.duplicates}

    trend_msgs = _messages(directory, trends.STREAM)
    stats["trends"] = {
        "responses": trends.store_snapshot(conn, store, trend_msgs, run_id) if trend_msgs else 0
    }
    stats["binance"] = binance.store_tail_pages(
        conn, store, _messages(directory, binance.STREAM), run_id
    )
    return stats
