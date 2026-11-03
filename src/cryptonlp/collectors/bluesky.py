"""Bluesky: a Jetstream WebSocket client filtered to ``app.bsky.feed.post``.

- **Reception time** is Jetstream's own ``time_us`` (the time it witnessed the event), never
  our clock and never the client-set ``createdAt``.
- **Cursor.** Saved at each flush as the ``time_us`` of the last event whose posts are on disk.
  Jetstream accepts a microsecond time on ``?cursor=`` and it works on every instance (its
  ``cursor`` sequence numbers are per instance), so we resume by time, a few seconds early,
  and drop the overlap by URI.
- **Lookback.** Jetstream keeps 36 hours and clamps an older cursor without an error, so a
  jump between consecutive events is logged in ``ops.collector_gaps``.
- **Deletions** in the stream are applied to posts already in the index (and to posts still
  in the buffer, which are simply dropped).
- Jetstream can't filter by text, so the full post stream (about 50 posts a second) is read
  and filtered here by the watch terms.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import websockets

from cryptonlp import ops
from cryptonlp.collectors.common import Message, Recorder, WatchFilter, read_recording
from cryptonlp.config import BlueskyConfig
from cryptonlp.rawstore import CollectedPost, RawStore
from cryptonlp.schemas import Origin, RawPost, Source

log = logging.getLogger(__name__)

COLLECTOR = "bluesky"
STREAM = "jetstream"
POST_COLLECTION = "app.bsky.feed.post"
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
# Consecutive events further apart than this mean the stream skipped data.
GAP_THRESHOLD = timedelta(minutes=2)


def from_us(time_us: int) -> datetime:
    return EPOCH + timedelta(microseconds=time_us)


def to_us(value: datetime) -> int:
    return (value - EPOCH) // timedelta(microseconds=1)


def post_uri(did: str, rkey: str) -> str:
    return f"at://{did}/{POST_COLLECTION}/{rkey}"


def _created_at(record: dict[str, Any], fallback: datetime) -> datetime:
    """``createdAt`` is set by the client: it can be malformed, naive or backdated."""
    value = record.get("createdAt")
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return fallback


def parse_post(event: dict[str, Any], run_id: str, origin: Origin) -> RawPost:
    commit = event["commit"]
    record = commit.get("record") or {}
    received_at = from_us(event["time_us"])
    langs = record.get("langs")
    return RawPost(
        source=Source.BLUESKY,
        uri=post_uri(event["did"], commit["rkey"]),
        author_id=event["did"],
        text=str(record.get("text", "")),
        created_at=_created_at(record, received_at),
        received_at=received_at,
        langs=tuple(str(x) for x in langs) if isinstance(langs, list) else (),
        origin=origin,
        collector_run_id=run_id,
    )


class BlueskyCollector:
    def __init__(
        self,
        conn: psycopg.Connection,
        store: RawStore,
        config: BlueskyConfig,
        watch: WatchFilter,
        run_id: str,
        *,
        origin: Origin = Origin.LIVE,
        recorder: Recorder | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if not conn.autocommit:
            raise ValueError("collectors need an autocommit connection")
        self.conn = conn
        self.store = store
        self.config = config
        self.watch = watch
        self.run_id = run_id
        self.origin = origin
        self.recorder = recorder
        self.rng = rng or random.Random()
        self.buffer: dict[str, CollectedPost] = {}
        self.pending_deletes: dict[str, Message] = {}
        self.last_time_us: int | None = None
        self._last_flush: datetime | None = None
        self._last_heartbeat: datetime | None = None
        self.stats = {"events": 0, "matched": 0, "written": 0, "deleted": 0, "gaps": 0}

    def saved_cursor(self) -> int | None:
        state = ops.get_state(self.conn, COLLECTOR)
        return int(state.cursor) if state and state.cursor else None

    async def run(self, messages: AsyncIterator[Message]) -> None:
        try:
            async for msg in messages:
                self.handle(msg)
        finally:
            self.flush()
            if self.recorder:
                self.recorder.close()

    def handle(self, msg: Message) -> None:
        event = msg.payload
        time_us = event.get("time_us")
        if not isinstance(time_us, int):
            return
        self.stats["events"] += 1
        now = from_us(time_us)
        if self.last_time_us is not None and now - from_us(self.last_time_us) > GAP_THRESHOLD:
            ops.record_gap(self.conn, COLLECTOR, from_us(self.last_time_us), now, self.run_id)
            self.stats["gaps"] += 1
        if self.last_time_us is None or time_us > self.last_time_us:
            self.last_time_us = time_us

        commit = event.get("commit")
        if event.get("kind") == "commit" and commit and commit.get("collection") == POST_COLLECTION:
            uri = post_uri(event["did"], commit["rkey"])
            if commit.get("operation") == "create":
                self._on_create(msg, event, uri)
            elif commit.get("operation") == "delete":
                if self.buffer.pop(uri, None) is not None:
                    self._record(msg)
                else:
                    self.pending_deletes[uri] = msg

        if self._last_flush is None:
            self._last_flush = self._last_heartbeat = now
        assert self._last_heartbeat is not None
        flush_due = (now - self._last_flush).total_seconds() >= self.config.flush_seconds
        if flush_due or len(self.buffer) >= self.config.max_rows:
            self.flush()
            self._last_flush = self._last_heartbeat = now
        elif (now - self._last_heartbeat).total_seconds() >= self.config.heartbeat_seconds:
            self.heartbeat()
            self._last_heartbeat = now

    def _on_create(self, msg: Message, event: dict[str, Any], uri: str) -> None:
        record = event["commit"].get("record") or {}
        matched = self.watch.matches(str(record.get("text", "")))
        if not matched:
            if self.recorder and self.rng.random() < self.config.replay_sample_rate:
                self._record(msg)  # lets a replay check that the filter drops it
            return
        self.stats["matched"] += 1
        self.buffer[uri] = CollectedPost(
            post=parse_post(event, self.run_id, self.origin), payload=event, matched=matched
        )
        self._record(msg)

    def _record(self, msg: Message) -> None:
        if self.recorder:
            self.recorder.write(msg)

    def _apply_deletes(self) -> None:
        if not self.pending_deletes:
            return
        rows = self.conn.execute(
            """
            INSERT INTO raw.deletions (source, uri, method, run_id)
            SELECT p.source, p.uri, 'stream', %s
            FROM raw.posts p
            WHERE p.source = 'bluesky' AND p.uri = ANY(%s)
            ON CONFLICT (source, uri) DO NOTHING
            RETURNING uri
            """,
            (self.run_id, list(self.pending_deletes)),
        ).fetchall()
        for (uri,) in rows:
            self._record(self.pending_deletes[uri])
        self.stats["deleted"] += len(rows)
        self.pending_deletes.clear()

    def heartbeat(self) -> None:
        """Report lag and apply deletions, without moving the cursor."""
        with self.conn.transaction():
            self._apply_deletes()
            ops.save_state(
                self.conn,
                COLLECTOR,
                "bluesky",
                self.run_id,
                last_event_at=from_us(self.last_time_us) if self.last_time_us else None,
            )
        if self.recorder:
            self.recorder.flush()

    def flush(self) -> None:
        """Write the buffer to the raw store and move the cursor past it, atomically."""
        if self.last_time_us is None:
            return
        with self.conn.transaction():
            result = self.store.store_posts(
                self.conn,
                self.buffer.values(),
                self.run_id,
                {"collector": COLLECTOR, "watch_sha256": self.watch.sha256},
            )
            self._apply_deletes()
            ops.save_state(
                self.conn,
                COLLECTOR,
                "bluesky",
                self.run_id,
                cursor=str(self.last_time_us),
                last_event_at=from_us(self.last_time_us),
            )
        self.stats["written"] += result.written
        self.buffer.clear()
        if self.recorder:
            self.recorder.flush()
        log.info("bluesky flush: %s new posts, totals %s", result.written, self.stats)


# --- sources -------------------------------------------------------------------------------


async def jetstream_messages(
    hosts: Iterable[str],
    cursor_us: int | None,
    rewind_seconds: float,
    *,
    idle_timeout: float = 120.0,
) -> AsyncIterator[Message]:
    """Live events, reconnecting forever; moves to the next host after 3 failures in a row."""
    hosts = list(hosts)
    host_index = failures = 0
    last_us = cursor_us
    while True:
        host = hosts[host_index % len(hosts)]
        url = f"wss://{host}/subscribe?wantedCollections={POST_COLLECTION}"
        if last_us is not None:
            url += f"&cursor={last_us - int(rewind_seconds * 1_000_000)}"
        try:
            async with websockets.connect(
                url, max_size=2**23, open_timeout=30, ping_interval=20, ping_timeout=30
            ) as ws:
                log.info("connected to %s", url)
                failures = 0
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=idle_timeout)
                    event = json.loads(raw)
                    time_us = event.get("time_us")
                    if isinstance(time_us, int):
                        last_us = time_us
                        yield Message(STREAM, from_us(time_us), event, {"host": host})
        except (OSError, TimeoutError, websockets.WebSocketException, json.JSONDecodeError) as exc:
            failures += 1
            if failures >= 3:
                host_index += 1
                failures = 0
            delay = min(60.0, 2.0**failures) + random.random()
            log.warning("jetstream %s: %r; reconnecting in %.1fs", host, exc, delay)
            await asyncio.sleep(delay)


async def replay_messages(paths: Iterable[Path]) -> AsyncIterator[Message]:
    for path in paths:
        for msg in read_recording(path):
            yield msg
