from __future__ import annotations

import random
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, timedelta
from pathlib import Path

import psycopg
import pyarrow.parquet as pq
import pytest

from cryptonlp.collectors.bluesky import BlueskyCollector, from_us, parse_post, to_us
from cryptonlp.collectors.common import Message, Recorder, WatchFilter, read_recording
from cryptonlp.config import CollectorsConfig
from cryptonlp.rawstore import RawStore
from cryptonlp.schemas import Origin
from tests.factories import T0, jetstream_create, jetstream_delete, jetstream_identity

pytestmark = pytest.mark.db


def collector(
    db: psycopg.Connection,
    store: RawStore,
    config: CollectorsConfig,
    watch: WatchFilter,
    run_id: str,
    **kwargs: object,
) -> BlueskyCollector:
    return BlueskyCollector(db, store, config.bluesky, watch, run_id, **kwargs)  # type: ignore[arg-type]


async def feed(msgs: Iterable[Message]) -> AsyncIterator[Message]:
    for m in msgs:
        yield m


def test_time_conversion_is_exact() -> None:
    us = 1793719925161578
    assert to_us(from_us(us)) == us
    assert from_us(us).tzinfo is UTC


def test_parse_post_uses_jetstream_time_and_tolerates_bad_created_at(run_id: str) -> None:
    msg = jetstream_create("bitcoin", T0)
    post = parse_post(msg.payload, run_id, Origin.LIVE)
    assert post.uri == "at://did:plc:alice/app.bsky.feed.post/3k1"
    assert post.received_at == T0 and post.created_at == T0 - timedelta(seconds=2)
    assert post.langs == ("en",)
    msg.payload["commit"]["record"]["createdAt"] = "not a date"
    assert parse_post(msg.payload, run_id, Origin.LIVE).created_at == T0
    msg.payload["commit"]["record"]["createdAt"] = "2026-11-03T13:00:00"  # naive: read as UTC
    assert parse_post(msg.payload, run_id, Origin.LIVE).created_at.tzinfo is not None


async def test_filters_buffers_flushes_and_saves_cursor(
    db: psycopg.Connection,
    store: RawStore,
    config: CollectorsConfig,
    watch: WatchFilter,
    run_id: str,
) -> None:
    c = collector(db, store, config, watch, run_id)
    last = T0 + timedelta(seconds=3)
    await c.run(
        feed(
            [
                jetstream_create("$BTC breaking out", T0, rkey="a"),
                jetstream_create("lunch was nice", T0 + timedelta(seconds=1), rkey="b"),
                jetstream_identity(T0 + timedelta(seconds=2)),
                jetstream_create("ETH gas", last, rkey="c"),
            ]
        )
    )
    uris = [r[0] for r in db.execute("SELECT uri FROM raw.posts ORDER BY uri")]
    assert uris == [
        "at://did:plc:alice/app.bsky.feed.post/a",
        "at://did:plc:alice/app.bsky.feed.post/c",
    ]
    assert c.saved_cursor() == to_us(last)
    row = db.execute("SELECT path FROM raw.files").fetchone()
    assert row is not None
    path = store.root / row[0]
    table = pq.read_table(path)
    assert table.column("matched").to_pylist() == [["btc", "$btc"], ["eth"]]
    assert '"time_us"' in table.column("payload")[0].as_py()  # the event exactly as received


async def test_flushes_on_schedule(
    db: psycopg.Connection,
    store: RawStore,
    config: CollectorsConfig,
    watch: WatchFilter,
    run_id: str,
) -> None:
    c = collector(db, store, config, watch, run_id)
    step = timedelta(seconds=config.bluesky.flush_seconds / 2 + 1)
    for i in range(5):
        c.handle(jetstream_create("bitcoin", T0 + i * step, rkey=f"k{i}"))
    # Events are flush_seconds/2 + 1 apart, so events 3 and 5 each come a full flush
    # interval after the previous flush and trigger one.
    assert db.execute("SELECT count(*) FROM raw.files").fetchone() == (2,)
    assert not c.buffer
    c.handle(jetstream_create("bitcoin", T0 + 5 * step, rkey="k5"))
    assert len(c.buffer) == 1  # waits for the next flush


async def test_deletions_in_buffer_and_in_store(
    db: psycopg.Connection,
    store: RawStore,
    config: CollectorsConfig,
    watch: WatchFilter,
    run_id: str,
) -> None:
    c = collector(db, store, config, watch, run_id)
    c.handle(jetstream_create("bitcoin a", T0, rkey="a"))
    c.flush()
    c.handle(jetstream_create("bitcoin b", T0 + timedelta(seconds=1), rkey="b"))
    c.handle(jetstream_delete(T0 + timedelta(seconds=2), rkey="b"))  # still in the buffer
    c.handle(jetstream_delete(T0 + timedelta(seconds=3), rkey="a"))  # already stored
    c.handle(jetstream_delete(T0 + timedelta(seconds=4), rkey="zzz"))  # never collected
    c.flush()
    assert db.execute("SELECT uri, method FROM raw.deletions").fetchall() == [
        ("at://did:plc:alice/app.bsky.feed.post/a", "stream")
    ]
    assert db.execute("SELECT count(*) FROM raw.posts").fetchone() == (1,)
    assert c.stats["deleted"] == 1


async def test_heartbeat_reports_lag_without_moving_cursor(
    db: psycopg.Connection,
    store: RawStore,
    config: CollectorsConfig,
    watch: WatchFilter,
    run_id: str,
) -> None:
    c = collector(db, store, config, watch, run_id)
    c.handle(jetstream_create("bitcoin", T0, rkey="a"))
    c.flush()
    later = T0 + timedelta(seconds=config.bluesky.heartbeat_seconds + 1)
    c.handle(jetstream_identity(later))
    row = db.execute(
        "SELECT cursor, last_event_at FROM ops.collector_state WHERE collector = 'bluesky'"
    ).fetchone()
    assert row == (str(to_us(T0)), later)
    lag = db.execute("SELECT lag_seconds FROM monitoring.collector_lag").fetchone()
    assert lag is not None and lag[0] > 0


async def test_jump_past_lookback_is_logged_as_gap(
    db: psycopg.Connection,
    store: RawStore,
    config: CollectorsConfig,
    watch: WatchFilter,
    run_id: str,
) -> None:
    c = collector(db, store, config, watch, run_id)
    c.last_time_us = to_us(T0)  # the saved cursor
    resumed = T0 + timedelta(hours=40)  # Jetstream clamped to its 36-hour window
    c.handle(jetstream_identity(resumed))
    assert db.execute("SELECT gap_start, gap_end FROM ops.collector_gaps").fetchall() == [
        (T0, resumed)
    ]


async def test_rewound_duplicates_are_dropped(
    db: psycopg.Connection,
    store: RawStore,
    config: CollectorsConfig,
    watch: WatchFilter,
    run_id: str,
) -> None:
    msgs = [jetstream_create("bitcoin", T0 + timedelta(seconds=i), rkey=f"k{i}") for i in range(3)]
    await collector(db, store, config, watch, run_id).run(feed(msgs))
    # A restart resumes a few seconds early and receives the same events again.
    c = collector(db, store, config, watch, run_id)
    await c.run(feed(msgs[1:]))
    assert db.execute("SELECT count(*) FROM raw.posts").fetchone() == (3,)
    assert db.execute("SELECT count(*) FROM raw.files").fetchone() == (1,)


async def test_record_then_replay_gives_the_same_posts(
    db: psycopg.Connection,
    store: RawStore,
    config: CollectorsConfig,
    watch: WatchFilter,
    run_id: str,
    tmp_path: Path,
) -> None:
    recorder = Recorder(tmp_path / "replay")
    c = collector(db, store, config, watch, run_id, recorder=recorder, rng=random.Random(0))
    msgs = [
        jetstream_create("bitcoin", T0, rkey="a"),
        jetstream_create("unrelated", T0 + timedelta(seconds=1), rkey="b"),
        jetstream_create("eth", T0 + timedelta(seconds=2), rkey="c"),
    ]
    c.handle(msgs[0])
    c.flush()
    c.handle(msgs[1])
    c.handle(jetstream_delete(T0 + timedelta(seconds=3), rkey="a"))
    await c.run(feed(msgs[2:]))
    recorded = list(read_recording(tmp_path / "replay" / "2026-11-03" / "jetstream.jsonl.gz"))
    # Matching posts, then the deletion that hit a stored post (recorded when it is applied,
    # at the flush; freeze sorts by time). The unrelated post was not sampled.
    ops = [(m.payload["commit"]["rkey"], m.payload["commit"]["operation"]) for m in recorded]
    assert ops == [("a", "create"), ("c", "create"), ("a", "delete")]

    alive_sql = (
        "SELECT p.uri FROM raw.posts p LEFT JOIN raw.deletions d USING (source, uri) "
        "WHERE d.uri IS NULL ORDER BY 1"
    )
    live = db.execute(alive_sql).fetchall()
    db.execute("TRUNCATE raw.deletions, raw.posts, raw.files, ops.collector_state CASCADE")
    replayed = collector(db, store, config, watch, run_id, origin=Origin.REPLAY)
    await replayed.run(feed(recorded))
    # The same posts survive. (In the replay, post a and its deletion share one buffer, so a
    # is dropped before it is ever written; live, it had been flushed and was then deleted.)
    assert (
        db.execute(alive_sql).fetchall() == live == [("at://did:plc:alice/app.bsky.feed.post/c",)]
    )
    assert db.execute("SELECT DISTINCT origin FROM raw.posts").fetchall() == [("replay",)]


def test_needs_autocommit(
    connect: object, store: RawStore, config: CollectorsConfig, watch: WatchFilter, run_id: str
) -> None:
    from cryptonlp.db import connect_test

    with connect_test() as conn, pytest.raises(ValueError, match="autocommit"):
        BlueskyCollector(conn, store, config.bluesky, watch, run_id)
