"""The replay day end to end: record, freeze, replay. Also trends, which only replay checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import psycopg
import pytest

from cryptonlp.collectors import replay, trends
from cryptonlp.collectors.common import Message, Recorder
from cryptonlp.config import CollectorsConfig
from cryptonlp.rawstore import RawStore
from tests.factories import jetstream_create, kline, mastodon_status

pytestmark = pytest.mark.db

DAY = datetime(2026, 11, 5, tzinfo=UTC)


def record_day(directory: Path, *, end: datetime) -> None:
    rec = Recorder(directory)
    t = DAY + timedelta(minutes=1)
    while t < end:
        rec.write(jetstream_create("bitcoin", t, rkey=f"k{int(t.timestamp())}"))
        rec.write(
            Message(
                "mastodon",
                t,
                [mastodon_status(int(t.timestamp()), host="a.social")],
                {"instance": "a.social", "timeline": "tag:bitcoin", "min_id": ""},
            )
        )
        rec.write(
            Message(
                "trends",
                t,
                {"topics": [{"topic": "bitcoin"}]},
                {"source": "bluesky", "instance": "appview", "endpoint": "getTrendingTopics"},
            )
        )
        bar = t.replace(minute=0) - timedelta(hours=1)
        rec.write(
            Message(
                "binance_klines",
                t,
                [kline(bar, 60)],
                {"symbol": "BTCUSDT", "interval": "1h", "startTime": "0"},
            )
        )
        t += timedelta(hours=1)
    # One message from the next day, which freeze must leave out.
    rec.write(jetstream_create("bitcoin", DAY + timedelta(days=1, minutes=5), rkey="next"))
    rec.close()


def test_freeze_writes_one_day_with_manifest(tmp_path: Path) -> None:
    record_day(tmp_path / "rec", end=DAY + timedelta(days=1))
    dest = tmp_path / "fixture"
    manifest = replay.freeze(tmp_path / "rec", DAY.date(), dest)
    assert manifest["streams"]["jetstream"]["messages"] == 24  # the next-day message is out
    assert json.loads((dest / "manifest.json").read_text()) == manifest
    # Deterministic: freezing again gives byte-identical files.
    again = replay.freeze(tmp_path / "rec", DAY.date(), tmp_path / "fixture2")
    assert again == manifest


def test_freeze_refuses_an_incomplete_day(tmp_path: Path) -> None:
    record_day(tmp_path / "rec", end=DAY + timedelta(hours=12))
    with pytest.raises(ValueError, match="not the whole day"):
        replay.freeze(tmp_path / "rec", DAY.date(), tmp_path / "fixture")
    replay.freeze(tmp_path / "rec", DAY.date(), tmp_path / "fixture", require_full_day=False)


async def test_replay_feeds_every_collector(
    db: psycopg.Connection,
    store: RawStore,
    config: CollectorsConfig,
    run_id: str,
    tmp_path: Path,
) -> None:
    record_day(tmp_path / "rec", end=DAY + timedelta(days=1))
    replay.freeze(tmp_path / "rec", DAY.date(), tmp_path / "fixture")
    stats = await replay.replay(db, store, config, tmp_path / "fixture", run_id)
    assert stats["bluesky"]["written"] == 24
    assert stats["mastodon"]["written"] == 24
    assert stats["trends"]["responses"] == 24
    assert stats["binance"]["candles"] == 24
    origins = db.execute("SELECT DISTINCT source, origin FROM raw.posts ORDER BY 1").fetchall()
    assert origins == [("bluesky", "replay"), ("mastodon", "replay")]
    # Replayed posts never show up as live collection on the dashboard.
    assert db.execute("SELECT count(*) FROM monitoring.posts_per_hour").fetchone() == (0,)


def test_replay_refuses_the_main_database(
    store: RawStore, config: CollectorsConfig, run_id: str, tmp_path: Path
) -> None:
    class FakeInfo:
        dsn = "host=localhost dbname=cryptonlp"

    class FakeConn:
        info = FakeInfo()

    import asyncio

    from cryptonlp.db import DatabaseGuardError

    with pytest.raises(DatabaseGuardError):
        asyncio.run(replay.replay(FakeConn(), store, config, tmp_path, run_id))  # type: ignore[arg-type]


def test_trend_snapshot(
    db: psycopg.Connection, store: RawStore, config: CollectorsConfig, run_id: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "mastodon.online":
            return httpx.Response(503)
        if "getTrends" in request.url.path:
            return httpx.Response(200, json={"trends": [{"topic": "x"}]})
        return httpx.Response(200, json=[{"name": "bitcoin"}])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        stats = trends.snapshot(db, store, client, config, run_id)
    assert stats == {"responses": 2 + len(config.mastodon.instances) - 1, "failed": 1}
    datasets = db.execute("SELECT dataset, rows FROM raw.files ORDER BY 1").fetchall()
    assert datasets == [
        ("bluesky_trends", 2),
        ("mastodon_trends", len(config.mastodon.instances) - 1),
    ]
    collectors = [r[0] for r in db.execute("SELECT collector FROM ops.collector_state ORDER BY 1")]
    assert collectors == ["trends:bluesky", "trends:mastodon"]
