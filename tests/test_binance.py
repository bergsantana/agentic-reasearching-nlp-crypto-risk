from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta

import httpx
import psycopg
import pyarrow.parquet as pq
import pytest

from cryptonlp.collectors import binance
from cryptonlp.collectors.common import Message
from cryptonlp.config import BinanceConfig
from cryptonlp.rawstore import RawStore
from cryptonlp.schemas import CandleSource, Interval, Symbol
from tests.factories import kline, vision_zip

DAY = datetime(2026, 11, 1, tzinfo=UTC)


def config(**overrides: object) -> BinanceConfig:
    fields: dict[str, object] = {
        "symbols": (Symbol.BTCUSDT,),
        "intervals": (Interval.HOUR,),
        "backfill_start": date(2026, 9, 15),
        "vision_base": "https://vision.test",
        "api_base": "https://api.test",
        "restate_hours": 3,
    }
    return BinanceConfig.model_validate(fields | overrides)


def test_parse_time_reads_ms_and_us() -> None:
    assert binance.parse_time(1730419200000) == datetime(2024, 11, 1, tzinfo=UTC)
    assert binance.parse_time(1793491200000000) == datetime(2026, 11, 1, tzinfo=UTC)
    assert binance.parse_time("1793494799999999") == datetime(
        2026, 11, 1, 0, 59, 59, 999999, tzinfo=UTC
    )


def test_archive_plan() -> None:
    plan = binance.archive_plan(date(2026, 9, 15), date(2026, 11, 3), daily_months=set())
    assert plan == [
        ("monthly", "2026-09"),
        ("monthly", "2026-10"),
        ("daily", "2026-11-01"),
        ("daily", "2026-11-02"),  # today's file isn't published yet
    ]
    # A month started with daily files keeps them, so no candle is imported twice.
    plan = binance.archive_plan(
        date(2026, 10, 1), date(2026, 11, 2), daily_months={date(2026, 10, 1)}
    )
    assert plan[0] == ("daily", "2026-10-01") and len(plan) == 31 + 1


class FakeVision:
    def __init__(self, files: dict[str, bytes], bad_checksum: bool = False) -> None:
        self.files = files
        self.bad_checksum = bad_checksum

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith(".CHECKSUM"):
            data = self.files.get(path.removesuffix(".CHECKSUM"))
            if data is None:
                return httpx.Response(404)
            digest = "0" * 64 if self.bad_checksum else hashlib.sha256(data).hexdigest()
            return httpx.Response(200, text=f"{digest}  {path.rsplit('/', 1)[-1]}")
        data = self.files.get(path)
        return httpx.Response(200, content=data) if data else httpx.Response(404)


def vision_files() -> dict[str, bytes]:
    files = {}
    for label, start, n in (
        ("2026-09", datetime(2026, 9, 30, 22, tzinfo=UTC), 2),
        ("2026-10", datetime(2026, 10, 31, 22, tzinfo=UTC), 2),
        ("2026-11-01", DAY, 3),
    ):
        kind = "daily" if len(label) == 10 else "monthly"
        name = f"BTCUSDT-1h-{label}.zip"
        rows = [kline(start + timedelta(hours=i), minutes=60) for i in range(n)]
        files[f"/data/spot/{kind}/klines/BTCUSDT/1h/{name}"] = vision_zip(
            name, rows, microseconds=label >= "2026-10"
        )
    return files


@pytest.mark.db
def test_backfill_keeps_zips_and_writes_parquet_by_day(
    db: psycopg.Connection, store: RawStore, run_id: str
) -> None:
    with httpx.Client(transport=httpx.MockTransport(FakeVision(vision_files()))) as client:
        stats = binance.backfill(db, store, client, config(), run_id, today=date(2026, 11, 3))
        again = binance.backfill(db, store, client, config(), run_id, today=date(2026, 11, 3))
    assert stats["archives"] == 3 and stats["candles"] == 7 and stats["invalid"] == 0
    assert stats["not_published"] == 1  # 2026-11-02 isn't on the fake server
    assert again["archives"] == 0 and again["skipped"] == 3

    zips = db.execute(
        "SELECT path FROM raw.files WHERE dataset = 'binance_vision' ORDER BY 1"
    ).fetchall()
    assert sorted(z[0].rsplit("/", 1)[-1] for z in zips) == [
        "BTCUSDT-1h-2026-09.zip",
        "BTCUSDT-1h-2026-10.zip",
        "BTCUSDT-1h-2026-11-01.zip",
    ]
    days = db.execute(
        "SELECT partition_date, rows FROM raw.files WHERE dataset = 'binance_klines' ORDER BY 1"
    ).fetchall()
    assert days == [(date(2026, 9, 30), 2), (date(2026, 10, 31), 2), (date(2026, 11, 1), 3)]
    path = db.execute(
        "SELECT path FROM raw.files WHERE dataset = 'binance_klines' AND partition_date = %s",
        (date(2026, 11, 1),),
    ).fetchone()
    assert path is not None
    assert path[0].startswith("binance_klines/symbol=BTCUSDT/interval=1h/date=2026-11-01/")
    table = pq.read_table(store.root / path[0])
    assert table.column("open_time").to_pylist()[0] == DAY  # microsecond file, read right
    state = db.execute("SELECT cursor, last_event_at FROM ops.collector_state").fetchone()
    assert state == ((DAY + timedelta(hours=2)).isoformat(), DAY + timedelta(hours=3))
    assert store.verify(db).ok


@pytest.mark.db
def test_backfill_rejects_bad_checksum(
    db: psycopg.Connection, store: RawStore, run_id: str
) -> None:
    transport = httpx.MockTransport(FakeVision(vision_files(), bad_checksum=True))
    with httpx.Client(transport=transport) as client, pytest.raises(ValueError, match="checksum"):
        binance.backfill(db, store, client, config(), run_id, today=date(2026, 11, 3))
    assert db.execute("SELECT count(*) FROM raw.files").fetchone() == (0,)


@pytest.mark.db
def test_tail_restates_recent_hours_and_drops_open_candle(
    db: psycopg.Connection, store: RawStore, run_id: str
) -> None:
    cfg = config()
    now = DAY + timedelta(hours=10, minutes=30)
    db.execute(
        "INSERT INTO ops.collector_state (collector, source, cursor, run_id) "
        "VALUES ('binance:BTCUSDT:1h', 'binance', %s, %s)",
        ((DAY + timedelta(hours=7)).isoformat(), run_id),
    )
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        start = binance.parse_time(int(request.url.params["startTime"]))
        hours = int((now - start).total_seconds() // 3600) + 1
        return httpx.Response(
            200, json=[kline(start + timedelta(hours=i), 60) for i in range(hours)]
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        pages = binance.fetch_tail(
            db, client, cfg, Symbol.BTCUSDT, Interval.HOUR, clock=lambda: now
        )
        stats = binance.store_tail_pages(db, store, pages, run_id)
    # Re-pulled from 3 hours before the cursor (04:00) up to the last closed bar (09:00).
    assert binance.parse_time(int(requests[0].url.params["startTime"])) == DAY + timedelta(hours=4)
    assert stats == {"candles": 6, "open_dropped": 1, "invalid": 0}
    cursor = db.execute("SELECT cursor FROM ops.collector_state").fetchone()
    assert cursor == ((DAY + timedelta(hours=9)).isoformat(),)


@pytest.mark.db
def test_tail_pages_until_short_page(db: psycopg.Connection, run_id: str) -> None:
    cfg = config(intervals=(Interval.MINUTE,))
    now = DAY + timedelta(hours=1)
    db.execute(
        "INSERT INTO ops.collector_state (collector, source, cursor, run_id) "
        "VALUES ('binance:BTCUSDT:1m', 'binance', %s, %s)",
        ((DAY - timedelta(hours=20)).isoformat(), run_id),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        start = binance.parse_time(int(request.url.params["startTime"]))
        n = min(binance.API_LIMIT, int((now - start).total_seconds() // 60))
        return httpx.Response(200, json=[kline(start + timedelta(minutes=i)) for i in range(n)])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        pages = binance.fetch_tail(
            db, client, cfg, Symbol.BTCUSDT, Interval.MINUTE, clock=lambda: now
        )
    # 3 hours of restatement before the cursor, then up to now: 24 hours of 1-minute candles.
    assert [len(p.payload) for p in pages] == [1000, 440]
    assert all(isinstance(p, Message) for p in pages)


@pytest.mark.db
def test_tail_leaves_old_gaps_to_the_backfill(db: psycopg.Connection, run_id: str) -> None:
    now = DAY + timedelta(hours=1)
    db.execute(
        "INSERT INTO ops.collector_state (collector, source, cursor, run_id) "
        "VALUES ('binance:BTCUSDT:1h', 'binance', %s, %s)",
        ((DAY - timedelta(days=200)).isoformat(), run_id),
    )
    starts = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(binance.parse_time(int(request.url.params["startTime"])))
        return httpx.Response(200, json=[])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        binance.fetch_tail(db, client, config(), Symbol.BTCUSDT, Interval.HOUR, clock=lambda: now)
    assert starts == [now - timedelta(days=7)]


def test_invalid_candles_are_counted_not_dropped(run_id: str) -> None:
    raw = kline(DAY, 60)
    raw[2] = "1.0"  # high below open and close
    row = binance.kline_row(
        raw, Symbol.BTCUSDT, Interval.HOUR, CandleSource.API, "api", DAY, run_id
    )
    assert binance.count_invalid([row]) == 1
