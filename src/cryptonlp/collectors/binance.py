"""Binance candles: backfill from the Vision archive, recent tail from the market-data API.

- **Backfill.** Monthly zips for past months and daily zips for the current month (a daily
  file appears the day after). Each zip is checked against its ``.CHECKSUM``, kept as is in
  the raw store, and converted to Parquet by day. A month already started with daily files
  keeps using daily files, so no candle is imported twice.
- **Tail.** ``/api/v3/klines`` from the last stored candle, re-pulling ``restate_hours`` before
  it because Binance restates candles, and never more than ``tail_max_days`` back (older gaps
  are left to the backfill). Only closed candles are stored. Re-pulled candles land
  in new files; readers keep the newest version of each ``(symbol, interval, open_time)``.
- Vision files switched from millisecond to microsecond times in 2025; both are read.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import psycopg
import pyarrow as pa
from pydantic import ValidationError

from cryptonlp import ops
from cryptonlp.collectors.common import Message, Recorder, request_with_backoff, utcnow
from cryptonlp.config import BinanceConfig
from cryptonlp.rawstore import RawStore
from cryptonlp.schemas import Candle, CandleSource, Interval, Symbol

log = logging.getLogger(__name__)

STREAM = "binance_klines"
KLINES_DATASET = "binance_klines"
VISION_DATASET = "binance_vision"
API_LIMIT = 1000
INTERVAL_LENGTH = {Interval.MINUTE: timedelta(minutes=1), Interval.HOUR: timedelta(hours=1)}
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_UTC_US = pa.timestamp("us", tz="UTC")
KLINE_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("interval", pa.string(), nullable=False),
        pa.field("open_time", _UTC_US, nullable=False),
        pa.field("close_time", _UTC_US, nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("quote_volume", pa.float64(), nullable=False),
        pa.field("trades", pa.int64(), nullable=False),
        pa.field("taker_buy_base_volume", pa.float64(), nullable=False),
        pa.field("taker_buy_quote_volume", pa.float64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),  # "vision" or "api"
        pa.field("source_ref", pa.string(), nullable=False),  # zip name, or "api"
        pa.field("fetched_at", _UTC_US, nullable=False),
        pa.field("collector_run_id", pa.string(), nullable=False),
    ]
)


def collector_key(symbol: Symbol, interval: Interval) -> str:
    return f"binance:{symbol.value}:{interval.value}"


def parse_time(value: int | str) -> datetime:
    """Binance times: milliseconds, or microseconds in Vision files since 2025."""
    n = int(value)
    return EPOCH + (timedelta(microseconds=n) if n >= 10**15 else timedelta(milliseconds=n))


def kline_row(
    raw: Sequence[Any],
    symbol: Symbol,
    interval: Interval,
    source: CandleSource,
    source_ref: str,
    fetched_at: datetime,
    run_id: str,
) -> dict[str, Any]:
    return {
        "symbol": symbol.value,
        "interval": interval.value,
        "open_time": parse_time(raw[0]),
        "open": float(raw[1]),
        "high": float(raw[2]),
        "low": float(raw[3]),
        "close": float(raw[4]),
        "volume": float(raw[5]),
        "close_time": parse_time(raw[6]),
        "quote_volume": float(raw[7]),
        "trades": int(raw[8]),
        "taker_buy_base_volume": float(raw[9]),
        "taker_buy_quote_volume": float(raw[10]),
        "source": source.value,
        "source_ref": source_ref,
        "fetched_at": fetched_at,
        "collector_run_id": run_id,
    }


def count_invalid(rows: Iterable[dict[str, Any]]) -> int:
    """Rows the Candle schema rejects. They are still stored: the raw store keeps what came."""
    invalid = 0
    fields = set(Candle.model_fields)
    for row in rows:
        try:
            Candle(**{k: v for k, v in row.items() if k in fields})
        except ValidationError as exc:
            invalid += 1
            log.warning("invalid candle %s %s: %s", row["symbol"], row["open_time"], exc)
    return invalid


def read_vision_zip(data: bytes) -> list[list[str]]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in the archive, found {names}")
        text = zf.read(names[0]).decode()
    rows = list(csv.reader(io.StringIO(text)))
    if rows and not rows[0][0].isdigit():  # some Vision files carry a header row
        rows = rows[1:]
    return [r for r in rows if r]


def write_klines(
    conn: psycopg.Connection,
    store: RawStore,
    rows: list[dict[str, Any]],
    run_id: str,
    meta: dict[str, str],
) -> int:
    """Write candles as Parquet, one file per symbol, interval and day of ``open_time``."""
    groups: dict[tuple[str, str, date], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["symbol"], row["interval"], row["open_time"].date())].append(row)
    for (symbol, interval, day), group in sorted(groups.items()):
        group.sort(key=lambda r: r["open_time"])
        table = pa.Table.from_pylist(group, schema=KLINE_SCHEMA)
        file_meta = {
            "dataset": KLINES_DATASET,
            "partition_date": day.isoformat(),
            "run_id": run_id,
            **meta,
        }
        directory = store.partition_dir(
            KLINES_DATASET, day, {"symbol": symbol, "interval": interval}
        )
        written = store.write_table(directory, table, file_meta)
        store.register(
            conn,
            written,
            dataset=KLINES_DATASET,
            day=day,
            run_id=run_id,
            received=[r["fetched_at"] for r in group],
        )
    return len(rows)


def _advance_cursor(
    conn: psycopg.Connection,
    symbol: Symbol,
    interval: Interval,
    rows: list[dict[str, Any]],
    run_id: str,
) -> None:
    if not rows:
        return
    key = collector_key(symbol, interval)
    newest = max(r["open_time"] for r in rows)
    state = ops.get_state(conn, key)
    if state and state.cursor and datetime.fromisoformat(state.cursor) >= newest:
        return
    ops.save_state(
        conn,
        key,
        "binance",
        run_id,
        cursor=newest.isoformat(),
        last_event_at=newest + INTERVAL_LENGTH[interval],
    )


# --- Vision backfill ------------------------------------------------------------------------


def _months(start: date, end: date) -> list[date]:
    """First days of the months from ``start``'s month to ``end``'s month, inclusive."""
    months, d = [], start.replace(day=1)
    while d <= end:
        months.append(d)
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def archive_plan(start: date, today: date, daily_months: set[date]) -> list[tuple[str, str]]:
    """(``monthly`` or ``daily``, label) for every archive from ``start`` up to yesterday."""
    plan: list[tuple[str, str]] = []
    this_month = today.replace(day=1)
    for month in _months(start, today):
        if month < this_month and month not in daily_months:
            plan.append(("monthly", month.strftime("%Y-%m")))
            continue
        day = max(month, start)
        while day.month == month.month and day < today:
            plan.append(("daily", day.isoformat()))
            day += timedelta(days=1)
    return plan


def _archive_day(url: str) -> date:
    """``.../BTCUSDT-1m-2026-11-01.zip`` -> 2026-11-01; a monthly archive -> its first day."""
    label = url.rsplit("/", 1)[-1].removesuffix(".zip").split("-", 2)[2]
    return date.fromisoformat(label if len(label) == 10 else f"{label}-01")


def vision_path(kind: str, symbol: Symbol, interval: Interval, label: str) -> str:
    name = f"{symbol.value}-{interval.value}-{label}.zip"
    return f"data/spot/{kind}/klines/{symbol.value}/{interval.value}/{name}"


def backfill(
    conn: psycopg.Connection,
    store: RawStore,
    client: httpx.Client,
    config: BinanceConfig,
    run_id: str,
    *,
    today: date | None = None,
    max_archives: int | None = None,
) -> dict[str, int]:
    today = today or utcnow().date()
    stats = {"archives": 0, "candles": 0, "invalid": 0, "not_published": 0, "skipped": 0}
    known = {
        r[0]
        for r in conn.execute(
            "SELECT source_url FROM raw.files WHERE dataset = %s AND source_url IS NOT NULL",
            (VISION_DATASET,),
        )
    }
    for symbol in config.symbols:
        for interval in config.intervals:
            prefix = f"{config.vision_base}/data/spot/daily/klines/{symbol.value}/{interval.value}/"
            daily_months = {_archive_day(u).replace(day=1) for u in known if u.startswith(prefix)}
            for kind, label in archive_plan(config.backfill_start, today, daily_months):
                if max_archives is not None and stats["archives"] >= max_archives:
                    return stats
                url = f"{config.vision_base}/{vision_path(kind, symbol, interval, label)}"
                if url in known:
                    stats["skipped"] += 1
                    continue
                if not _import_archive(
                    conn, store, client, url, kind, label, symbol, interval, run_id, stats
                ):
                    stats["not_published"] += 1
                    continue
                known.add(url)
    return stats


def _import_archive(
    conn: psycopg.Connection,
    store: RawStore,
    client: httpx.Client,
    url: str,
    kind: str,
    label: str,
    symbol: Symbol,
    interval: Interval,
    run_id: str,
    stats: dict[str, int],
) -> bool:
    response = request_with_backoff(client, url, ok_statuses=frozenset({200, 404}))
    if response.status_code == 404:
        return False
    checksum = request_with_backoff(client, url + ".CHECKSUM").text.split()[0]
    fetched_at = utcnow()
    name = url.rsplit("/", 1)[-1]
    rows = [
        kline_row(r, symbol, interval, CandleSource.VISION, name, fetched_at, run_id)
        for r in read_vision_zip(response.content)
    ]
    stats["invalid"] += count_invalid(rows)
    with conn.transaction():
        blob = store.write_blob(
            f"{VISION_DATASET}/{url.split('/data/', 1)[1]}", response.content, checksum
        )
        store.register(
            conn,
            blob,
            dataset=VISION_DATASET,
            day=_archive_day(url),
            run_id=run_id,
            source_url=url,
        )
        write_klines(conn, store, rows, run_id, {"source_url": url, "source_sha256": checksum})
        _advance_cursor(conn, symbol, interval, rows, run_id)
    stats["archives"] += 1
    stats["candles"] += len(rows)
    log.info("imported %s (%s candles)", name, len(rows))
    return True


# --- API tail -------------------------------------------------------------------------------


def fetch_tail(
    conn: psycopg.Connection,
    client: httpx.Client,
    config: BinanceConfig,
    symbol: Symbol,
    interval: Interval,
    clock: Callable[[], datetime] = utcnow,
) -> list[Message]:
    state = ops.get_state(conn, collector_key(symbol, interval))
    now = clock()
    last = (
        datetime.fromisoformat(state.cursor) if state and state.cursor else now - timedelta(days=2)
    )
    # Older gaps are the Vision backfill's job: it fills them from checksummed files.
    start = max(
        last - timedelta(hours=config.restate_hours), now - timedelta(days=config.tail_max_days)
    )
    pages = []
    while start < now:
        params = {
            "symbol": symbol.value,
            "interval": interval.value,
            "startTime": str((start - EPOCH) // timedelta(milliseconds=1)),
            "limit": str(API_LIMIT),
        }
        response = request_with_backoff(client, f"{config.api_base}/api/v3/klines", params=params)
        page = Message(
            STREAM,
            clock(),
            response.json(),
            {"symbol": symbol.value, "interval": interval.value, "startTime": params["startTime"]},
        )
        pages.append(page)
        if len(page.payload) < API_LIMIT:
            break
        start = parse_time(page.payload[-1][0]) + INTERVAL_LENGTH[interval]
    return pages


def store_tail_pages(
    conn: psycopg.Connection, store: RawStore, pages: Iterable[Message], run_id: str
) -> dict[str, int]:
    """Store the closed candles from API pages (live or replayed)."""
    stats = {"candles": 0, "open_dropped": 0, "invalid": 0}
    by_series: dict[tuple[Symbol, Interval], list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        symbol, interval = Symbol(page.meta["symbol"]), Interval(page.meta["interval"])
        for raw in page.payload:
            row = kline_row(
                raw, symbol, interval, CandleSource.API, "api", page.received_at, run_id
            )
            # A candle is closed once its close time has passed when the page arrived.
            if row["close_time"] >= page.received_at:
                stats["open_dropped"] += 1
                continue
            by_series[(symbol, interval)].append(row)
    with conn.transaction():
        for (symbol, interval), rows in sorted(by_series.items()):
            stats["invalid"] += count_invalid(rows)
            stats["candles"] += write_klines(conn, store, rows, run_id, {"source_url": "api"})
            _advance_cursor(conn, symbol, interval, rows, run_id)
    return stats


def tail(
    conn: psycopg.Connection,
    store: RawStore,
    client: httpx.Client,
    config: BinanceConfig,
    run_id: str,
    recorder: Recorder | None = None,
) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for symbol in config.symbols:
        for interval in config.intervals:
            pages = fetch_tail(conn, client, config, symbol, interval)
            if recorder:
                for p in pages:
                    recorder.write(p)
                recorder.flush()
            for k, v in store_tail_pages(conn, store, pages, run_id).items():
                totals[k] += v
    return dict(totals)
