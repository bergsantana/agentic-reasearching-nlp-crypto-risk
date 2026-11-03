"""Hourly realized volatility from the 1-minute Binance candles in the raw store.

RV of the bar opening at T is the square root of the sum of the 60 squared one-minute log
returns between T and T + 1 hour (claims M4): close-to-close, starting from the close of the
candle that opens at T - 1 minute (the price at T). A bar with any of those 61 candles missing
has no RV, rather than one computed from fewer returns.

The hourly log return of the same bar is the sum of those 60 returns, i.e. ln(P(T + 1h) / P(T)),
with the same completeness rule. ``load_hourly`` computes both for a whole range at once (arm 1
and its walk-forward harness); ``MinuteCloses`` does one bar at a time (the loop's scores).
Both follow the same definitions and are checked against each other in the tests.

Binance restates candles, so the raw store can hold several versions of one candle; the newest
fetch wins, as everywhere else that reads it (``cryptonlp.collectors.binance``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from cryptonlp.collectors.binance import KLINES_DATASET
from cryptonlp.rawstore import RawStore
from cryptonlp.schemas.common import BAR, Asset, floor_to_bar
from cryptonlp.schemas.ledger import DataFile
from cryptonlp.schemas.market import Interval, Symbol

MINUTE = timedelta(minutes=1)
RETURNS_PER_BAR = 60

FILES_SQL = """
SELECT path, sha256 FROM raw.files
WHERE dataset = %s AND removed_at IS NULL AND starts_with(path, %s)
  AND partition_date BETWEEN %s AND %s
ORDER BY file_id
"""


def symbol_for(asset: Asset) -> Symbol:
    return Symbol(f"{asset.value}USDT")


@dataclass(frozen=True)
class MinuteCloses:
    """Closing prices of 1-minute candles, by candle open time, and the file each came from."""

    closes: dict[datetime, float]
    files: dict[datetime, DataFile]

    def realized_vol(self, bar_open: datetime) -> float | None:
        prev = self.closes.get(bar_open - MINUTE)
        if prev is None:
            return None
        total = 0.0
        for i in range(RETURNS_PER_BAR):
            close = self.closes.get(bar_open + i * MINUTE)
            if close is None:
                return None
            total += math.log(close / prev) ** 2
            prev = close
        return math.sqrt(total)

    def log_return(self, bar_open: datetime) -> float | None:
        """ln(P(T + 1h) / P(T)) for the bar opening at T, under the same rule as the RV."""
        first = self.closes.get(bar_open - MINUTE)
        last = self.closes.get(bar_open + (RETURNS_PER_BAR - 1) * MINUTE)
        opens = (bar_open + i * MINUTE for i in range(RETURNS_PER_BAR))
        if first is None or last is None or any(t not in self.closes for t in opens):
            return None
        return math.log(last / first)

    def files_for(self, bar_open: datetime) -> tuple[DataFile, ...]:
        """The raw files the bar's RV was computed from, for the score's provenance."""
        opens = (bar_open + i * MINUTE for i in range(-1, RETURNS_PER_BAR))
        found = {self.files[t] for t in opens if t in self.files}
        return tuple(sorted(found, key=lambda f: f.path))


def load_minute_closes(
    conn: psycopg.Connection,
    store: RawStore,
    symbol: Symbol,
    start: datetime,
    end: datetime,
    *,
    now: datetime,
) -> MinuteCloses:
    """1-minute candles opening in [start, end) that had closed by ``now``."""
    prefix = f"{KLINES_DATASET}/symbol={symbol.value}/interval={Interval.MINUTE.value}/"
    files = conn.execute(
        FILES_SQL, (KLINES_DATASET, prefix, start.date(), (end - MINUTE).date())
    ).fetchall()
    newest: dict[datetime, tuple[datetime, float, DataFile]] = {}
    for path, sha256 in files:
        data_file = DataFile(path=path, sha256=sha256)
        table = pq.read_table(store.root / path, columns=["open_time", "close", "fetched_at"])
        for row in table.to_pylist():
            opened = row["open_time"]
            if not start <= opened < end or opened + MINUTE > now:
                continue
            seen = newest.get(opened)
            if seen is None or row["fetched_at"] >= seen[0]:
                newest[opened] = (row["fetched_at"], row["close"], data_file)
    return MinuteCloses(
        closes={t: v[1] for t, v in newest.items()},
        files={t: v[2] for t, v in newest.items()},
    )


def load_for_bars(
    conn: psycopg.Connection,
    store: RawStore,
    asset: Asset,
    bars: Iterable[datetime],
    *,
    now: datetime,
) -> MinuteCloses:
    """Every candle the RV of ``bars`` needs, as far as they had closed by ``now``."""
    opens = sorted(bars)
    if not opens:
        return MinuteCloses({}, {})
    return load_minute_closes(
        conn, store, symbol_for(asset), opens[0] - MINUTE, opens[-1] + BAR, now=now
    )


def realized_vol(
    conn: psycopg.Connection,
    store: RawStore,
    asset: Asset,
    bars: Iterable[datetime],
    *,
    now: datetime,
) -> dict[datetime, float]:
    """RV of each bar in ``bars`` that had closed by ``now`` with every candle present."""
    bars = list(bars)
    closes = load_for_bars(conn, store, asset, bars, now=now)
    out = {}
    for bar_open in bars:
        if bar_open + BAR <= now and (rv := closes.realized_vol(bar_open)) is not None:
            out[bar_open] = rv
    return out


# --- whole ranges at once (arm 1, walk-forward) -------------------------------------------------

BAR_COLUMNS = ("ret", "rv", "volume")


@dataclass(frozen=True)
class HourlyBars:
    """Hourly bars of one asset on a complete hourly grid (UTC bar opens as the index).

    Columns: ``ret`` (hourly log return), ``rv`` (realized volatility) and ``volume`` (base
    asset), all NaN for a bar with any candle missing. ``files`` are the raw-store files read.
    """

    asset: Asset
    frame: pd.DataFrame
    files: tuple[DataFile, ...]


def hourly_from_minutes(minutes: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Hourly bars opening in [start, end) from 1-minute candles.

    ``minutes`` has one row per candle open time (UTC, unique) with ``close`` and ``volume``.
    The candle opening at ``start - 1 minute`` gives the first bar's opening price.
    """
    if start != floor_to_bar(start) or end != floor_to_bar(end) or end <= start:
        raise ValueError("start and end must be bar opens, with end after start")
    n_bars = (end - start) // BAR
    grid = pd.date_range(start - MINUTE, end - MINUTE, freq="1min", inclusive="both")
    data = minutes.reindex(grid)
    log_close = np.log(data["close"].to_numpy(dtype=float))
    returns = np.diff(log_close).reshape(n_bars, RETURNS_PER_BAR)  # minute i of bar b
    volume = data["volume"].to_numpy(dtype=float)[1:].reshape(n_bars, RETURNS_PER_BAR)
    complete = ~np.isnan(returns).any(axis=1)
    rv = np.where(complete, np.sqrt(np.square(returns).sum(axis=1)), np.nan)
    ret = np.where(complete, returns.sum(axis=1), np.nan)
    vol = np.where(complete, volume.sum(axis=1), np.nan)
    index = pd.date_range(start, periods=n_bars, freq="1h", name="bar_open")
    return pd.DataFrame({"ret": ret, "rv": rv, "volume": vol}, index=index)


def load_hourly(
    conn: psycopg.Connection,
    store: RawStore,
    asset: Asset,
    start: datetime,
    end: datetime,
    *,
    now: datetime,
) -> HourlyBars:
    """Hourly bars opening in [start, end), from the 1-minute candles that had closed by
    ``now``. A bar that hadn't closed by ``now`` comes out NaN, like one with a missing candle:
    nothing after ``now`` can reach a forecast (the leakage rule for market data)."""
    symbol = symbol_for(asset)
    prefix = f"{KLINES_DATASET}/symbol={symbol.value}/interval={Interval.MINUTE.value}/"
    first, last = start - MINUTE, end - MINUTE
    rows = conn.execute(FILES_SQL, (KLINES_DATASET, prefix, first.date(), last.date())).fetchall()
    files = tuple(DataFile(path=path, sha256=sha256) for path, sha256 in rows)
    columns = ["open_time", "close", "volume", "fetched_at"]
    tables = [pq.read_table(store.root / f.path, columns=columns) for f in files]
    if tables:
        table = pa.concat_tables(tables)
        candles = table.to_pandas()
    else:
        candles = pd.DataFrame(columns=columns)
    if not candles.empty:
        opened = pd.to_datetime(candles["open_time"], utc=True)
        keep = (opened >= first) & (opened <= last) & (opened + MINUTE <= now)
        candles = candles.assign(open_time=opened)[keep]
        # Restated candles: the newest fetch wins.
        candles = candles.sort_values(["open_time", "fetched_at"], kind="stable")
        candles = candles.drop_duplicates("open_time", keep="last")
    minutes = candles.set_index("open_time")[["close", "volume"]]
    return HourlyBars(asset, hourly_from_minutes(minutes, start, end), files)
