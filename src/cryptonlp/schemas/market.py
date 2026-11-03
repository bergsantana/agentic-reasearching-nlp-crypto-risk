"""Binance candles (klines)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from cryptonlp.schemas.common import Asset, Record, RunId, UTCDatetime


class Symbol(StrEnum):
    BTCUSDT = "BTCUSDT"
    ETHUSDT = "ETHUSDT"

    @property
    def asset(self) -> Asset:
        return Asset(self.value.removesuffix("USDT"))


class Interval(StrEnum):
    MINUTE = "1m"
    HOUR = "1h"


class CandleSource(StrEnum):
    VISION = "vision"  # Binance Vision archive (backfill, with checksums)
    API = "api"  # data-api.binance.vision (recent tail)


class Candle(Record):
    symbol: Symbol
    interval: Interval
    open_time: UTCDatetime
    close_time: UTCDatetime  # Binance reports open_time + interval - 1 ms
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    quote_volume: float = Field(ge=0)
    trades: int = Field(ge=0)
    source: CandleSource
    collector_run_id: RunId

    @model_validator(mode="after")
    def _consistent(self) -> Candle:
        if self.close_time <= self.open_time:
            raise ValueError("close_time must be after open_time")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("low/high must bound open and close")
        return self
