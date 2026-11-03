"""Collector settings from ``config/collectors.toml``."""

from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import date
from pathlib import Path

from pydantic import Field

from cryptonlp.schemas.common import Record
from cryptonlp.schemas.market import Interval, Symbol
from cryptonlp.settings import Settings


class WatchConfig(Record):
    terms: tuple[str, ...] = Field(min_length=1)
    cashtags: tuple[str, ...] = ()

    @property
    def sha256(self) -> str:
        """Hash of the watch list, stored with every raw file the list filtered."""
        canonical = json.dumps(
            {"terms": sorted(self.terms), "cashtags": sorted(self.cashtags)}, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


class BlueskyConfig(Record):
    jetstream_hosts: tuple[str, ...] = Field(min_length=1)
    flush_seconds: float = Field(default=600, gt=0)
    max_rows: int = Field(default=5000, gt=0)
    heartbeat_seconds: float = Field(default=30, gt=0)
    rewind_seconds: float = Field(default=5, ge=0)
    replay_sample_rate: float = Field(default=0.001, ge=0, le=1)
    appview: str = "https://public.api.bsky.app"


class MastodonConfig(Record):
    instances: tuple[str, ...] = Field(min_length=1)
    hashtags: tuple[str, ...] = ()
    public_timeline: bool = False
    page_limit: int = Field(default=40, gt=0, le=40)
    max_pages: int = Field(default=10, gt=0)


class TrendsConfig(Record):
    mastodon_limit: int = Field(default=20, gt=0, le=20)


class BinanceConfig(Record):
    symbols: tuple[Symbol, ...] = Field(min_length=1)
    intervals: tuple[Interval, ...] = Field(min_length=1)
    vision_base: str = "https://data.binance.vision"
    api_base: str = "https://data-api.binance.vision"
    backfill_start: date
    restate_hours: int = Field(default=3, ge=0)
    tail_max_days: int = Field(default=7, gt=0)


class DeletionsConfig(Record):
    bluesky_batch: int = Field(default=10000, gt=0)
    mastodon_batch_per_instance: int = Field(default=600, gt=0)


class CollectorsConfig(Record):
    watch: WatchConfig
    bluesky: BlueskyConfig
    mastodon: MastodonConfig
    trends: TrendsConfig = TrendsConfig()
    binance: BinanceConfig
    deletions: DeletionsConfig = DeletionsConfig()


def load_config(path: Path | None = None) -> CollectorsConfig:
    path = path or Settings().collectors_config_path
    with path.open("rb") as f:
        return CollectorsConfig.model_validate(tomllib.load(f))
