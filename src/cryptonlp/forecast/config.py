"""Loop settings from ``config/loop.toml``."""

from __future__ import annotations

import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cryptonlp.schemas.common import Asset, BarOpen, Record
from cryptonlp.settings import Settings


class LoopSection(Record):
    assets: tuple[Asset, ...] = (Asset.BTC, Asset.ETH)
    features_settle_minutes: int = Field(default=20, ge=0, lt=60)
    features_lookback_hours: int = Field(default=48, gt=0)
    classify_batch: int = Field(default=500, gt=0)
    classify_budget_minutes: int = Field(default=15, gt=0, lt=60)

    @property
    def settle(self) -> timedelta:
        return timedelta(minutes=self.features_settle_minutes)

    @property
    def lookback(self) -> timedelta:
        return timedelta(hours=self.features_lookback_hours)

    @property
    def classify_budget(self) -> timedelta:
        return timedelta(minutes=self.classify_budget_minutes)


class BaselineSection(Record):
    window_hours: int = Field(default=24, gt=0)
    min_bars: int = Field(default=20, gt=0)

    @model_validator(mode="after")
    def _min_fits(self) -> BaselineSection:
        if self.min_bars > self.window_hours:
            raise ValueError("min_bars cannot exceed window_hours")
        return self


Arm1ModelName = Literal["garch", "gjr", "egarch", "har", "har_log", "hist_var"]


class Arm1Section(Record):
    """Arm 1 (statistics only). Comments on each setting are in ``config/loop.toml``."""

    models: tuple[Arm1ModelName, ...] = ("garch", "gjr", "egarch", "har", "har_log", "hist_var")
    # Issue arm 1 forecasts in the hourly loop (the walk-forward harness ignores this).
    in_loop: bool = True
    train_start: BarOpen = datetime.fromisoformat("2024-11-01T00:00:00+00:00")
    min_train_hours: int = Field(default=2160, gt=0)
    filter_hours: int = Field(default=2160, gt=0)
    min_coverage: float = Field(default=0.8, gt=0, le=1)
    var_levels: tuple[float, ...] = (0.01, 0.05)
    large_move_quantile: float = Field(default=0.95, gt=0, lt=1)
    tail_grid: int = Field(default=401, ge=11)
    har_windows: tuple[int, int, int] = (24, 168, 720)
    hist_window_hours: int = Field(default=720, gt=0)

    @field_validator("var_levels")
    @classmethod
    def _levels(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v or any(not 0 < a < 0.5 for a in v) or len(set(v)) != len(v):
            raise ValueError("var_levels must be distinct and in (0, 0.5)")
        return tuple(sorted(v))

    @field_validator("models")
    @classmethod
    def _models(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError("a model is listed twice")
        return v

    @field_validator("har_windows")
    @classmethod
    def _windows(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if not 0 < v[0] < v[1] < v[2]:
            raise ValueError("har_windows must be increasing: daily, weekly, monthly")
        return v

    @property
    def history_hours(self) -> int:
        """Closed bars a forecast needs before its block (the GARCH filter's warm-up), plus the
        block itself: what the loop loads every hour."""
        return max(self.filter_hours, self.har_windows[2], self.hist_window_hours) + 7 * 24 + 2


class WalkForwardSection(Record):
    # First test block (a Monday, 00:00 UTC). Bars before it are only ever training data.
    test_start: BarOpen = datetime.fromisoformat("2025-11-03T00:00:00+00:00")
    # Last bar to test (exclusive). Unset: up to the newest closed bar.
    end: BarOpen | None = None
    # Newey-West lags for the Diebold-Mariano and Clark-West tests. Unset: 4 (n/100)^(2/9).
    nw_lags: int | None = Field(default=None, ge=0)
    mlflow_experiment: str = "arm1-walkforward"

    @field_validator("test_start")
    @classmethod
    def _monday(cls, v: datetime) -> datetime:
        if v.weekday() != 0 or v.hour != 0:
            raise ValueError("test_start must be a Monday at 00:00 UTC (blocks are weeks)")
        return v


class LoopConfig(Record):
    loop: LoopSection = LoopSection()
    baseline: BaselineSection = BaselineSection()
    arm1: Arm1Section = Arm1Section()
    walkforward: WalkForwardSection = WalkForwardSection()

    @model_validator(mode="after")
    def _start_after_training(self) -> LoopConfig:
        if self.walkforward.test_start <= self.arm1.train_start:
            raise ValueError("walkforward.test_start must come after arm1.train_start")
        return self


def load_loop_config(path: Path | None = None) -> LoopConfig:
    path = path or Settings().loop_config_path
    with path.open("rb") as f:
        return LoopConfig.model_validate(tomllib.load(f))
