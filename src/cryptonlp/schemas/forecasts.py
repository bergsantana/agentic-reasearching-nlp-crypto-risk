"""Forecasts from the baseline and the three arms. The ledger stores them before the bar opens."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from cryptonlp.schemas.common import (
    Asset,
    BarOpen,
    NonEmptyStr,
    Phase,
    Record,
    RunId,
    UTCDatetime,
)


class Arm(StrEnum):
    BASELINE = "baseline"  # "yesterday's volatility"; every arm must beat it
    STATS = "arm1_stats"
    NLP = "arm2_nlp"
    COMBINED = "arm3_combined"


class Target(StrEnum):
    RV_1H = "rv_1h"  # next-hour realized volatility (confirmatory)
    P_LARGE_MOVE_1H = "p_large_move_1h"  # probability of a large move in the next hour
    VAR_1H = "var_1h"  # next-hour value at risk, scored by exceedance
    RV_1D = "rv_1d"  # next-day realized volatility (exploratory)


PROBABILITY_TARGETS = frozenset({Target.P_LARGE_MOVE_1H})
VAR_TARGETS = frozenset({Target.VAR_1H})


class Forecast(Record):
    asset: Asset
    arm: Arm
    model: NonEmptyStr  # e.g. "garch(1,1)", "har-rv", "lightgbm"
    model_version: NonEmptyStr
    target: Target
    bar_open: BarOpen  # first bar the forecast is about
    value: float
    var_level: float | None = Field(default=None, gt=0, lt=1)  # e.g. 0.01 for 99% VaR
    # Client-side time, checked here as a sanity check only. The ledger enforces the deadline
    # with the database clock (step 2) and never trusts this value.
    issued_at: UTCDatetime
    phase: Phase
    run_id: RunId

    @model_validator(mode="after")
    def _valid(self) -> Forecast:
        if self.issued_at >= self.bar_open:
            raise ValueError("a forecast must be issued before its bar opens")
        if self.target in VAR_TARGETS:
            if self.var_level is None:
                raise ValueError("VaR forecasts need var_level")
        elif self.var_level is not None:
            raise ValueError("var_level only applies to VaR targets")
        if self.target in PROBABILITY_TARGETS and not 0 <= self.value <= 1:
            raise ValueError("probability forecasts must lie in [0, 1]")
        if self.target in {Target.RV_1H, Target.RV_1D} and self.value < 0:
            raise ValueError("realized volatility forecasts must be non-negative")
        return self
