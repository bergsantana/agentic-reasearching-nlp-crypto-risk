"""The forecaster interface: the baseline now, arms 1 to 3 later, all fed the same request."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from cryptonlp.schemas.common import BAR, Asset, Phase, floor_to_bar
from cryptonlp.schemas.features import HourlyFeatures
from cryptonlp.schemas.forecasts import Arm, Forecast, Target

if TYPE_CHECKING:
    import pandas as pd


class LeakageError(ValueError):
    """A forecast was about to see data from after it was issued."""


@dataclass(frozen=True)
class ForecastRequest:
    """Everything a forecaster may use for one asset and bar. Built by the loop, which only
    puts in data from bars that had closed when the request was made; checked again here."""

    asset: Asset
    bar_open: datetime  # the bar to forecast
    issued_at: datetime  # the database clock when the request was built
    phase: Phase
    run_id: str
    # Realized volatility and log return of closed bars, by bar open. Bars with missing candles
    # are absent.
    rv: Mapping[datetime, float] = field(default_factory=dict)
    returns: Mapping[datetime, float] = field(default_factory=dict)
    # First bar the loop loaded (the grid ``frame`` starts from).
    history_start: datetime | None = None
    # NLP features of closed, final bars (arms 2 and 3).
    features: Sequence[HourlyFeatures] = ()

    def __post_init__(self) -> None:
        if self.issued_at >= self.bar_open:
            raise ValueError("a forecast must be issued before its bar opens")
        for name, series in (("RV", self.rv), ("returns", self.returns)):
            still_open = [b for b in series if b + BAR > self.issued_at]
            if still_open:
                raise LeakageError(
                    f"{name} of bars not closed at {self.issued_at}: {still_open[:3]}"
                )
        open_features = [f.bar_open for f in self.features if f.bar_close > self.issued_at]
        if open_features:
            raise LeakageError(
                f"features of bars not closed at {self.issued_at}: {open_features[:3]}"
            )
        if any(f.asset is not self.asset for f in self.features):
            raise ValueError("features of another asset")

    def frame(self) -> pd.DataFrame:
        """``ret`` and ``rv`` on a complete hourly grid, from ``history_start`` to the newest
        bar closed at issue time, with NaN where a bar is missing (what arm 1 reads)."""
        import pandas as pd  # only the forecasting jobs need it (see cryptonlp.cli._loop)

        newest = floor_to_bar(self.issued_at) - BAR
        keys = [*self.rv, *self.returns]
        start = self.history_start or (min(keys) if keys else newest)
        index = pd.date_range(start, newest, freq="1h", name="bar_open")
        return pd.DataFrame(
            {
                "ret": pd.Series(self.returns, dtype=float).reindex(index),
                "rv": pd.Series(self.rv, dtype=float).reindex(index),
            },
            index=index,
        )


class Forecaster(Protocol):
    arm: Arm
    model: str  # with the arm, target and bar, what makes a prediction unique in the ledger
    model_version: str
    target: Target
    # Closed bars of RV history the forecaster needs before the bar it forecasts.
    history_hours: int

    def forecast(self, request: ForecastRequest) -> Forecast | None:
        """The forecast, or None when there isn't enough data (e.g. missing candles)."""
        ...
