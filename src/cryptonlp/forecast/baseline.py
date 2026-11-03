"""The baseline every arm must beat: "yesterday's volatility" (design doc, Forecasting arms).

For the next hour: the root mean square of the realized volatility of the last ``window_hours``
closed bars (24 by default), i.e. the past day's realized variance spread evenly over its hours.
It ignores the hour-of-day pattern on purpose; arm 1 (HAR, GARCH) is where that belongs.
"""

from __future__ import annotations

import math

from cryptonlp.forecast.base import ForecastRequest
from cryptonlp.schemas.common import BAR, floor_to_bar
from cryptonlp.schemas.forecasts import Arm, Forecast, Target


class YesterdayVolatility:
    """Implements ``cryptonlp.forecast.Forecaster``."""

    arm = Arm.BASELINE
    target = Target.RV_1H
    model_version = "1"

    def __init__(self, window_hours: int = 24, min_bars: int = 20) -> None:
        if not 0 < min_bars <= window_hours:
            raise ValueError("min_bars must be between 1 and window_hours")
        self.window_hours = window_hours
        self.min_bars = min_bars
        self.model = f"yesterday_rv_{window_hours}h"
        self.history_hours = window_hours

    def forecast(self, request: ForecastRequest) -> Forecast | None:
        # The newest bar that had closed when the forecast was issued.
        newest = floor_to_bar(request.issued_at) - BAR
        window = [newest - i * BAR for i in range(self.window_hours)]
        values = [request.rv[b] for b in window if b in request.rv]
        if len(values) < self.min_bars:
            return None
        return Forecast(
            asset=request.asset,
            arm=self.arm,
            model=self.model,
            model_version=self.model_version,
            target=self.target,
            bar_open=request.bar_open,
            value=math.sqrt(sum(v * v for v in values) / len(values)),
            issued_at=request.issued_at,
            phase=request.phase,
            run_id=request.run_id,
        )
