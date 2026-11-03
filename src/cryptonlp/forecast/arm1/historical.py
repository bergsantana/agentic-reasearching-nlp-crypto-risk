"""Historical VaR (design doc, arm 1): the empirical quantile of the last ``hist_window_hours``
hourly returns, ending at the forecast's origin (t - 2), with no model of volatility.

It also gives a historical large-move probability: the share of returns in the same window
beyond the block's threshold. It forecasts no RV. Both serve as the reference the volatility
models' VaR and probabilities are compared against.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from cryptonlp.forecast.arm1.base import (
    LAG,
    Fit,
    Output,
    check_block,
    check_grid,
    large_move_threshold,
    min_bars,
    recent_coverage,
)
from cryptonlp.forecast.config import Arm1Section
from cryptonlp.schemas.common import BAR, Asset
from cryptonlp.schemas.forecasts import Target


class HistoricalVaR:
    """Implements ``Arm1Model``."""

    version = "1"

    def __init__(self, cfg: Arm1Section) -> None:
        self.cfg = cfg
        self.window = cfg.hist_window_hours
        self.name = f"hist_var_{self.window}h"
        self.outputs = (
            *(Output(Target.VAR_1H, a) for a in cfg.var_levels),
            Output(Target.P_LARGE_MOVE_1H),
        )

    def fit(self, train: pd.DataFrame, asset: Asset, block: datetime) -> Fit | None:
        """Nothing to estimate: the fit records the block's large-move threshold."""
        check_grid(train)
        n_obs = int(train["ret"].notna().sum())
        if n_obs < self.cfg.min_train_hours:
            return None
        return Fit(
            model=self.name,
            model_version=self.version,
            asset=asset,
            block_start=block,
            train_start=train.index[0].to_pydatetime(),
            train_end=train.index[-1].to_pydatetime(),
            n_obs=n_obs,
            params={"window_hours": float(self.window)},
            mu=0.0,
            threshold=large_move_threshold(train, self.cfg.large_move_quantile),
            var_z={},
            z_grid=(),
            diagnostics={"estimator": "empirical_quantile"},
        )

    def predict(self, fit: Fit, bars: pd.DataFrame, targets: pd.DatetimeIndex) -> pd.DataFrame:
        check_grid(bars)
        check_block(fit, targets)
        ret = bars["ret"]
        need = min_bars(self.window, self.cfg.min_coverage)
        rolling = ret.rolling(self.window, min_periods=need)
        origins = targets - LAG * BAR
        enough = recent_coverage(bars, targets, self.cfg)
        out: dict[str, np.ndarray] = {}
        for o in self.outputs:
            if o.target is Target.VAR_1H:
                assert o.var_level is not None
                q = rolling.quantile(o.var_level)
                out[o.key] = np.where(enough, -q.reindex(origins).to_numpy(dtype=float), np.nan)
            else:
                beyond = (ret.abs() > fit.threshold).astype(float).where(ret.notna())
                share = beyond.rolling(self.window, min_periods=need).mean()
                out[o.key] = np.where(enough, share.reindex(origins).to_numpy(dtype=float), np.nan)
        return pd.DataFrame(out, index=targets)
