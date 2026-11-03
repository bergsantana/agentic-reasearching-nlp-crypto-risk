"""HAR regressions on hourly realized volatility (Corsi 2009; article, Section 2.6, Equation 5).

For bar t, with origin o = t - 2 (the newest usable bar), the regressors are the mean RV over
the ``har_windows`` bars ending at o: 24 (daily), 168 (weekly) and 720 (monthly) by default.
A window with less than ``min_coverage`` of its bars (missing candles) gives no forecast.

* ``har`` is Equation 5 as written: RV_t on the three means, by ordinary least squares, in
  levels of volatility. A forecast below the training window's 1st-percentile RV is raised to
  it, since a linear model can go negative and QLIKE is undefined at zero.
* ``har_log`` is the log variant common in the literature: ln RV_t^2 on the logs of the mean
  variances, converted back with Duan's smearing factor (the mean of exp(residual)), which
  keeps forecasts positive. It is an extra model, not a replacement for Equation 5.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from cryptonlp.forecast.arm1.base import (
    LAG,
    Fit,
    check_block,
    check_grid,
    finish_fit,
    min_bars,
    outputs_from_sigma,
    volatility_outputs,
)
from cryptonlp.forecast.config import Arm1Section
from cryptonlp.schemas.common import BAR, Asset


class HarModel:
    """Implements ``Arm1Model``."""

    version = "1"

    def __init__(self, cfg: Arm1Section, *, log: bool = False) -> None:
        self.cfg = cfg
        self.log = log
        self.name = "har_log_rv" if log else "har_rv"
        self.outputs = volatility_outputs(cfg)

    def regressors(self, bars: pd.DataFrame, targets: pd.DatetimeIndex) -> np.ndarray:
        """The design matrix for ``targets`` (constant first), NaN rows where data are thin."""
        rv = bars["rv"]
        base = rv * rv if self.log else rv
        columns = [np.ones(len(targets))]
        for window in self.cfg.har_windows:
            mean = base.rolling(window, min_periods=min_bars(window, self.cfg.min_coverage)).mean()
            origin = mean.reindex(targets - LAG * BAR).to_numpy(dtype=float)
            if self.log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    origin = np.where(origin > 0, np.log(origin), np.nan)
            columns.append(origin)
        return np.column_stack(columns)

    def _sigma(self, params: dict[str, float], x: np.ndarray) -> np.ndarray:
        beta = np.array([params[f"b{i}"] for i in range(x.shape[1])])
        fitted = x @ beta
        if self.log:
            return np.asarray(np.sqrt(np.exp(fitted) * params["smearing"]), dtype=float)
        return np.asarray(np.maximum(fitted, params["floor"]), dtype=float)

    def fit(self, train: pd.DataFrame, asset: Asset, block: datetime) -> Fit | None:
        check_grid(train)
        if train["ret"].notna().sum() < self.cfg.min_train_hours:
            return None
        x = self.regressors(train, pd.DatetimeIndex(train.index))
        rv = train["rv"].to_numpy(dtype=float)
        if self.log:
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.where(rv > 0, np.log(rv * rv), np.nan)
        else:
            y = rv
        rows = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if rows.sum() < self.cfg.min_train_hours // 2:
            return None
        beta, *_ = np.linalg.lstsq(x[rows], y[rows], rcond=None)
        resid = y[rows] - x[rows] @ beta
        params = {f"b{i}": float(b) for i, b in enumerate(beta)}
        if self.log:
            params["smearing"] = float(np.mean(np.exp(resid)))
        else:
            params["floor"] = float(np.quantile(rv[np.isfinite(rv)], 0.01))
        sigma = np.where(np.isfinite(x).all(axis=1), self._sigma(params, x), np.nan)
        returns = train["ret"].to_numpy(dtype=float)
        return finish_fit(
            model=self.name,
            version=self.version,
            asset=asset,
            block=block,
            train=train,
            sigma=sigma,
            mu=float(np.nanmean(returns)),
            params=params,
            cfg=self.cfg,
            diagnostics={
                "n_regression": int(rows.sum()),
                "r2": float(1 - resid.var() / y[rows].var()),
                "windows": list(self.cfg.har_windows),
                "estimator": "ols",
            },
        )

    def predict(self, fit: Fit, bars: pd.DataFrame, targets: pd.DatetimeIndex) -> pd.DataFrame:
        check_grid(bars)
        check_block(fit, targets)
        x = self.regressors(bars, targets)
        sigma = np.where(np.isfinite(x).all(axis=1), self._sigma(fit.params, x), np.nan)
        return outputs_from_sigma(fit, sigma, self.outputs, targets)
