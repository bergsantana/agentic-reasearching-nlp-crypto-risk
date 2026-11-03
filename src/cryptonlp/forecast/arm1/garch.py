"""GARCH(1,1), GJR-GARCH(1,1) and EGARCH(1,1) on hourly log returns (article, Section 2.6).

Estimation uses ``arch`` (constant mean, Gaussian quasi-maximum likelihood, returns in percent
for numerical stability). The variance filter and the forecasts are computed here, from the
fitted parameters, so the live loop and the walk-forward harness share one deterministic
path:

* ``filter_variance`` runs the one-step recursion in ``arch``'s own form (checked against
  ``arch`` in the tests). A missing return (missing candles) enters as its expectation: no
  shock, rather than a jump over the gap.
* The forecast for bar t uses data up to t-2, so it is the two-step-ahead variance:
  GARCH and GJR: omega + (alpha + gamma/2 + beta) * h(t-1); EGARCH: exp(omega + beta ln h(t-1))
  times E[exp(alpha(|z| - E|z|) + gamma z)] for a standard normal z, in closed form.
* For a block, the filter starts ``filter_hours`` before the block, from a backcast of the
  first 75 squared residuals (weights 0.94^i, as ``arch`` does). The start's influence is
  gone long before the block: persistence near 0.99 decays it to ~1e-9 over 2,160 hours.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

from cryptonlp.forecast.arm1.base import (
    LAG,
    Fit,
    check_block,
    check_grid,
    finish_fit,
    outputs_from_sigma,
    recent_coverage,
    volatility_outputs,
)
from cryptonlp.forecast.config import Arm1Section
from cryptonlp.schemas.common import BAR, Asset

Kind = Literal["garch", "gjr", "egarch"]
SCALE = 100.0  # returns in percent while estimating and filtering
BACKCAST_OBS = 75
BACKCAST_DECAY = 0.94
E_ABS_Z = math.sqrt(2.0 / math.pi)  # E|z| for a standard normal z

NAMES: dict[Kind, str] = {"garch": "garch11", "gjr": "gjr_garch11", "egarch": "egarch11"}
SPECS: dict[Kind, dict[str, object]] = {
    "garch": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "gjr": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    "egarch": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
}


def backcast(kind: Kind, resid: np.ndarray) -> float:
    """``arch``'s starting value: an exponentially weighted mean of the first squared residuals
    (its log for EGARCH). Missing residuals are skipped."""
    first = resid[np.isfinite(resid)][:BACKCAST_OBS]
    if len(first) == 0:
        raise ValueError("no returns to start the variance filter from")
    weights = BACKCAST_DECAY ** np.arange(len(first))
    value = float(np.sum(weights * first**2) / np.sum(weights))
    return math.log(value) if kind == "egarch" else value


def filter_variance(kind: Kind, params: dict[str, float], resid: np.ndarray) -> np.ndarray:
    """One-step conditional variances h(0..n): h[k] uses residuals before k (percent units).

    ``resid`` has n entries; the result has n + 1, the last being the variance for the bar
    after the data end."""
    n = len(resid)
    omega, alpha, beta = params["omega"], params["alpha[1]"], params["beta[1]"]
    gamma = params.get("gamma[1]", 0.0)
    bc = backcast(kind, resid)
    h = np.empty(n + 1)
    if kind == "egarch":
        log_h = omega + beta * bc
        h[0] = math.exp(log_h)
        for k in range(n):
            e = resid[k]
            if math.isfinite(e):
                z = e / math.sqrt(h[k])
                log_h = omega + alpha * (abs(z) - E_ABS_Z) + gamma * z + beta * log_h
            else:
                log_h = omega + beta * log_h
            h[k + 1] = math.exp(log_h)
        return h
    h[0] = omega + (alpha + 0.5 * gamma) * bc + beta * bc
    for k in range(n):
        e = resid[k]
        if math.isfinite(e):
            shock = alpha * e * e + (gamma * e * e if e < 0 else 0.0)
        else:
            shock = (alpha + 0.5 * gamma) * h[k]
        h[k + 1] = omega + shock + beta * h[k]
    return h


def egarch_shock_factor(alpha: float, gamma: float) -> float:
    """E[exp(alpha (|z| - E|z|) + gamma z)] for a standard normal z."""
    a, g = alpha, gamma
    return math.exp(-a * E_ABS_Z) * (
        math.exp((a + g) ** 2 / 2) * norm.cdf(a + g) + math.exp((a - g) ** 2 / 2) * norm.cdf(a - g)
    )


def two_step(kind: Kind, params: dict[str, float], h_prev: np.ndarray) -> np.ndarray:
    """Variance of bar t given data to t-2, from h(t-1) (the one-step variance known at t-2)."""
    omega, alpha, beta = params["omega"], params["alpha[1]"], params["beta[1]"]
    gamma = params.get("gamma[1]", 0.0)
    if kind == "egarch":
        with np.errstate(divide="ignore"):
            return np.exp(omega + beta * np.log(h_prev)) * egarch_shock_factor(alpha, gamma)
    return omega + (alpha + 0.5 * gamma + beta) * h_prev


class GarchModel:
    """Implements ``Arm1Model``."""

    version = "1"

    def __init__(self, kind: Kind, cfg: Arm1Section) -> None:
        self.kind = kind
        self.cfg = cfg
        self.name = NAMES[kind]
        self.outputs = volatility_outputs(cfg)

    def sigma(
        self,
        params: dict[str, float],
        bars: pd.DataFrame,
        start: datetime,
        targets: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Two-step volatility forecasts (log-return units) for ``targets``, filtering from
        ``start``. Targets need start + 1 bar <= t <= the last bar + LAG."""
        if len(targets) == 0:
            return np.empty(0)
        last = targets.max() - LAG * BAR
        window = bars.loc[start:last, "ret"]
        resid = window.to_numpy(dtype=float) * SCALE - params["mu"]
        h = filter_variance(self.kind, params, resid)
        pos = ((targets - BAR) - pd.Timestamp(start)) // BAR
        pos_arr = np.asarray(pos, dtype=np.int64)
        if (pos_arr < 0).any() or (pos_arr > len(resid)).any():
            raise ValueError("a target lies outside the filtered window")
        variance = two_step(self.kind, params, h[pos_arr])
        return np.sqrt(variance) / SCALE

    def fit(self, train: pd.DataFrame, asset: Asset, block: datetime) -> Fit | None:
        check_grid(train)
        returns = train["ret"].dropna()
        if len(returns) < self.cfg.min_train_hours:
            return None
        model = arch_model(
            returns.to_numpy() * SCALE,
            mean="Constant",
            dist="normal",
            rescale=False,
            **SPECS[self.kind],  # type: ignore[arg-type]
        )
        result = model.fit(disp="off", show_warning=False, options={"maxiter": 1000})
        params = {str(k): float(v) for k, v in result.params.items()}
        start = train.index[0].to_pydatetime()
        targets = pd.DatetimeIndex(train.index[1:])
        sigma = np.concatenate([[np.nan], self.sigma(params, train, start, targets)])
        return finish_fit(
            model=self.name,
            version=self.version,
            asset=asset,
            block=block,
            train=train,
            sigma=sigma,
            mu=params["mu"] / SCALE,
            params=params,
            cfg=self.cfg,
            diagnostics={
                "loglik": float(result.loglikelihood),
                "converged": int(result.convergence_flag) == 0,
                "scale": SCALE,
                "estimator": "gaussian_qml",
            },
        )

    def predict(self, fit: Fit, bars: pd.DataFrame, targets: pd.DatetimeIndex) -> pd.DataFrame:
        check_grid(bars)
        if len(targets) == 0:
            return outputs_from_sigma(fit, np.empty(0), self.outputs, targets)
        check_block(fit, targets)
        start = max(fit.block_start - timedelta(hours=self.cfg.filter_hours), fit.train_start)
        if bars.index[0] > start:
            raise ValueError(f"bars start at {bars.index[0]}, after the filter start {start}")
        sigma = self.sigma(fit.params, bars, start, targets)
        sigma = np.where(recent_coverage(bars, targets, self.cfg), sigma, np.nan)
        return outputs_from_sigma(fit, sigma, self.outputs, targets)
