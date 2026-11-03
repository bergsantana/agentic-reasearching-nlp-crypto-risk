"""Losses and tests for the forecasts (article, Section 4.11). Own code, short on purpose so it
is easy to audit (tech/stack-and-build-guide.md, "Statistics and forecasting").

* Volatility: squared error on the variance and QLIKE (Patton 2011), as in ``scoring``.
* VaR (a positive loss; a violation is a return below -VaR): Kupiec's unconditional coverage
  (1995) and Christoffersen's independence and conditional coverage (1998) likelihood-ratio
  tests, plus the quantile ("tick") loss used to compare two VaR forecasts.
* Probabilities: the Brier score and Murphy's decomposition (reliability, resolution,
  uncertainty) over equal-width bins.
* Comparing two forecasts: Diebold-Mariano on the loss differences, with a Newey-West
  (Bartlett) long-run variance and the Harvey-Leybourne-Newbold small-sample correction;
  Clark-West for nested models (H1).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats
from scipy.special import xlogy


@dataclass(frozen=True)
class Backtest:
    stat: float
    pvalue: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- losses --------------------------------------------------------------------------------


def se_variance(realized_vol: np.ndarray, forecast_vol: np.ndarray) -> np.ndarray:
    return np.asarray((realized_vol**2 - forecast_vol**2) ** 2, dtype=float)


def qlike(realized_vol: np.ndarray, forecast_vol: np.ndarray) -> np.ndarray:
    """s/h - ln(s/h) - 1 on the variances; NaN where either is 0 (left out, as in scoring)."""
    s, h = realized_vol**2, forecast_vol**2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = s / h
        out = ratio - np.log(ratio) - 1
    return np.where((s > 0) & (h > 0), out, np.nan)


def var_hits(returns: np.ndarray, var: np.ndarray) -> np.ndarray:
    return np.asarray(returns < -var, dtype=bool)


def tick_loss(returns: np.ndarray, var: np.ndarray, level: float | np.ndarray) -> np.ndarray:
    """Quantile loss of the return quantile q = -VaR at ``level``: (level - 1{r < q})(r - q)."""
    q = -var
    return np.asarray((level - (returns < q)) * (returns - q), dtype=float)


def brier(prob: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    return np.asarray((prob - outcome) ** 2, dtype=float)


# --- VaR backtests -------------------------------------------------------------------------


def kupiec(hits: np.ndarray, level: float) -> Backtest:
    """Unconditional coverage: is the violation rate equal to ``level``?"""
    hits = np.asarray(hits, dtype=bool)
    n, x = len(hits), int(hits.sum())
    if n == 0:
        return Backtest(math.nan, math.nan, 0)
    rate = x / n
    lr = -2 * (xlogy(n - x, 1 - level) + xlogy(x, level) - xlogy(n - x, 1 - rate) - xlogy(x, rate))
    return Backtest(float(lr), float(stats.chi2.sf(lr, 1)), n)


def christoffersen(hits: np.ndarray, level: float) -> tuple[Backtest, Backtest]:
    """Independence (first-order Markov against independent violations) and conditional
    coverage (unconditional coverage and independence together, chi-squared with 2 df)."""
    hits = np.asarray(hits, dtype=bool)
    n = len(hits)
    if n < 2:
        nan = Backtest(math.nan, math.nan, n)
        return nan, nan
    prev, cur = hits[:-1], hits[1:]
    n00 = int((~prev & ~cur).sum())
    n01 = int((~prev & cur).sum())
    n10 = int((prev & ~cur).sum())
    n11 = int((prev & cur).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n - 1)
    restricted = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    unrestricted = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr_ind = float(max(-2 * (restricted - unrestricted), 0.0))
    lr_cc = kupiec(hits[1:], level).stat + lr_ind
    return (
        Backtest(lr_ind, float(stats.chi2.sf(lr_ind, 1)), n),
        Backtest(float(lr_cc), float(stats.chi2.sf(lr_cc, 2)), n),
    )


# --- probabilities -------------------------------------------------------------------------


@dataclass(frozen=True)
class BrierDecomposition:
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def murphy(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> BrierDecomposition:
    """Brier = reliability - resolution + uncertainty (exact when forecasts are constant within
    each bin; otherwise up to the within-bin variance)."""
    prob, outcome = np.asarray(prob, dtype=float), np.asarray(outcome, dtype=float)
    n = len(prob)
    if n == 0:
        return BrierDecomposition(math.nan, math.nan, math.nan, math.nan, 0)
    base = outcome.mean()
    which = np.minimum((prob * bins).astype(int), bins - 1)
    rel = res = 0.0
    for k in range(bins):
        mask = which == k
        if mask.any():
            nk = mask.sum()
            rel += nk * (prob[mask].mean() - outcome[mask].mean()) ** 2
            res += nk * (outcome[mask].mean() - base) ** 2
    return BrierDecomposition(
        float(brier(prob, outcome).mean()), rel / n, res / n, float(base * (1 - base)), n
    )


# --- comparing forecasts -------------------------------------------------------------------


def default_lags(n: int) -> int:
    """Newey and West's (1994) rule of thumb: floor(4 (n / 100)^(2/9))."""
    return math.floor(4 * (n / 100) ** (2 / 9)) if n > 0 else 0


def long_run_variance(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett kernel) long-run variance of ``x`` around its mean."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    lrv = float(x @ x) / n
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2 * (1 - k / (lags + 1)) * float(x[k:] @ x[:-k]) / n
    return lrv


@dataclass(frozen=True)
class Comparison:
    mean_diff: float  # mean of loss_a - loss_b: positive when b is better
    stat: float
    pvalue: float
    n: int
    lags: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, *, horizon: int = 2, lags: int | None = None
) -> Comparison:
    """Two-sided test of equal expected loss, HLN-corrected, against Student's t (n - 1 df).

    ``horizon`` is the number of bars between a forecast's data and its target (2 here: the
    forecast for t uses data to t - 2)."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    lags = default_lags(n) if lags is None else lags
    if n < 3:
        return Comparison(math.nan, math.nan, math.nan, n, lags)
    lrv = long_run_variance(d, lags)
    mean = float(d.mean())
    if lrv <= 0:
        return Comparison(mean, math.nan, math.nan, n, lags)
    dm = mean / math.sqrt(lrv / n)
    h = horizon
    hln = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0))
    stat = dm * hln
    return Comparison(mean, stat, float(2 * stats.t.sf(abs(stat), n - 1)), n, lags)


def clark_west(
    actual: np.ndarray, small: np.ndarray, large: np.ndarray, *, lags: int | None = None
) -> Comparison:
    """Clark and West (2007): does the larger model, which nests the smaller one, have lower
    mean squared error? One-sided; ``mean_diff`` is the mean adjusted MSPE difference."""
    y, f1, f2 = (np.asarray(a, dtype=float) for a in (actual, small, large))
    adj = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    adj = adj[np.isfinite(adj)]
    n = len(adj)
    lags = default_lags(n) if lags is None else lags
    if n < 3:
        return Comparison(math.nan, math.nan, math.nan, n, lags)
    lrv = long_run_variance(adj, lags)
    mean = float(adj.mean())
    if lrv <= 0:
        return Comparison(mean, math.nan, math.nan, n, lags)
    stat = mean / math.sqrt(lrv / n)
    return Comparison(mean, stat, float(stats.norm.sf(stat)), n, lags)
