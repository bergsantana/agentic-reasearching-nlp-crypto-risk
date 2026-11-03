"""What every arm 1 model shares: the timing rule, weekly blocks, the fitted state, and the
tails that turn a volatility forecast into VaR and large-move probabilities.

**Timing (claims M14).** The forecast for bar t is issued at minute 25 of bar t-1, which is
still open, so the newest bar it can use is t-2 (``LAG`` bars back). Every model forecasts
two steps ahead of its data, in the walk-forward harness exactly as in the live loop.

**Blocks.** Parameters are re-estimated once a week (article, Section 4.4). A block starts on
Monday at 00:00 UTC; its fit uses every bar from ``train_start`` up to the newest bar the
block's first forecast could see (block start - 2 hours), an expanding window. The fit is a
pure function of those bars, so the live loop and the harness get the same parameters.

**Tails (filtered historical simulation).** Parameters come from Gaussian quasi-maximum
likelihood or least squares, but VaR and the large-move probability use the empirical
distribution of the training window's standardized returns z = (r - mu) / sigma_hat, where
sigma_hat is the model's own two-step forecast. Crypto returns are fat-tailed; a normal
quantile would understate the 1% VaR.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd

from cryptonlp.forecast.config import Arm1Section
from cryptonlp.schemas.common import BAR, Asset
from cryptonlp.schemas.forecasts import Target

LAG = 2  # bars between a forecast's newest data and its target (claims M14)
WEEK = timedelta(days=7)
# Bars at the start of a training window left out of the tails, while a filter warms up.
BURN_IN = 168


@dataclass(frozen=True, order=True)
class Output:
    target: Target
    var_level: float | None = None

    @property
    def key(self) -> str:
        return (
            self.target.value if self.var_level is None else f"{self.target.value}@{self.var_level}"
        )


def volatility_outputs(cfg: Arm1Section) -> tuple[Output, ...]:
    return (
        Output(Target.RV_1H),
        *(Output(Target.VAR_1H, a) for a in cfg.var_levels),
        Output(Target.P_LARGE_MOVE_1H),
    )


def block_start(bar_open: datetime) -> datetime:
    """Monday 00:00 UTC of the week holding ``bar_open``."""
    day = bar_open.replace(hour=0, minute=0, second=0, microsecond=0)
    return day - timedelta(days=day.weekday())


def training_end(block: datetime) -> datetime:
    """The newest bar a block's fit may use: what its first forecast sees."""
    return block - LAG * BAR


def at_origin(series: pd.Series, targets: pd.DatetimeIndex) -> np.ndarray:
    """Values of ``series`` at each target's origin (t - LAG bars): the newest usable bar."""
    return series.reindex(targets - LAG * BAR).to_numpy(dtype=float)


def min_bars(window: int, coverage: float) -> int:
    return max(1, math.ceil(window * coverage - 1e-9))


@dataclass(frozen=True)
class Fit:
    """A model fitted for one asset and block. Stored as the block's experiment package
    (``to_config``) so every forecast in the ledger points at the parameters it used."""

    model: str
    model_version: str
    asset: Asset
    block_start: datetime
    train_start: datetime  # first bar of the window
    train_end: datetime  # last bar of the window (inclusive)
    n_obs: int  # bars with a return in the window
    params: dict[str, float]
    mu: float  # mean hourly log return used for the tails
    threshold: float  # large move: |return| above this (training quantile)
    var_z: dict[str, float]  # VaR level -> quantile of z
    z_grid: tuple[float, ...]  # quantiles of z at evenly spaced levels from 0 to 1
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_version": self.model_version,
            "asset": self.asset.value,
            "block_start": self.block_start.isoformat(),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "n_obs": self.n_obs,
            "params": self.params,
            "mu": self.mu,
            "large_move_threshold": self.threshold,
            "var_z": self.var_z,
            "z_grid": list(self.z_grid),
            # JSON has no NaN: a diagnostic that came out undefined is stored as null.
            "diagnostics": {k: _finite_or_none(v) for k, v in self.diagnostics.items()},
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Fit:
        return cls(
            model=config["model"],
            model_version=config["model_version"],
            asset=Asset(config["asset"]),
            block_start=datetime.fromisoformat(config["block_start"]),
            train_start=datetime.fromisoformat(config["train_start"]),
            train_end=datetime.fromisoformat(config["train_end"]),
            n_obs=int(config["n_obs"]),
            params={k: float(v) for k, v in config["params"].items()},
            mu=float(config["mu"]),
            threshold=float(config["large_move_threshold"]),
            var_z={k: float(v) for k, v in config["var_z"].items()},
            z_grid=tuple(float(v) for v in config["z_grid"]),
            diagnostics=dict(config.get("diagnostics", {})),
        )


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Arm1Model(Protocol):
    name: str  # the ledger's ``model``
    version: str
    outputs: tuple[Output, ...]

    def fit(self, train: pd.DataFrame, asset: Asset, block: datetime) -> Fit | None:
        """Fit on the block's training bars (hourly grid with ``ret`` and ``rv``), or None
        when there are too few."""
        ...

    def predict(self, fit: Fit, bars: pd.DataFrame, targets: pd.DatetimeIndex) -> pd.DataFrame:
        """Forecasts for ``targets`` (one block), one column per output key, NaN where the data
        are too thin. For each target t only bars up to t - LAG are used."""
        ...


# --- shared pieces -------------------------------------------------------------------------


def check_grid(bars: pd.DataFrame) -> None:
    index = bars.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise TypeError("bars need a UTC DatetimeIndex")
    if len(index) > 1 and not (np.diff(index.asi8) == np.diff(index.asi8)[0]).all():
        raise ValueError("bars must be a complete hourly grid (missing bars as NaN rows)")
    if len(index) > 1 and index[1] - index[0] != BAR:
        raise ValueError("bars must be hourly")


def check_block(fit: Fit, targets: pd.DatetimeIndex) -> None:
    """Forecasts come from the fit of their own block."""
    blocks = {block_start(t.to_pydatetime()) for t in targets}
    if blocks - {fit.block_start}:
        raise ValueError(f"targets outside the block of {fit.block_start}: {sorted(blocks)}")


def recent_coverage(bars: pd.DataFrame, targets: pd.DatetimeIndex, cfg: Arm1Section) -> np.ndarray:
    """Whether enough of the last day's bars (ending at each target's origin) have a return,
    as the baseline requires (``min_coverage`` of 24)."""
    count = bars["ret"].notna().rolling(24, min_periods=1).sum()
    origin = count.reindex(targets - LAG * BAR).to_numpy(dtype=float)
    return np.asarray(np.nan_to_num(origin, nan=0.0) >= min_bars(24, cfg.min_coverage))


def large_move_threshold(train: pd.DataFrame, quantile: float) -> float:
    returns = train["ret"].dropna().to_numpy()
    return float(np.quantile(np.abs(returns), quantile))


def finish_fit(
    *,
    model: str,
    version: str,
    asset: Asset,
    block: datetime,
    train: pd.DataFrame,
    sigma: np.ndarray,  # in-sample two-step forecast for every training bar (NaN allowed)
    mu: float,
    params: dict[str, float],
    cfg: Arm1Section,
    diagnostics: dict[str, Any] | None = None,
) -> Fit:
    returns = train["ret"].to_numpy(dtype=float)
    z = (returns - mu) / sigma
    usable = np.isfinite(z)
    usable[: min(BURN_IN, len(usable) // 4)] = False
    z = z[usable]
    if len(z) < 100:
        raise ValueError(f"{model}: only {len(z)} standardized returns for the tails")
    if not all(math.isfinite(v) for v in params.values()):
        raise ValueError(f"{model}: non-finite parameters {params}")
    levels = np.linspace(0.0, 1.0, cfg.tail_grid)
    return Fit(
        model=model,
        model_version=version,
        asset=asset,
        block_start=block,
        train_start=train.index[0].to_pydatetime(),
        train_end=train.index[-1].to_pydatetime(),
        n_obs=int(np.isfinite(returns).sum()),
        params=params,
        mu=mu,
        threshold=large_move_threshold(train, cfg.large_move_quantile),
        var_z={str(a): float(np.quantile(z, a)) for a in cfg.var_levels},
        z_grid=tuple(float(v) for v in np.quantile(z, levels)),
        diagnostics={"n_z": len(z), **(diagnostics or {})},
    )


def z_cdf(fit: Fit, x: np.ndarray) -> np.ndarray:
    """Empirical CDF of the training z, interpolated on the stored quantile grid."""
    grid = np.asarray(fit.z_grid)
    levels = np.linspace(0.0, 1.0, len(grid))
    return np.asarray(np.interp(x, grid, levels, left=0.0, right=1.0), dtype=float)


def outputs_from_sigma(
    fit: Fit, sigma: np.ndarray, outputs: Sequence[Output], targets: pd.DatetimeIndex
) -> pd.DataFrame:
    """RV, VaR and large-move probability from a volatility forecast and the fit's tails.

    VaR is a positive loss in log-return units: a violation is a return below -VaR."""
    out: dict[str, np.ndarray] = {}
    for o in outputs:
        if o.target is Target.RV_1H:
            out[o.key] = sigma
        elif o.target is Target.VAR_1H:
            assert o.var_level is not None
            out[o.key] = -(fit.mu + sigma * fit.var_z[str(o.var_level)])
        elif o.target is Target.P_LARGE_MOVE_1H:
            with np.errstate(divide="ignore", invalid="ignore"):
                upper = (fit.threshold - fit.mu) / sigma
                lower = (-fit.threshold - fit.mu) / sigma
            p = z_cdf(fit, lower) + 1.0 - z_cdf(fit, upper)
            out[o.key] = np.where(np.isfinite(sigma), np.clip(p, 0.0, 1.0), np.nan)
        else:  # pragma: no cover - outputs are fixed per model
            raise ValueError(f"unsupported output {o}")
    return pd.DataFrame(out, index=targets)
