"""Losses for one forecast, written with its score in the ledger.

Volatility, on the variance scale (Patton 2011, claims L32):

``se_variance`` is the squared error between realized and forecast variance; its mean over a
period is the MSE. ``qlike`` is s/h - ln(s/h) - 1 with s the realized and h the forecast
variance: 0 for a perfect forecast, and it penalizes underestimating more than overestimating.
It is undefined when either is 0, and then left out.

VaR (a positive loss; the realized value is the bar's log return r): ``exceedance`` is 1 when
r < -VaR, and ``tick`` is the quantile loss (level - 1{r < -VaR})(r + VaR).

Large-move probability (the realized value is 1 when |r| passed the threshold of the
forecast's experiment package, else 0): ``brier`` is (p - outcome)^2.
"""

from __future__ import annotations

import math


def volatility_losses(realized: float, forecast: float) -> dict[str, float]:
    rv2, h = realized * realized, forecast * forecast
    losses = {"se_variance": (rv2 - h) ** 2}
    if rv2 > 0 and h > 0:
        ratio = rv2 / h
        losses["qlike"] = ratio - math.log(ratio) - 1
    return losses


def var_losses(realized_return: float, var: float, level: float) -> dict[str, float]:
    hit = realized_return < -var
    return {
        "exceedance": 1.0 if hit else 0.0,
        "tick": (level - (1.0 if hit else 0.0)) * (realized_return + var),
    }


def probability_losses(outcome: float, prob: float) -> dict[str, float]:
    return {"brier": (prob - outcome) ** 2}
