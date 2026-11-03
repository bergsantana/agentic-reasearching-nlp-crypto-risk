"""Arm 1, statistics only: GARCH, GJR-GARCH, EGARCH, HAR and historical VaR (step 5).

``models`` builds the configured models. Each implements ``Arm1Model``: a weekly ``fit`` on an
expanding window and a ``predict`` that uses, for bar t, only bars up to t - 2. The live loop
(``live``) and the walk-forward harness (``cryptonlp.forecast.walkforward``) call the same two
methods, so a forecast in the ledger can be recomputed offline from its experiment package.
"""

from __future__ import annotations

from cryptonlp.forecast.arm1.base import LAG, Arm1Model, Fit, Output, block_start, training_end
from cryptonlp.forecast.arm1.garch import GarchModel
from cryptonlp.forecast.arm1.har import HarModel
from cryptonlp.forecast.arm1.historical import HistoricalVaR
from cryptonlp.forecast.config import Arm1Section


def models(cfg: Arm1Section) -> list[Arm1Model]:
    built: dict[str, Arm1Model] = {
        "garch": GarchModel("garch", cfg),
        "gjr": GarchModel("gjr", cfg),
        "egarch": GarchModel("egarch", cfg),
        "har": HarModel(cfg),
        "har_log": HarModel(cfg, log=True),
        "hist_var": HistoricalVaR(cfg),
    }
    return [built[name] for name in cfg.models]


__all__ = [
    "LAG",
    "Arm1Model",
    "Fit",
    "GarchModel",
    "HarModel",
    "HistoricalVaR",
    "Output",
    "block_start",
    "models",
    "training_end",
]
