"""Arm 1 in the hourly loop: one experiment package per model, asset and weekly block.

The first run that forecasts a bar of a new block fits each model on the block's training
window (``base.training_end``) and appends the fit to the ledger as an experiment package:
parameters, tails, large-move threshold, the arm 1 settings and a manifest of the raw files
read. Every forecast of the block cites that package (``predictions.package_seq``), and the
loop reads the parameters back from it instead of fitting again. So the parameters behind
each forecast are fixed in the ledger before the forecast, and the validator can refit them
from the manifest and compare.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg

from cryptonlp import ledger
from cryptonlp.features import bars
from cryptonlp.forecast.arm1 import base
from cryptonlp.forecast.arm1.base import Arm1Model, Fit
from cryptonlp.forecast.config import Arm1Section
from cryptonlp.forecast.provenance import write_manifest
from cryptonlp.rawstore import RawStore
from cryptonlp.schemas.common import BAR, Asset, Phase
from cryptonlp.schemas.ledger import ExperimentPackage

log = logging.getLogger(__name__)

PACKAGE_SQL = "SELECT seq, config FROM ledger.experiment_packages WHERE package_key = %s"


def package_key(model: Arm1Model, asset: Asset, block: datetime, phase: Phase) -> str:
    return f"arm1.{model.name}.v{model.version}.{asset.value}.{block:%Y-%m-%d}.{phase.value}"


def commit_of(run_id: str) -> str:
    commit = run_id.split("-")[1]
    if commit == "nogit":
        raise ValueError("arm 1 packages need the git commit of the code (run outside a repo?)")
    return commit


@dataclass(frozen=True)
class FitPackage:
    seq: int
    fit: Fit
    created: bool  # appended by this call (the block's first forecast)


class Arm1Runner:
    """Finds or creates each model's package for a block. One per loop run."""

    def __init__(
        self,
        cfg: Arm1Section,
        models: list[Arm1Model],
        store: RawStore,
        ledger_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.models = models
        self.store = store
        self.ledger_dir = ledger_dir
        self._training: dict[tuple[Asset, datetime], bars.HourlyBars] = {}

    @property
    def history_hours(self) -> int:
        return self.cfg.history_hours

    def training(
        self, conn: psycopg.Connection, asset: Asset, block: datetime, now: datetime
    ) -> bars.HourlyBars:
        key = (asset, block)
        if key not in self._training:
            end = base.training_end(block) + BAR
            if end > now:
                raise ValueError(f"block {block} cannot be fitted before {end}")
            self._training[key] = bars.load_hourly(
                conn, self.store, asset, self.cfg.train_start, end, now=now
            )
        return self._training[key]

    def package(
        self,
        conn: psycopg.Connection,
        model: Arm1Model,
        asset: Asset,
        block: datetime,
        *,
        now: datetime,
        phase: Phase,
        run_id: str,
    ) -> FitPackage | None:
        """The block's package for ``model``, created (fit and appended) if it is missing.
        None when the training window is still too short."""
        key = package_key(model, asset, block, phase)
        row = conn.execute(PACKAGE_SQL, (key,)).fetchone()
        if row is not None:
            return FitPackage(row[0], Fit.from_config(row[1]["fit"]), created=False)
        training = self.training(conn, asset, block, now)
        fit = model.fit(training.frame, asset, block)
        if fit is None:
            log.info("%s %s: too little training data for block %s", model.name, asset.value, block)
            return None
        manifest = write_manifest(training.files, self.ledger_dir)
        config: dict[str, Any] = {
            "fit": fit.to_config(),
            "arm1": self.cfg.model_dump(mode="json"),
            "data_manifest": {"path": manifest.path, "files": len(training.files)},
        }
        entry = ledger.append(
            conn,
            ExperimentPackage(
                package_key=key,
                description=(
                    f"Arm 1 {model.name} v{model.version} fitted for {asset.value}, block from "
                    f"{block:%Y-%m-%d %H:%M} UTC, on bars {fit.train_start:%Y-%m-%d %H:%M} to "
                    f"{fit.train_end:%Y-%m-%d %H:%M} UTC ({fit.n_obs} returns)"
                ),
                code_commit=commit_of(run_id),
                data=(manifest,),
                model_revision=f"{model.name} v{model.version}",
                config=config,
                phase=phase,
                run_id=run_id,
            ),
        )
        log.info("package %s appended as seq %d", key, entry.seq)
        return FitPackage(entry.seq, fit, created=True)
