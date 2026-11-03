"""The hourly loop (step 3b): classify new posts, compute features, forecast the next bar into
the ledger before it opens, and score every forecast whose bar has closed.

Forecasters: the baseline, and arm 1 (step 5) when ``[arm1] in_loop`` is true. Arm 1 models
are fitted once per weekly block, into an experiment package each forecast cites
(``cryptonlp.forecast.arm1.live``).

Every step reads the time from the database clock (``ledger.clock()``), the same clock the
ledger checks deadlines against. In the test database a session can set that clock, so
``simulate`` runs a recorded day through the loop in minutes; anywhere else the ledger refuses a
simulated time.

A failing step doesn't stop the others: a classifier error leaves the features waiting but
still lets the baseline forecast and the scores go in. The run then fails as a whole, so the
error shows up in ``ops.job_runs`` and on the dashboard.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

from cryptonlp import ledger
from cryptonlp.classify import Classifier, classify_pending
from cryptonlp.features import bars, nlp
from cryptonlp.forecast import arm1
from cryptonlp.forecast.arm1.live import Arm1Runner
from cryptonlp.forecast.base import Forecaster, ForecastRequest
from cryptonlp.forecast.baseline import YesterdayVolatility
from cryptonlp.forecast.config import LoopConfig
from cryptonlp.forecast.scoring import probability_losses, var_losses, volatility_losses
from cryptonlp.ledger.writer import LatePredictionError
from cryptonlp.rawstore import RawStore
from cryptonlp.schemas.common import BAR, Asset, Phase, floor_to_bar
from cryptonlp.schemas.forecasts import Arm, Forecast, Target
from cryptonlp.schemas.ledger import Score
from cryptonlp.settings import Settings

log = logging.getLogger(__name__)

# One loop at a time; a second one skips instead of racing the first for the same bar.
LOCK_ID = 7352460150482437303
# Targets the loop scores. Next-day RV (exploratory) is not forecast yet.
SCORED_TARGETS = (Target.RV_1H, Target.VAR_1H, Target.P_LARGE_MOVE_1H)

ISSUED_SQL = """
SELECT 1 FROM ledger.predictions
WHERE phase = %s AND asset = %s AND arm = %s AND model = %s AND target = %s
  AND var_level IS NOT DISTINCT FROM %s AND bar_open = %s
"""

# With the large-move threshold of the forecast's experiment package (arm 1).
DUE_SQL = """
SELECT p.seq, p.asset, p.target, p.bar_open, p.value, p.phase, p.var_level,
       (k.config -> 'fit' ->> 'large_move_threshold')::float8
FROM ledger.predictions p
LEFT JOIN ledger.experiment_packages k ON k.seq = p.package_seq
WHERE p.target = ANY(%s)
  AND p.bar_open + ledger.horizon(p.target) <= %s
  AND NOT EXISTS (SELECT 1 FROM ledger.scores s WHERE s.prediction_seq = p.seq)
ORDER BY p.seq
"""


@dataclass
class LoopStats:
    started_at: str = ""
    labeled: int = 0
    features_written: int = 0
    features_waiting: int = 0
    forecasts: int = 0
    already_issued: int = 0
    no_forecast: int = 0  # not enough data (missing candles)
    late: int = 0  # refused by the database clock, and logged as a violation
    packages: int = 0  # arm 1 fits appended (a new weekly block)
    scored: int = 0
    score_waiting: int = 0  # bar closed, but candles still missing
    target_bar: str = ""
    errors: list[str] = field(default_factory=list)
    skipped_locked: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoopError(RuntimeError):
    def __init__(self, stats: LoopStats) -> None:
        super().__init__(f"loop steps failed: {stats.errors}")
        self.stats = stats


class HourlyLoop:
    def __init__(
        self,
        config: LoopConfig,
        classifier: Classifier,
        forecasters: Sequence[Forecaster],
        store: RawStore,
        *,
        phase: Phase,
        run_id: str,
        arm1_runner: Arm1Runner | None = None,
    ) -> None:
        self.config = config
        self.classifier = classifier
        self.forecasters = list(forecasters)
        self.store = store
        self.phase = phase
        self.run_id = run_id
        self.arm1 = arm1_runner

    @classmethod
    def default(
        cls,
        config: LoopConfig,
        classifier: Classifier,
        store: RawStore,
        *,
        phase: Phase,
        run_id: str,
        ledger_dir: Path | None = None,
    ) -> HourlyLoop:
        """The loop with the forecasters built so far: the baseline, and arm 1 unless
        ``[arm1] in_loop`` is false. Arm 1's data manifests go under ``ledger_dir``."""
        baseline = YesterdayVolatility(config.baseline.window_hours, config.baseline.min_bars)
        runner = None
        if config.arm1.in_loop and config.arm1.models:
            runner = Arm1Runner(
                config.arm1,
                arm1.models(config.arm1),
                store,
                ledger_dir or Settings().ledger_path,
            )
        return cls(
            config, classifier, [baseline], store, phase=phase, run_id=run_id, arm1_runner=runner
        )

    def run(self, conn: psycopg.Connection) -> LoopStats:
        """One cycle. ``conn`` must be autocommit (each ledger append is its own transaction)."""
        stats = LoopStats(started_at=ledger.database_clock(conn).isoformat())
        row = conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_ID,)).fetchone()
        if not (row and row[0]):
            log.warning("another loop is running; skipping")
            stats.skipped_locked = True
            return stats
        try:
            steps: list[tuple[str, Callable[[psycopg.Connection, LoopStats], None]]] = [
                ("classify", self.classify),
                ("features", self.features),
                ("forecast", self.forecast),
                ("score", self.score),
            ]
            for name, step in steps:
                if name == "features" and any(e.startswith("classify") for e in stats.errors):
                    continue  # posts would be missing labels; the bars wait for the next run
                try:
                    step(conn, stats)
                except Exception as exc:
                    log.exception("loop step %s failed", name)
                    stats.errors.append(f"{name}: {type(exc).__name__}: {exc}")
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))
        return stats

    # --- steps -------------------------------------------------------------------------------

    def classify(self, conn: psycopg.Connection, stats: LoopStats) -> None:
        stats.labeled = classify_pending(
            conn,
            self.classifier,
            now=ledger.database_clock(conn),
            run_id=self.run_id,
            batch_size=self.config.loop.classify_batch,
        )

    def features(self, conn: psycopg.Connection, stats: LoopStats) -> None:
        cfg = self.config.loop
        stats.features_written, stats.features_waiting = nlp.update(
            conn,
            cfg.assets,
            now=ledger.database_clock(conn),
            settle=cfg.settle,
            lookback=cfg.lookback,
            classifier_version=self.classifier.model_version,
            n_tasks=len(self.classifier.tasks),
            run_id=self.run_id,
        )

    def forecast(self, conn: psycopg.Connection, stats: LoopStats) -> None:
        now = ledger.database_clock(conn)
        target_bar = floor_to_bar(now) + BAR
        stats.target_bar = target_bar.isoformat()
        history = max(f.history_hours for f in self.forecasters)
        if self.arm1 is not None:
            history = max(history, self.arm1.history_hours)
        newest_closed = floor_to_bar(now) - BAR
        for asset in self.config.loop.assets:
            todo = [f for f in self.forecasters if not self._issued(conn, asset, f, target_bar)]
            stats.already_issued += len(self.forecasters) - len(todo)
            arm1_todo = self._arm1_todo(conn, asset, target_bar, stats)
            if not todo and not arm1_todo:
                continue
            start = newest_closed - (history - 1) * BAR
            hourly = bars.load_hourly(conn, self.store, asset, start, newest_closed + BAR, now=now)
            request = ForecastRequest(
                asset=asset,
                bar_open=target_bar,
                issued_at=now,
                phase=self.phase,
                run_id=self.run_id,
                rv=_finite(hourly.frame["rv"]),
                returns=_finite(hourly.frame["ret"]),
                history_start=start,
                features=nlp.load(
                    conn,
                    asset,
                    since=start,
                    now=now,
                    classifier_version=self.classifier.model_version,
                ),
            )
            for forecaster in todo:
                self._record(conn, forecaster.forecast(request), None, stats)
            if arm1_todo:
                self._forecast_arm1(conn, request, arm1_todo, stats)

    def _record(
        self,
        conn: psycopg.Connection,
        forecast: Forecast | None,
        package_seq: int | None,
        stats: LoopStats,
    ) -> None:
        if forecast is None:
            stats.no_forecast += 1
            return
        try:
            ledger.record_prediction(conn, forecast, package_seq=package_seq)
            stats.forecasts += 1
        except LatePredictionError:
            log.error(
                "late forecast for %s %s (%s): refused",
                forecast.asset.value,
                forecast.bar_open,
                forecast.model,
            )
            stats.late += 1

    def _arm1_todo(
        self, conn: psycopg.Connection, asset: Asset, bar_open: datetime, stats: LoopStats
    ) -> list[tuple[arm1.Arm1Model, list[arm1.Output]]]:
        if self.arm1 is None:
            return []
        todo = []
        for model in self.arm1.models:
            missing = [
                o
                for o in model.outputs
                if not self._issued_key(
                    conn, asset, Arm.STATS, model.name, o.target, o.var_level, bar_open
                )
            ]
            stats.already_issued += len(model.outputs) - len(missing)
            if missing:
                todo.append((model, missing))
        return todo

    def _forecast_arm1(
        self,
        conn: psycopg.Connection,
        request: ForecastRequest,
        todo: list[tuple[arm1.Arm1Model, list[arm1.Output]]],
        stats: LoopStats,
    ) -> None:
        assert self.arm1 is not None
        block = arm1.block_start(request.bar_open)
        frame = request.frame()
        targets = pd.DatetimeIndex([request.bar_open])
        for model, outputs in todo:
            package = self.arm1.package(
                conn,
                model,
                request.asset,
                block,
                now=request.issued_at,
                phase=self.phase,
                run_id=self.run_id,
            )
            if package is None:
                stats.no_forecast += len(outputs)
                continue
            stats.packages += package.created
            values = model.predict(package.fit, frame, targets).iloc[0]
            for o in outputs:
                value = float(values[o.key])
                forecast = None
                if math.isfinite(value):
                    forecast = Forecast(
                        asset=request.asset,
                        arm=Arm.STATS,
                        model=model.name,
                        model_version=model.version,
                        target=o.target,
                        var_level=o.var_level,
                        bar_open=request.bar_open,
                        value=value,
                        issued_at=request.issued_at,
                        phase=self.phase,
                        run_id=self.run_id,
                    )
                self._record(conn, forecast, package.seq, stats)

    def score(self, conn: psycopg.Connection, stats: LoopStats) -> None:
        now = ledger.database_clock(conn)
        due = conn.execute(DUE_SQL, ([t.value for t in SCORED_TARGETS], now)).fetchall()
        by_asset: dict[Asset, list[tuple[Any, ...]]] = defaultdict(list)
        for row in due:
            by_asset[Asset(row[1])].append(row)
        for asset, rows in by_asset.items():
            closes = bars.load_for_bars(conn, self.store, asset, {r[3] for r in rows}, now=now)
            for seq, _asset, target, bar_open, value, phase, level, threshold in rows:
                outcome = _realized(closes, Target(target), bar_open, value, level, threshold)
                if outcome is None:
                    stats.score_waiting += 1
                    continue
                realized, losses = outcome
                ledger.append(
                    conn,
                    Score(
                        prediction_seq=seq,
                        realized=realized,
                        losses=losses,
                        data=closes.files_for(bar_open),
                        phase=Phase(phase),
                        run_id=self.run_id,
                    ),
                )
                stats.scored += 1

    def _issued(
        self, conn: psycopg.Connection, asset: Asset, f: Forecaster, bar_open: datetime
    ) -> bool:
        return self._issued_key(conn, asset, f.arm, f.model, f.target, None, bar_open)

    def _issued_key(
        self,
        conn: psycopg.Connection,
        asset: Asset,
        arm: Arm,
        model: str,
        target: Target,
        var_level: float | None,
        bar_open: datetime,
    ) -> bool:
        row = conn.execute(
            ISSUED_SQL,
            (self.phase.value, asset.value, arm.value, model, target.value, var_level, bar_open),
        ).fetchone()
        return row is not None


def _finite(series: pd.Series) -> dict[datetime, float]:
    clean = series.dropna()
    return {
        t.to_pydatetime(): float(v)
        for t, v in zip(pd.DatetimeIndex(clean.index), clean, strict=True)
    }


def _realized(
    closes: bars.MinuteCloses,
    target: Target,
    bar_open: datetime,
    value: float,
    level: float | None,
    threshold: float | None,
) -> tuple[float, dict[str, float]] | None:
    """The realized value and losses of one forecast, or None while candles are missing."""
    if target is Target.RV_1H:
        rv = closes.realized_vol(bar_open)
        return None if rv is None else (rv, volatility_losses(rv, value))
    ret = closes.log_return(bar_open)
    if ret is None:
        return None
    if target is Target.VAR_1H:
        assert level is not None
        return ret, var_losses(ret, value, level)
    if threshold is None:
        log.error("large-move forecast for %s without a package threshold", bar_open)
        return None
    outcome = 1.0 if abs(ret) > threshold else 0.0
    return outcome, probability_losses(outcome, value)


def simulate(
    conn: psycopg.Connection,
    loop: HourlyLoop,
    start: datetime,
    end: datetime,
    *,
    minute: int = 25,
) -> list[LoopStats]:
    """Run the loop once an hour, at ``minute`` past, from ``start`` to ``end`` on the simulated
    clock. Test database only: ``set_simulated_clock`` refuses any other."""
    runs = []
    at = floor_to_bar(start) + timedelta(minutes=minute)
    if at < start:
        at += BAR
    try:
        while at <= end:
            ledger.set_simulated_clock(conn, at)
            runs.append(loop.run(conn))
            at += BAR
    finally:
        ledger.set_simulated_clock(conn, None)
    return runs
