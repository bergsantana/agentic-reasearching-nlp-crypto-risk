"""Walk-forward harness for arm 1 and the baseline (step 5; article, Section 4.4, Figure 3).

Expanding window, weekly blocks, no shuffling. For each block (Monday 00:00 UTC):

1. every model is fitted on the bars from ``train_start`` to block start - 2 hours, the
   newest bar the block's first forecast could see in the live loop;
2. each bar t of the block is forecast with the bars up to t - 2 only (``arm1.LAG``), using
   the same ``predict`` the live loop calls;
3. forecasts are scored against the bar's realized volatility, its return (VaR) and whether
   its absolute return passed the block's large-move threshold.

Development data only: nothing here writes to the ledger. Runs are logged to MLflow.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from cryptonlp.forecast import evaluation as ev
from cryptonlp.forecast.arm1 import LAG, Arm1Model, Fit, models
from cryptonlp.forecast.arm1.base import WEEK, check_grid, large_move_threshold, training_end
from cryptonlp.forecast.baseline import YesterdayVolatility
from cryptonlp.forecast.config import LoopConfig
from cryptonlp.forecast.provenance import manifest_text
from cryptonlp.schemas.common import BAR, Asset
from cryptonlp.schemas.forecasts import Arm, Target
from cryptonlp.schemas.ledger import DataFile, sha256_text

log = logging.getLogger(__name__)

FORECAST_COLUMNS = ["bar_open", "block", "arm", "model", "target", "var_level", "value"]


@dataclass
class WalkForwardResult:
    asset: Asset
    forecasts: pd.DataFrame  # long: one row per bar, model, target and VaR level
    realized: pd.DataFrame  # per bar: rv, ret, threshold, large
    fits: list[Fit] = field(default_factory=list)
    fit_seconds: dict[str, float] = field(default_factory=dict)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)


def baseline_values(
    bars: pd.DataFrame, targets: pd.DatetimeIndex, baseline: YesterdayVolatility
) -> np.ndarray:
    """ "Yesterday's volatility" for every target at once: the same rule as
    ``YesterdayVolatility.forecast`` (RMS of the RV of the window ending at t - 2)."""
    rv2 = bars["rv"] ** 2
    rms = np.sqrt(rv2.rolling(baseline.window_hours, min_periods=baseline.min_bars).mean())
    return np.asarray(rms.reindex(targets - LAG * BAR).to_numpy(dtype=float))


def _long(
    values: pd.DataFrame, block: datetime, arm: Arm, model: str, outputs: dict[str, Any]
) -> list[pd.DataFrame]:
    frames = []
    for key, (target, level) in outputs.items():
        frames.append(
            pd.DataFrame(
                {
                    "bar_open": values.index,
                    "block": block,
                    "arm": arm.value,
                    "model": model,
                    "target": target,
                    "var_level": np.nan if level is None else level,
                    "value": values[key].to_numpy(dtype=float),
                }
            )
        )
    return frames


def run(
    bars: pd.DataFrame,
    asset: Asset,
    config: LoopConfig,
    arm1_models: Sequence[Arm1Model] | None = None,
) -> WalkForwardResult:
    """Walk forward over ``bars`` (complete hourly grid from ``arm1.train_start``)."""
    check_grid(bars)
    cfg, wf = config.arm1, config.walkforward
    arm1_models = list(arm1_models) if arm1_models is not None else models(cfg)
    baseline = YesterdayVolatility(config.baseline.window_hours, config.baseline.min_bars)
    have = bars.index[bars["ret"].notna().to_numpy()]
    if len(have) == 0:
        raise ValueError("no bars with data")
    end = pd.Timestamp(wf.end) if wf.end else have[-1] + BAR
    frames: list[pd.DataFrame] = []
    realized: list[pd.DataFrame] = []
    fits: list[Fit] = []
    seconds: dict[str, float] = {m.name: 0.0 for m in arm1_models}
    block = pd.Timestamp(wf.test_start)
    while block < end:
        block_dt = block.to_pydatetime()
        in_block = (bars.index >= block) & (bars.index < min(block + WEEK, end))
        targets = pd.DatetimeIndex(bars.index[in_block])
        train = bars.loc[pd.Timestamp(cfg.train_start) : pd.Timestamp(training_end(block_dt))]
        block += WEEK
        if len(targets) == 0 or len(train) == 0:
            continue
        visible = bars.loc[: targets.max() - LAG * BAR]
        threshold = large_move_threshold(train, cfg.large_move_quantile)
        ret = bars["ret"].reindex(targets)
        realized.append(
            pd.DataFrame(
                {
                    "rv": bars["rv"].reindex(targets).to_numpy(),
                    "ret": ret.to_numpy(),
                    "threshold": threshold,
                    "large": (ret.abs() > threshold).astype(float).where(ret.notna()).to_numpy(),
                },
                index=targets,
            )
        )
        base = pd.DataFrame({"rv_1h": baseline_values(visible, targets, baseline)}, index=targets)
        frames += _long(
            base, block_dt, Arm.BASELINE, baseline.model, {"rv_1h": (Target.RV_1H.value, None)}
        )
        for model in arm1_models:
            started = time.perf_counter()
            fit = model.fit(train, asset, block_dt)
            seconds[model.name] += time.perf_counter() - started
            if fit is None:
                continue
            fits.append(fit)
            values = model.predict(fit, visible, targets)
            outputs = {o.key: (o.target.value, o.var_level) for o in model.outputs}
            frames += _long(values, block_dt, Arm.STATS, model.name, outputs)
        log.info(
            "%s block %s: %d bars, %d fits", asset.value, block_dt.date(), len(targets), len(fits)
        )
    forecasts = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FORECAST_COLUMNS)
    )
    real = (
        pd.concat(realized)
        if realized
        else pd.DataFrame(columns=["rv", "ret", "threshold", "large"])
    )
    real.index.name = "bar_open"
    result = WalkForwardResult(asset, forecasts, real, fits, seconds)
    result.metrics, result.comparisons = evaluate(result, lags=wf.nw_lags)
    return result


# --- evaluation ----------------------------------------------------------------------------


def _scored(result: WalkForwardResult) -> pd.DataFrame:
    df = result.forecasts.merge(
        result.realized[["rv", "ret", "large"]], left_on="bar_open", right_index=True, how="left"
    )
    df["loss_se"] = np.nan
    df["loss_qlike"] = np.nan
    df["loss_tick"] = np.nan
    df["loss_brier"] = np.nan
    df["hit"] = np.nan
    rv = df["target"] == Target.RV_1H.value
    df.loc[rv, "loss_se"] = ev.se_variance(
        df.loc[rv, "rv"].to_numpy(), df.loc[rv, "value"].to_numpy()
    )
    df.loc[rv, "loss_qlike"] = ev.qlike(df.loc[rv, "rv"].to_numpy(), df.loc[rv, "value"].to_numpy())
    var = df["target"] == Target.VAR_1H.value
    sub = df[var]
    r, v, a = (sub[c].to_numpy(dtype=float) for c in ("ret", "value", "var_level"))
    df.loc[var, "loss_tick"] = ev.tick_loss(r, v, a)
    df.loc[var, "hit"] = np.where(np.isfinite(r) & np.isfinite(v), ev.var_hits(r, v), np.nan)
    p = df["target"] == Target.P_LARGE_MOVE_1H.value
    df.loc[p, "loss_brier"] = ev.brier(df.loc[p, "value"].to_numpy(), df.loc[p, "large"].to_numpy())
    return df


def _metric_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (model, target, level), g in df.groupby(["model", "target", "var_level"], dropna=False):
        g = g.sort_values("bar_open")
        row: dict[str, Any] = {"model": model, "target": target, "var_level": level}
        if target == Target.RV_1H.value:
            ok = g["value"].notna() & g["rv"].notna()
            row |= {
                "n": int(ok.sum()),
                "mse_variance": float(g.loc[ok, "loss_se"].mean()),
                "qlike": float(g.loc[ok, "loss_qlike"].mean()),
                "mean_forecast": float(g.loc[ok, "value"].mean()),
                "mean_realized": float(g.loc[ok, "rv"].mean()),
            }
        elif target == Target.VAR_1H.value:
            ok = g["hit"].notna()
            hits = g.loc[ok, "hit"].to_numpy(dtype=bool)
            alpha = float(cast(float, level))
            ind, cc = ev.christoffersen(hits, alpha)
            uc = ev.kupiec(hits, alpha)
            row |= {
                "n": int(ok.sum()),
                "violation_rate": float(hits.mean()) if len(hits) else math.nan,
                "kupiec_lr": uc.stat,
                "kupiec_p": uc.pvalue,
                "christoffersen_ind_lr": ind.stat,
                "christoffersen_ind_p": ind.pvalue,
                "christoffersen_cc_lr": cc.stat,
                "christoffersen_cc_p": cc.pvalue,
                "tick_loss": float(g.loc[ok, "loss_tick"].mean()),
            }
        else:
            ok = g["value"].notna() & g["large"].notna()
            dec = ev.murphy(g.loc[ok, "value"].to_numpy(), g.loc[ok, "large"].to_numpy())
            row |= {
                "n": dec.n,
                "brier": dec.brier,
                "reliability": dec.reliability,
                "resolution": dec.resolution,
                "uncertainty": dec.uncertainty,
                "mean_forecast": float(g.loc[ok, "value"].mean()),
                "event_rate": float(g.loc[ok, "large"].mean()),
            }
        rows.append(row)
    return rows


def _compare(
    df: pd.DataFrame, target: str, level: float | None, loss: str, reference: str, lags: int | None
) -> list[dict[str, Any]]:
    sub = df[(df["target"] == target)]
    sub = sub[sub["var_level"].isna()] if level is None else sub[sub["var_level"] == level]
    wide = sub.pivot_table(index="bar_open", columns="model", values=loss, aggfunc="first")
    if reference not in wide:
        return []
    rows = []
    for model in wide.columns:
        if model == reference:
            continue
        both = wide[[reference, model]].dropna()
        res = ev.diebold_mariano(both[reference].to_numpy(), both[model].to_numpy(), lags=lags)
        rows.append(
            {
                "target": target,
                "var_level": level,
                "loss": loss.removeprefix("loss_"),
                "model": model,
                "reference": reference,
                "mean_loss_model": float(both[model].mean()) if len(both) else math.nan,
                "mean_loss_reference": float(both[reference].mean()) if len(both) else math.nan,
                **{f"dm_{k}": v for k, v in res.as_dict().items()},
            }
        )
    return rows


def evaluate(
    result: WalkForwardResult, *, lags: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics per model and target, and Diebold-Mariano comparisons: each RV forecast against
    the baseline (QLIKE and squared error), each VaR and probability forecast against
    historical VaR (tick loss, Brier). ``dm_mean_diff`` > 0 means the model beat the reference."""
    if result.forecasts.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = _scored(result)
    metrics = pd.DataFrame(_metric_rows(df))
    baseline = df.loc[df["arm"] == Arm.BASELINE.value, "model"].iloc[0]
    hist = next((m for m in df["model"].unique() if str(m).startswith("hist_var")), None)
    rows: list[dict[str, Any]] = []
    for loss in ("loss_qlike", "loss_se"):
        rows += _compare(df, Target.RV_1H.value, None, loss, baseline, lags)
    if hist is not None:
        for level in sorted(
            df.loc[df["target"] == Target.VAR_1H.value, "var_level"].dropna().unique()
        ):
            rows += _compare(df, Target.VAR_1H.value, float(level), "loss_tick", hist, lags)
        rows += _compare(df, Target.P_LARGE_MOVE_1H.value, None, "loss_brier", hist, lags)
    return metrics, pd.DataFrame(rows)


def summary(result: WalkForwardResult) -> pd.DataFrame:
    """RV forecasts ranked by QLIKE, with the Diebold-Mariano test against the baseline."""
    m = result.metrics
    rv = m[m["target"] == Target.RV_1H.value][["model", "n", "qlike", "mse_variance"]]
    c = result.comparisons
    if not c.empty:
        dm = c[(c["target"] == Target.RV_1H.value) & (c["loss"] == "qlike")]
        dm = dm[["model", "dm_mean_diff", "dm_stat", "dm_pvalue"]]
        rv = rv.merge(dm, on="model", how="left")
    return rv.sort_values("qlike").reset_index(drop=True)


# --- MLflow --------------------------------------------------------------------------------


def _metric_key(*parts: Any) -> str:
    return ".".join(
        str(p) for p in parts if p is not None and not (isinstance(p, float) and math.isnan(p))
    )


def _flat(prefix: str, value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        out: dict[str, str] = {}
        for k, v in value.items():
            out |= _flat(f"{prefix}.{k}" if prefix else str(k), v)
        return out
    return {prefix: json.dumps(value, default=str) if not isinstance(value, str) else value}


def fits_json(fits: Sequence[Fit]) -> list[dict[str, Any]]:
    return [{k: v for k, v in f.to_config().items() if k != "z_grid"} for f in fits]


def log_to_mlflow(
    results: Sequence[WalkForwardResult],
    config: LoopConfig,
    *,
    run_id: str,
    data_files: Sequence[DataFile],
    tracking_uri: str,
    artifact_location: str | None = None,
) -> str:
    """One MLflow run for the invocation: config as parameters, every metric and comparison,
    and the forecasts, fits and data manifest as artifacts. Returns the MLflow run ID.

    ``artifact_location`` applies when the experiment is created on a local store (a server
    keeps its own); without it MLflow writes artifacts to ./mlruns in the working directory."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    name = config.walkforward.mlflow_experiment
    if artifact_location and mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)
    manifest = manifest_text(data_files)
    with mlflow.start_run(run_name=f"walkforward-{run_id}") as active:
        mlflow.set_tags(
            {
                "cryptonlp.run_id": run_id,
                "cryptonlp.phase": "development",
                "cryptonlp.code_commit": run_id.split("-", 1)[1],
                "assets": ",".join(r.asset.value for r in results),
                "data_manifest_sha256": sha256_text(manifest),
            }
        )
        params = _flat(
            "", config.model_dump(mode="json", include={"baseline", "arm1", "walkforward"})
        )
        mlflow.log_params(params)
        metrics: dict[str, float] = {}
        for r in results:
            for row in r.metrics.to_dict("records"):
                level = row["var_level"]
                for key, value in row.items():
                    if key in {"model", "target", "var_level"} or not isinstance(
                        value, int | float
                    ):
                        continue
                    if math.isfinite(value):
                        metrics[
                            _metric_key(r.asset.value, row["model"], row["target"], level, key)
                        ] = float(value)
            for row in r.comparisons.to_dict("records"):
                for key in ("dm_stat", "dm_pvalue", "dm_mean_diff"):
                    value = row[key]
                    if isinstance(value, float) and math.isfinite(value):
                        name = _metric_key(
                            r.asset.value,
                            row["model"],
                            row["target"],
                            row["var_level"],
                            f"vs_{row['reference']}",
                            row["loss"],
                            key,
                        )
                        metrics[name] = value
            for name, secs in r.fit_seconds.items():
                metrics[_metric_key(r.asset.value, name, "fit_seconds")] = secs
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data_manifest.json").write_text(manifest, encoding="utf-8")
            for r in results:
                d = root / r.asset.value
                d.mkdir()
                r.forecasts.to_parquet(d / "forecasts.parquet", index=False)
                r.realized.to_parquet(d / "realized.parquet")
                r.metrics.to_csv(d / "metrics.csv", index=False)
                r.comparisons.to_csv(d / "comparisons.csv", index=False)
                (d / "fits.json").write_text(
                    json.dumps(fits_json(r.fits), indent=1), encoding="utf-8"
                )
            mlflow.log_artifacts(str(root))
        return str(active.info.run_id)
