"""Step 5: hourly bars, arm 1 (GARCH family, HAR, historical VaR), the walk-forward harness, the
forecast tests, MLflow logging, and arm 1 in the hourly loop across a weekly block boundary.

The leakage tests are the ones that matter most: a forecast for bar t must not change when
anything from bar t-1 onwards changes (bar t-1 is still open when the loop issues it).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import pytest
from arch import arch_model

from cryptonlp import ledger
from cryptonlp.collectors.binance import write_klines
from cryptonlp.features import bars as bars_mod
from cryptonlp.features.bars import MinuteCloses, hourly_from_minutes
from cryptonlp.forecast import ForecastRequest, LeakageError, YesterdayVolatility
from cryptonlp.forecast import evaluation as ev
from cryptonlp.forecast import walkforward as wf
from cryptonlp.forecast.arm1 import LAG, Fit, block_start, models, training_end
from cryptonlp.forecast.arm1.garch import (
    SCALE,
    SPECS,
    GarchModel,
    Kind,
    backcast,
    egarch_shock_factor,
    filter_variance,
    two_step,
)
from cryptonlp.forecast.config import (
    Arm1Section,
    BaselineSection,
    LoopConfig,
    LoopSection,
    WalkForwardSection,
    load_loop_config,
)
from cryptonlp.forecast.loop import HourlyLoop, simulate
from cryptonlp.forecast.scoring import probability_losses, var_losses
from cryptonlp.rawstore import RawStore
from cryptonlp.schemas.common import BAR, Asset, Phase
from cryptonlp.schemas.ledger import DataFile
from cryptonlp.stubs import StubClassifier

RUN = "20261105T000000Z-0123456789ab"
MINUTE = timedelta(minutes=1)
START = datetime(2025, 1, 6, tzinfo=UTC)  # a Monday


def garch_bars(n: int, seed: int = 1, start: datetime = START) -> pd.DataFrame:
    """Hourly bars from a GARCH(1,1) with Student-t shocks; RV is the true volatility with
    multiplicative noise."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-7, 0.07, 0.9
    s2 = omega / (1 - alpha - beta)
    ret, rv = np.empty(n), np.empty(n)
    for t in range(n):
        s = math.sqrt(s2)
        ret[t] = s * rng.standard_t(5) / math.sqrt(5 / 3)
        rv[t] = s * math.exp(0.3 * rng.standard_normal() - 0.045)
        s2 = omega + alpha * ret[t] ** 2 + beta * s2
    index = pd.date_range(start, periods=n, freq="1h", name="bar_open")
    return pd.DataFrame({"ret": ret, "rv": rv, "volume": 1.0}, index=index)


def small_cfg(**kwargs: Any) -> Arm1Section:
    base: dict[str, Any] = {
        "train_start": START,
        "min_train_hours": 600,
        "filter_hours": 400,
        "har_windows": (6, 24, 72),
        "hist_window_hours": 72,
        "tail_grid": 201,
    }
    return Arm1Section(**(base | kwargs))


# --- hourly bars ---------------------------------------------------------------------------


def minute_prices(n: int, seed: int, start: datetime) -> dict[datetime, float]:
    rng = np.random.default_rng(seed)
    p, out = 100.0, {}
    for i in range(n):
        p *= math.exp(rng.normal(0, 0.001))
        out[start + i * MINUTE] = p
    return out


def test_hourly_bars_match_the_per_bar_definitions() -> None:
    t0 = START
    prices = minute_prices(4 * 60 + 1, 3, t0 - MINUTE)
    del prices[t0 + 2 * BAR + 17 * MINUTE]  # one candle missing in the third bar
    minutes = pd.DataFrame(
        {"close": list(prices.values()), "volume": 2.0},
        index=pd.DatetimeIndex(list(prices.keys())),
    )
    frame = hourly_from_minutes(minutes, t0, t0 + 4 * BAR)
    closes = MinuteCloses(prices, {})
    for i in range(4):
        bar = t0 + i * BAR
        rv, ret = closes.realized_vol(bar), closes.log_return(bar)
        if i == 2:
            assert rv is None and ret is None
            assert frame.loc[bar].isna().all()
            continue
        assert rv is not None and ret is not None
        row = frame.loc[pd.Timestamp(bar)]
        assert row["rv"] == pytest.approx(rv, rel=1e-12)
        assert row["ret"] == pytest.approx(ret, rel=1e-9)
        assert row["volume"] == 120.0
        # The hourly return is the sum of the 60 one-minute returns: price at T+1h over price at T.
        assert ret == pytest.approx(math.log(prices[bar + 59 * MINUTE] / prices[bar - MINUTE]))


# --- the variance filter and two-step forecasts --------------------------------------------


@pytest.fixture(scope="module")
def sim_returns() -> np.ndarray:
    return garch_bars(6000, seed=7)["ret"].to_numpy() * SCALE


@pytest.mark.parametrize("kind", ["garch", "gjr", "egarch"])
def test_filter_matches_arch(kind: Kind, sim_returns: np.ndarray) -> None:
    am = arch_model(sim_returns, mean="Constant", dist="normal", rescale=False, **SPECS[kind])  # type: ignore[arg-type]
    res = am.fit(disp="off", show_warning=False)
    params = {str(k): float(v) for k, v in res.params.items()}
    resid = sim_returns - params["mu"]
    assert backcast(kind, resid) == pytest.approx(am.volatility.backcast(resid), rel=1e-12)
    fixed = am.fix(res.params)
    h = filter_variance(kind, params, resid)
    theirs = np.asarray(fixed.conditional_volatility) ** 2
    # Same recursion. arch starts from a backcast of the residuals around the sample mean, this
    # code from the fitted mean, so they agree once the start has decayed.
    np.testing.assert_allclose(h[500:-1], theirs[500:], rtol=1e-8)
    if kind != "egarch":
        forecast = fixed.forecast(horizon=2, reindex=False).variance.to_numpy()[-1]
        assert h[-1] == pytest.approx(forecast[0], rel=1e-8)
        assert two_step(kind, params, np.array([h[-1]]))[0] == pytest.approx(forecast[1], rel=1e-8)


def test_egarch_two_step_factor_is_the_normal_expectation() -> None:
    z = np.random.default_rng(0).standard_normal(4_000_000)
    for alpha, gamma in [(0.15, -0.05), (0.3, 0.1), (0.0, 0.0)]:
        mc = np.mean(np.exp(alpha * (np.abs(z) - math.sqrt(2 / math.pi)) + gamma * z))
        assert egarch_shock_factor(alpha, gamma) == pytest.approx(mc, rel=2e-3)


def test_a_missing_return_enters_as_no_shock() -> None:
    params = {"mu": 0.0, "omega": 0.1, "alpha[1]": 0.1, "beta[1]": 0.8}
    resid = np.array([1.0, np.nan, 2.0])
    h = filter_variance("garch", params, resid)
    assert h[2] == pytest.approx(0.1 + 0.9 * h[1])


# --- models: fit, predict, the leakage rule -------------------------------------------------


@pytest.fixture(scope="module")
def sim() -> pd.DataFrame:
    frame = garch_bars(24 * 7 * 9)
    frame.iloc[700:705] = np.nan  # a gap in the candles
    return frame


def test_models_fit_and_round_trip_through_json(sim: pd.DataFrame) -> None:
    cfg = small_cfg()
    block = START + 6 * 7 * 24 * BAR
    train = sim.loc[: training_end(block)]
    for model in models(cfg):
        fit = model.fit(train, Asset.BTC, block)
        assert fit is not None, model.name
        assert fit.train_end == block - LAG * BAR
        assert fit.block_start == block and fit.n_obs == train["ret"].notna().sum()
        config = json.loads(json.dumps(fit.to_config()))
        assert Fit.from_config(config) == fit
        if model.name != cfg_hist_name(cfg):
            assert set(fit.var_z) == {"0.01", "0.05"} and fit.var_z["0.01"] < fit.var_z["0.05"] < 0
    # Too little training data: no fit.
    assert GarchModel("garch", cfg).fit(sim.iloc[:500], Asset.BTC, START + 3 * 168 * BAR) is None


def cfg_hist_name(cfg: Arm1Section) -> str:
    return f"hist_var_{cfg.hist_window_hours}h"


@pytest.mark.parametrize("name", ["garch", "gjr", "egarch", "har", "har_log", "hist_var"])
def test_a_forecast_never_sees_bar_t_minus_1_or_later(name: str, sim: pd.DataFrame) -> None:
    cfg = small_cfg(models=(name,))
    (model,) = models(cfg)
    block = START + 6 * 7 * 24 * BAR
    fit = model.fit(sim.loc[: training_end(block)], Asset.BTC, block)
    assert fit is not None
    target = block + 30 * BAR
    targets = pd.DatetimeIndex([target])
    seen = sim.loc[: target - LAG * BAR]
    base = model.predict(fit, seen, targets)
    # Wild values from bar t-1 onwards change nothing.
    future = sim.copy()
    future.loc[target - BAR :, ["ret", "rv"]] = [-0.5, 0.5]
    assert model.predict(fit, future, targets).equals(base)
    # The newest usable bar, t-2, does count.
    moved = seen.copy()
    moved.loc[target - LAG * BAR, ["ret", "rv"]] = [-0.5, 0.5]
    changed = model.predict(fit, moved, targets)
    assert not np.allclose(changed.to_numpy(), base.to_numpy())
    assert np.isfinite(base.to_numpy()).all()
    # Forecasting the whole block at once gives each bar what it gets alone.
    block_targets = pd.DatetimeIndex(
        sim.index[(sim.index >= block) & (sim.index < block + 50 * BAR)]
    )
    together = model.predict(fit, sim.loc[: block_targets[-1] - LAG * BAR], block_targets)
    for t in block_targets[::7]:
        alone = model.predict(fit, sim.loc[: t - LAG * BAR], pd.DatetimeIndex([t]))
        np.testing.assert_allclose(together.loc[[t]].to_numpy(), alone.to_numpy(), rtol=1e-12)
    # A fit only serves its own block.
    with pytest.raises(ValueError, match="outside the block"):
        model.predict(fit, sim, pd.DatetimeIndex([block + 8 * 24 * BAR]))


def test_thin_data_gives_no_forecast(sim: pd.DataFrame) -> None:
    cfg = small_cfg()
    block = START + 6 * 7 * 24 * BAR
    target = block + 30 * BAR
    holed = sim.copy()
    holed.loc[target - 10 * BAR : target - LAG * BAR, ["ret", "rv"]] = np.nan
    for model in models(cfg):
        fit = model.fit(sim.loc[: training_end(block)], Asset.BTC, block)
        assert fit is not None
        out = model.predict(fit, holed.loc[: target - LAG * BAR], pd.DatetimeIndex([target]))
        assert out.isna().all(axis=None), model.name


def test_tails_turn_volatility_into_var_and_probability(sim: pd.DataFrame) -> None:
    cfg = small_cfg(models=("garch",))
    (model,) = models(cfg)
    block = START + 6 * 7 * 24 * BAR
    fit = model.fit(sim.loc[: training_end(block)], Asset.BTC, block)
    assert fit is not None
    targets = pd.DatetimeIndex(sim.index[(sim.index >= block) & (sim.index < block + 168 * BAR)])
    out = model.predict(fit, sim.loc[: targets[-1] - LAG * BAR], targets)
    sigma = out["rv_1h"].to_numpy()
    np.testing.assert_allclose(
        out["var_1h@0.01"].to_numpy(), -(fit.mu + sigma * fit.var_z["0.01"]), rtol=1e-12
    )
    assert (out["var_1h@0.01"] > out["var_1h@0.05"]).all()
    p = out["p_large_move_1h"]
    assert p.between(0, 1).all()
    # More volatility, more chance of a large move.
    assert np.corrcoef(sigma, p)[0, 1] > 0.9


# --- walk-forward ----------------------------------------------------------------------------


def wf_config(**arm1: Any) -> LoopConfig:
    return LoopConfig(
        arm1=small_cfg(**arm1),
        walkforward=WalkForwardSection(test_start=START + 6 * 7 * 24 * BAR),
    )


@pytest.fixture(scope="module")
def walked(sim: pd.DataFrame) -> wf.WalkForwardResult:
    return wf.run(sim, Asset.BTC, wf_config())


def test_walk_forward_refits_weekly_on_an_expanding_window(
    walked: wf.WalkForwardResult, sim: pd.DataFrame
) -> None:
    blocks = sorted({f.block_start for f in walked.fits})
    assert blocks == [START + w * 7 * 24 * BAR for w in (6, 7, 8)]
    for fit in walked.fits:
        assert fit.train_start == START
        assert fit.train_end == fit.block_start - LAG * BAR
    # Every bar of every block was forecast by every model, with the block's own fit.
    fc = walked.forecasts
    assert fc["bar_open"].min() == blocks[0]
    assert fc["bar_open"].max() == sim.index[-1]
    assert (fc["block"] == fc["bar_open"].map(lambda t: block_start(t.to_pydatetime()))).all()
    n_outputs = 1 + 5 * 4 + 3  # baseline; 5 volatility models x 4 outputs; historical x 3
    assert len(fc) == 3 * 168 * n_outputs


def test_walk_forward_baseline_is_the_loop_baseline(
    walked: wf.WalkForwardResult, sim: pd.DataFrame
) -> None:
    base = walked.forecasts[walked.forecasts["arm"] == "baseline"].set_index("bar_open")["value"]
    forecaster = YesterdayVolatility(24, 20)
    for t in base.index[::37]:
        issued = t - BAR + timedelta(minutes=25)
        closed = sim.loc[: t - LAG * BAR, "rv"].dropna()
        request = ForecastRequest(
            asset=Asset.BTC,
            bar_open=t.to_pydatetime(),
            issued_at=issued.to_pydatetime(),
            phase=Phase.DEVELOPMENT,
            run_id=RUN,
            rv={k.to_pydatetime(): float(v) for k, v in closed.items()},
        )
        forecast = forecaster.forecast(request)
        assert forecast is not None
        assert base[t] == pytest.approx(forecast.value, rel=1e-12)


def test_walk_forward_metrics_and_comparisons(walked: wf.WalkForwardResult) -> None:
    m = walked.metrics.set_index(["model", "target", "var_level"], drop=False)
    rv = m[m["target"] == "rv_1h"]
    assert set(rv["model"]) == {
        "yesterday_rv_24h",
        "garch11",
        "gjr_garch11",
        "egarch11",
        "har_rv",
        "har_log_rv",
    }
    var = m[m["target"] == "var_1h"]
    assert var["violation_rate"].between(0, 0.2).all()
    assert var["kupiec_p"].between(0, 1).all()
    c = walked.comparisons
    assert {"qlike", "se", "tick", "brier"} <= set(c["loss"])
    # On GARCH data, GARCH beats "yesterday's volatility".
    garch = c[(c["model"] == "garch11") & (c["loss"] == "qlike")].iloc[0]
    assert garch["dm_mean_diff"] > 0
    table = wf.summary(walked)
    assert table.iloc[0]["model"] != "yesterday_rv_24h"


def test_walk_forward_scores_with_the_block_threshold(walked: wf.WalkForwardResult) -> None:
    real = walked.realized
    thresholds = {f.block_start: f.threshold for f in walked.fits}
    sample = real.iloc[::50]
    for bar, (_, row) in zip(pd.DatetimeIndex(sample.index), sample.iterrows(), strict=True):
        block = block_start(bar.to_pydatetime())
        assert row["threshold"] == pytest.approx(thresholds[block])
        assert row["large"] == float(abs(row["ret"]) > row["threshold"])


def test_walk_forward_logs_to_mlflow(walked: wf.WalkForwardResult, tmp_path: Path) -> None:
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    files = [DataFile(path="binance_klines/x.parquet", sha256="a" * 64)]
    run_id = wf.log_to_mlflow(
        [walked],
        wf_config(),
        run_id=RUN,
        data_files=files,
        tracking_uri=uri,
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    run = MlflowClient(uri).get_run(run_id)
    assert run.data.tags["cryptonlp.run_id"] == RUN
    assert run.data.params["arm1.filter_hours"] == "400"
    assert "BTC.garch11.rv_1h.qlike" in run.data.metrics
    assert "BTC.garch11.var_1h.0.01.kupiec_p" in run.data.metrics
    assert "BTC.garch11.rv_1h.vs_yesterday_rv_24h.qlike.dm_pvalue" in run.data.metrics
    names = {a.path for a in MlflowClient(uri).list_artifacts(run_id)}
    assert {"BTC", "data_manifest.json"} <= names
    assert (tmp_path / "artifacts").is_dir()


# --- tests and losses ----------------------------------------------------------------------


def test_kupiec() -> None:
    hits = np.zeros(1000, dtype=bool)
    hits[:10] = True
    assert ev.kupiec(hits, 0.01).stat == pytest.approx(0.0, abs=1e-9)
    hits[:25] = True
    x, n, a = 25, 1000, 0.01
    lr = -2 * (
        (n - x) * math.log(1 - a)
        + x * math.log(a)
        - (n - x) * math.log(1 - x / n)
        - x * math.log(x / n)
    )
    res = ev.kupiec(hits, a)
    assert res.stat == pytest.approx(lr) and res.pvalue < 0.001
    assert ev.kupiec(np.zeros(100, dtype=bool), 0.01).stat == pytest.approx(
        -2 * 100 * math.log(0.99)
    )


def test_christoffersen_catches_clustered_violations() -> None:
    hits = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
    # Transitions: n00 = 10, n01 = 4, n10 = 3, n11 = 2 (19 pairs).
    pi01, pi11, pi = 4 / 14, 2 / 5, 6 / 19
    ll1 = 10 * math.log(1 - pi01) + 4 * math.log(pi01) + 3 * math.log(1 - pi11) + 2 * math.log(pi11)
    ll0 = 13 * math.log(1 - pi) + 6 * math.log(pi)
    ind, cc = ev.christoffersen(hits, 0.05)
    assert ind.stat == pytest.approx(-2 * (ll0 - ll1))
    assert cc.stat == pytest.approx(ev.kupiec(hits[1:], 0.05).stat + ind.stat)
    clustered = np.zeros(2000, dtype=bool)
    for start in range(0, 2000, 200):
        clustered[start : start + 10] = True  # a 5% rate, but in runs
    ind_cluster, cc_cluster = ev.christoffersen(clustered, 0.05)
    assert ev.kupiec(clustered, 0.05).pvalue > 0.5
    assert ind_cluster.pvalue < 1e-6 and cc_cluster.pvalue < 1e-6


def test_diebold_mariano_is_a_hac_t_test_with_the_hln_correction() -> None:
    import statsmodels.api as sm  # type: ignore[import-untyped]

    rng = np.random.default_rng(1)
    e = rng.standard_normal(500)
    d = 0.1 + e + 0.5 * np.concatenate([[0], e[:-1]])  # MA(1) loss differences
    a, b = d + 1.0, np.ones_like(d)
    res = ev.diebold_mariano(a, b, horizon=2, lags=4)
    ols = sm.OLS(d, np.ones_like(d)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": False}
    )
    n, h = 500, 2
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    assert res.mean_diff == pytest.approx(d.mean())
    assert res.stat == pytest.approx(float(ols.tvalues[0]) * hln, rel=1e-9)
    assert ev.default_lags(500) == 5


def test_clark_west_adjusts_for_the_nested_model_noise() -> None:
    rng = np.random.default_rng(2)
    n = 3000
    x = rng.standard_normal(n)
    y = 0.3 * x + rng.standard_normal(n)
    small = np.zeros(n)
    large = 0.3 * x
    assert ev.clark_west(y, small, large).pvalue < 1e-6
    noisy = 0.05 * rng.standard_normal(n)  # a useless extra regressor
    assert ev.clark_west(y - 0.3 * x, small, noisy).pvalue > 0.01


def test_murphy_decomposition_adds_up() -> None:
    rng = np.random.default_rng(3)
    p = rng.choice([0.05, 0.15, 0.35, 0.75], size=5000)
    y = (rng.random(5000) < p).astype(float)
    dec = ev.murphy(p, y)
    assert dec.brier == pytest.approx(dec.reliability - dec.resolution + dec.uncertainty, rel=1e-12)


def test_losses_for_var_and_probabilities() -> None:
    assert var_losses(-0.03, 0.02, 0.01) == pytest.approx(
        {"exceedance": 1.0, "tick": (0.01 - 1) * (-0.01)}
    )
    assert var_losses(0.01, 0.02, 0.05) == pytest.approx({"exceedance": 0.0, "tick": 0.05 * 0.03})
    assert ev.tick_loss(np.array([-0.03]), np.array([0.02]), 0.01)[0] == pytest.approx(0.0099)
    assert probability_losses(1.0, 0.2) == {"brier": pytest.approx(0.64)}


def test_config_loads() -> None:
    cfg = load_loop_config()
    assert cfg.walkforward.test_start.weekday() == 0
    assert cfg.arm1.var_levels == (0.01, 0.05)
    assert len(models(cfg.arm1)) == len(cfg.arm1.models)
    with pytest.raises(ValueError, match="Monday"):
        WalkForwardSection(test_start=START + BAR)


def test_the_request_refuses_returns_of_open_bars() -> None:
    now = START + 5 * BAR + timedelta(minutes=25)
    with pytest.raises(LeakageError):
        ForecastRequest(
            asset=Asset.BTC,
            bar_open=START + 6 * BAR,
            issued_at=now,
            phase=Phase.DEVELOPMENT,
            run_id=RUN,
            returns={START + 5 * BAR: 0.01},
        )


# --- arm 1 in the loop (test database) -----------------------------------------------------

# Candles from DAY0 (a Monday) for 15 days; the loop runs from Sunday 22:25 of the second week
# into the third, so it forecasts bars of two blocks and has to fit the second one live.
DAY0 = datetime(2026, 9, 7, tzinfo=UTC)
BLOCK2 = DAY0 + timedelta(days=14)
CANDLES_END = BLOCK2 + 4 * BAR


def candle_rows(asset: Asset, seed: int) -> list[dict[str, Any]]:
    hourly = garch_bars((CANDLES_END - DAY0) // BAR + 1, seed=seed, start=DAY0 - BAR)
    rng = np.random.default_rng(seed + 100)
    price = 70_000.0 if asset is Asset.BTC else 2_500.0
    rv = hourly["rv"].to_numpy(dtype=float)
    rows = []
    t = DAY0 - MINUTE
    while t < CANDLES_END:
        sigma = float(rv[(t - (DAY0 - BAR)) // BAR]) / math.sqrt(60)
        price *= math.exp(rng.normal(0, sigma))
        rows.append(
            {
                "symbol": f"{asset.value}USDT",
                "interval": "1m",
                "open_time": t,
                "close_time": t + MINUTE - timedelta(milliseconds=1),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1.0,
                "quote_volume": price,
                "trades": 1,
                "taker_buy_base_volume": 0.5,
                "taker_buy_quote_volume": price / 2,
                "source": "api",
                "source_ref": "api",
                "fetched_at": t + MINUTE + timedelta(seconds=5),
                "collector_run_id": RUN,
            }
        )
        t += MINUTE
    return rows


def clear_ledger(conn: psycopg.Connection) -> None:
    with conn.transaction():
        conn.execute("SET LOCAL session_replication_role = replica")
        conn.execute("TRUNCATE ledger.entries RESTART IDENTITY CASCADE")


def loop_config() -> LoopConfig:
    return LoopConfig(
        loop=LoopSection(features_lookback_hours=2),
        baseline=BaselineSection(window_hours=24, min_bars=20),
        arm1=Arm1Section(
            train_start=DAY0,
            min_train_hours=150,
            filter_hours=120,
            har_windows=(6, 24, 48),
            hist_window_hours=48,
            tail_grid=101,
        ),
        walkforward=WalkForwardSection(test_start=DAY0 + timedelta(days=7)),
    )


@pytest.fixture
def market(
    db: psycopg.Connection, store: RawStore
) -> Iterator[tuple[psycopg.Connection, RawStore]]:
    clear_ledger(db)
    db.execute("TRUNCATE features.hourly")
    for i, asset in enumerate((Asset.BTC, Asset.ETH)):
        write_klines(db, store, candle_rows(asset, seed=11 + i), RUN, {})
    yield db, store
    clear_ledger(db)


@pytest.mark.db
def test_arm1_in_the_loop_across_a_block_boundary(
    market: tuple[psycopg.Connection, RawStore], tmp_path: Path
) -> None:
    conn, store = market
    cfg = loop_config()
    loop = HourlyLoop.default(
        cfg, StubClassifier(), store, phase=Phase.DEVELOPMENT, run_id=RUN, ledger_dir=tmp_path
    )
    start = BLOCK2 - 2 * BAR + timedelta(minutes=25)  # Sunday 22:25: forecasts Sunday 23:00
    runs = simulate(conn, loop, start, start + 3 * BAR)
    assert all(not r.errors for r in runs), [r.errors for r in runs]
    per_bar = 1 + 5 * 4 + 3  # baseline + arm 1
    assert [r.forecasts for r in runs] == [2 * per_bar] * 4
    # Six models x two assets, for the old block (first run) and the new one (second run).
    assert [r.packages for r in runs] == [12, 12, 0, 0]

    packages = {
        key: (seq, cfg_)
        for seq, key, cfg_, data in conn.execute(
            "SELECT seq, package_key, config, data FROM ledger.experiment_packages"
        )
        if (manifest := data[0]) and (tmp_path / manifest["path"]).exists()
    }
    assert len(packages) == 24
    assert "arm1.garch11.v1.BTC.2026-09-21.development" in packages

    # Each forecast cites its block's package, and recomputing it offline from the raw store
    # (fit on the package's window, predict with bars up to t - 2) gives the same number.
    rows = conn.execute(
        "SELECT p.asset, p.model, p.target, p.var_level, p.bar_open, p.value, k.package_key, "
        "k.config FROM ledger.predictions p JOIN ledger.experiment_packages k "
        "ON k.seq = p.package_seq WHERE p.arm = 'arm1_stats' ORDER BY p.seq"
    ).fetchall()
    assert len(rows) == 4 * 2 * (per_bar - 1)
    by_model = {m.name: m for m in models(cfg.arm1)}
    now = CANDLES_END + BAR
    for asset_s, model_name, target, level, bar_open, value, key, config in rows[::5]:
        asset = Asset(asset_s)
        block = block_start(bar_open)
        assert key.endswith(f".{asset_s}.{block:%Y-%m-%d}.development")
        hourly = bars_mod.load_hourly(conn, store, asset, DAY0, bar_open - BAR, now=now).frame
        model = by_model[model_name]
        fit = model.fit(hourly.loc[: training_end(block)], asset, block)
        assert fit is not None
        assert fit.to_config()["params"] == pytest.approx(config["fit"]["params"], rel=1e-9)
        out_key = target if level is None else f"{target}@{level}"
        offline = model.predict(fit, hourly, pd.DatetimeIndex([bar_open]))[out_key].iloc[0]
        assert value == pytest.approx(offline, rel=1e-9)

    # Bars that closed by the last run are scored: VaR by the return, the probability by the
    # package's threshold.
    scored = conn.execute(
        "SELECT p.target, p.var_level, p.value, s.realized, s.losses, "
        "(k.config -> 'fit' ->> 'large_move_threshold')::float8 "
        "FROM ledger.scores s JOIN ledger.predictions p ON p.seq = s.prediction_seq "
        "LEFT JOIN ledger.experiment_packages k ON k.seq = p.package_seq"
    ).fetchall()
    targets = {r[0] for r in scored}
    assert targets == {"rv_1h", "var_1h", "p_large_move_1h"}
    for target, level, value, realized, losses, threshold in scored:
        if target == "var_1h":
            assert losses == pytest.approx(var_losses(realized, value, level))
        elif target == "p_large_move_1h":
            assert realized in (0.0, 1.0)
            assert losses == pytest.approx(probability_losses(realized, value))
            assert threshold is not None
    assert ledger.check_chain(conn).ok

    # A second run in the same hour adds nothing.
    again = simulate(conn, loop, start + 3 * BAR, start + 3 * BAR)
    assert again[0].forecasts == 0 and again[0].already_issued == 2 * per_bar


def test_scheduled_jobs_import_without_the_statistics_stack() -> None:
    """The scheduler's image only gets new dependencies when it is rebuilt, but it runs the
    bind-mounted code at once. Only the forecasting jobs may need pandas, arch or MLflow, so a
    code update can't stop collection and normalization (3 Nov 2026: it did, for ten minutes)."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "for m in ('pandas', 'arch', 'mlflow', 'statsmodels'): sys.modules[m] = None\n"
        "import cryptonlp.flows.collect, cryptonlp.cli\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
