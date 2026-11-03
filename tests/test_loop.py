"""Step 3b: the thin end-to-end loop. Stub classifier, features with the leakage rule, the
baseline, scoring, and a synthetic day run through the whole loop on the simulated clock.

The frozen replay day (tests/fixtures/replay/) is not recorded yet. Until it is, the
end-to-end test runs a synthetic day built here: posts through the real normalization, and
1-minute candles through the real raw store, with forecasts and scores checked against values
computed independently from the same prices.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from cryptonlp import ledger
from cryptonlp.classify import PostText
from cryptonlp.collectors.binance import write_klines
from cryptonlp.db import DatabaseGuardError
from cryptonlp.features import nlp
from cryptonlp.features.bars import MinuteCloses
from cryptonlp.flows import collect
from cryptonlp.forecast import ForecastRequest, LeakageError, YesterdayVolatility
from cryptonlp.forecast.config import BaselineSection, LoopConfig, LoopSection, load_loop_config
from cryptonlp.forecast.loop import HourlyLoop, simulate
from cryptonlp.forecast.scoring import volatility_losses
from cryptonlp.normalize.config import load_normalize_config
from cryptonlp.normalize.pipeline import Normalizer
from cryptonlp.normalize.pseudonym import PseudonymKey
from cryptonlp.rawstore import CollectedPost, RawStore
from cryptonlp.schemas import Origin, RawPost, Source
from cryptonlp.schemas.common import BAR, Asset, Phase
from cryptonlp.schemas.features import HourlyFeatures
from cryptonlp.schemas.labels import Task
from cryptonlp.settings import Settings
from cryptonlp.stubs import StubClassifier

RUN = "20261105T000000Z-0123456789ab"
DAY = datetime(2026, 11, 5, tzinfo=UTC)
MINUTE = timedelta(minutes=1)
DEV = Phase.DEVELOPMENT
KEY = PseudonymKey(bytes(range(32)))


# --- stub classifier -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def stub() -> StubClassifier:
    return StubClassifier()


def labels_of(stub: StubClassifier, text: str) -> dict[Task, tuple[str, ...]]:
    post = PostText("bluesky", "at://x/1", text, DAY)
    return {label.task: label.values for label in stub.classify([post], labeled_at=DAY, run_id=RUN)}


def test_stub_gives_one_label_per_task(stub: StubClassifier) -> None:
    posts = [PostText("bluesky", f"at://x/{i}", t, DAY) for i, t in enumerate(["a", "b c"])]
    labels = stub.classify(posts, labeled_at=DAY, run_id=RUN)
    assert len(labels) == 2 * len(stub.tasks)
    assert {label.model_version for label in labels} == {"stub"}
    assert all(label.confidence is not None for label in labels)
    assert Task.ENTITY not in stub.tasks  # entities come from normalization


def test_stub_rules(stub: StubClassifier) -> None:
    good = labels_of(stub, "Bitcoin is pumping, great rally and strong support. Love it!")
    assert good[Task.SENTIMENT] == ("positive",)
    assert "price_talk" in good[Task.TOPIC]
    assert good[Task.SCAM] == ("organic",)
    bad = labels_of(stub, "Awful: the bridge was hacked and funds were stolen overnight")
    assert bad[Task.SENTIMENT] == ("negative",)
    assert "security_incident" in bad[Task.TOPIC]
    assert bad[Task.EVENT] == ("hack",)
    scam = labels_of(stub, "Huge giveaway! Send 0.1 BTC and receive 0.2 back, DM me")
    assert scam[Task.SCAM] == ("likely_scam",)
    plain = labels_of(stub, "Reading a paper about distributed systems this afternoon")
    assert (plain[Task.TOPIC], plain[Task.EVENT]) == (("other",), ("none",))


# --- realized volatility, baseline, losses -------------------------------------------------


def closes_from(prices: dict[datetime, float]) -> MinuteCloses:
    return MinuteCloses(prices, {})


def test_realized_vol_is_close_to_close_over_the_hour() -> None:
    bar = DAY + 3 * BAR
    prices = {bar - MINUTE + i * MINUTE: 100.0 * (1.001 ** (i % 2)) for i in range(61)}
    # Alternating up and down by 0.1%: 60 returns of ln(1.001) in absolute value.
    expected = math.sqrt(60) * math.log(1.001)
    assert closes_from(prices).realized_vol(bar) == pytest.approx(expected)
    # The price at the open (close of the previous minute) is required, and so is every minute.
    assert (
        closes_from({t: p for t, p in prices.items() if t != bar - MINUTE}).realized_vol(bar)
        is None
    )
    assert (
        closes_from({t: p for t, p in prices.items() if t != bar + 30 * MINUTE}).realized_vol(bar)
        is None
    )


def request(rv: dict[datetime, float], issued_at: datetime, **kwargs: Any) -> ForecastRequest:
    return ForecastRequest(
        asset=Asset.BTC,
        bar_open=issued_at.replace(minute=0) + BAR,
        issued_at=issued_at,
        phase=DEV,
        run_id=RUN,
        rv=rv,
        **kwargs,
    )


def test_baseline_is_the_rms_of_the_last_24_closed_bars() -> None:
    now = DAY + 30 * BAR + timedelta(minutes=25)
    newest = DAY + 29 * BAR  # closed at DAY + 30h
    rv = {newest - i * BAR: 0.001 * (i + 1) for i in range(30)}
    forecast = YesterdayVolatility(24, 20).forecast(request(rv, now))
    assert forecast is not None
    expected = math.sqrt(sum((0.001 * (i + 1)) ** 2 for i in range(24)) / 24)
    assert forecast.value == pytest.approx(expected)
    assert (forecast.arm.value, forecast.model, forecast.bar_open) == (
        "baseline",
        "yesterday_rv_24h",
        DAY + 31 * BAR,
    )
    # Fewer than min_bars complete bars in the window: no forecast.
    sparse = {b: v for b, v in rv.items() if b > newest - 19 * BAR}
    assert YesterdayVolatility(24, 20).forecast(request(sparse, now)) is None


def test_a_request_refuses_data_from_open_bars() -> None:
    """The leakage rule for forecasts: nothing from a bar that hadn't closed at issue time."""
    now = DAY + 5 * BAR + timedelta(minutes=25)
    with pytest.raises(LeakageError):
        request({DAY + 5 * BAR: 0.01}, now)  # the current bar, still open
    request({DAY + 4 * BAR: 0.01}, now)  # closed at 05:00: allowed
    open_bar = HourlyFeatures(
        asset=Asset.BTC,
        bar_open=DAY + 5 * BAR,
        post_count=0,
        source_cutoff=None,
        bot_filtered=False,
        classifier_version="stub",
        run_id=RUN,
    )
    with pytest.raises(LeakageError):
        request({}, now, features=[open_bar])
    with pytest.raises(ValueError, match="before its bar opens"):
        ForecastRequest(
            asset=Asset.BTC, bar_open=DAY + 5 * BAR, issued_at=now, phase=DEV, run_id=RUN
        )


def test_losses() -> None:
    assert volatility_losses(0.01, 0.01) == {"se_variance": 0.0, "qlike": 0.0}
    under = volatility_losses(0.02, 0.01)["qlike"]
    over = volatility_losses(0.01, 0.02)["qlike"]
    assert under > over > 0  # QLIKE punishes underestimating more
    assert "qlike" not in volatility_losses(0.0, 0.01)


def test_loop_config_loads() -> None:
    cfg = load_loop_config()
    assert cfg.loop.assets == (Asset.BTC, Asset.ETH)
    assert cfg.baseline.min_bars <= cfg.baseline.window_hours


def test_forecast_loop_is_scheduled_only_when_enabled() -> None:
    off = [f for f, _ in collect.schedules(Settings(forecast_loop=False))]
    on = dict(collect.schedules(Settings(forecast_loop=True)))
    assert collect.forecast_loop not in off
    assert on[collect.forecast_loop] == "25 * * * *"


# --- the synthetic day (test database) -----------------------------------------------------


def prices(seed: int, start: datetime, end: datetime, base: float) -> dict[datetime, float]:
    """A deterministic random walk of 1-minute closes."""
    rng = random.Random(seed)
    out, p, t = {}, base, start
    while t < end:
        p *= math.exp(rng.gauss(0, 0.0008))
        out[t] = p
        t += MINUTE
    return out


def kline_rows(symbol: str, closes: dict[datetime, float]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": symbol,
            "interval": "1m",
            "open_time": t,
            "close_time": t + MINUTE - timedelta(milliseconds=1),
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1.0,
            "quote_volume": c,
            "trades": 1,
            "taker_buy_base_volume": 0.5,
            "taker_buy_quote_volume": c / 2,
            "source": "api",
            "source_ref": "api",
            "fetched_at": t + MINUTE + timedelta(seconds=5),
            "collector_run_id": RUN,
        }
        for t, c in sorted(closes.items())
    ]


def rv_of(closes: dict[datetime, float], bar: datetime) -> float | None:
    minutes = [bar - MINUTE + i * MINUTE for i in range(61)]
    if any(t not in closes for t in minutes):
        return None
    return math.sqrt(
        sum(math.log(closes[b] / closes[a]) ** 2 for a, b in itertools.pairwise(minutes))
    )


def post(n: int, text: str, received_at: datetime) -> CollectedPost:
    return CollectedPost(
        post=RawPost(
            source=Source.BLUESKY,
            uri=f"at://did:plc:a{n}/app.bsky.feed.post/{n}",
            author_id=f"did:plc:a{n}",
            text=text,
            created_at=received_at - timedelta(seconds=2),
            received_at=received_at,
            langs=("en",),
            origin=Origin.LIVE,
            collector_run_id=RUN,
        ),
        payload={"n": n},
    )


POSTS = [
    post(
        1,
        "Bitcoin is pumping hard today, great rally and strong support at the lows",
        DAY + timedelta(minutes=10),
    ),
    post(
        2,
        "Terrible news: the exchange was hacked and bitcoin holders lost their funds",
        DAY + timedelta(minutes=40),
    ),
    post(
        3,
        "Ethereum developers schedule the next network upgrade for early December",
        DAY + timedelta(minutes=50),
    ),
    # Received exactly when the 00:00 bar closes: it belongs to the 01:00 bar, never to 00:00.
    post(
        4, "Bitcoin dips again while traders wait for the Fed decision on interest rates", DAY + BAR
    ),
    post(
        5,
        "Quiet afternoon for ether, staking yields barely moved since last week",
        DAY + 2 * BAR + timedelta(minutes=5),
    ),
]
HISTORY_START = DAY - timedelta(hours=26)
CANDLES_END = DAY + timedelta(hours=7)
MISSING = DAY + 3 * BAR + timedelta(minutes=17)  # one ETH candle never arrives


def clear_ledger(conn: psycopg.Connection) -> None:
    # Replica mode skips the append-only triggers: the test database only.
    with conn.transaction():
        conn.execute("SET LOCAL session_replication_role = replica")
        conn.execute("TRUNCATE ledger.entries RESTART IDENTITY CASCADE")


@pytest.fixture
def day(
    db: psycopg.Connection, store: RawStore
) -> Iterator[tuple[psycopg.Connection, RawStore, dict[Asset, dict[datetime, float]]]]:
    clear_ledger(db)
    db.execute("TRUNCATE features.hourly")
    closes = {
        Asset.BTC: prices(1, HISTORY_START, CANDLES_END, 70_000.0),
        Asset.ETH: prices(2, HISTORY_START, CANDLES_END, 2_500.0),
    }
    del closes[Asset.ETH][MISSING]
    for asset, series in closes.items():
        write_klines(db, store, kline_rows(f"{asset.value}USDT", series), RUN, {})
    store.store_posts(db, POSTS, RUN)
    Normalizer(load_normalize_config(), KEY, store, RUN).run(db)
    yield db, store, closes
    clear_ledger(db)


def make_loop(store: RawStore) -> HourlyLoop:
    config = LoopConfig(
        loop=LoopSection(features_lookback_hours=4),
        baseline=BaselineSection(window_hours=24, min_bars=20),
    )
    return HourlyLoop.default(config, StubClassifier(), store, phase=DEV, run_id=RUN)


@pytest.mark.db
def test_the_loop_runs_the_day_end_to_end(
    day: tuple[psycopg.Connection, RawStore, dict[Asset, dict[datetime, float]]],
) -> None:
    conn, store, closes = day
    runs = simulate(conn, make_loop(store), DAY, DAY + timedelta(hours=5, minutes=30))
    assert [r.started_at[11:16] for r in runs] == [f"0{h}:25" for h in range(6)]
    assert all(not r.errors for r in runs), [r.errors for r in runs]

    # Every run forecast the next bar for both assets, before it opened.
    assert [r.forecasts for r in runs] == [2] * 6
    predictions = conn.execute(
        "SELECT seq, asset, bar_open, value, issued_at, recorded_at FROM ledger.predictions "
        "ORDER BY bar_open, asset"
    ).fetchall()
    assert [(p[2] - DAY) // BAR for p in predictions] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    for _seq, asset, bar_open, value, issued_at, recorded_at in predictions:
        assert recorded_at == issued_at == bar_open - BAR + timedelta(minutes=25)
        newest = bar_open - 2 * BAR
        window = [rv_of(closes[Asset(asset)], newest - i * BAR) for i in range(24)]
        complete = [v for v in window if v is not None]
        assert value == pytest.approx(math.sqrt(sum(v * v for v in complete) / len(complete)))

    # Bars 01:00 to 04:00 closed before the last run: scored, except ETH 03:00 (missing candle).
    scores = conn.execute(
        "SELECT p.asset, p.bar_open, p.value, s.realized, s.losses, s.data, s.recorded_at "
        "FROM ledger.scores s JOIN ledger.predictions p ON p.seq = s.prediction_seq "
        "ORDER BY p.bar_open, p.asset"
    ).fetchall()
    scored = [(a, (b - DAY) // BAR) for a, b, *_ in scores]
    assert scored == [
        ("BTC", 1),
        ("ETH", 1),
        ("BTC", 2),
        ("ETH", 2),
        ("BTC", 3),
        ("BTC", 4),
        ("ETH", 4),
    ]
    for asset, bar_open, value, realized, losses, data, recorded_at in scores:
        assert realized == pytest.approx(rv_of(closes[Asset(asset)], bar_open))
        assert losses == pytest.approx(volatility_losses(realized, value))
        assert recorded_at >= bar_open + BAR
        assert data and all(
            f["path"].startswith(f"binance_klines/symbol={asset}USDT/") for f in data
        )
    assert runs[-1].score_waiting == 1

    # Features: final bars only (closed 20 minutes before the run), leakage rule applied.
    feats = {
        (a, (b - DAY) // BAR): (n, cutoff, sentiment, topics, events)
        for a, b, n, cutoff, sentiment, topics, events in conn.execute(
            "SELECT asset, bar_open, post_count, source_cutoff, sentiment_shares, topic_shares, "
            "event_counts FROM features.hourly WHERE bar_open >= %s",
            (DAY,),
        )
    }
    assert max(b for _, b in feats) == 4  # the 05:00 bar was still open at 05:25
    n, cutoff, sentiment, topics, events = feats[("BTC", 0)]
    assert n == 2  # posts 1 and 2; post 4 arrived as the bar closed
    assert cutoff == DAY + timedelta(minutes=40)
    assert sentiment["positive"] == sentiment["negative"] == 0.5
    assert topics["security_incident"] == 0.5 and events["hack"] == 1
    assert feats[("BTC", 1)][0] == 1  # post 4
    assert feats[("ETH", 0)][0] == 1 and feats[("ETH", 2)][0] == 1
    assert feats[("ETH", 3)] == (0, None, {}, {}, {})

    # Grafana's view counts the same.
    bars = conn.execute(
        "SELECT sum(on_time), sum(late), sum(scored) FROM monitoring.forecast_bars "
        "WHERE phase = 'development'"
    ).fetchone()
    assert bars == (12, 0, 7)
    assert ledger.check_chain(conn).ok


@pytest.mark.db
def test_a_second_run_in_the_same_hour_changes_nothing(
    day: tuple[psycopg.Connection, RawStore, dict[Asset, dict[datetime, float]]],
) -> None:
    conn, store, _ = day
    loop = make_loop(store)
    first = simulate(conn, loop, DAY + 2 * BAR, DAY + 2 * BAR + timedelta(minutes=30))
    again = simulate(conn, loop, DAY + 2 * BAR, DAY + 2 * BAR + timedelta(minutes=30))
    assert (first[0].forecasts, again[0].forecasts, again[0].already_issued) == (2, 0, 2)
    assert again[0].labeled == again[0].features_written == 0
    assert conn.execute("SELECT count(*) FROM ledger.predictions").fetchone() == (2,)


@pytest.mark.db
def test_features_wait_for_unlabeled_or_unnormalized_posts(
    day: tuple[psycopg.Connection, RawStore, dict[Asset, dict[datetime, float]]],
) -> None:
    conn, store, _ = day
    late = post(
        9,
        "Bitcoin miners report record hashrate for the third week in a row",
        DAY + timedelta(minutes=55),
    )
    store.store_posts(conn, [late], RUN)  # on disk, not normalized yet
    ledger.set_simulated_clock(conn, DAY + timedelta(hours=1, minutes=25))
    try:
        written, waiting = nlp.update(
            conn,
            (Asset.BTC,),
            now=ledger.database_clock(conn),
            settle=timedelta(minutes=20),
            lookback=timedelta(hours=2),
            classifier_version="stub",
            n_tasks=4,
            run_id=RUN,
        )
    finally:
        ledger.set_simulated_clock(conn, None)
    assert written == 1 and waiting == 1  # 23:00 is final; 00:00 waits for post 9


@pytest.mark.db
def test_a_deleted_post_takes_its_labels_with_it(
    day: tuple[psycopg.Connection, RawStore, dict[Asset, dict[datetime, float]]],
) -> None:
    conn, store, _ = day
    simulate(conn, make_loop(store), DAY + BAR, DAY + BAR + timedelta(minutes=30))
    uri = POSTS[0].post.uri
    assert conn.execute("SELECT count(*) FROM clean.labels WHERE uri = %s", (uri,)).fetchone() == (
        4,
    )
    conn.execute(
        "INSERT INTO raw.deletions (source, uri, method, run_id) "
        "VALUES ('bluesky', %s, 'stream', %s)",
        (uri, RUN),
    )
    assert conn.execute("SELECT count(*) FROM clean.labels WHERE uri = %s", (uri,)).fetchone() == (
        0,
    )


@pytest.mark.db
def test_simulate_is_refused_outside_the_test_database(
    connect: Callable[[], psycopg.Connection], monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = connect()
    monkeypatch.setattr(conn.info.__class__, "dbname", property(lambda self: "cryptonlp"))
    with pytest.raises(DatabaseGuardError):
        simulate(conn, make_loop(RawStore(Settings().raw_path)), DAY, DAY + BAR)
    conn.close()
