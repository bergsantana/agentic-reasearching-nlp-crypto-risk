from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from cryptonlp.schemas import (
    AgentRole,
    Annotator,
    Arm,
    Asset,
    Candle,
    CandleSource,
    Finding,
    Forecast,
    HourlyFeatures,
    Interval,
    Label,
    NormalizedPost,
    Phase,
    RawPost,
    Source,
    Symbol,
    Target,
    Task,
    floor_to_bar,
)

RUN_ID = "20261103T120000Z-0123456789ab"
BAR_OPEN = datetime(2026, 11, 3, 14, tzinfo=UTC)


def raw_post(**overrides: Any) -> RawPost:
    fields: dict[str, Any] = {
        "source": Source.BLUESKY,
        "uri": "at://did:plc:abc/app.bsky.feed.post/3k",
        "author_id": "did:plc:abc",
        "text": "$BTC looking strong",
        "created_at": datetime(2026, 11, 3, 13, 59, tzinfo=UTC),
        "received_at": datetime(2026, 11, 3, 14, 0, 1, tzinfo=UTC),
        "collector_run_id": RUN_ID,
    }
    return RawPost(**(fields | overrides))


def features(**overrides: Any) -> HourlyFeatures:
    fields: dict[str, Any] = {
        "asset": Asset.BTC,
        "bar_open": BAR_OPEN,
        "post_count": 10,
        "source_cutoff": BAR_OPEN + timedelta(minutes=59),
        "sentiment_shares": {"negative": 0.2, "neutral": 0.5, "positive": 0.3},
        "topic_shares": {"price_talk": 0.8, "macro": 0.4},
        "scam_shares": {"organic": 0.9, "likely_scam": 0.1},
        "event_counts": {"none": 10},
        "bot_filtered": True,
        "classifier_version": "stub",
        "run_id": RUN_ID,
    }
    return HourlyFeatures(**(fields | overrides))


def forecast(**overrides: Any) -> Forecast:
    fields: dict[str, Any] = {
        "asset": Asset.ETH,
        "arm": Arm.BASELINE,
        "model": "yesterday",
        "model_version": "1",
        "target": Target.RV_1H,
        "bar_open": BAR_OPEN,
        "value": 0.004,
        "issued_at": BAR_OPEN - timedelta(minutes=5),
        "phase": Phase.DEVELOPMENT,
        "run_id": RUN_ID,
    }
    return Forecast(**(fields | overrides))


def label(**overrides: Any) -> Label:
    fields: dict[str, Any] = {
        "post_uri": "at://x/1",
        "task": Task.SENTIMENT,
        "values": ("positive",),
        "probabilities": {"negative": 0.1, "neutral": 0.2, "positive": 0.7},
        "confidence": 0.7,
        "annotator": Annotator.MODEL,
        "model_version": "stub",
        "labeled_at": BAR_OPEN,
        "run_id": RUN_ID,
    }
    return Label(**(fields | overrides))


# --- times ---------------------------------------------------------------------------------


def test_times_are_converted_to_utc() -> None:
    manaus = timezone(timedelta(hours=-4))
    post = raw_post(received_at=datetime(2026, 11, 3, 10, 0, 1, tzinfo=manaus))
    assert post.received_at == datetime(2026, 11, 3, 14, 0, 1, tzinfo=UTC)
    assert post.received_at.utcoffset() == timedelta(0)


def test_naive_times_rejected() -> None:
    with pytest.raises(ValidationError):
        raw_post(received_at=datetime(2026, 11, 3, 14))  # noqa: DTZ001


def test_floor_to_bar() -> None:
    assert floor_to_bar(datetime(2026, 11, 3, 14, 59, 59, tzinfo=UTC)) == BAR_OPEN


def test_bar_open_must_be_on_the_hour() -> None:
    with pytest.raises(ValidationError, match="on the hour"):
        features(bar_open=BAR_OPEN + timedelta(minutes=1))


# --- records -------------------------------------------------------------------------------


def test_records_are_immutable_and_strict() -> None:
    post = raw_post()
    with pytest.raises(ValidationError):
        post.text = "edited"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs"):
        raw_post(unexpected="x")


def test_json_round_trip() -> None:
    for record in (raw_post(), features(), forecast(), label()):
        assert type(record).model_validate_json(record.model_dump_json()) == record


def test_bad_run_id_rejected() -> None:
    with pytest.raises(ValidationError):
        raw_post(collector_run_id="yesterday")


def test_normalized_post_author_must_be_hashed() -> None:
    base: dict[str, Any] = raw_post().model_dump(exclude={"author_id", "langs", "collector_run_id"})
    extra: dict[str, Any] = {"lang": "en", "lang_method": "detected", "normalize_run_id": RUN_ID}
    ok = NormalizedPost(**base, author_hash="a" * 64, **extra)
    assert ok.author_hash == "a" * 64
    assert ok.in_study
    with pytest.raises(ValidationError):
        NormalizedPost(**base, author_hash="did:plc:abc", **extra)
    with pytest.raises(ValidationError):  # a duplicate names its kind
        NormalizedPost(**base, author_hash="a" * 64, duplicate_of="at://x", **extra)
    with pytest.raises(ValidationError):  # spam is spam because a rule matched
        NormalizedPost(**base, author_hash="a" * 64, is_spam=True, **extra)
    spam = NormalizedPost(
        **base, author_hash="a" * 64, is_spam=True, spam_rules=("double_your_coins",), **extra
    )
    assert not spam.in_study


def test_candle_consistency() -> None:
    fields: dict[str, Any] = {
        "symbol": Symbol.BTCUSDT,
        "interval": Interval.HOUR,
        "open_time": BAR_OPEN,
        "close_time": BAR_OPEN + timedelta(hours=1) - timedelta(milliseconds=1),
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 1.0,
        "quote_volume": 100.0,
        "trades": 3,
        "source": CandleSource.VISION,
        "collector_run_id": RUN_ID,
    }
    assert Candle(**fields).symbol.asset is Asset.BTC
    with pytest.raises(ValidationError, match="bound"):
        Candle(**(fields | {"high": 101.0}))


# --- labels --------------------------------------------------------------------------------


def test_label_valid() -> None:
    assert label().values == ("positive",)


@pytest.mark.parametrize(
    "overrides",
    [
        {"values": ("bullish",)},  # not in the label set
        {"values": ("positive", "neutral")},  # single-label task
        {"confidence": None},  # model labels need confidence
        {"probabilities": {"bullish": 0.5}},
        {"probabilities": {"positive": 1.5}},
    ],
)
def test_label_invalid(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        label(**overrides)


def test_multi_label_and_entity_tasks() -> None:
    topic = label(task=Task.TOPIC, values=("price_talk", "macro"), probabilities={})
    assert len(topic.values) == 2
    entity = label(task=Task.ENTITY, values=("token:BTC", "exchange:binance"), probabilities={})
    assert len(entity.values) == 2
    with pytest.raises(ValidationError):
        label(task=Task.ENTITY, values=("BTC",), probabilities={})
    assert label(task=Task.ENTITY, values=("none",), probabilities={}).values == ("none",)
    with pytest.raises(ValidationError, match="cannot go with"):
        label(task=Task.ENTITY, values=("none", "token:ETH"), probabilities={})


def test_human_label_needs_no_confidence() -> None:
    human = label(
        annotator=Annotator.HUMAN, confidence=None, probabilities={}, model_version="human:pass1"
    )
    assert human.confidence is None


# --- features: the leakage rule ------------------------------------------------------------


def test_features_valid() -> None:
    assert features().bar_close == BAR_OPEN + timedelta(hours=1)


def test_leakage_rule_rejects_post_received_at_or_after_bar_close() -> None:
    with pytest.raises(ValidationError, match="leakage"):
        features(source_cutoff=BAR_OPEN + timedelta(hours=1))


def test_features_need_cutoff_when_posts_used() -> None:
    with pytest.raises(ValidationError, match="source_cutoff"):
        features(source_cutoff=None)
    empty = features(post_count=0, source_cutoff=None, sentiment_shares={}, scam_shares={})
    assert empty.post_count == 0


def test_single_label_shares_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        features(sentiment_shares={"negative": 0.2, "positive": 0.2})


# --- forecasts -----------------------------------------------------------------------------


def test_forecast_valid() -> None:
    assert forecast().arm is Arm.BASELINE


def test_forecast_after_bar_open_rejected() -> None:
    with pytest.raises(ValidationError, match="before its bar opens"):
        forecast(issued_at=BAR_OPEN)


def test_var_forecasts_need_level() -> None:
    with pytest.raises(ValidationError, match="var_level"):
        forecast(target=Target.VAR_1H, value=-0.02)
    assert forecast(target=Target.VAR_1H, value=-0.02, var_level=0.01).var_level == 0.01
    with pytest.raises(ValidationError, match="only applies"):
        forecast(var_level=0.01)


def test_forecast_value_ranges() -> None:
    with pytest.raises(ValidationError):
        forecast(target=Target.P_LARGE_MOVE_1H, value=1.2)
    with pytest.raises(ValidationError):
        forecast(value=-0.1)


# --- findings ------------------------------------------------------------------------------


def test_finding_needs_evidence() -> None:
    fields: dict[str, Any] = {
        "agent": AgentRole.ANALYST,
        "claim": "Mentions of an exchange outage rose sharply",
        "evidence_post_uris": ("at://x/1", "at://x/2"),
        "confidence": 0.6,
        "created_at": BAR_OPEN,
        "phase": Phase.DEVELOPMENT,
        "run_id": RUN_ID,
    }
    assert Finding(**fields).finding_id is not None
    with pytest.raises(ValidationError):
        Finding(**(fields | {"evidence_post_uris": ()}))
