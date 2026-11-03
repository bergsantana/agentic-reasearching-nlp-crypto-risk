"""Hourly NLP features, one row per asset and bar."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from cryptonlp.schemas.common import BAR, Asset, BarOpen, NonEmptyStr, Record, RunId, UTCDatetime
from cryptonlp.schemas.labels import TASK_LABELS, Task

_SUM_TOLERANCE = 1e-6


class HourlyFeatures(Record):
    asset: Asset
    bar_open: BarOpen
    post_count: int = Field(ge=0)
    # Latest received_at among the posts used. The leakage rule: it must be before the bar closes.
    source_cutoff: UTCDatetime | None
    sentiment_shares: dict[str, float] = Field(default_factory=dict)
    topic_shares: dict[str, float] = Field(default_factory=dict)  # multi-label: need not sum to 1
    scam_shares: dict[str, float] = Field(default_factory=dict)
    event_counts: dict[str, int] = Field(default_factory=dict)
    bot_filtered: bool  # the Verifier's bot/shill filter applied (the ablation compares both)
    classifier_version: NonEmptyStr
    run_id: RunId

    @property
    def bar_close(self) -> datetime:
        return self.bar_open + BAR

    @model_validator(mode="after")
    def _no_leakage_and_valid_shares(self) -> HourlyFeatures:
        if self.post_count > 0 and self.source_cutoff is None:
            raise ValueError("source_cutoff is required when posts were used")
        if self.source_cutoff is not None and self.source_cutoff >= self.bar_close:
            raise ValueError(
                f"leakage: a post received at {self.source_cutoff.isoformat()} was used "
                f"for the bar closing at {self.bar_close.isoformat()}"
            )
        for task, shares, sums_to_one in (
            (Task.SENTIMENT, self.sentiment_shares, True),
            (Task.TOPIC, self.topic_shares, False),
            (Task.SCAM, self.scam_shares, True),
        ):
            if not set(shares) <= TASK_LABELS[task]:
                raise ValueError(f"{task} shares name unknown labels")
            if any(not 0 <= s <= 1 for s in shares.values()):
                raise ValueError(f"{task} shares must lie in [0, 1]")
            total = sum(shares.values())
            if sums_to_one and self.post_count > 0 and shares and abs(total - 1) > _SUM_TOLERANCE:
                raise ValueError(f"{task} shares must sum to 1")
        if not set(self.event_counts) <= TASK_LABELS[Task.EVENT]:
            raise ValueError("event counts name unknown labels")
        if any(c < 0 for c in self.event_counts.values()):
            raise ValueError("event counts must be non-negative")
        return self
