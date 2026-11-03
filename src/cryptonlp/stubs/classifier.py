"""``StubClassifier``: VADER for sentiment, keyword rules for topic, scam and event.

A stand-in, so the hourly loop runs end to end before SemIf is served (step 6). Its labels are
not results: they carry ``model_version = "stub"`` and never enter H1 to H4. The rules are crude
on purpose. Every rule label gets the same confidence, 0.5, since the rules carry none.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from cryptonlp.classify.base import PostText
from cryptonlp.schemas.labels import Annotator, Label, Task

STUB_VERSION = "stub"
RULE_CONFIDENCE = 0.5
# VADER's usual cut-offs on the compound score.
VADER_THRESHOLD = 0.05


def _rx(*words: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w$])(?:" + "|".join(words) + r")(?!\w)", re.IGNORECASE)


# Multi-label: every topic whose pattern matches; "other" when none does.
TOPIC_RULES: dict[str, re.Pattern[str]] = {
    "price_talk": _rx(
        r"price",
        r"pump\w*",
        r"dump\w*",
        r"moon\w*",
        r"ath",
        r"bull\w*",
        r"bear\w*",
        r"rally",
        r"crash\w*",
        r"dip",
        r"breakout",
        r"support",
        r"resistance",
        r"liquidat\w+",
        r"longs?",
        r"shorts?",
        r"\$\d[\d,.]*k?",
    ),
    "regulation": _rx(
        r"sec",
        r"cftc",
        r"regulat\w+",
        r"congress",
        r"senate",
        r"bill",
        r"law\w*",
        r"compliance",
        r"mica",
        r"etfs?",
        r"ban(ned|s)?",
        r"lawsuit",
    ),
    "security_incident": _rx(
        r"hack\w*",
        r"exploit\w*",
        r"drain\w*",
        r"stolen",
        r"breach\w*",
        r"phish\w*",
        r"rug ?pull\w*",
        r"vulnerab\w+",
    ),
    "protocol_tech": _rx(
        r"upgrade\w*",
        r"hard ?fork",
        r"soft ?fork",
        r"layer ?2",
        r"l2s?",
        r"rollups?",
        r"nodes?",
        r"validators?",
        r"mainnet",
        r"testnets?",
        r"eip-?\d*",
        r"bip-?\d*",
        r"lightning",
        r"staking",
        r"smart contracts?",
    ),
    "launch_airdrop": _rx(
        r"airdrops?",
        r"launch\w*",
        r"presale",
        r"ido",
        r"ico",
        r"token sale",
        r"mint\w*",
        r"whitelist\w*",
    ),
    "macro": _rx(
        r"fed",
        r"fomc",
        r"inflation",
        r"cpi",
        r"interest rates?",
        r"rate cuts?",
        r"recession",
        r"treasur\w+",
        r"dxy",
        r"tariffs?",
        r"jobs report",
    ),
}

# Single label, checked in this order.
SCAM_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "likely_scam",
        _rx(
            r"giveaway",
            r"double your",
            r"seed phrase",
            r"private key",
            r"guaranteed (profit|returns?)",
            r"recover (your )?(funds|wallet)",
            r"dm me",
            r"send \S+ (btc|eth) .{0,40}(back|receive)",
        ),
    ),
    (
        "promotion",
        _rx(
            r"join (us|our|now)",
            r"telegram",
            r"discord",
            r"referral",
            r"use my code",
            r"sign ?up",
            r"presale",
            r"100x",
            r"gem",
            r"airdrops?",
        ),
    ),
)

EVENT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hack", _rx(r"hack(ed|s)?", r"exploit(ed|s)?", r"drained", r"stolen")),
    (
        "regulatory_action",
        _rx(r"sec (sues|charges|approves)", r"lawsuit", r"fined", r"enforcement", r"banned"),
    ),
    ("outage", _rx(r"outage", r"halted", r"suspend(ed|s) withdrawals", r"downtime")),
    ("listing", _rx(r"listed on", r"listing on", r"lists \$\w+", r"delist\w*")),
)


class StubClassifier:
    """Implements ``cryptonlp.classify.Classifier``."""

    model_version = STUB_VERSION
    tasks = frozenset({Task.SENTIMENT, Task.TOPIC, Task.SCAM, Task.EVENT})

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()

    def classify(
        self, posts: Sequence[PostText], *, labeled_at: datetime, run_id: str
    ) -> list[Label]:
        labels: list[Label] = []
        for post in posts:
            for task, values, confidence in self._label(post.text):
                labels.append(
                    Label(
                        post_uri=post.uri,
                        task=task,
                        values=values,
                        confidence=confidence,
                        annotator=Annotator.MODEL,
                        model_version=self.model_version,
                        labeled_at=labeled_at,
                        run_id=run_id,
                    )
                )
        return labels

    def _label(self, text: str) -> list[tuple[Task, tuple[str, ...], float]]:
        compound = float(self._vader.polarity_scores(text)["compound"])
        if compound >= VADER_THRESHOLD:
            sentiment, confidence = "positive", compound
        elif compound <= -VADER_THRESHOLD:
            sentiment, confidence = "negative", -compound
        else:
            sentiment, confidence = "neutral", 1 - abs(compound)
        topics = tuple(t for t, rx in TOPIC_RULES.items() if rx.search(text)) or ("other",)
        scam = next((label for label, rx in SCAM_RULES if rx.search(text)), "organic")
        event = next((label for label, rx in EVENT_RULES if rx.search(text)), "none")
        return [
            (Task.SENTIMENT, (sentiment,), confidence),
            (Task.TOPIC, topics, RULE_CONFIDENCE),
            (Task.SCAM, (scam,), RULE_CONFIDENCE),
            (Task.EVENT, (event,), RULE_CONFIDENCE),
        ]
