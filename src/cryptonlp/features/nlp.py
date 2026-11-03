"""Hourly NLP features per asset, from labeled in-study posts, into ``features.hourly``.

**The leakage rule.** The features of a bar use only posts received during it: on or after the
bar opens and strictly before it closes. ``HourlyFeatures`` refuses a row whose latest post
arrived at or after the close, and so does the table.

A bar's row is written once, when the bar is final: closed for ``settle`` (so every post
received during it is on disk and normalized), with no raw post of the bar still waiting for
normalization and no in-study post of the bar still waiting for a label. Until then the bar is
left for the next run, so a row never changes after it is written.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

import psycopg
from psycopg.types.json import Jsonb

from cryptonlp.schemas.common import BAR, Asset, floor_to_bar
from cryptonlp.schemas.features import HourlyFeatures
from cryptonlp.schemas.labels import TASK_LABELS, Task

DONE_SQL = """
SELECT bar_open FROM features.hourly
WHERE asset = %s AND bot_filtered = false AND classifier_version = %s
  AND bar_open BETWEEN %s AND %s
"""

# Raw posts received during the bar that normalization hasn't reached yet.
UNNORMALIZED_SQL = """
SELECT count(*) FROM raw.posts r
LEFT JOIN clean.posts c USING (source, uri)
LEFT JOIN raw.deletions d USING (source, uri)
WHERE c.uri IS NULL AND d.uri IS NULL AND r.received_at >= %s AND r.received_at < %s
"""

# In-study posts received during the bar without every label from this classifier.
UNLABELED_SQL = """
SELECT count(*) FROM clean.study_posts p
WHERE p.received_at >= %(start)s AND p.received_at < %(end)s
  AND (SELECT count(*) FROM clean.labels l
       WHERE l.source = p.source AND l.uri = p.uri AND l.model_version = %(version)s) < %(tasks)s
"""

LABELS_SQL = """
SELECT p.uri, p.received_at, l.task, l.label_values
FROM clean.study_posts p
JOIN clean.labels l ON l.source = p.source AND l.uri = p.uri AND l.model_version = %(version)s
WHERE %(asset)s = ANY(p.assets) AND p.received_at >= %(start)s AND p.received_at < %(end)s
"""

INSERT_SQL = """
INSERT INTO features.hourly (
    asset, bar_open, bot_filtered, classifier_version, post_count, source_cutoff,
    sentiment_shares, topic_shares, scam_shares, event_counts, run_id, computed_at
) VALUES (
    %(asset)s, %(bar_open)s, %(bot_filtered)s, %(classifier_version)s, %(post_count)s,
    %(source_cutoff)s, %(sentiment_shares)s, %(topic_shares)s, %(scam_shares)s,
    %(event_counts)s, %(run_id)s, %(computed_at)s
)
ON CONFLICT DO NOTHING
"""

SELECT_SQL = """
SELECT asset, bar_open, post_count, source_cutoff, sentiment_shares, topic_shares, scam_shares,
       event_counts, bot_filtered, classifier_version, run_id
FROM features.hourly
WHERE asset = %s AND classifier_version = %s AND bot_filtered = false
  AND bar_open >= %s AND bar_open + interval '1 hour' <= %s
ORDER BY bar_open
"""


def latest_final_bar(now: datetime, settle: timedelta) -> datetime:
    """Open of the newest bar that closed at least ``settle`` before ``now``."""
    return floor_to_bar(now - settle) - BAR


def compute(
    conn: psycopg.Connection,
    asset: Asset,
    bar_open: datetime,
    *,
    classifier_version: str,
    run_id: str,
) -> HourlyFeatures:
    """Features of one bar from the labels in the database. Doesn't check the bar is final."""
    rows = conn.execute(
        LABELS_SQL,
        {
            "version": classifier_version,
            "asset": asset.value,
            "start": bar_open,
            "end": bar_open + BAR,
        },
    ).fetchall()
    labels: dict[str, dict[str, list[str]]] = defaultdict(dict)
    received: dict[str, datetime] = {}
    for uri, received_at, task, values in rows:
        labels[uri][task] = values
        received[uri] = received_at
    n = len(labels)

    def shares(task: Task) -> dict[str, float]:
        if not n:
            return {}
        counts = Counter(v for post in labels.values() for v in post.get(task.value, ()))
        return {label: counts[label] / n for label in sorted(TASK_LABELS[task])}

    events = Counter(v for post in labels.values() for v in post.get(Task.EVENT.value, ()))
    return HourlyFeatures(
        asset=asset,
        bar_open=bar_open,
        post_count=n,
        source_cutoff=max(received.values()) if received else None,
        sentiment_shares=shares(Task.SENTIMENT),
        topic_shares=shares(Task.TOPIC),
        scam_shares=shares(Task.SCAM),
        event_counts={label: events[label] for label in sorted(TASK_LABELS[Task.EVENT])}
        if n
        else {},
        bot_filtered=False,
        classifier_version=classifier_version,
        run_id=run_id,
    )


def _ready(conn: psycopg.Connection, bar_open: datetime, version: str, n_tasks: int) -> bool:
    end = bar_open + BAR
    row = conn.execute(UNNORMALIZED_SQL, (bar_open, end)).fetchone()
    if row and row[0]:
        return False
    row = conn.execute(
        UNLABELED_SQL, {"start": bar_open, "end": end, "version": version, "tasks": n_tasks}
    ).fetchone()
    return not (row and row[0])


def write(conn: psycopg.Connection, features: HourlyFeatures, *, computed_at: datetime) -> None:
    conn.execute(
        INSERT_SQL,
        {
            "asset": features.asset.value,
            "bar_open": features.bar_open,
            "bot_filtered": features.bot_filtered,
            "classifier_version": features.classifier_version,
            "post_count": features.post_count,
            "source_cutoff": features.source_cutoff,
            "sentiment_shares": Jsonb(features.sentiment_shares),
            "topic_shares": Jsonb(features.topic_shares),
            "scam_shares": Jsonb(features.scam_shares),
            "event_counts": Jsonb(features.event_counts),
            "run_id": features.run_id,
            "computed_at": computed_at,
        },
    )


def update(
    conn: psycopg.Connection,
    assets: tuple[Asset, ...],
    *,
    now: datetime,
    settle: timedelta,
    lookback: timedelta,
    classifier_version: str,
    n_tasks: int,
    run_id: str,
) -> tuple[int, int]:
    """Write features for every final bar in the lookback that doesn't have them yet.
    Returns (bars written, bars still waiting for posts or labels)."""
    last = latest_final_bar(now, settle)
    first = floor_to_bar(now - lookback)
    bars = []
    t = first
    while t <= last:
        bars.append(t)
        t += BAR
    written = waiting = 0
    ready: dict[datetime, bool] = {}
    for asset in assets:
        done = {
            r[0] for r in conn.execute(DONE_SQL, (asset.value, classifier_version, first, last))
        }
        for bar_open in bars:
            if bar_open in done:
                continue
            if bar_open not in ready:
                ready[bar_open] = _ready(conn, bar_open, classifier_version, n_tasks)
            if not ready[bar_open]:
                waiting += 1
                continue
            row = compute(
                conn, asset, bar_open, classifier_version=classifier_version, run_id=run_id
            )
            write(conn, row, computed_at=now)
            written += 1
    return written, waiting


def load(
    conn: psycopg.Connection,
    asset: Asset,
    *,
    since: datetime,
    now: datetime,
    classifier_version: str,
) -> list[HourlyFeatures]:
    """Stored features of the bars from ``since`` that had closed by ``now``."""
    rows = conn.execute(SELECT_SQL, (asset.value, classifier_version, since, now)).fetchall()
    cols = [
        "asset",
        "bar_open",
        "post_count",
        "source_cutoff",
        "sentiment_shares",
        "topic_shares",
        "scam_shares",
        "event_counts",
        "bot_filtered",
        "classifier_version",
        "run_id",
    ]
    return [HourlyFeatures.model_validate(dict(zip(cols, r, strict=True))) for r in rows]
