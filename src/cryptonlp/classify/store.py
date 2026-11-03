"""Labels in ``clean.labels``: which in-study posts still need them, and writing them."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime

import psycopg
from psycopg.types.json import Jsonb

from cryptonlp.classify.base import Classifier, PostText
from cryptonlp.schemas.labels import Label

# In-study posts received by ``now`` with no label yet from this classifier version. A batch's
# labels are written in one transaction, so a post has all of them or none.
PENDING_SQL = """
SELECT p.source, p.uri, p.text, p.received_at
FROM clean.study_posts p
WHERE p.received_at <= %(now)s
  AND NOT EXISTS (
      SELECT 1 FROM clean.labels l
      WHERE l.source = p.source AND l.uri = p.uri AND l.model_version = %(version)s
  )
ORDER BY p.received_at, p.source, p.uri
LIMIT %(limit)s
"""

# A post deleted at the source while its batch was being classified is skipped.
INSERT_SQL = """
INSERT INTO clean.labels (
    source, uri, task, model_version, annotator, label_values, probabilities, confidence,
    labeled_at, run_id
)
SELECT %(source)s, %(uri)s, %(task)s, %(model_version)s, %(annotator)s, %(label_values)s,
       %(probabilities)s, %(confidence)s, %(labeled_at)s, %(run_id)s
WHERE EXISTS (SELECT 1 FROM clean.posts WHERE source = %(source)s AND uri = %(uri)s)
ON CONFLICT DO NOTHING
"""


def write_labels(conn: psycopg.Connection, posts: Sequence[PostText], labels: list[Label]) -> int:
    """Insert labels in one transaction. Rows already there are kept as they are."""
    source_of = {p.uri: p.source for p in posts}
    rows = [
        {
            "source": source_of[label.post_uri],
            "uri": label.post_uri,
            "task": label.task.value,
            "model_version": label.model_version,
            "annotator": label.annotator.value,
            "label_values": list(label.values),
            "probabilities": Jsonb(label.probabilities),
            "confidence": label.confidence,
            "labeled_at": label.labeled_at,
            "run_id": label.run_id,
        }
        for label in labels
    ]
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(INSERT_SQL, rows)
    return len(rows)


def _check(classifier: Classifier, posts: Sequence[PostText], labels: list[Label]) -> None:
    """Every post got exactly one label per task, from this classifier version."""
    expected = {(p.uri, t) for p in posts for t in classifier.tasks}
    got = Counter((label.post_uri, label.task) for label in labels)
    if set(got) != expected or any(n != 1 for n in got.values()):
        raise RuntimeError(
            f"{type(classifier).__name__} returned {len(labels)} labels for {len(posts)} posts "
            f"and {len(classifier.tasks)} tasks: not one per post and task"
        )
    versions = {label.model_version for label in labels}
    if versions - {classifier.model_version}:
        raise RuntimeError(f"labels carry model versions {sorted(versions)}")


def classify_pending(
    conn: psycopg.Connection,
    classifier: Classifier,
    *,
    now: datetime,
    run_id: str,
    batch_size: int = 2000,
) -> int:
    """Label every in-study post received by ``now`` that this classifier hasn't labeled yet.
    ``conn`` must be autocommit. Returns the number of posts labeled."""
    total = 0
    while True:
        rows = conn.execute(
            PENDING_SQL,
            {"now": now, "version": classifier.model_version, "limit": batch_size},
        ).fetchall()
        if not rows:
            return total
        posts = [PostText(*r) for r in rows]
        labels = classifier.classify(posts, labeled_at=now, run_id=run_id)
        _check(classifier, posts, labels)
        write_labels(conn, posts, labels)
        total += len(posts)
