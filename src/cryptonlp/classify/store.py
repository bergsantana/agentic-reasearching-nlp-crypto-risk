"""Labels in ``clean.labels``: which in-study posts still need them, and writing them."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

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


def register_version(conn: psycopg.Connection, classifier: Classifier, run_id: str) -> None:
    """Record the classifier version's spec, once. A version string already recorded with a
    different spec is an error: the labels would no longer say what produced them."""
    spec = dict(classifier.spec)
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    prompt = spec.get("prompt_sha256")
    conn.execute(
        "INSERT INTO clean.classifier_versions "
        "(model_version, spec, spec_sha256, prompt_sha256, run_id) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (model_version) DO NOTHING",
        (classifier.model_version, Jsonb(spec), digest, prompt, run_id),
    )
    row = conn.execute(
        "SELECT spec_sha256 FROM clean.classifier_versions WHERE model_version = %s",
        (classifier.model_version,),
    ).fetchone()
    if row is None or row[0] != digest:
        raise RuntimeError(
            f"classifier version {classifier.model_version} is recorded with another spec"
        )


def classify_pending(
    conn: psycopg.Connection,
    classifier: Classifier,
    *,
    now: datetime,
    run_id: str,
    batch_size: int = 2000,
    clock: Callable[[], datetime] | None = None,
    limit: int | None = None,
) -> int:
    """Label every in-study post received by ``now`` that this classifier hasn't labeled yet,
    up to ``limit`` posts. ``conn`` must be autocommit. Each batch's labels carry the time
    ``clock`` gives when the batch is done (the database clock in the loop; ``now`` if no clock
    is given). Returns the number of posts labeled."""
    register_version(conn, classifier, run_id)
    total = 0
    while limit is None or total < limit:
        size = batch_size if limit is None else min(batch_size, limit - total)
        rows = conn.execute(
            PENDING_SQL,
            {"now": now, "version": classifier.model_version, "limit": size},
        ).fetchall()
        if not rows:
            break
        posts = [PostText(*r) for r in rows]
        started = datetime.now(UTC)
        labels = classifier.classify(posts, labeled_at=now, run_id=run_id)
        _check(classifier, posts, labels)
        if clock is not None:
            at = clock()
            labels = [label.model_copy(update={"labeled_at": at}) for label in labels]
        write_labels(conn, posts, labels)
        conn.execute(
            "INSERT INTO clean.classify_batches (model_version, posts, started_at, finished_at, "
            "run_id) VALUES (%s, %s, %s, %s, %s)",
            (classifier.model_version, len(posts), started, datetime.now(UTC), run_id),
        )
        total += len(posts)
    return total
