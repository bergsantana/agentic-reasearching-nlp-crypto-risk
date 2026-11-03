"""Intra-annotator agreement: the first pass against the blind re-labeling (claims R2).

Single-label tasks (sentiment, scam, event): Cohen's kappa over the paired posts. Multi-label
tasks (topic, entity): Cohen's kappa for each label as a yes/no decision, their unweighted
mean over the labels used, and the share of posts with exactly the same set. Pairs where either
pass flagged a problem (not English, unreadable) are counted, not scored.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from datetime import timedelta
from typing import Any

import psycopg

from cryptonlp.labeling.sample import PASS1, RELABEL
from cryptonlp.labeling.tasks import SPECS
from cryptonlp.schemas.labels import Task


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa for two ratings of the same items; None when chance agreement is 1
    (both passes used a single category), where kappa is undefined."""
    if len(a) != len(b):
        raise ValueError("the two passes must rate the same items")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def task_agreement(
    task: Task, first: Sequence[Sequence[str]], second: Sequence[Sequence[str]]
) -> dict[str, Any]:
    n = len(first)
    same = sum(set(x) == set(y) for x, y in zip(first, second, strict=True))
    out: dict[str, Any] = {"n": n, "exact_agreement": same / n if n else None}
    spec = SPECS[task]
    if not spec.multiple:
        out["kappa"] = cohen_kappa([x[0] for x in first], [y[0] for y in second])
        return out
    per_label: dict[str, float | None] = {}
    for option in spec.options:
        v = option.value
        per_label[v] = cohen_kappa([v in x for x in first], [v in y for y in second])
    defined = [k for k in per_label.values() if k is not None]
    out["kappa_per_label"] = per_label
    out["kappa"] = sum(defined) / len(defined) if defined else None
    return out


def agreement(conn: psycopg.Connection, min_gap: timedelta) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT a.labels, a.problem, a.updated_at, b.labels, b.problem, b.created_at
        FROM labeling.items r
        JOIN labeling.current b ON b.item_id = r.item_id
        JOIN labeling.items f ON f.pass = %s AND f.source = r.source AND f.uri = r.uri
        JOIN labeling.current a ON a.item_id = f.item_id
        WHERE r.pass = %s AND NOT b.cancelled AND NOT a.cancelled
        """,
        (PASS1, RELABEL),
    ).fetchall()
    scored = [(a, b) for a, pa, _, b, pb, _ in rows if a is not None and b is not None]
    gaps = [created - first_updated for _, _, first_updated, _, _, created in rows]
    report: dict[str, Any] = {
        "pairs": len(rows),
        "scored": len(scored),
        "problem_in_either": sum(1 for r in rows if r[1] is not None or r[4] is not None),
        "problem_agreement": sum(1 for r in rows if r[1] == r[4]) / len(rows) if rows else None,
        "min_gap_days": min(gaps).total_seconds() / 86400 if gaps else None,
        "gap_violations": sum(1 for g in gaps if g < min_gap),
        "tasks": {},
    }
    for task in Task:
        report["tasks"][task.value] = task_agreement(
            task, [a[task.value] for a, _ in scored], [b[task.value] for _, b in scored]
        )
    return report
