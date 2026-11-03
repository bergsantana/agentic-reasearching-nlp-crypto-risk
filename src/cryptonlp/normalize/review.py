"""A sample of normalization decisions for the hand review (step 3 check).

The CSV holds post text, so it is written under data/ (git-ignored) and never committed. Fill
in ``correct`` (y/n) and ``note`` for each row; the error rate per group goes in the record of
the stage 2 approval.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import psycopg

GROUPS: dict[str, str] = {
    "kept_for_study": "c.in_study",
    "not_english": "c.lang <> 'en' AND c.lang_method = 'detected'",
    "english_low_confidence": "c.lang = 'en' AND c.lang_confidence < 0.6",
    "language_not_detected": "c.lang_method <> 'detected'",
    "spam": "c.is_spam",
    "exact_duplicate": "c.duplicate_kind = 'exact'",
    "near_duplicate": "c.duplicate_kind = 'near'",
    "has_address": "cardinality(c.addresses) > 0",
}

COLUMNS = [
    "group",
    "correct",
    "note",
    "source",
    "uri",
    "received_at",
    "lang",
    "lang_method",
    "lang_confidence",
    "langs_declared",
    "assets",
    "cashtags",
    "addresses",
    "spam_rules",
    "duplicate_kind",
    "near_duplicate_score",
    "text",
    "original_text",
]


def sample(conn: psycopg.Connection, per_group: int = 30, seed: str = "1") -> list[dict[str, Any]]:
    """Posts from each decision group, in a seeded pseudo-random order: the same data and seed
    always give the same sample."""
    out: list[dict[str, Any]] = []
    for group, condition in GROUPS.items():
        rows = conn.execute(
            f"""
            SELECT c.source, c.uri, c.received_at, c.lang, c.lang_method,
                   round(c.lang_confidence::numeric, 3), c.langs_declared, c.assets,
                   c.cashtags, c.addresses, c.spam_rules, c.duplicate_kind,
                   round(c.near_duplicate_score::numeric, 3), c.text, o.text
            FROM clean.posts c
            LEFT JOIN clean.posts o ON o.uri = c.duplicate_of
            WHERE {condition}
            ORDER BY md5(%s || c.uri)
            LIMIT %s
            """,
            (seed, per_group),
        ).fetchall()
        for r in rows:
            out.append(
                {
                    "group": group,
                    "correct": "",
                    "note": "",
                    **dict(zip(COLUMNS[3:], [_cell(v) for v in r], strict=True)),
                }
            )
    return out


def _cell(value: Any) -> Any:
    if isinstance(value, list):
        return " ".join(value)
    return "" if value is None else value


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
