"""Verifying the hash chain, exporting it, and the weekly anchor sent to the advisor.

The chain can be checked two ways:

- ``check_chain``: against the live database. Besides recomputing every hash and link, the
  database confirms each row still matches the canonical text it was hashed from, so an edited
  column is caught too.
- ``verify_export``: against an export file, with nothing but SHA-256. This is what the advisor
  or the validator agent can run without access to the database.

Neither can catch a rewrite of the whole tail with every hash recomputed. That is what anchors
are for: the head hash sent to the advisor each week must still be in the chain.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from cryptonlp.ledger.writer import append, head
from cryptonlp.schemas.common import Phase
from cryptonlp.schemas.ledger import Violation, ViolationKind

GENESIS = "0" * 64


def entry_hash(prev_hash: str, canonical: str) -> str:
    """SHA-256 of prev_hash || canonical, as the database computes it."""
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChainEntry:
    seq: int
    table: str
    canonical: str
    prev_hash: str
    hash: str
    # Whether the row still matches its canonical text. Only the database can tell, so it is
    # None for entries read from an export.
    row_matches: bool | None = None


@dataclass(frozen=True)
class Anchor:
    head_seq: int
    head_hash: str
    entries: int
    created_at: str | None = None
    run_id: str | None = None

    @classmethod
    def load(cls, path: Path) -> Anchor:
        data = json.loads(path.read_text())
        return cls(
            head_seq=int(data["head_seq"]),
            head_hash=str(data["head_hash"]),
            entries=int(data["entries"]),
            created_at=data.get("created_at"),
            run_id=data.get("run_id"),
        )


@dataclass(frozen=True)
class Problem:
    seq: int | None
    table: str | None
    issue: str


@dataclass
class ChainReport:
    entries: int
    head_seq: int | None
    head_hash: str | None
    anchors_checked: int
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def first_problem(self) -> Problem | None:
        return self.problems[0] if self.problems else None

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "entries": self.entries,
            "head_seq": self.head_seq,
            "head_hash": self.head_hash,
            "anchors_checked": self.anchors_checked,
            "problems": [asdict(p) for p in self.problems[:20]],
            "problem_count": len(self.problems),
        }


def _canonical_issue(entry: ChainEntry) -> str | None:
    try:
        data = json.loads(entry.canonical)
    except json.JSONDecodeError:
        return "canonical text is not valid JSON"
    if not isinstance(data, dict):
        return "canonical text is not a JSON object"
    if data.get("seq") != entry.seq or data.get("table") != entry.table:
        return "canonical text names a different sequence number or table"
    return None


def verify_entries(entries: Iterable[ChainEntry], anchors: Iterable[Anchor] = ()) -> ChainReport:
    ordered = sorted(entries, key=lambda e: e.seq)
    problems: list[Problem] = []
    by_seq: dict[int, ChainEntry] = {}
    expected_seq = 1
    prev_hash = GENESIS
    for e in ordered:
        if e.seq in by_seq:
            problems.append(Problem(e.seq, e.table, "sequence number used twice"))
            continue
        by_seq[e.seq] = e
        if e.seq != expected_seq:
            problems.append(
                Problem(e.seq, e.table, f"entries {expected_seq} to {e.seq - 1} are missing")
            )
        expected_seq = e.seq + 1
        if e.row_matches is False:
            problems.append(
                Problem(e.seq, e.table, "row no longer matches its canonical text (edited column)")
            )
        if issue := _canonical_issue(e):
            problems.append(Problem(e.seq, e.table, issue))
        if entry_hash(e.prev_hash, e.canonical) != e.hash:
            problems.append(Problem(e.seq, e.table, "hash is not SHA-256(prev_hash || canonical)"))
        if e.prev_hash != prev_hash:
            problems.append(
                Problem(e.seq, e.table, "prev_hash is not the hash of the entry before it")
            )
        prev_hash = e.hash

    checked = 0
    for a in anchors:
        checked += 1
        anchored = by_seq.get(a.head_seq)
        when = a.created_at or "an earlier anchor"
        if anchored is None:
            problems.append(
                Problem(a.head_seq, None, f"entry anchored on {when} is missing (chain cut back)")
            )
        elif anchored.hash != a.head_hash:
            problems.append(
                Problem(
                    a.head_seq,
                    anchored.table,
                    f"hash differs from the anchor of {when} (history rewritten)",
                )
            )

    last = ordered[-1] if ordered else None
    return ChainReport(
        entries=len(ordered),
        head_seq=last.seq if last else None,
        head_hash=last.hash if last else None,
        anchors_checked=checked,
        problems=problems,
    )


# --- live database -------------------------------------------------------------------------


def load_chain(conn: psycopg.Connection) -> list[ChainEntry]:
    rows = conn.execute(
        "SELECT seq, table_name, canonical, prev_hash, hash, row_matches "
        "FROM ledger.verify_rows() ORDER BY seq"
    ).fetchall()
    return [ChainEntry(*row) for row in rows]


def stored_anchors(conn: psycopg.Connection) -> list[Anchor]:
    rows = conn.execute(
        "SELECT head_seq, head_hash, entries, created_at, run_id FROM ops.ledger_anchors "
        "ORDER BY anchor_id"
    ).fetchall()
    return [Anchor(r[0], r[1], r[2], r[3].isoformat(), r[4]) for r in rows]


def check_chain(conn: psycopg.Connection, anchors: Iterable[Anchor] = ()) -> ChainReport:
    return verify_entries(load_chain(conn), anchors)


def record_check(conn: psycopg.Connection, report: ChainReport, run_id: str, phase: Phase) -> None:
    """Store the result for the dashboard. A newly found break also enters the ledger.

    Only the first failing check after a passing one records a violation, so a break that
    stays unrepaired is counted once, not once a day.
    """
    previous = conn.execute(
        "SELECT ok FROM ops.ledger_checks ORDER BY checked_at DESC, check_id DESC LIMIT 1"
    ).fetchone()
    conn.execute(
        "INSERT INTO ops.ledger_checks "
        "(run_id, ok, entries, head_seq, head_hash, anchors_checked, problems) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (
            run_id,
            report.ok,
            report.entries,
            report.head_seq,
            report.head_hash,
            report.anchors_checked,
            Jsonb([asdict(p) for p in report.problems]),
        ),
    )
    if not report.ok and (previous is None or previous[0]):
        append(
            conn,
            Violation(
                kind=ViolationKind.CHAIN_BREAK,
                reported_by="ledger_check",
                detail=report.summary(),
                phase=phase,
                run_id=run_id,
            ),
        )


# --- export and anchors --------------------------------------------------------------------


def export(conn: psycopg.Connection, path: Path) -> ChainReport:
    """Write the chain as JSON lines (seq, table, prev_hash, hash, canonical), oldest first."""
    entries = load_chain(conn)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            line = {
                "seq": e.seq,
                "table": e.table,
                "prev_hash": e.prev_hash,
                "hash": e.hash,
                "canonical": e.canonical,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return verify_entries(entries)


def read_export(path: Path) -> list[ChainEntry]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                entries.append(
                    ChainEntry(d["seq"], d["table"], d["canonical"], d["prev_hash"], d["hash"])
                )
    return entries


def verify_export(path: Path, anchors: Iterable[Anchor] = ()) -> ChainReport:
    return verify_entries(read_export(path), anchors)


def make_anchor(conn: psycopg.Connection, directory: Path, run_id: str) -> tuple[Anchor, Path]:
    """Record the current head and write it to a file to send to the advisor."""
    last = head(conn)
    if last is None:
        raise RuntimeError("the ledger is empty: nothing to anchor yet")
    count = conn.execute("SELECT count(*) FROM ledger.entries").fetchone()
    assert count is not None
    now = datetime.now(UTC)
    anchor = Anchor(last.seq, last.hash, count[0], now.isoformat(timespec="seconds"), run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ledger-anchor-{now:%Y-%m-%d}-seq{last.seq}.json"
    path.write_text(
        json.dumps(
            {
                **asdict(anchor),
                "note": (
                    "Head of the cryptonlp pre-registration ledger. Keep this message: any "
                    "later rewrite of entries up to head_seq changes head_hash."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    conn.execute(
        "INSERT INTO ops.ledger_anchors (run_id, head_seq, head_hash, entries, path) "
        "VALUES (%s, %s, %s, %s, %s)",
        (run_id, anchor.head_seq, anchor.head_hash, anchor.entries, str(path)),
    )
    return anchor, path


def anchor_message(anchor: Anchor) -> str:
    return (
        f"cryptonlp ledger anchor, {anchor.created_at}: entry {anchor.head_seq} of "
        f"{anchor.entries}, hash {anchor.head_hash}"
    )
