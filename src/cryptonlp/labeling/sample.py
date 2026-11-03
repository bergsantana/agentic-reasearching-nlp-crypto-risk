"""Drawing posts for labeling: the daily first-pass batches and the re-labeling batch.

First pass: one batch per UTC day, from that day's in-study posts. Strata are disjoint: per
platform, scam or promotion candidates (``config/labeling.toml``, ``[enrich]``) and all other
posts. Candidates get ``enrich.share`` of the batch, the other posts the rest, each split across
platforms in proportion to the day's volume. Within a stratum, a simple random sample. Each
batch records every stratum's population N and sample n, so estimates can be re-weighted by
N/n, and the seed, so the draw can be repeated on the same rows.

Re-labeling: posts whose first-pass label was last edited at least ``relabel.min_gap_days``
ago, in proportion to platform.
"""

from __future__ import annotations

import json
import random
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import psycopg
from psycopg.types.json import Jsonb

from cryptonlp.labeling.config import LabelingConfig

PASS1 = "pass1"
RELABEL = "relabel"


class DrawError(RuntimeError):
    pass


def allocate(total: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Split ``total`` across strata in proportion to their sizes (largest remainder), never
    more than a stratum holds. Ties go to the larger stratum, then by name."""
    total = min(total, sum(sizes.values()))
    alloc = dict.fromkeys(sizes, 0)
    while total > 0:
        open_ = {k: n - alloc[k] for k, n in sizes.items() if n > alloc[k]}
        weight = sum(open_.values())
        quotas = {k: total * n / weight for k, n in open_.items()}
        given = 0
        for k, q in quotas.items():
            take = min(int(q), open_[k])
            alloc[k] += take
            given += take
        rest = total - given
        order = sorted(open_, key=lambda k: (-(quotas[k] - int(quotas[k])), -sizes[k], k))
        for k in order:
            if rest == 0:
                break
            if alloc[k] < sizes[k]:
                alloc[k] += 1
                rest -= 1
        total = rest
    return alloc


@dataclass(frozen=True)
class Candidate:
    source: str
    uri: str
    stratum: str


@dataclass(frozen=True)
class Batch:
    batch_id: int
    pass_: str
    stratum_day: date | None
    size: int
    strata: dict[str, dict[str, int]]
    seed: int


def stratified_sample(
    population: Sequence[Candidate], quotas: Mapping[str, int], rng: random.Random
) -> list[Candidate]:
    by_stratum: dict[str, list[Candidate]] = {}
    for c in sorted(population, key=lambda c: (c.source, c.uri)):
        by_stratum.setdefault(c.stratum, []).append(c)
    chosen: list[Candidate] = []
    for stratum in sorted(quotas):
        chosen.extend(rng.sample(by_stratum.get(stratum, []), quotas[stratum]))
    rng.shuffle(chosen)  # labeling order: strata mixed, so the annotator can't tell them apart
    return chosen


def plan_pass1(
    config: LabelingConfig, population: Sequence[Candidate], size: int
) -> dict[str, int]:
    """Sample size per stratum for one first-pass batch."""
    counts: dict[str, int] = {}
    for c in population:
        counts[c.stratum] = counts.get(c.stratum, 0) + 1
    cand = {k: n for k, n in counts.items() if k.endswith(":candidate")}
    other = {k: n for k, n in counts.items() if k.endswith(":other")}
    n_cand = min(round(size * config.enrich.share), sum(cand.values()))
    quotas = allocate(n_cand, cand)
    quotas |= allocate(size - sum(quotas.values()), other)
    return quotas


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time(), tzinfo=UTC)
    return start, start + timedelta(days=1)


def _insert(
    conn: psycopg.Connection,
    *,
    pass_: str,
    day: date | None,
    chosen: Sequence[Candidate],
    strata: dict[str, dict[str, int]],
    seed: int,
    config: LabelingConfig,
    run_id: str,
) -> Batch:
    with conn.transaction():
        row = conn.execute(
            "INSERT INTO labeling.batches (pass, stratum_day, size, seed, strata, config_sha256, "
            "run_id) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING batch_id",
            (pass_, day, len(chosen), seed, Jsonb(strata), config.sha256, run_id),
        ).fetchone()
        assert row is not None
        batch_id = int(row[0])
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO labeling.items (batch_id, pass, source, uri, stratum, position) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [(batch_id, pass_, c.source, c.uri, c.stratum, i) for i, c in enumerate(chosen)],
            )
    return Batch(batch_id, pass_, day, len(chosen), strata, seed)


def draw_pass1(
    conn: psycopg.Connection,
    config: LabelingConfig,
    day: date,
    *,
    now: datetime,
    run_id: str,
    size: int | None = None,
    seed: int | None = None,
) -> Batch:
    """Draw a first-pass batch from the in-study posts received on ``day`` (UTC). A second
    batch from the same day (a top-up) needs an explicit ``size``."""
    start, end = _day_bounds(day)
    if now < end + config.sample.settle:
        raise DrawError(
            f"{day} can be drawn from {end + config.sample.settle:%Y-%m-%d %H:%M} UTC, "
            "once every post of the day is on disk and normalized"
        )
    existing = conn.execute(
        "SELECT count(*) FROM labeling.batches WHERE pass = %s AND stratum_day = %s",
        (PASS1, day),
    ).fetchone()
    if existing and existing[0] and size is None:
        raise DrawError(f"{day} already has a batch: pass an explicit size for a top-up")
    size = size or config.sample.batch_size

    rows = conn.execute(
        """
        SELECT p.source, p.uri, p.text, p.addresses
        FROM clean.study_posts p
        WHERE p.received_at >= %s AND p.received_at < %s AND p.source = ANY(%s)
          AND NOT EXISTS (
              SELECT 1 FROM labeling.items i
              WHERE i.pass = %s AND i.source = p.source AND i.uri = p.uri
          )
        """,
        (start, end, [s.value for s in config.sample.sources], PASS1),
    ).fetchall()
    population = [
        Candidate(
            source,
            uri,
            f"{source}:{'candidate' if config.is_candidate(text, addresses) else 'other'}",
        )
        for source, uri, text, addresses in rows
    ]
    if len(population) < size:
        raise DrawError(f"{day} has {len(population)} in-study posts left, fewer than {size}")
    quotas = plan_pass1(config, population, size)
    seed = secrets.randbits(63) if seed is None else seed
    chosen = stratified_sample(population, quotas, random.Random(seed))
    strata = {
        k: {"N": sum(c.stratum == k for c in population), "n": quotas.get(k, 0)}
        for k in sorted({c.stratum for c in population})
    }
    return _insert(
        conn,
        pass_=PASS1,
        day=day,
        chosen=chosen,
        strata=strata,
        seed=seed,
        config=config,
        run_id=run_id,
    )


def relabel_eligible(
    conn: psycopg.Connection, config: LabelingConfig, now: datetime
) -> list[Candidate]:
    """First-pass posts with a usable label last edited at least the minimum gap ago, not yet
    re-labeled."""
    rows = conn.execute(
        """
        SELECT i.source, i.uri
        FROM labeling.items i
        JOIN labeling.current c USING (item_id)
        WHERE i.pass = %s AND c.labels IS NOT NULL AND c.updated_at <= %s
          AND NOT EXISTS (
              SELECT 1 FROM labeling.items r
              WHERE r.pass = %s AND r.source = i.source AND r.uri = i.uri
          )
        """,
        (PASS1, now - config.relabel.min_gap, RELABEL),
    ).fetchall()
    return [Candidate(source, uri, source) for source, uri in rows]


def draw_relabel(
    conn: psycopg.Connection,
    config: LabelingConfig,
    *,
    now: datetime,
    run_id: str,
    seed: int | None = None,
) -> Batch:
    done = conn.execute("SELECT count(*) FROM labeling.batches WHERE pass = %s", (RELABEL,))
    row = done.fetchone()
    if row and row[0]:
        raise DrawError("the re-labeling batch has been drawn already")
    population = relabel_eligible(conn, config, now)
    size = config.relabel.size
    if len(population) < size:
        raise DrawError(
            f"only {len(population)} posts were labeled at least "
            f"{config.relabel.min_gap_days} days ago; {size} are needed"
        )
    counts: dict[str, int] = {}
    for c in population:
        counts[c.stratum] = counts.get(c.stratum, 0) + 1
    quotas = allocate(size, counts)
    seed = secrets.randbits(63) if seed is None else seed
    chosen = stratified_sample(population, quotas, random.Random(seed))
    strata = {k: {"N": counts[k], "n": quotas[k]} for k in sorted(counts)}
    return _insert(
        conn,
        pass_=RELABEL,
        day=None,
        chosen=chosen,
        strata=strata,
        seed=seed,
        config=config,
        run_id=run_id,
    )


def describe(batch: Batch) -> str:
    return json.dumps(
        {
            "batch_id": batch.batch_id,
            "pass": batch.pass_,
            "day": batch.stratum_day.isoformat() if batch.stratum_day else None,
            "size": batch.size,
            "strata": batch.strata,
            "seed": batch.seed,
        },
        indent=2,
    )
