"""Embeddings of in-study posts into ``clean.embeddings`` (pgvector), for the Analyst's semantic
search (step 8). The model is served by vLLM (Compose service ``embed``).

Posts are embedded as they are, with no instruction prefix: Qwen3-Embedding takes the
instruction on the query side only. A post longer than ``max_tokens`` is cut to its first
``max_tokens`` tokens (the server's context; posts that long are rare).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import psycopg

from cryptonlp.classify.config import EmbedSection
from cryptonlp.classify.vllm import VllmClient, VllmError

PENDING_SQL = """
SELECT p.source, p.uri, p.text
FROM clean.study_posts p
WHERE NOT EXISTS (
    SELECT 1 FROM clean.embeddings e
    WHERE e.source = p.source AND e.uri = p.uri AND e.model_version = %(version)s
)
ORDER BY p.received_at, p.source, p.uri
LIMIT %(limit)s
"""

# A post deleted at the source while its batch was being embedded is skipped.
INSERT_SQL = """
INSERT INTO clean.embeddings (source, uri, model_version, embedding, run_id)
SELECT %(source)s, %(uri)s, %(version)s, %(embedding)s::vector, %(run_id)s
WHERE EXISTS (SELECT 1 FROM clean.posts WHERE source = %(source)s AND uri = %(uri)s)
ON CONFLICT DO NOTHING
"""


def vector_literal(values: Sequence[float]) -> str:
    if any(not math.isfinite(v) for v in values):
        raise VllmError("embedding has non-finite values")
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def embed_pending(
    conn: psycopg.Connection,
    client: VllmClient,
    cfg: EmbedSection,
    *,
    run_id: str,
    limit: int | None = None,
) -> int:
    """Embed every in-study post that has no vector from this model yet (up to ``limit``).
    ``conn`` must be autocommit; each batch is one transaction. Returns posts embedded."""
    total = 0
    while limit is None or total < limit:
        size = cfg.batch if limit is None else min(cfg.batch, limit - total)
        rows = conn.execute(PENDING_SQL, {"version": cfg.model_version, "limit": size}).fetchall()
        if not rows:
            break
        vectors = client.embed([r[2] for r in rows], max_tokens=cfg.max_tokens)
        if any(len(v) != cfg.dimensions for v in vectors):
            raise VllmError(f"expected {cfg.dimensions} dimensions from {cfg.model}")
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                INSERT_SQL,
                [
                    {
                        "source": source,
                        "uri": uri,
                        "version": cfg.model_version,
                        "embedding": vector_literal(vec),
                        "run_id": run_id,
                    }
                    for (source, uri, _), vec in zip(rows, vectors, strict=True)
                ],
            )
        total += len(rows)
    return total


def connect(cfg: EmbedSection, url: str | None = None) -> VllmClient:
    return VllmClient(url or cfg.url, cfg.served_name, concurrency=4)
