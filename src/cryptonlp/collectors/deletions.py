"""Deletion sync: find posts deleted at the source, then purge them from the raw store.

Bluesky deletions mostly arrive in the stream (see ``bluesky.py``). This sweep also catches
the ones missed while the collector was down, and Mastodon deletions, which have no stream
here. Each run checks the posts checked least recently first, so the whole store is covered
over successive runs.

- Bluesky: ``app.bsky.feed.getPosts`` on the public AppView, 25 URIs per request. A URI that
  doesn't come back was deleted, or its account was deleted, deactivated or taken down. All
  of these mean the post should no longer be kept.
- Mastodon: ``GET /api/v1/statuses/:id`` on the instance the post was read from. 404 or 410
  means deleted.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx
import psycopg

from cryptonlp.collectors.common import RateLimitedError, request_with_backoff
from cryptonlp.config import CollectorsConfig
from cryptonlp.rawstore import RawStore

log = logging.getLogger(__name__)

GET_POSTS_BATCH = 25


def _mark(
    conn: psycopg.Connection, source: str, alive: list[str], gone: list[str], run_id: str
) -> None:
    with conn.transaction():
        if alive:
            conn.execute(
                "UPDATE raw.posts SET last_checked_at = now() WHERE source = %s AND uri = ANY(%s)",
                (source, alive),
            )
        if gone:
            conn.execute(
                "INSERT INTO raw.deletions (source, uri, method, run_id) "
                "SELECT %s, u, 'sweep', %s FROM unnest(%s::text[]) AS u "
                "ON CONFLICT (source, uri) DO NOTHING",
                (source, run_id, gone),
            )


def _due(
    conn: psycopg.Connection, source: str, limit: int, instance: str | None = None
) -> list[tuple[str, str | None]]:
    rows = conn.execute(
        """
        SELECT p.uri, p.native_id FROM raw.posts p
        WHERE p.source = %s AND p.origin <> 'replay'
          AND (%s::text IS NULL OR p.instance = %s)
          AND NOT EXISTS (SELECT 1 FROM raw.deletions d WHERE d.source = p.source AND d.uri = p.uri)
        ORDER BY p.last_checked_at NULLS FIRST, p.received_at
        LIMIT %s
        """,
        (source, instance, instance, limit),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def sweep_bluesky(
    conn: psycopg.Connection, client: httpx.Client, config: CollectorsConfig, run_id: str
) -> dict[str, int]:
    stats = {"checked": 0, "deleted": 0}
    uris = [u for u, _ in _due(conn, "bluesky", config.deletions.bluesky_batch)]
    url = f"{config.bluesky.appview}/xrpc/app.bsky.feed.getPosts"
    for i in range(0, len(uris), GET_POSTS_BATCH):
        batch = uris[i : i + GET_POSTS_BATCH]
        response = request_with_backoff(client, url, params=[("uris", u) for u in batch])
        found = {p["uri"] for p in response.json().get("posts", [])}
        gone = [u for u in batch if u not in found]
        _mark(conn, "bluesky", [u for u in batch if u in found], gone, run_id)
        stats["checked"] += len(batch)
        stats["deleted"] += len(gone)
    return stats


def sweep_mastodon(
    conn: psycopg.Connection,
    client: httpx.Client,
    config: CollectorsConfig,
    run_id: str,
    sleep: Callable[[float], None] = time.sleep,
    spacing: float = 0.5,
) -> dict[str, int]:
    """One request per status; ``spacing`` keeps each instance well under 300 requests/5 min."""
    stats = {"checked": 0, "deleted": 0, "errors": 0}
    instances = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT instance FROM raw.posts WHERE source = 'mastodon' "
            "AND instance IS NOT NULL ORDER BY 1"
        )
    ]
    for instance in instances:
        due = _due(conn, "mastodon", config.deletions.mastodon_batch_per_instance, instance)
        alive: list[str] = []
        gone: list[str] = []
        for uri, native_id in due:
            try:
                response = request_with_backoff(
                    client,
                    f"https://{instance}/api/v1/statuses/{native_id}",
                    ok_statuses=frozenset({200, 404, 410}),
                    max_tries=3,
                )
            except RateLimitedError:
                log.warning("rate limited by %s; stopping this instance for now", instance)
                break
            except httpx.HTTPError as exc:
                stats["errors"] += 1
                log.warning("status check %s failed: %r", uri, exc)
                continue
            (alive if response.status_code == 200 else gone).append(uri)
            stats["checked"] += 1
            sleep(spacing)
        _mark(conn, "mastodon", alive, gone, run_id)
        stats["deleted"] += len(gone)
    return stats


def sync_deletions(
    conn: psycopg.Connection,
    store: RawStore,
    client: httpx.Client,
    config: CollectorsConfig,
    run_id: str,
) -> dict[str, int]:
    stats: dict[str, int] = {}
    for name, sweep in (("bluesky", sweep_bluesky), ("mastodon", sweep_mastodon)):
        for k, v in sweep(conn, client, config, run_id).items():
            stats[f"{name}_{k}"] = v
    stats.update(store.purge_deleted(conn, run_id))
    return stats
