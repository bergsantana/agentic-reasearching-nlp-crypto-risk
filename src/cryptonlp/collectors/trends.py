"""Hourly snapshots of trending topics. Neither platform keeps a trend history to query later.

Each snapshot stores the responses as received: Bluesky's ``getTrendingTopics`` and
``getTrends``, and each Mastodon instance's ``/api/v1/trends/tags``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime

import httpx
import psycopg
import pyarrow as pa

from cryptonlp import ops
from cryptonlp.collectors.common import Message, Recorder, request_with_backoff, utcnow
from cryptonlp.config import CollectorsConfig
from cryptonlp.rawstore import RawStore

log = logging.getLogger(__name__)

STREAM = "trends"
BLUESKY_ENDPOINTS = (
    ("app.bsky.unspecced.getTrendingTopics", {"limit": "25"}),
    ("app.bsky.unspecced.getTrends", {"limit": "25"}),
)

TREND_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string(), nullable=False),
        pa.field("instance", pa.string(), nullable=False),
        pa.field("endpoint", pa.string(), nullable=False),
        pa.field("fetched_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("payload", pa.string(), nullable=False),
        pa.field("collector_run_id", pa.string(), nullable=False),
    ]
)


def fetch_snapshot(
    client: httpx.Client,
    config: CollectorsConfig,
    clock: Callable[[], datetime] = utcnow,
) -> tuple[list[Message], int]:
    """Fetch every trend endpoint. Returns the responses and the number that failed."""
    requests: list[tuple[str, str, str, dict[str, str]]] = []
    appview = config.bluesky.appview
    for endpoint, params in BLUESKY_ENDPOINTS:
        requests.append(("bluesky", appview, f"{appview}/xrpc/{endpoint}", params))
    for instance in config.mastodon.instances:
        url = f"https://{instance}/api/v1/trends/tags"
        requests.append(("mastodon", instance, url, {"limit": str(config.trends.mastodon_limit)}))

    msgs, failed = [], 0
    for source, instance, url, params in requests:
        try:
            response = request_with_backoff(client, url, params=params, max_tries=3)
        except (httpx.HTTPError, RuntimeError) as exc:
            failed += 1
            log.warning("trends %s failed: %r", url, exc)
            continue
        endpoint = url.rsplit("/", 1)[-1] if source == "bluesky" else "trends/tags"
        msgs.append(
            Message(
                STREAM,
                clock(),
                response.json(),
                {"source": source, "instance": instance, "endpoint": endpoint},
            )
        )
    return msgs, failed


def store_snapshot(
    conn: psycopg.Connection, store: RawStore, msgs: Iterable[Message], run_id: str
) -> int:
    msgs = list(msgs)
    groups: dict[tuple[str, date], list[Message]] = defaultdict(list)
    for m in msgs:
        groups[(m.meta["source"], m.received_at.astimezone(UTC).date())].append(m)
    written = 0
    with conn.transaction():
        for (source, day), group in sorted(groups.items()):
            dataset = f"{source}_trends"
            table = pa.Table.from_pylist(
                [
                    {
                        "source": source,
                        "instance": m.meta["instance"],
                        "endpoint": m.meta["endpoint"],
                        "fetched_at": m.received_at,
                        "payload": json.dumps(m.payload, ensure_ascii=False),
                        "collector_run_id": run_id,
                    }
                    for m in group
                ],
                schema=TREND_SCHEMA,
            )
            meta = {"dataset": dataset, "partition_date": day.isoformat(), "run_id": run_id}
            file = store.write_table(store.partition_dir(dataset, day), table, meta)
            store.register(
                conn,
                file,
                dataset=dataset,
                day=day,
                run_id=run_id,
                received=[m.received_at for m in group],
            )
            written += len(group)
        for source in sorted({m.meta["source"] for m in msgs}):
            ops.save_state(
                conn,
                f"trends:{source}",
                "trends",
                run_id,
                last_event_at=max(m.received_at for m in msgs if m.meta["source"] == source),
            )
    return written


def snapshot(
    conn: psycopg.Connection,
    store: RawStore,
    client: httpx.Client,
    config: CollectorsConfig,
    run_id: str,
    recorder: Recorder | None = None,
) -> dict[str, int]:
    msgs, failed = fetch_snapshot(client, config)
    if recorder:
        for m in msgs:
            recorder.write(m)
        recorder.flush()
    if not msgs:
        raise RuntimeError("every trend endpoint failed")
    return {"responses": store_snapshot(conn, store, msgs, run_id), "failed": failed}
