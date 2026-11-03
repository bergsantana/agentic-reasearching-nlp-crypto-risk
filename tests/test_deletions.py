from __future__ import annotations

from datetime import timedelta

import httpx
import psycopg
import pytest

from cryptonlp.collectors import deletions
from cryptonlp.config import CollectorsConfig
from cryptonlp.rawstore import CollectedPost, RawStore
from cryptonlp.schemas import RawPost, Source
from tests.factories import T0

pytestmark = pytest.mark.db


def bsky(n: int, run_id: str) -> CollectedPost:
    return CollectedPost(
        post=RawPost(
            source=Source.BLUESKY,
            uri=f"at://did:plc:a/app.bsky.feed.post/{n}",
            author_id="did:plc:a",
            text="bitcoin",
            created_at=T0,
            received_at=T0 + timedelta(seconds=n),
            collector_run_id=run_id,
        ),
        payload={},
    )


def masto(n: int, run_id: str) -> CollectedPost:
    return CollectedPost(
        post=RawPost(
            source=Source.MASTODON,
            uri=f"https://origin.social/users/a/statuses/{n}",
            author_id="https://origin.social/users/a",
            text="bitcoin",
            created_at=T0,
            received_at=T0 + timedelta(seconds=n),
            collector_run_id=run_id,
        ),
        payload={},
        instance="a.social",
        native_id=str(1000 + n),
    )


def test_sync_finds_deleted_posts_on_both_sources_and_purges(
    db: psycopg.Connection, store: RawStore, config: CollectorsConfig, run_id: str
) -> None:
    store.store_posts(db, [bsky(n, run_id) for n in range(30)], run_id)
    store.store_posts(db, [masto(n, run_id) for n in range(3)], run_id)
    gone_bsky = {bsky(n, run_id).post.uri for n in (3, 27)}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("getPosts"):
            uris = request.url.params.get_list("uris")
            return httpx.Response(
                200, json={"posts": [{"uri": u} for u in uris if u not in gone_bsky]}
            )
        if request.url.path == "/api/v1/statuses/1001":
            return httpx.Response(404, json={"error": "Record not found"})
        return httpx.Response(200, json={"id": request.url.path.rsplit("/", 1)[-1]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        stats = deletions.sync_deletions(db, store, client, config, run_id)
    assert stats["bluesky_checked"] == 30 and stats["bluesky_deleted"] == 2
    assert stats["mastodon_checked"] == 3 and stats["mastodon_deleted"] == 1
    assert stats["files_rewritten"] == 2 and stats["posts_purged"] == 3
    batches = [len(r.url.params.get_list("uris")) for r in seen if "getPosts" in r.url.path]
    assert batches == [25, 5]
    assert {r.url.host for r in seen if "statuses" in r.url.path} == {"a.social"}

    deleted = {
        r[0] for r in db.execute("SELECT uri FROM raw.deletions WHERE purged_at IS NOT NULL")
    }
    assert deleted == gone_bsky | {masto(1, run_id).post.uri}
    unchecked = db.execute(
        "SELECT count(*) FROM raw.posts p WHERE last_checked_at IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM raw.deletions d WHERE d.uri = p.uri)"
    ).fetchone()
    assert unchecked == (0,)
    assert store.verify(db).ok


def test_sweep_checks_least_recently_checked_first(
    db: psycopg.Connection, store: RawStore, config: CollectorsConfig, run_id: str
) -> None:
    store.store_posts(db, [bsky(n, run_id) for n in range(3)], run_id)
    small = config.model_copy(
        update={"deletions": config.deletions.model_copy(update={"bluesky_batch": 2})}
    )
    asked: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        uris = request.url.params.get_list("uris")
        asked.append(uris)
        return httpx.Response(200, json={"posts": [{"uri": u} for u in uris]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        deletions.sweep_bluesky(db, client, small, run_id)
        deletions.sweep_bluesky(db, client, small, run_id)
    assert asked[1][0] == bsky(2, run_id).post.uri  # the one not checked in the first run
