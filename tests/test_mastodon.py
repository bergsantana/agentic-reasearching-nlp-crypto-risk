from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import psycopg
import pytest

from cryptonlp.collectors.common import Message, WatchFilter
from cryptonlp.collectors.mastodon import MastodonCollector, html_to_text
from cryptonlp.config import CollectorsConfig, MastodonConfig
from cryptonlp.rawstore import RawStore
from tests.factories import mastodon_status

pytestmark = pytest.mark.db

NOW = datetime(2026, 11, 3, 14, 0, tzinfo=UTC)


def test_html_to_text() -> None:
    html = '<p>Big <a href="x">#<span>BTC</span></a> news &amp; more</p><p>line<br>two</p>'
    assert html_to_text(html) == "Big #BTC news & more\n\nline\ntwo"


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class FakeTimeline:
    """Serves statuses newest first, honoring ``min_id`` and ``limit`` like Mastodon."""

    def __init__(self, statuses: dict[str, list[dict[str, Any]]]) -> None:
        self.statuses = statuses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        items = self.statuses.get(f"{request.url.host}{request.url.path}", [])
        min_id = int(request.url.params.get("min_id", 0))
        limit = int(request.url.params["limit"])
        newer = sorted((s for s in items if int(s["id"]) > min_id), key=lambda s: int(s["id"]))
        page = newer[:limit] if min_id else newer[-limit:]  # min_id pages forward
        return httpx.Response(200, json=sorted(page, key=lambda s: -int(s["id"])))


def make(
    db: psycopg.Connection,
    store: RawStore,
    watch: WatchFilter,
    run_id: str,
    **overrides: Any,
) -> MastodonCollector:
    fields: dict[str, Any] = {
        "instances": ("a.social", "b.social"),
        "hashtags": ("bitcoin",),
        "page_limit": 2,
    }
    config = MastodonConfig(**(fields | overrides))
    return MastodonCollector(db, store, config, watch, run_id)


def test_first_poll_takes_one_page_then_pages_forward(
    db: psycopg.Connection, store: RawStore, watch: WatchFilter, run_id: str
) -> None:
    timeline = FakeTimeline(
        {
            "a.social/api/v1/timelines/tag/bitcoin": [
                mastodon_status(i, host="a.social") for i in (1, 2, 3)
            ]
        }
    )
    c = make(db, store, watch, run_id)
    with mock_client(timeline) as client:
        first = c.poll(client, clock=lambda: NOW)
        assert first["written"] == 2  # the newest page only: no backfill
        timeline.statuses["a.social/api/v1/timelines/tag/bitcoin"] += [
            mastodon_status(i, host="a.social") for i in (4, 5, 6, 7, 8)
        ]
        second = c.poll(client, clock=lambda: NOW)
    assert second["written"] == 5  # 4..8, in pages of 2, with no gap
    state = db.execute(
        "SELECT cursor FROM ops.collector_state WHERE collector = 'mastodon:a.social:tag:bitcoin'"
    ).fetchone()
    assert state == ("8",)
    sent = [r.url.params.get("min_id") for r in timeline.requests if r.url.host == "a.social"]
    assert sent == [None, "3", "5", "7"]


def test_same_status_on_two_instances_is_kept_once(
    db: psycopg.Connection, store: RawStore, watch: WatchFilter, run_id: str
) -> None:
    shared = mastodon_status(10, host="origin.social")
    timeline = FakeTimeline(
        {
            "a.social/api/v1/timelines/tag/bitcoin": [shared | {"id": "100"}],
            "b.social/api/v1/timelines/tag/bitcoin": [shared | {"id": "200"}],
        }
    )
    with mock_client(timeline) as client:
        stats = make(db, store, watch, run_id).poll(client, clock=lambda: NOW)
    assert (stats["written"], stats["duplicates"]) == (1, 1)
    row = db.execute("SELECT uri, instance, native_id FROM raw.posts").fetchone()
    assert row == ("https://origin.social/users/alice/statuses/10", "a.social", "100")


def test_skips_reblogs_and_non_public_and_records_tags(
    db: psycopg.Connection, store: RawStore, watch: WatchFilter, run_id: str
) -> None:
    timeline = FakeTimeline(
        {
            "a.social/api/v1/timelines/tag/bitcoin": [
                mastodon_status(1, host="a.social", tags=("bitcoin", "news")),
                mastodon_status(2, host="a.social", reblog={"id": "9"}),
                mastodon_status(3, host="a.social", visibility="unlisted"),
            ]
        }
    )
    c = make(db, store, watch, run_id, max_pages=1)
    c.config = c.config.model_copy(update={"page_limit": 40})
    with mock_client(timeline) as client:
        c.poll(client, clock=lambda: NOW)
    assert db.execute("SELECT native_id FROM raw.posts").fetchall() == [("1",)]
    posts = c.posts_from_page(
        Message(
            "mastodon",
            NOW,
            [mastodon_status(1, tags=("bitcoin", "news"))],
            {"instance": "a.social", "timeline": "tag:bitcoin"},
        )
    )
    assert posts[0].matched == ("#bitcoin",)
    assert posts[0].post.text == "$BTC to the moon"
    assert posts[0].post.received_at == NOW


def test_public_timeline_is_filtered_by_watch_terms(
    db: psycopg.Connection, store: RawStore, watch: WatchFilter, run_id: str
) -> None:
    timeline = FakeTimeline(
        {
            "a.social/api/v1/timelines/public": [
                mastodon_status(1, content="<p>ethereum upgrade</p>", tags=()),
                mastodon_status(2, content="<p>my cat</p>", tags=()),
            ]
        }
    )
    c = make(db, store, watch, run_id, hashtags=(), public_timeline=True, page_limit=40)
    with mock_client(timeline) as client:
        c.poll(client, clock=lambda: NOW)
    assert db.execute("SELECT native_id FROM raw.posts").fetchall() == [("1",)]
    assert timeline.requests[0].url.params["local"] == "true"


def test_one_failing_instance_does_not_stop_the_others(
    db: psycopg.Connection, store: RawStore, watch: WatchFilter, run_id: str
) -> None:
    good = FakeTimeline(
        {"b.social/api/v1/timelines/tag/bitcoin": [mastodon_status(1, host="b.social")]}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.social":
            return httpx.Response(403, json={"error": "login required"})
        return good(request)

    with mock_client(handler) as client:
        stats = make(db, store, watch, run_id).poll(client, clock=lambda: NOW)
    assert (stats["failed"], stats["written"]) == (1, 1)


def test_all_failing_raises(
    db: psycopg.Connection, store: RawStore, watch: WatchFilter, run_id: str
) -> None:
    with mock_client(lambda r: httpx.Response(403)) as client, pytest.raises(RuntimeError):
        make(db, store, watch, run_id).poll(client, clock=lambda: NOW)


def test_repo_instances_are_configured(config: CollectorsConfig) -> None:
    assert config.mastodon.instances and config.mastodon.hashtags
