"""Mastodon: polls hashtag timelines (and, optionally, local public timelines) per instance.

- Pages forward with ``min_id``, so a busy timeline is read without gaps (``since_id`` would
  return only the newest page and skip the rest). The cursor is the newest status id seen.
- A first poll with no cursor takes only the newest page: no backfill of older posts.
- The same status reaches several instances and tags; it is kept once, keyed by its canonical
  ActivityPub ``uri``. The instance and local id it was read from are kept for the deletion
  sweep.
- Reception time is our clock when the page arrived. Mastodon has no stream time.
- Hashtag timelines are the query, so their posts are kept as they are; public timelines are
  filtered by the watch terms.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Callable, Iterable
from datetime import datetime
from html.parser import HTMLParser
from typing import Any

import httpx
import psycopg

from cryptonlp import ops
from cryptonlp.collectors.common import (
    Message,
    Recorder,
    WatchFilter,
    request_with_backoff,
    utcnow,
)
from cryptonlp.config import MastodonConfig
from cryptonlp.rawstore import CollectedPost, PostWriteResult, RawStore
from cryptonlp.schemas import Origin, RawPost, Source

log = logging.getLogger(__name__)

STREAM = "mastodon"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "p":
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(content: str) -> str:
    parser = _TextExtractor()
    parser.feed(content)
    parser.close()
    text = html.unescape("".join(parser.parts))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def timeline_key(instance: str, timeline: str) -> str:
    return f"mastodon:{instance}:{timeline}"


def timeline_path(timeline: str) -> tuple[str, dict[str, str]]:
    if timeline.startswith("tag:"):
        return f"/api/v1/timelines/tag/{timeline.removeprefix('tag:')}", {}
    if timeline == "public:local":
        return "/api/v1/timelines/public", {"local": "true"}
    raise ValueError(f"unknown timeline {timeline!r}")


def parse_status(
    status: dict[str, Any], instance: str, received_at: datetime, run_id: str, origin: Origin
) -> RawPost:
    language = status.get("language")
    text = html_to_text(status.get("content") or "")
    if status.get("spoiler_text"):
        text = f"{status['spoiler_text']}\n\n{text}"
    return RawPost(
        source=Source.MASTODON,
        uri=status["uri"],
        author_id=status["account"]["uri"],
        text=text,
        created_at=datetime.fromisoformat(status["created_at"].replace("Z", "+00:00")),
        received_at=received_at,
        langs=(language,) if language else (),
        origin=origin,
        collector_run_id=run_id,
    )


class MastodonCollector:
    def __init__(
        self,
        conn: psycopg.Connection,
        store: RawStore,
        config: MastodonConfig,
        watch: WatchFilter,
        run_id: str,
        *,
        origin: Origin = Origin.LIVE,
        recorder: Recorder | None = None,
    ) -> None:
        if not conn.autocommit:
            raise ValueError("collectors need an autocommit connection")
        self.conn = conn
        self.store = store
        self.config = config
        self.watch = watch
        self.run_id = run_id
        self.origin = origin
        self.recorder = recorder

    def timelines(self) -> list[str]:
        timelines = [f"tag:{t}" for t in self.config.hashtags]
        if self.config.public_timeline:
            timelines.append("public:local")
        return timelines

    def posts_from_page(self, msg: Message) -> list[CollectedPost]:
        instance, timeline = msg.meta["instance"], msg.meta["timeline"]
        posts = []
        for status in msg.payload:
            if status.get("reblog") or status.get("visibility") != "public":
                continue
            post = parse_status(status, instance, msg.received_at, self.run_id, self.origin)
            if timeline.startswith("tag:"):
                tags = {t.get("name", "").lower() for t in status.get("tags", [])}
                tags.add(timeline.removeprefix("tag:"))
                matched = tuple(f"#{t}" for t in sorted(tags) if t in self.config.hashtags)
            else:
                matched = self.watch.matches(post.text)
                if not matched:
                    continue
            posts.append(
                CollectedPost(
                    post=post,
                    payload=status,
                    matched=matched,
                    instance=instance,
                    native_id=str(status["id"]),
                )
            )
        return posts

    def store_pages(self, pages: Iterable[Message]) -> PostWriteResult:
        posts = [p for msg in pages for p in self.posts_from_page(msg)]
        return self.store.store_posts(
            self.conn,
            posts,
            self.run_id,
            {"collector": STREAM, "watch_sha256": self.watch.sha256},
        )

    def poll(self, client: httpx.Client, clock: Callable[[], datetime] = utcnow) -> dict[str, int]:
        """One pass over every instance and timeline. Each timeline commits on its own."""
        stats = {"pages": 0, "statuses": 0, "written": 0, "duplicates": 0, "failed": 0}
        for instance in self.config.instances:
            for timeline in self.timelines():
                try:
                    self._poll_timeline(client, instance, timeline, clock, stats)
                except (httpx.HTTPError, RuntimeError) as exc:
                    stats["failed"] += 1
                    log.warning("mastodon %s %s failed: %r", instance, timeline, exc)
        if stats["failed"] and stats["failed"] == len(self.config.instances) * len(
            self.timelines()
        ):
            raise RuntimeError("every Mastodon timeline failed")
        return stats

    def _poll_timeline(
        self,
        client: httpx.Client,
        instance: str,
        timeline: str,
        clock: Callable[[], datetime],
        stats: dict[str, int],
    ) -> None:
        key = timeline_key(instance, timeline)
        state = ops.get_state(self.conn, key)
        cursor = state.cursor if state else None
        path, extra = timeline_path(timeline)
        pages: list[Message] = []
        for _ in range(self.config.max_pages):
            params = {**extra, "limit": str(self.config.page_limit)}
            if cursor:
                params["min_id"] = cursor
            response = request_with_backoff(client, f"https://{instance}{path}", params=params)
            msg = Message(
                STREAM,
                clock(),
                response.json(),
                {"instance": instance, "timeline": timeline, "min_id": cursor or ""},
            )
            if self.recorder:
                self.recorder.write(msg)
            pages.append(msg)
            stats["pages"] += 1
            stats["statuses"] += len(msg.payload)
            if msg.payload:
                cursor = str(max(int(s["id"]) for s in msg.payload))
            # A first poll takes one page only; later polls page forward until caught up.
            if state is None or len(msg.payload) < self.config.page_limit:
                break
        with self.conn.transaction():
            result = self.store_pages(pages)
            ops.save_state(
                self.conn,
                key,
                "mastodon",
                self.run_id,
                cursor=cursor,
                last_event_at=pages[-1].received_at if pages else None,
            )
        stats["written"] += result.written
        stats["duplicates"] += result.duplicates
        if self.recorder:
            self.recorder.flush()
