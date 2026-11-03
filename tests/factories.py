"""Builders for the payloads the sources deliver, shaped like real ones (checked Nov 2026)."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any

from cryptonlp.collectors.bluesky import STREAM, to_us
from cryptonlp.collectors.common import Message

T0 = datetime(2026, 11, 3, 14, 0, tzinfo=UTC)


def jetstream_create(
    text: str,
    at: datetime,
    did: str = "did:plc:alice",
    rkey: str = "3k1",
    langs: list[str] | None = None,
) -> Message:
    event = {
        "did": did,
        "time_us": to_us(at),
        "cursor": 26743695402,
        "kind": "commit",
        "commit": {
            "rev": "3mzg5vijulz2z",
            "operation": "create",
            "collection": "app.bsky.feed.post",
            "rkey": rkey,
            "record": {
                "$type": "app.bsky.feed.post",
                "createdAt": (at - timedelta(seconds=2)).isoformat().replace("+00:00", "Z"),
                "langs": langs or ["en"],
                "text": text,
            },
            "cid": "bafyrei",
        },
    }
    return Message(STREAM, at, event)


def jetstream_delete(at: datetime, did: str = "did:plc:alice", rkey: str = "3k1") -> Message:
    event = {
        "did": did,
        "time_us": to_us(at),
        "kind": "commit",
        "commit": {
            "rev": "3mzg5vh3h422c",
            "operation": "delete",
            "collection": "app.bsky.feed.post",
            "rkey": rkey,
        },
    }
    return Message(STREAM, at, event)


def jetstream_identity(at: datetime) -> Message:
    event = {"did": "did:plc:bob", "time_us": to_us(at), "kind": "identity", "identity": {}}
    return Message(STREAM, at, event)


def mastodon_status(
    status_id: int,
    content: str = "<p>$BTC to the moon</p>",
    tags: tuple[str, ...] = ("bitcoin",),
    host: str = "mastodon.social",
    user: str = "alice",
    **overrides: Any,
) -> dict[str, Any]:
    status = {
        "id": str(status_id),
        "created_at": "2026-11-03T13:59:00.000Z",
        "visibility": "public",
        "language": "en",
        "uri": f"https://{host}/users/{user}/statuses/{status_id}",
        "url": f"https://{host}/@{user}/{status_id}",
        "content": content,
        "spoiler_text": "",
        "reblog": None,
        "account": {"acct": user, "uri": f"https://{host}/users/{user}"},
        "tags": [{"name": t} for t in tags],
    }
    return status | overrides


def kline(open_time: datetime, minutes: int = 1, price: float = 77000.0) -> list[Any]:
    """A klines row as the API returns it (millisecond times, decimal strings)."""
    ms = (open_time - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)
    close_ms = ms + minutes * 60_000 - 1
    p = f"{price:.8f}"
    return [
        ms,
        p,
        f"{price + 10:.8f}",
        f"{price - 10:.8f}",
        p,
        "1.5",
        close_ms,
        "115500.0",
        42,
        "0.7",
        "53900.0",
        "0",
    ]


def vision_zip(name: str, rows: list[list[Any]], microseconds: bool = True) -> bytes:
    """A Vision archive: one headerless CSV, microsecond times as in files since 2025."""
    lines = []
    for r in rows:
        r = list(r)
        if microseconds:
            r[0], r[6] = int(r[0]) * 1000, int(r[6]) * 1000 + 999
        lines.append(",".join(str(x) for x in r))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name.replace(".zip", ".csv"), "\n".join(lines) + "\n")
    return buf.getvalue()
