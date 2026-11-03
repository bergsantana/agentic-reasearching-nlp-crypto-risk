"""Pieces every collector shares: messages, recording and replay, the watch filter, HTTP."""

from __future__ import annotations

import gzip
import json
import logging
import random
import re
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from cryptonlp.config import WatchConfig

log = logging.getLogger(__name__)

USER_AGENT = "cryptonlp-research/0.1 (academic research; non-commercial)"


def utcnow() -> datetime:
    return datetime.now(UTC)


# --- messages, recording and replay --------------------------------------------------------


@dataclass(frozen=True)
class Message:
    """One unit a source delivered, exactly as received: a Jetstream event, a page of statuses,
    a klines response. Collectors process messages, so live data, recordings and replays all
    take the same path."""

    stream: str  # "jetstream", "mastodon", "trends", "binance_klines"
    received_at: datetime
    payload: Any
    meta: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "stream": self.stream,
                "received_at": self.received_at.isoformat(),
                "meta": dict(self.meta),
                "payload": self.payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> Message:
        d = json.loads(line)
        return cls(
            stream=d["stream"],
            received_at=datetime.fromisoformat(d["received_at"]).astimezone(UTC),
            payload=d["payload"],
            meta=d.get("meta", {}),
        )


class Recorder:
    """Appends messages to ``<dir>/<UTC day>/<stream>.jsonl.gz``, for the replay day.

    Recordings go to ``data/replay`` (git-ignored). ``cryptonlp replay freeze`` copies one
    complete day into ``tests/fixtures/replay``.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._files: dict[tuple[str, str], gzip.GzipFile] = {}

    def write(self, msg: Message) -> None:
        day = msg.received_at.astimezone(UTC).date().isoformat()
        key = (day, msg.stream)
        f = self._files.get(key)
        if f is None:
            for old in [k for k in self._files if k[0] != day]:
                self._files.pop(old).close()
            path = self.directory / day / f"{msg.stream}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            f = self._files[key] = gzip.GzipFile(path, "ab")
        f.write((msg.to_json() + "\n").encode())

    def flush(self) -> None:
        for f in self._files.values():
            f.flush()

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()


def read_recording(path: Path) -> Iterator[Message]:
    """Messages from one recorded stream file, in the order they were received."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Message.from_json(line)


def recording_files(directory: Path, stream: str) -> list[Path]:
    """Every recorded file for a stream under a replay directory (one day or many)."""
    return sorted(directory.rglob(f"{stream}.jsonl.gz"))


# --- watch filter --------------------------------------------------------------------------


class WatchFilter:
    """Whole-word, case-insensitive match on the watch terms; cashtags need the ``$``."""

    def __init__(self, config: WatchConfig) -> None:
        self.sha256 = config.sha256
        terms = sorted({t.lower() for t in config.terms}, key=len, reverse=True)
        words = "|".join(re.escape(t).replace(r"\ ", r"[\s-]+") for t in terms)
        tags = "|".join(
            re.escape(t.lower()) for t in sorted(config.cashtags, key=len, reverse=True)
        )
        self._terms = re.compile(rf"(?<![\w$#])[$#]?({words})(?![\w])", re.IGNORECASE)
        self._cashtags = (
            re.compile(rf"(?<![\w$])\$({tags})(?![\w])", re.IGNORECASE) if tags else None
        )

    def matches(self, text: str) -> tuple[str, ...]:
        """The watch terms found in ``text``, lower-cased and de-duplicated, in order found."""
        found: dict[str, None] = {}
        for m in self._terms.finditer(text):
            found[re.sub(r"[\s-]+", " ", m.group(1).lower())] = None
        if self._cashtags:
            for m in self._cashtags.finditer(text):
                found["$" + m.group(1).lower()] = None
        return tuple(found)


# --- HTTP ----------------------------------------------------------------------------------


class RateLimitedError(RuntimeError):
    pass


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value) - utcnow()).total_seconds())
        except (TypeError, ValueError):
            return None


def request_with_backoff(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, Any] | list[tuple[str, Any]] | None = None,
    max_tries: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 120.0,
    ok_statuses: frozenset[int] = frozenset({200}),
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """GET with exponential backoff on network errors, 429, 418 and 5xx.

    Honors ``Retry-After``. Any status in ``ok_statuses`` is returned to the caller (e.g. 404
    for "deleted" in the deletion sweep); other 4xx statuses raise at once.
    """
    for attempt in range(1, max_tries + 1):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError as exc:
            if attempt == max_tries:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * (1 + random.random() / 4)
            log.warning("GET %s failed (%s); retrying in %.1fs", url, exc, delay)
            sleep(delay)
            continue
        if response.status_code in ok_statuses:
            return response
        retryable = response.status_code in (418, 429) or response.status_code >= 500
        if not retryable or attempt == max_tries:
            if response.status_code in (418, 429):
                raise RateLimitedError(f"rate limited by {url}: HTTP {response.status_code}")
            response.raise_for_status()
            return response
        delay = _retry_after(response) or min(max_delay, base_delay * 2 ** (attempt - 1))
        log.warning("GET %s: HTTP %s; retrying in %.1fs", url, response.status_code, delay)
        sleep(min(delay, max_delay * 5))
    raise AssertionError("unreachable")


def http_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
