"""Posts as collected (raw store) and after normalization and pseudonymization."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints, model_validator

from cryptonlp.schemas.common import NonEmptyStr, Origin, Record, RunId, Source, UTCDatetime

AuthorHash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RawPost(Record):
    """A post exactly as the collector received it. Written once to the raw store, never edited."""

    source: Source
    uri: NonEmptyStr  # at://... for Bluesky, the status URL for Mastodon; kept for deletion sync
    author_id: NonEmptyStr  # DID or account URI as received; replaced by a keyed hash in step 3
    text: str
    # Set by the posting client. On Bluesky it can be backdated, so never use it for leakage rules.
    created_at: UTCDatetime
    # When the source delivered the post (Jetstream's time_us on Bluesky). Leakage rules use this.
    received_at: UTCDatetime
    langs: tuple[str, ...] = ()  # language tags declared by the client, if any
    origin: Origin = Origin.LIVE
    collector_run_id: RunId


class NormalizedPost(Record):
    """A post after language filtering, dedupe, entity extraction and pseudonymization."""

    source: Source
    uri: NonEmptyStr
    author_hash: AuthorHash  # keyed hash of author_id; the key lives outside the repo
    text: str
    created_at: UTCDatetime
    received_at: UTCDatetime
    lang: NonEmptyStr  # detected language
    origin: Origin
    cashtags: tuple[str, ...] = ()
    tickers: tuple[str, ...] = ()
    contract_addresses: tuple[str, ...] = ()
    is_spam: bool = False
    duplicate_of: str | None = None  # URI of the post this one exactly or nearly duplicates
    near_duplicate_score: float | None = Field(default=None, ge=0, le=1)
    normalize_run_id: RunId

    @model_validator(mode="after")
    def _duplicate_not_self(self) -> NormalizedPost:
        if self.duplicate_of == self.uri:
            raise ValueError("a post cannot be a duplicate of itself")
        return self
