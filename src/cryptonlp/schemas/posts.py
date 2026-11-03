"""Posts as collected (raw store) and after normalization and pseudonymization."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints, model_validator

from cryptonlp.schemas.common import (
    Asset,
    NonEmptyStr,
    Origin,
    Record,
    RunId,
    Source,
    UTCDatetime,
)

AuthorHash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class LangMethod(StrEnum):
    DETECTED = "detected"  # from the text
    DECLARED = "declared"  # too little text: the client's tag
    NONE = "none"  # neither: "und"


class DuplicateKind(StrEnum):
    EXACT = "exact"
    NEAR = "near"


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
    """A row of ``clean.posts``: after the language filter, dedupe, entity rules, spam rules and
    pseudonymization (step 3). Filters set flags; ``in_study`` combines them."""

    source: Source
    uri: NonEmptyStr  # kept for deletion sync and for citing evidence
    author_hash: AuthorHash  # HMAC-SHA256 of the author ID; the key lives outside the repo
    text: str
    created_at: UTCDatetime
    received_at: UTCDatetime
    origin: Origin
    lang: NonEmptyStr  # ISO 639-1, or "und"
    lang_method: LangMethod
    lang_confidence: float | None = Field(default=None, ge=0, le=1)
    langs_declared: tuple[str, ...] = ()
    assets: tuple[Asset, ...] = ()
    cashtags: tuple[str, ...] = ()  # "$BTC"
    hashtags: tuple[str, ...] = ()  # lower case, without "#"
    addresses: tuple[str, ...] = ()  # "evm:0x...", "btc:...", "sol:..."
    has_url: bool = False
    spam_rules: tuple[str, ...] = ()
    is_spam: bool = False
    duplicate_of: str | None = None  # URI of the original this post exactly or nearly copies
    duplicate_kind: DuplicateKind | None = None
    near_duplicate_score: float | None = Field(default=None, ge=0, le=1)
    normalize_run_id: RunId

    @property
    def in_study(self) -> bool:
        return (
            self.lang == "en"
            and not self.is_spam
            and self.duplicate_of is None
            and self.origin is Origin.LIVE
        )

    @model_validator(mode="after")
    def _consistent(self) -> NormalizedPost:
        if self.duplicate_of == self.uri:
            raise ValueError("a post cannot be a duplicate of itself")
        if (self.duplicate_of is None) != (self.duplicate_kind is None):
            raise ValueError("duplicate_of and duplicate_kind go together")
        if self.duplicate_kind is DuplicateKind.NEAR and self.near_duplicate_score is None:
            raise ValueError("a near duplicate records its similarity")
        if self.is_spam != bool(self.spam_rules):
            raise ValueError("is_spam is set exactly when a spam rule matched")
        return self
