"""Types shared by every schema."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, StringConstraints

from cryptonlp.run_id import RUN_ID_PATTERN

BAR = timedelta(hours=1)


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _on_the_hour(value: datetime) -> datetime:
    if (value.minute, value.second, value.microsecond) != (0, 0, 0):
        raise ValueError(f"bar times must fall on the hour (UTC), got {value.isoformat()}")
    return value


# Naive datetimes are rejected; aware ones are converted to UTC.
UTCDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
# The open of an hourly UTC bar.
BarOpen = Annotated[AwareDatetime, AfterValidator(_to_utc), AfterValidator(_on_the_hour)]
RunId = Annotated[str, StringConstraints(pattern=RUN_ID_PATTERN)]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


def floor_to_bar(value: datetime) -> datetime:
    """Open of the hourly UTC bar that contains ``value``."""
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


class Record(BaseModel):
    """Base for all schemas: immutable, and unknown fields are an error, not silently dropped."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Source(StrEnum):
    BLUESKY = "bluesky"
    MASTODON = "mastodon"


class Origin(StrEnum):
    """Where a post came from. Only ``live`` posts may enter the confirmatory study."""

    LIVE = "live"
    BACKFILL = "backfill"
    REPLAY = "replay"


class Phase(StrEnum):
    """Keeps development, pilot and confirmatory rows apart in the ledger."""

    DEVELOPMENT = "development"
    PILOT = "pilot"
    CONFIRMATORY = "confirmatory"


class Asset(StrEnum):
    BTC = "BTC"
    ETH = "ETH"
