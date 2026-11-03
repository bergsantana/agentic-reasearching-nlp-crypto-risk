"""Findings produced by agents. Every finding cites the posts it rests on."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field

from cryptonlp.schemas.common import NonEmptyStr, Phase, Record, RunId, UTCDatetime


class AgentRole(StrEnum):
    ORCHESTRATOR = "orchestrator"
    TREND_SCOUT = "trend_scout"
    ANALYST = "analyst"
    VERIFIER = "verifier"
    CURATOR = "curator"
    REPORTER = "reporter"
    VALIDATOR = "validator"


class Finding(Record):
    finding_id: UUID = Field(default_factory=uuid4)
    agent: AgentRole
    claim: NonEmptyStr
    # A finding with no evidence is rejected, not stored. The Verifier checks single-source ones.
    evidence_post_uris: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    created_at: UTCDatetime
    phase: Phase
    run_id: RunId
