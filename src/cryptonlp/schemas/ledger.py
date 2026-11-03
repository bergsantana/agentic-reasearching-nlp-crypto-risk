"""Records the pre-registration ledger accepts, besides forecasts (``schemas.forecasts``).

The database checks all of this again and adds what only it can know: the sequence number,
the time (its own clock) and the hash chain. These models catch mistakes before the insert.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from pydantic import Field, StringConstraints, model_validator

from cryptonlp.schemas.common import NonEmptyStr, Phase, Record, RunId

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Key = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Seq = Annotated[int, Field(gt=0)]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DataFile(Record):
    """A file an experiment or a score rests on, by path and SHA-256 (e.g. a raw-store file)."""

    path: NonEmptyStr
    sha256: Sha256


class HypothesisKind(StrEnum):
    CONFIRMATORY = "confirmatory"
    EXPLORATORY = "exploratory"


class Proposer(StrEnum):
    RESEARCHER = "researcher"
    ORCHESTRATOR = "orchestrator"
    ANALYST = "analyst"


class Hypothesis(Record):
    hypothesis_key: Key  # "H1", or e.g. "X-2026-11-01" for an exploratory one
    kind: HypothesisKind
    statement: NonEmptyStr  # verbatim; results quote it word for word
    analysis_plan: dict[str, Any] = Field(default_factory=dict)
    proposed_by: Proposer
    supersedes: Seq | None = None  # the entry this revision replaces
    deviation_reason: NonEmptyStr | None = None
    phase: Phase
    run_id: RunId

    @model_validator(mode="after")
    def _valid(self) -> Hypothesis:
        if self.kind is HypothesisKind.CONFIRMATORY and self.proposed_by is not Proposer.RESEARCHER:
            raise ValueError("only the researcher registers confirmatory hypotheses")
        if (self.supersedes is None) != (self.deviation_reason is None):
            raise ValueError("a revision needs both supersedes and deviation_reason")
        return self


class DocumentKind(StrEnum):
    LABELING_MANUAL = "labeling_manual"
    RUBRIC = "rubric"
    ANALYSIS_PLAN = "analysis_plan"
    POWER_SIMULATION = "power_simulation"
    RESULT_TABLES = "result_tables"
    INTERPRETATION_PLAN = "interpretation_plan"
    OTHER = "other"


class Document(Record):
    document_key: Key  # e.g. "labeling-manual"
    kind: DocumentKind
    title: NonEmptyStr
    version: NonEmptyStr
    path: str | None = None
    content_sha256: Sha256
    content: str | None = None
    supersedes: Seq | None = None
    change_reason: NonEmptyStr | None = None
    phase: Phase
    run_id: RunId

    @model_validator(mode="after")
    def _valid(self) -> Document:
        if self.content is not None and sha256_text(self.content) != self.content_sha256:
            raise ValueError("content_sha256 is not the SHA-256 of the content")
        if (self.supersedes is None) != (self.change_reason is None):
            raise ValueError("a new version needs both supersedes and change_reason")
        return self

    @classmethod
    def from_file(cls, path: Path, *, repo_root: Path | None = None, **fields: Any) -> Document:
        """Register a text file with its full content, by its path relative to the repo."""
        content = path.read_text(encoding="utf-8")
        shown = path.resolve()
        if repo_root is not None and shown.is_relative_to(repo_root.resolve()):
            shown = shown.relative_to(repo_root.resolve())
        return cls(path=str(shown), content=content, content_sha256=sha256_text(content), **fields)


class ExperimentPackage(Record):
    """Everything needed to re-run an experiment. The validator agent gets only this."""

    package_key: Key
    hypothesis_seq: Seq | None = None
    description: NonEmptyStr
    code_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{12,40}$")]
    data: tuple[DataFile, ...] = ()
    model_revision: NonEmptyStr | None = None
    prompt_sha256: Sha256 | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    phase: Phase
    run_id: RunId


class Score(Record):
    prediction_seq: Seq
    # Realized volatility, the 0/1 large-move indicator, or the realized return for VaR.
    realized: float = Field(allow_inf_nan=False)
    losses: dict[str, float] = Field(default_factory=dict)  # e.g. {"qlike": ..., "se": ...}
    data: tuple[DataFile, ...] = ()
    phase: Phase
    run_id: RunId


class VerdictKind(StrEnum):
    VERIFICATION = "verification"  # the Verifier on a finding (stage 6)
    REPLICATION = "replication"  # the validator agent on a package (stage 7)


VERDICTS: dict[VerdictKind, frozenset[str]] = {
    VerdictKind.VERIFICATION: frozenset({"supported", "rejected", "needs_review"}),
    VerdictKind.REPLICATION: frozenset({"reproduced", "divergent"}),
}


class Verdict(Record):
    kind: VerdictKind
    agent: Annotated[str, StringConstraints(pattern=r"^(verifier|validator|researcher)$")]
    package_seq: Seq | None = None
    finding_id: UUID | None = None
    verdict: NonEmptyStr
    confidence: float | None = Field(default=None, ge=0, le=1)
    cause: NonEmptyStr | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    phase: Phase
    run_id: RunId

    @model_validator(mode="after")
    def _valid(self) -> Verdict:
        if self.verdict not in VERDICTS[self.kind]:
            raise ValueError(f"a {self.kind} verdict must be one of {sorted(VERDICTS[self.kind])}")
        if self.kind is VerdictKind.REPLICATION and self.package_seq is None:
            raise ValueError("a replication verdict names the package it re-ran")
        if self.kind is VerdictKind.VERIFICATION and self.finding_id is None:
            raise ValueError("a verification verdict names its finding")
        if self.verdict == "divergent" and self.cause is None:
            raise ValueError("a divergent replication records its cause")
        return self


class InterventionType(StrEnum):
    CORRECTION = "correction"
    INTERRUPTION = "interruption"
    APPROVAL = "approval"


class Intervention(Record):
    stage: int = Field(ge=1, le=9)  # research stage, Quadro 5
    type: InterventionType
    at_checkpoint: bool  # a planned stage approval, or an intervention outside the checkpoints
    reason: NonEmptyStr
    actor: NonEmptyStr = "researcher"
    phase: Phase
    run_id: RunId

    @model_validator(mode="after")
    def _valid(self) -> Intervention:
        if self.at_checkpoint and self.type is not InterventionType.APPROVAL:
            raise ValueError("what happens at a checkpoint is an approval")
        return self


class ViolationKind(StrEnum):
    LATE_PREDICTION = "late_prediction"
    CHAIN_BREAK = "chain_break"
    PERMISSION_DENIED = "permission_denied"
    OTHER = "other"


class Violation(Record):
    kind: ViolationKind
    reported_by: NonEmptyStr
    detail: dict[str, Any] = Field(default_factory=dict)
    phase: Phase
    run_id: RunId
