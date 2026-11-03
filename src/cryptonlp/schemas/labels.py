"""Labels for the five NLP tasks, from a model or from the human annotator.

The label sets follow the design doc. They become final when the labeling manual is
registered in the ledger (step 4); change them there first, then here.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field, model_validator

from cryptonlp.schemas.common import NonEmptyStr, Record, RunId, UTCDatetime


class Task(StrEnum):
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    SCAM = "scam"
    ENTITY = "entity"
    EVENT = "event"


TASK_LABELS: dict[Task, frozenset[str]] = {
    Task.SENTIMENT: frozenset({"negative", "neutral", "positive"}),
    Task.TOPIC: frozenset(
        {
            "price_talk",
            "regulation",
            "security_incident",
            "protocol_tech",
            "launch_airdrop",
            "macro",
            "other",
        }
    ),
    Task.SCAM: frozenset({"organic", "promotion", "likely_scam"}),
    Task.EVENT: frozenset({"hack", "outage", "listing", "regulatory_action", "none"}),
}
MULTI_LABEL_TASKS = frozenset({Task.TOPIC, Task.ENTITY})
# Entity labels name the type and the linked entity, e.g. "token:BTC" or "exchange:binance";
# "none" alone when the post names no entity in scope (the labeling manual, task 5).
ENTITY_LABEL_RE = re.compile(r"^(token|project|exchange):[a-z0-9_.-]+$", re.IGNORECASE)
NO_ENTITY = "none"


class Annotator(StrEnum):
    MODEL = "model"
    HUMAN = "human"


class Label(Record):
    """One task's label for one post. Labels are appended, never overwritten."""

    post_uri: NonEmptyStr
    task: Task
    values: tuple[str, ...] = Field(min_length=1)
    # Per-option probabilities from the model (SemIf reads these in one forward pass).
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0, le=1)
    annotator: Annotator
    # "stub", the pinned model revision, or the human pass (e.g. "human:pass1", "human:relabel").
    model_version: NonEmptyStr
    labeled_at: UTCDatetime
    run_id: RunId

    @model_validator(mode="after")
    def _valid_for_task(self) -> Label:
        if len(set(self.values)) != len(self.values):
            raise ValueError("duplicate label values")
        if self.task not in MULTI_LABEL_TASKS and len(self.values) != 1:
            raise ValueError(f"task {self.task} takes exactly one label")
        if self.task is Task.ENTITY:
            if NO_ENTITY in self.values and len(self.values) > 1:
                raise ValueError(f"entity {NO_ENTITY!r} cannot go with other entities")
            bad = [v for v in self.values if v != NO_ENTITY and not ENTITY_LABEL_RE.match(v)]
        else:
            bad = [v for v in self.values if v not in TASK_LABELS[self.task]]
        if bad:
            raise ValueError(f"invalid labels for task {self.task}: {bad}")
        if self.task is not Task.ENTITY and not set(self.probabilities) <= TASK_LABELS[self.task]:
            raise ValueError("probabilities name options outside the task's label set")
        if any(not 0 <= p <= 1 for p in self.probabilities.values()):
            raise ValueError("probabilities must lie in [0, 1]")
        if self.annotator is Annotator.MODEL and self.confidence is None:
            raise ValueError("model labels need a confidence score")
        return self
