"""The classifier interface. SemIf scores four tasks in one pass; entities come from rules and a
lexicon in normalization (claims M6), so no classifier labels ``entity``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from cryptonlp.schemas.labels import Label, Task


@dataclass(frozen=True)
class PostText:
    """What a classifier sees of a post: its text, and nothing about the author."""

    source: str
    uri: str
    text: str
    received_at: datetime


class Classifier(Protocol):
    # Written on every label: "stub", or the pinned model revision.
    model_version: str
    # Every post gets exactly one label for each of these tasks.
    tasks: frozenset[Task]

    def classify(
        self, posts: Sequence[PostText], *, labeled_at: datetime, run_id: str
    ) -> list[Label]: ...
