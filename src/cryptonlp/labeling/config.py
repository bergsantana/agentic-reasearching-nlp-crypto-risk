"""Labeling settings from ``config/labeling.toml``."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from datetime import timedelta
from functools import cached_property
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from cryptonlp.schemas.common import Record, Source
from cryptonlp.settings import Settings


class SampleSection(Record):
    target: int = Field(default=2000, gt=0)
    batch_size: int = Field(default=500, gt=0)
    settle_minutes: int = Field(default=60, ge=0)
    sources: tuple[Source, ...] = (Source.BLUESKY, Source.MASTODON)

    @property
    def settle(self) -> timedelta:
        return timedelta(minutes=self.settle_minutes)


class EnrichSection(Record):
    share: float = Field(default=0.10, ge=0, lt=1)
    addresses: bool = True
    patterns: tuple[str, ...] = ()

    @field_validator("patterns")
    @classmethod
    def _compiles(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for p in patterns:
            re.compile(p)
        return patterns


class RelabelSection(Record):
    size: int = Field(default=200, gt=0)
    min_gap_days: int = Field(default=14, ge=0)

    @property
    def min_gap(self) -> timedelta:
        return timedelta(days=self.min_gap_days)


class SplitSection(Record):
    calibration: int = Field(default=500, gt=0)


class StudioSection(Record):
    url: str = "http://127.0.0.1:8080"


class LabelingConfig(Record):
    sample: SampleSection = SampleSection()
    enrich: EnrichSection = EnrichSection()
    relabel: RelabelSection = RelabelSection()
    split: SplitSection = SplitSection()
    studio: StudioSection = StudioSection()

    @model_validator(mode="after")
    def _sizes_fit(self) -> LabelingConfig:
        if self.split.calibration >= self.sample.target:
            raise ValueError("the calibration slice must be smaller than the labeled set")
        if self.relabel.size > self.sample.target:
            raise ValueError("cannot re-label more posts than the labeled set holds")
        return self

    @property
    def sha256(self) -> str:
        """Hash of the settings as loaded (the studio URL left out: it is not the method)."""
        method = self.model_dump(mode="json", exclude={"studio"})
        canonical = json.dumps(method, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @cached_property
    def candidate_re(self) -> re.Pattern[str] | None:
        if not self.enrich.patterns:
            return None
        return re.compile("|".join(f"(?:{p})" for p in self.enrich.patterns), re.IGNORECASE)

    def is_candidate(self, text: str, addresses: list[str] | tuple[str, ...]) -> bool:
        """Does the post look like a scam or promotion candidate (the enrichment stratum)?"""
        if self.enrich.addresses and addresses:
            return True
        rx = self.candidate_re
        return rx is not None and rx.search(text) is not None


def load_labeling_config(path: Path | None = None) -> LabelingConfig:
    path = path or Settings().labeling_config_path
    with path.open("rb") as f:
        return LabelingConfig.model_validate(tomllib.load(f))
