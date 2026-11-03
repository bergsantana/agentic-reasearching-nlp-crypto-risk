"""Normalization settings from ``config/normalize.toml``."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import Field, StringConstraints, field_validator, model_validator

from cryptonlp.schemas.common import NonEmptyStr, Record
from cryptonlp.settings import Settings

RuleName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


class LanguageConfig(Record):
    min_letters: int = Field(default=15, ge=0)
    min_confidence: float = Field(default=0.0, ge=0, le=1)


class DedupeConfig(Record):
    threshold: float = Field(default=0.8, gt=0, le=1)
    window_days: int = Field(default=7, gt=0)
    shingle_chars: int = Field(default=5, gt=0)
    num_perm: int = Field(default=64, gt=0)
    bands: int = Field(default=16, gt=0)
    min_chars: int = Field(default=30, ge=0)

    @model_validator(mode="after")
    def _bands_divide(self) -> DedupeConfig:
        if self.num_perm % self.bands:
            raise ValueError("bands must divide num_perm")
        return self

    @property
    def rows(self) -> int:
        return self.num_perm // self.bands


class EntitiesConfig(Record):
    assets: dict[NonEmptyStr, tuple[NonEmptyStr, ...]]


class SpamRule(Record):
    name: RuleName
    pattern: NonEmptyStr
    requires_url: bool = False
    # The rule does not apply when this also matches, e.g. a warning ("never share your seed
    # phrase") that uses the same words as the scam it warns about.
    unless: NonEmptyStr | None = None

    @field_validator("pattern", "unless")
    @classmethod
    def _compiles(cls, value: str | None) -> str | None:
        if value is not None:
            re.compile(value)
        return value


class SpamConfig(Record):
    max_cashtags: int = Field(default=6, gt=0)
    max_hashtags: int = Field(default=10, gt=0)
    rules: tuple[SpamRule, ...] = ()

    @model_validator(mode="after")
    def _unique_names(self) -> SpamConfig:
        names = [r.name for r in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("spam rule names must be unique")
        return self


class NormalizeConfig(Record):
    language: LanguageConfig = LanguageConfig()
    dedupe: DedupeConfig = DedupeConfig()
    entities: EntitiesConfig
    spam: SpamConfig = SpamConfig()

    @property
    def sha256(self) -> str:
        """Hash of the settings as loaded, stored with every row they produced."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_normalize_config(path: Path | None = None) -> NormalizeConfig:
    path = path or Settings().normalize_config_path
    with path.open("rb") as f:
        return NormalizeConfig.model_validate(tomllib.load(f))
