"""Classifier settings (``config/classify.toml``) and the SemIf rubric (``config/semif/``)."""

from __future__ import annotations

import hashlib
import json
import tomllib
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from cryptonlp.classify.prompt import DIRECT_SYSTEM, PROMPT_VERSION, ChatTemplate, Option
from cryptonlp.schemas.common import Record
from cryptonlp.schemas.labels import TASK_LABELS, Task
from cryptonlp.settings import REPO_ROOT, Settings

# Tasks SemIf labels; entities come from rules in normalization (claims M6).
SEMIF_TASKS = (Task.SENTIMENT, Task.TOPIC, Task.SCAM, Task.EVENT)
TOPIC_OTHER = "other"
NO, YES = "no", "yes"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# --- rubric --------------------------------------------------------------------------------


class RubricOption(Record):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class RubricDecision(Record):
    id: str = Field(min_length=1)
    task: Task
    # Topic only: the topic this yes/no decision is about.
    label: str | None = None
    question: str = Field(min_length=1)
    options: tuple[RubricOption, ...] = Field(min_length=2, max_length=16)

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(o.id for o in self.options)

    def prompt_options(self) -> tuple[Option, ...]:
        return tuple(Option(o.id, o.description) for o in self.options)


class Rubric(Record):
    version: str = Field(min_length=1)
    decisions: tuple[RubricDecision, ...]

    @model_validator(mode="after")
    def _covers_the_tasks(self) -> Rubric:
        ids = [d.id for d in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("decision ids must be unique")
        topics: list[str] = []
        for task in SEMIF_TASKS:
            mine = [d for d in self.decisions if d.task is task]
            if task is Task.TOPIC:
                for d in mine:
                    if d.label is None or set(d.option_ids) != {NO, YES}:
                        raise ValueError(f"{d.id}: a topic decision needs a label and no/yes")
                    topics.append(d.label)
                continue
            if len(mine) != 1:
                raise ValueError(f"the rubric needs exactly one {task.value} decision")
            if mine[0].label is not None or set(mine[0].option_ids) != TASK_LABELS[task]:
                raise ValueError(f"{task.value}: options must be {sorted(TASK_LABELS[task])}")
        expected = TASK_LABELS[Task.TOPIC] - {TOPIC_OTHER}
        if sorted(topics) != sorted(expected):
            raise ValueError(f"topic decisions must cover {sorted(expected)} once each")
        if any(d.task not in SEMIF_TASKS for d in self.decisions):
            raise ValueError(f"the rubric labels only {[t.value for t in SEMIF_TASKS]}")
        return self

    def for_task(self, task: Task) -> tuple[RubricDecision, ...]:
        return tuple(d for d in self.decisions if d.task is task)

    @property
    def sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def load_rubric(path: Path) -> Rubric:
    with path.open("rb") as f:
        return Rubric.model_validate(tomllib.load(f))


# --- settings ------------------------------------------------------------------------------


class LoopSection(Record):
    classifier: Literal["semif", "stub"] = "semif"


class Temperatures(Record):
    sentiment: float = Field(default=1.0, gt=0)
    topic: float = Field(default=1.0, gt=0)
    scam: float = Field(default=1.0, gt=0)
    event: float = Field(default=1.0, gt=0)

    def of(self, task: Task) -> float:
        return float(getattr(self, task.value))


class SemIfSection(Record):
    url: str = "http://127.0.0.1:8001"
    model: str = "Qwen/Qwen3.5-4B"
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dtype: str = "bfloat16"
    chat_template: str
    rubric: str
    concurrency: int = Field(default=48, gt=0)
    timeout_seconds: float = Field(default=120, gt=0)
    temperatures: Temperatures = Temperatures()

    @property
    def served_name(self) -> str:
        """The name the server must serve the model under (Compose passes the same): a server
        running another revision refuses every request instead of labeling with it."""
        return f"{self.model}@{self.revision}"


class EmbedSection(Record):
    url: str = "http://127.0.0.1:8002"
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dimensions: int = Field(default=1024, gt=0)
    batch: int = Field(default=64, gt=0)
    max_tokens: int = Field(default=2048, gt=0)

    @property
    def served_name(self) -> str:
        return f"{self.model}@{self.revision}"

    @property
    def model_version(self) -> str:
        return f"{self.model}@{self.revision[:12]}"


class HfModel(Record):
    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    subfolder: str | None = None
    max_tokens: int = Field(default=512, gt=0)
    dtype: Literal["float32", "bfloat16"] = "float32"
    # The model's own label -> ours (sentiment baselines only).
    labels: dict[str, str] = Field(default_factory=dict)


class BakeoffSection(Record):
    mlflow_experiment: str = "classifier-bakeoff"
    ece_bins: int = Field(default=15, gt=1)
    repeat: int = Field(default=200, ge=0)
    cryptobert: HfModel
    finbert: HfModel
    openjev: HfModel

    @model_validator(mode="after")
    def _sentiment_maps(self) -> BakeoffSection:
        for name in ("cryptobert", "finbert"):
            m: HfModel = getattr(self, name)
            if sorted(m.labels.values()) != sorted(TASK_LABELS[Task.SENTIMENT]):
                raise ValueError(f"{name}: labels must map onto negative, neutral, positive")
        return self


class H4Section(Record):
    decided: bool = False
    margin: float = Field(ge=0, lt=1)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    bootstrap: int = Field(default=10000, ge=100)
    seed: int = 20261104


class FallbackSection(Record):
    """When a task stops using SemIf (config/classify.toml explains the three rules)."""

    alpha: float = Field(default=0.025, gt=0, lt=0.5)
    min_macro_f1: float = Field(ge=0, le=1)
    min_scam_recall: float = Field(ge=0, le=1)


class ClassifyConfig(Record):
    loop: LoopSection = LoopSection()
    semif: SemIfSection
    embed: EmbedSection
    bakeoff: BakeoffSection
    h4: H4Section
    fallback: FallbackSection

    @cached_property
    def rubric(self) -> Rubric:
        return load_rubric(_resolve(self.semif.rubric))

    @cached_property
    def chat_template(self) -> ChatTemplate:
        return ChatTemplate(_resolve(self.semif.chat_template))

    @property
    def prompt_spec(self) -> dict[str, Any]:
        """Everything that decides the prompts SemIf sees, besides the post."""
        return {
            "system": DIRECT_SYSTEM,
            "prompt_version": PROMPT_VERSION,
            "chat_template_sha256": self.chat_template.sha256,
            "rubric": self.rubric.model_dump(mode="json"),
        }

    @property
    def prompt_sha256(self) -> str:
        """The prompt hash registered with H4 and cited by every SemIf label's model version."""
        return sha256_json(self.prompt_spec)

    @property
    def semif_spec(self) -> dict[str, Any]:
        """Everything that decides a SemIf label: model, readout, prompts and temperatures."""
        s = self.semif
        return {
            "classifier": "semif",
            "model": s.model,
            "revision": s.revision,
            "dtype": s.dtype,
            "readout": "last-position logits of the option letters, softmax(logits / T)",
            "prompt_sha256": self.prompt_sha256,
            "rubric_version": self.rubric.version,
            "temperatures": s.temperatures.model_dump(mode="json"),
        }

    @property
    def semif_version(self) -> str:
        """The model version on SemIf's labels: the revision and a hash of the whole spec."""
        return f"semif-{self.semif.revision[:8]}-{sha256_json(self.semif_spec)[:12]}"


def load_classify_config(path: Path | None = None) -> ClassifyConfig:
    path = path or Settings().classify_config_path
    with path.open("rb") as f:
        return ClassifyConfig.model_validate(tomllib.load(f))
