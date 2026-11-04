"""SemIf (Lee, 2026) as the study's classifier, behind the ``Classifier`` interface.

Each post is the state of nine SemIf decisions from the rubric (``config/semif/rubric.toml``):
sentiment, scam and event, one decision each, and one yes/no decision per topic, since topic
takes several labels and SemIf picks one option (claims M21). The nine prompts share the post as
their prefix, which vLLM's prefix cache computes once.

Readout, as SemIf's direct mode: the raw logits of the option letters at the last position,
softmaxed. Calibration divides them by one temperature per task before the softmax (SemIf's
``CALIBRATION.md``), which never changes the chosen option.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cryptonlp.classify.base import PostText
from cryptonlp.classify.config import (
    NO,
    SEMIF_TASKS,
    TOPIC_OTHER,
    YES,
    ClassifyConfig,
    RubricDecision,
)
from cryptonlp.classify.prompt import (
    LETTERS,
    ChatTemplate,
    Decision,
    direct_messages,
    render_prompt,
    sha256_text,
)
from cryptonlp.classify.vllm import VllmClient, VllmError
from cryptonlp.schemas.labels import Annotator, Label, Task

log = logging.getLogger(__name__)


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("need finite scores")
    scaled = [v / temperature for v in values]
    top = max(scaled)
    weights = [math.exp(v - top) for v in scaled]
    total = sum(weights)
    return [w / total for w in weights]


@dataclass(frozen=True)
class Scored:
    """One decision as SemIf answered it: raw logits per option, and the exact prompt's hash."""

    decision_id: str
    option_ids: tuple[str, ...]
    logits: tuple[float, ...]
    prompt_sha256: str


class SemIfScorer:
    """Scores any SemIf decision on the vLLM server. Generic: the conformance check runs
    SemIf's own benchmark rows through it."""

    def __init__(self, client: VllmClient, template: ChatTemplate) -> None:
        self.client = client
        self.template = template
        self._letters: list[int] | None = None

    def letter_ids(self, n: int) -> list[int]:
        """Token IDs of the answer letters, each one exact token (SemIf's ``_slot_ids``)."""
        if self._letters is None:
            ids = []
            for letter in LETTERS:
                tokens = self.client.tokenize(letter)
                if len(tokens) != 1:
                    raise VllmError(f"answer letter {letter!r} is not one token: {tokens}")
                ids.append(tokens[0])
            if len(set(ids)) != len(ids):
                raise VllmError("answer-letter tokens collide")
            self._letters = ids
        return self._letters[:n]

    def check_template(self, decision: Decision) -> dict[str, Any]:
        """Our rendering of the chat template against the server's, token for token, and
        SemIf's answer-boundary check: appending a letter adds exactly that letter's token."""
        prompt = render_prompt(self.template, decision)
        ours = self.client.tokenize(prompt)
        theirs = self.client.tokenize_chat(direct_messages(decision))
        if ours != theirs:
            raise VllmError(
                "the server renders the chat template differently from config/semif "
                f"({len(ours)} vs {len(theirs)} tokens)"
            )
        letters = self.letter_ids(len(decision.options))
        for letter, token in zip(LETTERS, letters, strict=False):
            if self.client.tokenize(prompt + letter) != [*ours, token]:
                raise VllmError(f"answer letter {letter} changes the tokenization of the prompt")
        return {"prompt_tokens": len(ours), "letter_ids": letters}

    def score_one(self, decision: Decision) -> Scored:
        prompt = render_prompt(self.template, decision)
        logits = self.client.option_logits(prompt, self.letter_ids(len(decision.options)))
        return Scored(decision.id, decision.option_ids, tuple(logits), sha256_text(prompt))

    def score(self, decisions: Sequence[Decision]) -> list[Scored]:
        self.letter_ids(len(LETTERS))  # once, before the threads start
        return self.client.map(self.score_one, decisions)


def decisions_for(post: PostText, rubric: Sequence[RubricDecision]) -> list[Decision]:
    """The post's nine decisions, in rubric order. Decision IDs are ``<uri>#<rubric id>``."""
    return [
        Decision(f"{post.uri}#{d.id}", post.text, d.question, d.prompt_options()) for d in rubric
    ]


@dataclass(frozen=True)
class TaskLabel:
    values: tuple[str, ...]
    probabilities: dict[str, float]
    confidence: float


def single_label(option_ids: Sequence[str], logits: Sequence[float], t: float) -> TaskLabel:
    """Argmax of softmax(logits / T); ties go to the first option, as SemIf's argmax."""
    p = softmax(logits, t)
    best = max(range(len(p)), key=lambda i: (p[i], -i))
    return TaskLabel((option_ids[best],), dict(zip(option_ids, p, strict=True)), p[best])


def topic_label(answers: dict[str, tuple[float, float]], t: float) -> TaskLabel:
    """Topics from one yes/no decision each, ``answers[topic] = (logit no, logit yes)``. A
    topic is chosen when "yes" wins; none chosen is "other". Probabilities: P(yes) per topic,
    and for "other" the chance that no topic is a yes (decisions taken as independent).
    Confidence: the chance of the whole chosen set under the same assumption."""
    p_yes = {topic: softmax(pair, t)[1] for topic, pair in answers.items()}
    chosen = tuple(sorted(topic for topic, p in p_yes.items() if p > 0.5))
    p_other = math.prod(1 - p for p in p_yes.values())
    confidence = math.prod(max(p, 1 - p) for p in p_yes.values())
    return TaskLabel(chosen or (TOPIC_OTHER,), {**p_yes, TOPIC_OTHER: p_other}, confidence)


def task_labels(config: ClassifyConfig, scored: dict[str, Scored]) -> dict[Task, TaskLabel]:
    """Turn one post's scored decisions (keyed by rubric decision ID) into task labels, with
    the configured temperatures."""
    temps = config.semif.temperatures
    out: dict[Task, TaskLabel] = {}
    topics: dict[str, tuple[float, float]] = {}
    for d in config.rubric.decisions:
        s = scored[d.id]
        if d.task is Task.TOPIC:
            by_id = dict(zip(s.option_ids, s.logits, strict=True))
            assert d.label is not None
            topics[d.label] = (by_id[NO], by_id[YES])
        else:
            out[d.task] = single_label(s.option_ids, s.logits, temps.of(d.task))
    out[Task.TOPIC] = topic_label(topics, temps.of(Task.TOPIC))
    return out


class SemIfClassifier:
    """Implements ``cryptonlp.classify.Classifier``."""

    tasks = frozenset(SEMIF_TASKS)

    def __init__(self, config: ClassifyConfig, client: VllmClient) -> None:
        self.config = config
        self.client = client
        self.scorer = SemIfScorer(client, config.chat_template)
        self.model_version = config.semif_version

    @property
    def spec(self) -> dict[str, Any]:
        return self.config.semif_spec

    @classmethod
    def connect(cls, config: ClassifyConfig, url: str | None = None) -> SemIfClassifier:
        s = config.semif
        client = VllmClient(
            url or s.url, s.served_name, timeout=s.timeout_seconds, concurrency=s.concurrency
        )
        return cls(config, client)

    def close(self) -> None:
        self.client.close()

    def check(self) -> dict[str, Any]:
        """The server serves the pinned revision and renders prompts as we do."""
        served = self.client.served()
        sentiment = self.config.rubric.for_task(Task.SENTIMENT)[0]
        canary = Decision(
            "canary", "$BTC just broke 100k, LFG", sentiment.question, sentiment.prompt_options()
        )
        return {
            "served": served.id,
            "max_model_len": served.max_model_len,
            **self.scorer.check_template(canary),
            "model_version": self.model_version,
            "prompt_sha256": self.config.prompt_sha256,
        }

    def score_posts(self, posts: Sequence[PostText]) -> dict[str, dict[str, Scored]]:
        """Raw SemIf output: ``{uri: {rubric decision ID: Scored}}`` (the bake-off keeps it)."""
        rubric = self.config.rubric.decisions
        decisions = [d for p in posts for d in decisions_for(p, rubric)]
        scored = self.scorer.score(decisions)
        out: dict[str, dict[str, Scored]] = {p.uri: {} for p in posts}
        for s in scored:
            uri, _, rubric_id = s.decision_id.rpartition("#")
            out[uri][rubric_id] = s
        return out

    def classify(
        self, posts: Sequence[PostText], *, labeled_at: datetime, run_id: str
    ) -> list[Label]:
        raw = self.score_posts(posts)
        labels: list[Label] = []
        for post in posts:
            for task, tl in task_labels(self.config, raw[post.uri]).items():
                labels.append(
                    Label(
                        post_uri=post.uri,
                        task=task,
                        values=tl.values,
                        probabilities=tl.probabilities,
                        confidence=tl.confidence,
                        annotator=Annotator.MODEL,
                        model_version=self.model_version,
                        labeled_at=labeled_at,
                        run_id=run_id,
                    )
                )
        return labels
