"""Step 6: SemIf's prompt, rubric, readout and labels, against a fake vLLM server."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from cryptonlp.classify import conformance
from cryptonlp.classify.base import PostText
from cryptonlp.classify.config import (
    ClassifyConfig,
    Rubric,
    load_classify_config,
    load_rubric,
)
from cryptonlp.classify.prompt import ChatTemplate, Decision, Option, render_prompt, sha256_text
from cryptonlp.classify.semif import (
    SemIfClassifier,
    SemIfScorer,
    single_label,
    topic_label,
)
from cryptonlp.classify.vllm import VllmClient, VllmError
from cryptonlp.schemas.labels import Task
from cryptonlp.settings import REPO_ROOT
from tests.fakes import FakeVllm

RUN = "20261104T000000Z-0123456789ab"
AT = datetime(2026, 11, 4, tzinfo=UTC)


@pytest.fixture(scope="module")
def cfg() -> ClassifyConfig:
    return load_classify_config()


def fixture_rows() -> list[dict[str, object]]:
    return conformance.load_fixture()


# --- the prompt is SemIf's, byte for byte ----------------------------------------------------


def test_prompts_match_semif_reference_hashes(cfg: ClassifyConfig) -> None:
    rows = conformance.load_fixture()
    assert len(rows) == 144
    for r in rows:
        prompt = render_prompt(cfg.chat_template, conformance.to_decision(r))
        assert sha256_text(prompt) == r["reference"]["prompt_sha256"], r["id"]


def test_chat_template_is_the_pinned_revisions() -> None:
    t = ChatTemplate(REPO_ROOT / "config" / "semif" / "qwen3.5-4b-chat_template.jinja")
    # Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
    assert t.sha256 == "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"


def test_thinking_is_off_and_the_post_comes_first(cfg: ClassifyConfig) -> None:
    d = Decision("x", 'Say "hi" $BTC', "Q?", (Option("a", "A"), Option("b", "B")))
    prompt = render_prompt(cfg.chat_template, d)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert '{"evidence": "Say \\"hi\\" $BTC", "criterion": "Q?"' in prompt


# --- rubric and versions -----------------------------------------------------------------------


def test_rubric_covers_the_four_tasks(cfg: ClassifyConfig) -> None:
    r = cfg.rubric
    assert [d.task for d in r.decisions].count(Task.TOPIC) == 6
    assert {d.task for d in r.decisions} == {Task.SENTIMENT, Task.TOPIC, Task.SCAM, Task.EVENT}


def test_rubric_rejects_a_wrong_label_set(tmp_path: Path) -> None:
    raw = (REPO_ROOT / "config" / "semif" / "rubric.toml").read_text()
    bad = tmp_path / "rubric.toml"
    bad.write_text(raw.replace('id = "likely_scam"', 'id = "scam"'))
    with pytest.raises(ValueError, match="scam"):
        load_rubric(bad)


def test_version_changes_with_the_rubric_and_the_temperatures(cfg: ClassifyConfig) -> None:
    hotter = cfg.model_copy(
        update={
            "semif": cfg.semif.model_copy(
                update={"temperatures": cfg.semif.temperatures.model_copy(update={"topic": 2.0})}
            )
        }
    )
    assert hotter.prompt_sha256 == cfg.prompt_sha256  # temperatures are not the prompt
    assert hotter.semif_version != cfg.semif_version
    assert cfg.semif_version.startswith("semif-851bf6e8-")
    data = cfg.rubric.model_dump(mode="json")
    data["decisions"][0]["question"] += " Really."
    edited = cfg.model_copy()
    edited.__dict__["rubric"] = Rubric.model_validate(data)  # the cached property
    assert edited.prompt_sha256 != cfg.prompt_sha256
    assert edited.semif_version != cfg.semif_version


# --- readout -----------------------------------------------------------------------------------


def test_single_label_is_softmax_argmax_and_temperature_keeps_the_choice() -> None:
    a = single_label(("negative", "neutral", "positive"), [1.0, 3.0, 2.0], 1.0)
    b = single_label(("negative", "neutral", "positive"), [1.0, 3.0, 2.0], 2.5)
    assert a.values == b.values == ("neutral",)
    assert sum(a.probabilities.values()) == pytest.approx(1)
    assert b.confidence < a.confidence  # a higher temperature, a flatter distribution


def test_topic_label_takes_every_yes_and_other_when_none() -> None:
    two = topic_label({"macro": (0.0, 2.0), "regulation": (1.0, 3.0), "price_talk": (2.0, 0.0)}, 1)
    assert two.values == ("macro", "regulation")
    none = topic_label({"macro": (2.0, 0.0), "regulation": (3.0, 1.0)}, 1)
    assert none.values == ("other",)
    p = none.probabilities
    assert p["other"] == pytest.approx((1 - p["macro"]) * (1 - p["regulation"]))


def serve(cfg: ClassifyConfig, **kw: object) -> tuple[FakeVllm, SemIfClassifier]:
    fake = FakeVllm(cfg.semif.served_name, cfg.chat_template, **kw)  # type: ignore[arg-type]
    client = VllmClient(
        "http://semif", cfg.semif.served_name, transport=fake.transport(), concurrency=4
    )
    return fake, SemIfClassifier(cfg, client)


POSTS = [
    PostText("bluesky", f"at://did:plc:a/app.bsky.feed.post/{i}", f"$BTC post {i}", AT)
    for i in range(5)
]


def test_classifier_labels_every_post_and_task(cfg: ClassifyConfig) -> None:
    fake, clf = serve(cfg)
    check = clf.check()
    assert check["model_version"] == cfg.semif_version
    labels = clf.classify(POSTS, labeled_at=AT, run_id=RUN)
    assert fake.completions == 9 * len(POSTS)
    assert len(labels) == 4 * len(POSTS)
    assert {lb.model_version for lb in labels} == {cfg.semif_version}
    for lb in labels:
        assert lb.confidence is not None and 0 <= lb.confidence <= 1
        if lb.task is not Task.TOPIC:
            assert sum(lb.probabilities.values()) == pytest.approx(1)
    # Every prompt for a post starts with the same prefix: the post (vLLM's cache reuses it).
    first = [p for p in fake.prompts if '"$BTC post 0"' in p]
    assert len(first) == 9
    prefix = first[0][: first[0].index('"criterion"')]
    assert all(p.startswith(prefix) for p in first)


def test_labels_follow_the_logits(cfg: ClassifyConfig) -> None:
    def logits(prompt: str, ids: Sequence[int]) -> list[float]:
        # "positive" is the third sentiment option (C); every topic says yes (B); scam
        # picks promotion (B); event picks hack (B).
        best = 2 if "holding the crypto" in prompt else 1
        return [5.0 if i == best else 0.0 for i in range(len(ids))]

    _, clf = serve(cfg, logits_for=logits)
    labels = {lb.task: lb for lb in clf.classify(POSTS[:1], labeled_at=AT, run_id=RUN)}
    assert labels[Task.SENTIMENT].values == ("positive",)
    assert set(labels[Task.TOPIC].values) == {d.label for d in cfg.rubric.for_task(Task.TOPIC)}
    assert labels[Task.SCAM].values == ("promotion",)
    assert labels[Task.EVENT].values == ("hack",)


def test_a_server_on_another_revision_is_refused(cfg: ClassifyConfig) -> None:
    fake = FakeVllm("Qwen/Qwen3.5-4B@0000000000000000000000000000000000000000", cfg.chat_template)
    client = VllmClient("http://semif", cfg.semif.served_name, transport=fake.transport())
    with pytest.raises(VllmError, match="does not serve"):
        SemIfClassifier(cfg, client).check()


def test_conformance_compares_with_semif_reference(cfg: ClassifyConfig) -> None:
    rows = conformance.load_fixture()[:12]
    ref = {r["id"]: r["reference"]["option_logits"] for r in rows}

    def as_reference(prompt: str, ids: Sequence[int]) -> list[float]:
        payload = json.loads(prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0])
        row = next(
            r
            for r in rows
            if r["question"] == payload["criterion"] and r["state"] == payload["evidence"]
        )
        return [float(x) + 0.01 for x in ref[row["id"]]]

    fake = FakeVllm(cfg.semif.served_name, cfg.chat_template, logits_for=as_reference)
    client = VllmClient("http://semif", cfg.semif.served_name, transport=fake.transport())
    result = conformance.check(SemIfScorer(client, cfg.chat_template), rows)
    assert result.passed and result.prompt_hash_matches == 12 and result.agreement == 1
    assert result.max_abs_logit_diff == pytest.approx(0.01)
    flipped = [
        dict(
            r,
            reference=dict(
                r["reference"], option_logits=list(reversed(r["reference"]["option_logits"]))
            ),
        )
        for r in rows
    ]
    worse = conformance.check(SemIfScorer(client, cfg.chat_template), flipped)
    assert not worse.passed


def test_softmax_matches_numpy() -> None:
    from cryptonlp.classify.semif import softmax

    x = [22.0, 26.375, 24.375]
    e = np.exp(np.array(x) - max(x))
    assert softmax(x) == pytest.approx(list(e / e.sum()))
