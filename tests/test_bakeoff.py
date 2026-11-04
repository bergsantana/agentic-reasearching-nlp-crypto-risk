"""Step 6 on the test database: SemIf labels and embeddings in the store, and the bake-off from
the frozen labeled set to the H4 test, with a fake vLLM server and fake baseline scores."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest

from cryptonlp.classify import bakeoff, classify_pending, embed, register
from cryptonlp.classify.config import ClassifyConfig, load_classify_config
from cryptonlp.classify.semif import SemIfClassifier
from cryptonlp.classify.vllm import VllmClient
from cryptonlp.labeling import sample, store
from cryptonlp.schemas.common import Phase
from cryptonlp.stubs import StubClassifier
from tests.fakes import FakeVllm
from tests.test_labeling import AFTER_DAY, DAY, SMALL, FakeStudio, day, region, studio
from tests.test_loop import clear_ledger

__all__ = ["day"]  # the labeled day's fixture, shared with test_labeling

RUN = "20261105T000000Z-0123456789ab"


@pytest.fixture
def cfg(day: psycopg.Connection) -> ClassifyConfig:
    day.execute("TRUNCATE clean.classify_batches, clean.classifier_versions")
    return load_classify_config()


def semif(cfg: ClassifyConfig) -> tuple[FakeVllm, SemIfClassifier]:
    fake = FakeVllm(cfg.semif.served_name, cfg.chat_template)
    client = VllmClient(
        "http://semif", cfg.semif.served_name, transport=fake.transport(), concurrency=8
    )
    return fake, SemIfClassifier(cfg, client)


# --- labels and embeddings in the store ------------------------------------------------------


@pytest.mark.db
def test_semif_labels_the_store_once_per_version(
    day: psycopg.Connection, cfg: ClassifyConfig
) -> None:
    _, clf = semif(cfg)
    clock = iter([AFTER_DAY, AFTER_DAY + timedelta(minutes=1), AFTER_DAY + timedelta(minutes=2)])
    n = classify_pending(
        day, clf, now=AFTER_DAY, run_id=RUN, batch_size=25, clock=lambda: next(clock)
    )
    assert n == 40
    assert classify_pending(day, clf, now=AFTER_DAY, run_id=RUN) == 0  # nothing left
    rows = day.execute(
        "SELECT task, count(*), min(labeled_at), max(labeled_at) FROM clean.labels "
        "WHERE model_version = %s GROUP BY 1 ORDER BY 1",
        (cfg.semif_version,),
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [
        ("event", 40),
        ("scam", 40),
        ("sentiment", 40),
        ("topic", 40),
    ]
    # Two batches (25 + 15), each stamped when it was done.
    assert rows[0][2] == AFTER_DAY and rows[0][3] == AFTER_DAY + timedelta(minutes=1)
    spec = day.execute(
        "SELECT spec, prompt_sha256 FROM clean.classifier_versions WHERE model_version = %s",
        (cfg.semif_version,),
    ).fetchone()
    assert spec is not None and spec[0] == cfg.semif_spec and spec[1] == cfg.prompt_sha256
    batches = day.execute(
        "SELECT posts FROM clean.classify_batches WHERE model_version = %s ORDER BY batch_id",
        (cfg.semif_version,),
    ).fetchall()
    assert [b[0] for b in batches] == [25, 15]
    # The stub labels next to SemIf's, never over them.
    assert classify_pending(day, StubClassifier(), now=AFTER_DAY, run_id=RUN) == 40
    with pytest.raises(psycopg.errors.RaiseException, match="never changed"):
        day.execute("UPDATE clean.classifier_versions SET run_id = 'x'")


@pytest.mark.db
def test_a_version_string_cannot_change_its_spec(
    day: psycopg.Connection, cfg: ClassifyConfig
) -> None:
    _, clf = semif(cfg)
    classify_pending(day, clf, now=AFTER_DAY, run_id=RUN, limit=1)
    other = cfg.model_copy(update={"h4": cfg.h4.model_copy(update={"margin": 0.2})})
    assert other.semif_spec == cfg.semif_spec  # H4's margin is not part of the labels
    clf.config = cfg.model_copy(
        update={"semif": cfg.semif.model_copy(update={"dtype": "float16"})}
    )  # a different spec under the old version string
    with pytest.raises(RuntimeError, match="another spec"):
        classify_pending(day, clf, now=AFTER_DAY, run_id=RUN, limit=1)


@pytest.mark.db
def test_embeddings_go_into_pgvector(day: psycopg.Connection, cfg: ClassifyConfig) -> None:
    e = cfg.embed
    fake = FakeVllm(e.served_name, cfg.chat_template, dimensions=e.dimensions)
    client = VllmClient("http://embed", e.served_name, transport=fake.transport())
    assert embed.embed_pending(day, client, e, run_id=RUN, limit=10) == 10
    assert embed.embed_pending(day, client, e, run_id=RUN) == 30
    assert embed.embed_pending(day, client, e, run_id=RUN) == 0
    row = day.execute(
        "SELECT count(*), vector_dims(min(embedding::text)::vector) FROM clean.embeddings"
    ).fetchone()
    assert row == (40, 1024)
    near = day.execute(
        "SELECT uri FROM clean.embeddings ORDER BY embedding <=> "
        "(SELECT embedding FROM clean.embeddings ORDER BY uri LIMIT 1) LIMIT 1"
    ).fetchone()
    first = day.execute("SELECT min(uri) FROM clean.embeddings").fetchone()
    assert near == first
    waiting = day.execute("SELECT waiting, embedded FROM monitoring.embedding_backlog").fetchone()
    assert waiting == (0, 40)


# --- the bake-off ----------------------------------------------------------------------------

SENTIMENTS = ("negative", "neutral", "positive")
SCAMS = ("organic", "promotion", "likely_scam")
EVENTS = ("none", "hack", "listing")
TOPICS = (("price_talk",), ("regulation", "macro"), ("other",))


def varied(i: int) -> list[dict[str, Any]]:
    return [
        region("sentiment", SENTIMENTS[i % 3]),
        region("topic", *TOPICS[i % 3]),
        region("scam", SCAMS[(i // 2) % 3]),
        region("event", EVENTS[(i // 3) % 3]),
        region("entity", "token:BTC"),
    ]


@pytest.fixture
def labeled(day: psycopg.Connection, tmp_path: Path) -> Iterator[tuple[psycopg.Connection, Path]]:
    clear_ledger(day)
    fake = FakeStudio()
    for size in (None, 4):
        batch = sample.draw_pass1(day, SMALL, DAY.date(), now=AFTER_DAY, run_id=RUN, size=size)
        project = store.push(day, studio(fake), batch.batch_id)
        for i in range(len(fake.projects[project])):
            fake.annotate(project, result=varied(i + 10 * project), only=i)
    store.pull(day, studio(fake), now=AFTER_DAY, run_id=RUN)
    frozen = store.freeze(day, SMALL, tmp_path / "labeling", run_id=RUN)
    assert (frozen.calibration, frozen.test) == (5, 9)
    yield day, frozen.manifest
    clear_ledger(day)


def fake_hf_scores(run: Path, cfg: ClassifyConfig) -> None:
    """What scripts/hf_score.py writes, with random numbers."""
    rng = random.Random(4)
    inputs = [json.loads(line) for line in (run / "inputs.jsonl").read_text().splitlines()]
    for name, labels in (
        ("cryptobert", ["Bearish", "Neutral", "Bullish"]),
        ("finbert", ["positive", "negative", "neutral"]),
    ):
        rows = [
            {
                "item_id": r["item_id"],
                "labels": labels,
                "logits": [rng.uniform(-3, 3) for _ in labels],
            }
            for r in inputs
        ]
        (run / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        (run / f"{name}-meta.json").write_text(
            json.dumps({"posts": len(rows), "score_seconds": 1.5})
        )
    rubric = json.loads((run / "rubric.json").read_text())
    rows = [
        {
            "item_id": r["item_id"],
            "decision": d["id"],
            "option_ids": [o["id"] for o in d["options"]],
            "entailment": [rng.random() for _ in d["options"]],
        }
        for r in inputs
        for d in rubric["decisions"]
    ]
    (run / "openjev.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


@pytest.mark.db
def test_bakeoff_from_the_labeled_set_to_h4(
    labeled: tuple[psycopg.Connection, Path], cfg: ClassifyConfig, tmp_path: Path
) -> None:
    conn, labeled_set = labeled
    _, clf = semif(cfg)

    # Before registration: the calibration slice only.
    cal_run = tmp_path / "cal"
    manifest = bakeoff.export(conn, cfg, cal_run, splits=["calibration"], run_id=RUN)
    assert manifest["counts"] == {"calibration": 5} and manifest["h4_registration"] is None
    assert "text" not in bakeoff.read_items(cal_run)[0]
    with pytest.raises(bakeoff.BakeoffError, match="registered"):
        bakeoff.export(conn, cfg, tmp_path / "early", splits=["test"], run_id=RUN)
    bakeoff.score_semif(cal_run, clf, repeat=3)
    bakeoff.score_vader(cal_run)
    ev = bakeoff.evaluate(cal_run, cfg, eval_run_id=RUN)
    assert ev.split == "calibration" and ev.h4 is None
    assert set(ev.temperatures) == {"semif", "vader"}

    # Registration needs the margin decided, and goes in all at once. Both states are set here
    # explicitly, so the test does not depend on what config/classify.toml currently says.
    undecided = cfg.model_copy(update={"h4": cfg.h4.model_copy(update={"decided": False})})
    with pytest.raises(register.RegistrationError, match="decided"):
        register.register(
            conn, undecided, labeled_set=labeled_set, phase=Phase.DEVELOPMENT, run_id=RUN
        )
    decided = cfg.model_copy(update={"h4": cfg.h4.model_copy(update={"decided": True})})
    entries = register.register(
        conn, decided, labeled_set=labeled_set, phase=Phase.DEVELOPMENT, run_id=RUN
    )
    assert [e.table for e in entries] == ["documents", "hypotheses", "experiment_packages"]
    h4 = conn.execute(
        "SELECT statement, analysis_plan ->> 'margin' FROM ledger.hypotheses "
        "WHERE hypothesis_key = 'H4'"
    ).fetchone()
    assert h4 == (register.H4_STATEMENT, "0.05")

    # A different rubric or temperature than the registered one is still refused.
    hotter = decided.model_copy(
        update={
            "semif": decided.semif.model_copy(
                update={
                    "temperatures": decided.semif.temperatures.model_copy(update={"sentiment": 1.3})
                }
            )
        }
    )
    with pytest.raises(bakeoff.BakeoffError, match="registered"):
        bakeoff.export(conn, hotter, tmp_path / "hot", splits=["test"], run_id=RUN)

    run = tmp_path / "all"
    manifest = bakeoff.export(conn, decided, run, splits=["calibration", "test"], run_id=RUN)
    assert manifest["counts"] == {"calibration": 5, "test": 9}
    assert manifest["h4_registration"]["seq"] == entries[2].seq
    meta = bakeoff.score_semif(run, clf, repeat=4)
    assert meta["posts"] == 14 and meta["decisions"] == 14 * 9 and meta["repeat_posts"] == 4
    bakeoff.score_vader(run)
    fake_hf_scores(run, decided)
    ev = bakeoff.evaluate(run, decided, eval_run_id=RUN)
    assert ev.split == "test" and ev.models == [
        "cryptobert",
        "finbert",
        "openjev",
        "semif",
        "vader",
    ]
    tasks = {(r["model"], r["task"]) for r in ev.metrics}
    assert ("semif", "topic") in tasks and ("openjev", "event") in tasks
    assert ("cryptobert", "topic") not in tasks
    one = next(
        r
        for r in ev.metrics
        if r["model"] == "semif"
        and r["task"] == "sentiment"
        and r["platform"] == "all"
        and r["weighting"] == "unweighted"
    )
    assert one["n"] == 9 and 0 <= one["macro_f1"] <= 1
    platforms = {r["platform"] for r in ev.metrics}
    assert platforms <= {"all", "bluesky", "mastodon"} and "all" in platforms
    assert ev.h4 is not None and ev.h4["n"] == 9 and ev.h4["decided"]
    assert ev.reproducibility and ev.reproducibility["same_all_labels"] == 1.0  # deterministic
    assert set(ev.fallback["tasks"]) == {"sentiment", "topic", "scam", "event"}
    sentiment_fb = ev.fallback["tasks"]["sentiment"]
    assert set(sentiment_fb["others"]) == {"cryptobert", "finbert", "openjev", "vader"}
    assert set(ev.fallback["tasks"]["topic"]["others"]) == {"openjev"}
    assert all(t["use"] in ev.models for t in ev.fallback["tasks"].values())
    assert ev.latency["semif"]["posts"] == 14
    report = bakeoff.write_report(run, ev)
    assert "## H4" in (report / "report.md").read_text()
    assert (report / "metrics.csv").read_text().startswith("model,task,split")


@pytest.mark.db
def test_bakeoff_logs_and_registers_in_mlflow(
    labeled: tuple[psycopg.Connection, Path], cfg: ClassifyConfig, tmp_path: Path
) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    conn, _ = labeled
    _, clf = semif(cfg)
    run = tmp_path / "cal"
    bakeoff.export(conn, cfg, run, splits=["calibration"], run_id=RUN)
    bakeoff.score_semif(run, clf, repeat=0)
    bakeoff.score_vader(run)
    ev = bakeoff.evaluate(run, cfg, eval_run_id=RUN)
    report = bakeoff.write_report(run, ev)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(
        cfg.bakeoff.mlflow_experiment, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow_run = bakeoff.log_to_mlflow(run, report, ev, cfg, tracking_uri=uri, register=True)
    client = MlflowClient(uri)
    logged = client.get_run(mlflow_run)
    assert logged.data.tags["semif_version"] == cfg.semif_version
    assert "semif.sentiment.all.unweighted.macro_f1" in logged.data.metrics
    live = client.get_model_version_by_alias("cryptonlp-classifier-sentiment", "live")
    assert live.run_id == mlflow_run and live.tags["classifier"] in {"semif", "vader"}
    artifacts = {a.path for a in client.list_artifacts(mlflow_run, "raw")}
    assert "raw/inputs.jsonl" not in artifacts  # no post text in MLflow
