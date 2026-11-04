"""Step 4: sampling, the labeling interface, Label Studio round trips, agreement, the split."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import psycopg
import pyarrow.parquet as pq
import pytest

from cryptonlp.flows import collect
from cryptonlp.labeling import agreement as agr
from cryptonlp.labeling import sample, store
from cryptonlp.labeling.config import LabelingConfig, load_labeling_config
from cryptonlp.labeling.studio import Studio, StudioError
from cryptonlp.labeling.tasks import TASKS, label_config, read_result
from cryptonlp.normalize.config import load_normalize_config
from cryptonlp.normalize.pipeline import Normalizer
from cryptonlp.normalize.pseudonym import PseudonymKey
from cryptonlp.rawstore import CollectedPost, RawStore
from cryptonlp.schemas import Origin, RawPost, Source
from cryptonlp.schemas.labels import TASK_LABELS, Task
from cryptonlp.settings import Settings

RUN = "20261105T000000Z-0123456789ab"
DAY = datetime(2026, 11, 5, tzinfo=UTC)
KEY = PseudonymKey(bytes(range(32)))


# --- allocation and strata -------------------------------------------------------------------


def test_allocate_is_proportional_and_exact() -> None:
    alloc = sample.allocate(500, {"bluesky:other": 4500, "mastodon:other": 1500})
    assert alloc == {"bluesky:other": 375, "mastodon:other": 125}
    alloc = sample.allocate(10, {"a": 1, "b": 1, "c": 1})
    assert alloc == {"a": 1, "b": 1, "c": 1}  # capped by what the strata hold
    alloc = sample.allocate(7, {"a": 5, "b": 3, "c": 2})
    assert sum(alloc.values()) == 7 and alloc["a"] >= alloc["b"] >= alloc["c"]
    assert sample.allocate(0, {"a": 3}) == {"a": 0}


def test_plan_gives_candidates_their_share_and_never_more_than_exist() -> None:
    config = LabelingConfig()
    pop = [sample.Candidate("bluesky", f"b{i}", "bluesky:other") for i in range(900)]
    pop += [sample.Candidate("mastodon", f"m{i}", "mastodon:other") for i in range(300)]
    pop += [sample.Candidate("bluesky", f"bc{i}", "bluesky:candidate") for i in range(60)]
    pop += [sample.Candidate("mastodon", f"mc{i}", "mastodon:candidate") for i in range(20)]
    quotas = sample.plan_pass1(config, pop, 500)
    assert sum(quotas.values()) == 500
    assert quotas["bluesky:candidate"] + quotas["mastodon:candidate"] == 50
    assert quotas["bluesky:candidate"] == 38 and quotas["mastodon:candidate"] == 12
    # A day with 5 candidates gives all 5, and the other posts fill the batch.
    few = [c for c in pop if not c.stratum.endswith("candidate")] + pop[-5:]
    quotas = sample.plan_pass1(config, few, 500)
    assert quotas["mastodon:candidate"] == 5 and sum(quotas.values()) == 500


def test_sample_is_reproducible_from_the_seed_and_shuffled() -> None:
    pop = [sample.Candidate("bluesky", f"b{i:03}", "x") for i in range(100)]
    a = sample.stratified_sample(pop, {"x": 20}, random.Random(7))
    b = sample.stratified_sample(list(reversed(pop)), {"x": 20}, random.Random(7))
    assert a == b  # the population's order does not matter
    assert a != sorted(a, key=lambda c: c.uri)


def test_config_loads_and_flags_candidates() -> None:
    config = load_labeling_config()
    assert config.sample.batch_size == 500 and config.split.calibration == 500
    assert config.is_candidate("Join our Telegram for the next 100x gem", [])
    assert config.is_candidate("plain words", ["evm:0xabc"])
    assert not config.is_candidate("Bitcoin closed the week higher", [])
    assert len(config.sha256) == 64
    with pytest.raises(ValueError):
        LabelingConfig.model_validate({"split": {"calibration": 5000}})


# --- the interface and reading annotations ---------------------------------------------------


def region(name: str, *choices: str) -> dict[str, Any]:
    value = {"choices": list(choices)}  # JSON gives lists
    return {"from_name": name, "to_name": "text", "type": "choices", "value": value}


FULL = [
    region("sentiment", "positive"),
    region("topic", "price_talk", "macro"),
    region("scam", "organic"),
    region("event", "none"),
    region("entity", "token:BTC"),
]


def test_interface_offers_exactly_the_schema_labels() -> None:
    xml = label_config()
    for spec in TASKS:
        assert f'name="{spec.task.value}"' in xml
        if spec.task is not Task.ENTITY:
            assert {o.value for o in spec.options} == TASK_LABELS[spec.task]
    assert "$text" in xml and "$platform" not in xml


def test_read_a_valid_annotation() -> None:
    read = read_result(FULL)
    assert read.error is None and read.problem is None and not read.uncertain
    assert read.labels == {
        "sentiment": ["positive"],
        "topic": ["macro", "price_talk"],
        "scam": ["organic"],
        "event": ["none"],
        "entity": ["token:BTC"],
    }
    # The shown text is accepted too, and the unsure tick is kept.
    shown = [region("sentiment", "Positive"), *FULL[1:], region("uncertain", "uncertain")]
    read = read_result(shown)
    assert read.labels is not None and read.labels["sentiment"] == ["positive"] and read.uncertain


@pytest.mark.parametrize(
    ("result", "error"),
    [
        (FULL[1:], "no answer for sentiment"),
        ([*FULL[:4], region("entity", "none", "token:ETH")], "goes alone"),
        ([region("sentiment", "positive", "negative"), *FULL[1:]], "takes one answer"),
        ([*FULL[:3], region("event", "rugpull"), FULL[4]], "unknown event"),
        ([*FULL, region("colour", "red")], "unknown fields"),
    ],
)
def test_read_rejects_invalid_annotations(result: list[dict[str, Any]], error: str) -> None:
    read = read_result(result)
    assert read.labels is None and read.error is not None and error in read.error


def test_a_problem_flag_replaces_the_labels() -> None:
    read = read_result([region("problem", "not_english")])
    assert read.labels is None and read.problem == "not_english" and read.error is None


# --- agreement -------------------------------------------------------------------------------


def test_cohen_kappa() -> None:
    a = ["x", "y", "y", "x", "y"]
    b = ["x", "y", "x", "x", "y"]
    assert agr.cohen_kappa(a, b) == pytest.approx((0.8 - 0.48) / 0.52)
    assert agr.cohen_kappa(a, a) == pytest.approx(1.0)
    assert agr.cohen_kappa(["x", "x"], ["x", "x"]) is None  # undefined: one category


def test_multi_label_agreement_is_per_label() -> None:
    first = [["price_talk"], ["price_talk", "macro"], ["other"], ["regulation"]]
    second = [["price_talk"], ["price_talk"], ["other"], ["regulation"]]
    out = agr.task_agreement(Task.TOPIC, first, second)
    assert out["exact_agreement"] == pytest.approx(0.75)
    assert out["kappa_per_label"]["price_talk"] == pytest.approx(1.0)
    assert out["kappa_per_label"]["macro"] == pytest.approx(0.0)
    assert out["kappa_per_label"]["security_incident"] is None  # never used: undefined


# --- the Label Studio client -----------------------------------------------------------------


def test_client_exchanges_the_token_and_retries_once_on_401() -> None:
    calls: list[str] = []
    issued = iter(["a1", "a2"])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/api/token/refresh/":
            assert json.loads(request.content) == {"refresh": "pat"}
            return httpx.Response(200, json={"access": next(issued)})
        if request.headers["Authorization"] == "Bearer a1":
            return httpx.Response(401)
        page = int(request.url.params["page"])
        tasks = [{"id": 10 + page, "data": {"item_id": page}}]
        return httpx.Response(200, json={"total": 2, "tasks": tasks})

    with Studio("http://ls", "pat", transport=httpx.MockTransport(handler)) as studio:
        assert studio.task_ids(5) == {1: 11, 2: 12}
    assert calls[:3] == ["POST /api/token/refresh/", "GET /api/tasks", "POST /api/token/refresh/"]


def test_client_needs_a_token_and_reports_refusals() -> None:
    with pytest.raises(StudioError, match="personal access token"):
        Studio("http://ls", "")
    refuse = httpx.MockTransport(lambda r: httpx.Response(401))
    with pytest.raises(StudioError, match="refused the token"):
        Studio("http://ls", "bad", transport=refuse).export(1)


def test_labeling_sync_is_scheduled_only_with_a_token() -> None:
    off = [f.name for f, _ in collect.schedules(Settings(label_studio_api_key=""))]
    on = [f.name for f, _ in collect.schedules(Settings(label_studio_api_key="x"))]
    assert "labeling-sync" not in off and "labeling-sync" in on


# --- the workflow on the test database -------------------------------------------------------

VOCABULARY = """
market traders watch the price after a long week while analysts expect volatility to rise again
because funds keep buying and selling across many exchanges during quiet evening hours some
holders feel nervous others remain calm about the coming months and new rules from lawmakers
could change how banks treat digital assets
"""
WORDS = VOCABULARY.split()


def text_for(rng: random.Random, candidate: bool) -> str:
    body = " ".join(rng.choice(WORDS) for _ in range(16))
    return f"Bitcoin {body}" + (" join our telegram for the next gem" if candidate else "")


def make_post(n: int, source: Source, candidate: bool, minute: int) -> CollectedPost:
    rng = random.Random(n)
    uri = (
        f"at://did:plc:a{n}/app.bsky.feed.post/{n}"
        if source is Source.BLUESKY
        else f"https://mastodon.social/users/u{n}/statuses/{n}"
    )
    return CollectedPost(
        post=RawPost(
            source=source,
            uri=uri,
            author_id=uri.rsplit("/", 1)[0],
            text=text_for(rng, candidate),
            created_at=DAY + timedelta(minutes=minute),
            received_at=DAY + timedelta(minutes=minute),
            langs=("en",),
            origin=Origin.LIVE,
            collector_run_id=RUN,
        ),
        payload={"n": n},
    )


# 30 Bluesky and 10 Mastodon posts, of which 4 and 2 are candidates, across the day.
POSTS = [
    make_post(n, Source.BLUESKY if n < 30 else Source.MASTODON, n % 7 == 0, n * 30)
    for n in range(40)
]
SMALL = LabelingConfig.model_validate(
    {
        "sample": {"target": 12, "batch_size": 10},
        "enrich": {"share": 0.2, "patterns": [r"\btelegram\b"]},
        "relabel": {"size": 4, "min_gap_days": 14},
        "split": {"calibration": 5},
    }
)
AFTER_DAY = DAY + timedelta(days=1, hours=2)


class FakeStudio:
    """Label Studio in memory: projects of tasks, annotations set by the test."""

    def __init__(self) -> None:
        self.projects: dict[int, list[dict[str, Any]]] = {}
        self.deleted: list[int] = []
        self._next_task = 100

    def validate_config(self, label_config: str) -> None:
        assert "<View>" in label_config

    def create_project(self, title: str, label_config: str, description: str = "") -> int:
        pid = len(self.projects) + 1
        self.projects[pid] = []
        return pid

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> None:
        for t in tasks:
            self._next_task += 1
            self.projects[project_id].append({**t, "id": self._next_task, "annotations": []})

    def task_ids(self, project_id: int) -> dict[int, int]:
        return {t["data"]["item_id"]: t["id"] for t in self.projects[project_id]}

    def export(self, project_id: int) -> list[dict[str, Any]]:
        return self.projects[project_id]

    def delete_task(self, task_id: int) -> None:
        self.deleted.append(task_id)

    def annotate(
        self,
        project_id: int,
        result: list[dict[str, Any]] | None = None,
        at: datetime = AFTER_DAY,
        only: int | None = None,
    ) -> None:
        for i, t in enumerate(self.projects[project_id]):
            if only is not None and i != only:
                continue
            stamp = at.isoformat()
            t["annotations"] = [
                {
                    "id": t["id"] * 10,
                    "result": FULL if result is None else result,
                    "was_cancelled": False,
                    "created_at": stamp,
                    "updated_at": stamp,
                    "lead_time": 6.0,
                }
            ]


@pytest.fixture
def day(db: psycopg.Connection, tmp_path: Path) -> Iterator[psycopg.Connection]:
    db.execute("TRUNCATE labeling.batches, labeling.studio_deletions RESTART IDENTITY CASCADE")
    db.execute("TRUNCATE clean.labels, clean.embeddings, clean.minhash, clean.posts")
    store_ = RawStore(tmp_path / "raw")
    store_.store_posts(db, POSTS, RUN)
    Normalizer(load_normalize_config(), KEY, store_, RUN).run(db)
    in_study = db.execute("SELECT count(*) FROM clean.study_posts").fetchone()
    assert in_study == (40,), "the synthetic posts must all be in the study"
    yield db
    db.execute("TRUNCATE labeling.batches, labeling.studio_deletions RESTART IDENTITY CASCADE")


def studio(fake: FakeStudio) -> Studio:
    return cast(Studio, fake)


@pytest.mark.db
def test_draw_waits_for_the_day_to_settle_and_records_strata(day: psycopg.Connection) -> None:
    with pytest.raises(sample.DrawError, match="can be drawn from"):
        sample.draw_pass1(day, SMALL, DAY.date(), now=DAY + timedelta(hours=24, minutes=5),
                          run_id=RUN)  # fmt: skip
    batch = sample.draw_pass1(day, SMALL, DAY.date(), now=AFTER_DAY, run_id=RUN, seed=3)
    assert batch.size == 10
    assert batch.strata == {
        "bluesky:candidate": {"N": 5, "n": 2},
        "bluesky:other": {"N": 25, "n": 6},
        "mastodon:candidate": {"N": 1, "n": 0},
        "mastodon:other": {"N": 9, "n": 2},
    }
    items = day.execute(
        "SELECT stratum, position FROM labeling.items WHERE batch_id = %s", (batch.batch_id,)
    ).fetchall()
    assert Counter(s for s, _ in items) == {k: v["n"] for k, v in batch.strata.items() if v["n"]}
    assert sorted(p for _, p in items) == list(range(10))
    # The same day again needs an explicit size, and never repeats a post.
    with pytest.raises(sample.DrawError, match="top-up"):
        sample.draw_pass1(day, SMALL, DAY.date(), now=AFTER_DAY, run_id=RUN)
    sample.draw_pass1(day, SMALL, DAY.date(), now=AFTER_DAY, run_id=RUN, size=30)
    with pytest.raises(sample.DrawError, match="fewer than"):
        sample.draw_pass1(day, SMALL, DAY.date(), now=AFTER_DAY, run_id=RUN, size=1)


@pytest.mark.db
def test_push_pull_and_invalid_annotations(day: psycopg.Connection) -> None:
    fake = FakeStudio()
    batch = sample.draw_pass1(day, SMALL, DAY.date(), now=AFTER_DAY, run_id=RUN, seed=3)
    project = store.push(day, studio(fake), batch.batch_id)
    tasks = fake.projects[project]
    assert [set(t["data"]) for t in tasks] == [{"item_id", "text"}] * 10  # text only
    with pytest.raises(ValueError, match="already"):
        store.push(day, studio(fake), batch.batch_id)

    fake.annotate(project)
    fake.annotate(project, result=FULL[1:], only=0)  # missing sentiment
    stats = store.pull(day, studio(fake), now=AFTER_DAY, run_id=RUN)
    assert (stats.projects, stats.new_versions) == (1, 10)
    assert store.pull(day, studio(fake), now=AFTER_DAY, run_id=RUN).new_versions == 0
    report = store.status(day, SMALL, AFTER_DAY)
    first = report["batches"][0]
    assert (first["annotated"], first["usable"], first["invalid"]) == (10, 9, 1)
    assert first["posts_per_hour"] == pytest.approx(600.0)

    # A correction is a new version; the latest counts. A deleted annotation is withdrawn.
    fake.annotate(project, only=0, at=AFTER_DAY + timedelta(hours=1))
    tasks[1]["annotations"] = []
    stats = store.pull(day, studio(fake), now=AFTER_DAY + timedelta(hours=1), run_id=RUN)
    assert (stats.new_versions, stats.withdrawn) == (2, 1)
    row = day.execute(
        "SELECT count(*) FILTER (WHERE labels IS NOT NULL), count(*) FILTER (WHERE error IS "
        "NOT NULL) FROM labeling.current"
    ).fetchone()
    assert row == (9, 1)


@pytest.mark.db
def test_a_deletion_at_the_source_removes_the_item_and_its_studio_task(
    day: psycopg.Connection,
) -> None:
    fake = FakeStudio()
    batch = sample.draw_pass1(day, SMALL, DAY.date(), now=AFTER_DAY, run_id=RUN, seed=3)
    project = store.push(day, studio(fake), batch.batch_id)
    fake.annotate(project)
    store.pull(day, studio(fake), now=AFTER_DAY, run_id=RUN)
    source, uri, task_id = day.execute(
        "SELECT source, uri, studio_task_id FROM labeling.items WHERE position = 0"
    ).fetchone() or (None, None, None)
    day.execute(
        "INSERT INTO raw.deletions (source, uri, method, run_id) VALUES (%s, %s, 'stream', %s)",
        (source, uri, RUN),
    )
    left = day.execute("SELECT count(*) FROM labeling.items WHERE uri = %s", (uri,)).fetchone()
    assert left == (0,)
    assert store.sync_deletions(day, studio(fake)) == 1 and fake.deleted == [task_id]
    assert store.sync_deletions(day, studio(fake)) == 0
    # Pulling again ignores the deleted post's task, still in the export.
    assert store.pull(day, studio(fake), now=AFTER_DAY, run_id=RUN).new_versions == 0


@pytest.mark.db
def test_relabel_freeze_and_agreement(day: psycopg.Connection, tmp_path: Path) -> None:
    fake = FakeStudio()
    for size in (None, 4):  # a batch and a top-up: 14 posts, one flagged
        batch = sample.draw_pass1(day, SMALL, DAY.date(), now=AFTER_DAY, run_id=RUN, size=size)
        fake.annotate(store.push(day, studio(fake), batch.batch_id))
    fake.annotate(1, result=[region("problem", "not_english")], only=3)
    store.pull(day, studio(fake), now=AFTER_DAY, run_id=RUN)

    # Re-labeling waits 14 days after the last edit.
    with pytest.raises(sample.DrawError, match="14 days"):
        sample.draw_relabel(day, SMALL, now=AFTER_DAY + timedelta(days=13), run_id=RUN)
    later = AFTER_DAY + timedelta(days=15)
    relabel = sample.draw_relabel(day, SMALL, now=later, run_id=RUN, seed=1)
    assert relabel.size == 4 and sum(v["n"] for v in relabel.strata.values()) == 4
    rp = store.push(day, studio(fake), relabel.batch_id)
    flipped = [region("sentiment", "negative"), *FULL[1:]]
    fake.annotate(rp, at=later)
    fake.annotate(rp, result=flipped, at=later, only=0)
    store.pull(day, studio(fake), now=later, run_id=RUN)
    with pytest.raises(sample.DrawError, match="already"):
        sample.draw_relabel(day, SMALL, now=later, run_id=RUN)

    report = agr.agreement(day, SMALL.relabel.min_gap)
    assert (report["pairs"], report["scored"], report["gap_violations"]) == (4, 4, 0)
    sentiment = report["tasks"]["sentiment"]
    assert sentiment["exact_agreement"] == pytest.approx(0.75)

    frozen = store.freeze(day, SMALL, tmp_path, run_id=RUN)
    assert (frozen.labeled, frozen.calibration, frozen.test, frozen.problems) == (13, 5, 8, 1)
    table = pq.read_table(frozen.manifest).to_pylist()
    received = [r["received_at"] for r in table]
    assert received == sorted(received)
    assert {r["split"] for r in table[:5]} == {"calibration"}
    assert {r["split"] for r in table[5:]} == {"test"}
    labels = day.execute(
        "SELECT model_version, annotator, count(*) FROM clean.labels GROUP BY 1, 2"
    ).fetchall()
    assert labels == [("human:pass1", "human", 13 * len(Task))]
    with pytest.raises(store.FreezeError, match="frozen already"):
        store.freeze(day, SMALL, tmp_path, run_id=RUN)
    assert store.freeze_relabel(day, run_id=RUN) == 4
    # Human labels make `normalize rebuild` refuse (cli.normalize_job), so they stay.


@pytest.mark.db
def test_freeze_refuses_an_incomplete_or_short_set(day: psycopg.Connection, tmp_path: Path) -> None:
    fake = FakeStudio()
    batch = sample.draw_pass1(day, SMALL, DAY.date(), now=AFTER_DAY, run_id=RUN)
    project = store.push(day, studio(fake), batch.batch_id)
    fake.annotate(project, only=0)
    store.pull(day, studio(fake), now=AFTER_DAY, run_id=RUN)
    with pytest.raises(store.FreezeError, match="not labeled yet"):
        store.freeze(day, SMALL, tmp_path, run_id=RUN)
    fake.annotate(project)
    store.pull(day, studio(fake), now=AFTER_DAY, run_id=RUN)
    with pytest.raises(store.FreezeError, match="below the target"):
        store.freeze(day, SMALL, tmp_path, run_id=RUN)
    assert store.freeze(day, SMALL, tmp_path, run_id=RUN, allow_short=True).test == 5
