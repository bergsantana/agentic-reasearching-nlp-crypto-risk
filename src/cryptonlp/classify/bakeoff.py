"""The classifier bake-off (step 6; article Section 4.7; H4).

One run directory (``data/bakeoff/<run ID>/``) holds everything:

1. ``export``: the frozen labeled set (``cryptonlp labeling freeze``) for the splits asked for:
   ``items.parquet`` (IDs, platform, split, stratum weight, human labels, text hash),
   ``inputs.jsonl`` (ID and text, for the scorers), ``rubric.json`` and ``manifest.json``.
   The test split is exported only once H4 is in the ledger with this exact SemIf spec (rubric,
   prompt hash, revision, temperatures): the rubric may be tuned on the calibration slice only,
   and is registered before the test posts are scored (claims P18).
2. Scoring, one model at a time: ``semif`` (vLLM), ``vader`` (here), and CryptoBERT, FinBERT and
   Wortega's OpenJev (``scripts/hf_score.py`` in the ``hf-scorer`` container). Raw outputs only.
3. ``evaluate``: temperatures fitted on the calibration rows, every metric on the target split
   per platform, unweighted and re-weighted by the strata, H4's non-inferiority test,
   SemIf's reproducibility, latency, and the fallback decision per task.

SemIf, OpenJev: all four tasks. CryptoBERT, FinBERT, VADER: sentiment only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from cryptonlp.classify import metrics as m
from cryptonlp.classify.base import PostText
from cryptonlp.classify.config import NO, TOPIC_OTHER, YES, ClassifyConfig, canonical_json
from cryptonlp.classify.errors import ClassifyError
from cryptonlp.classify.semif import SemIfClassifier
from cryptonlp.labeling.tasks import SPECS
from cryptonlp.schemas.labels import Task

HUMAN = "human:pass1"
SPLITS = ("calibration", "test")
PLATFORMS = ("all", "bluesky", "mastodon")
SINGLE_TASKS = (Task.SENTIMENT, Task.SCAM, Task.EVENT)
EVAL_TASKS = (Task.SENTIMENT, Task.TOPIC, Task.SCAM, Task.EVENT)
SENTIMENT_ONLY = ("cryptobert", "finbert", "vader")
ALL_TASKS = ("semif", "openjev")
MODELS = (*ALL_TASKS, *SENTIMENT_ONLY)
H4_PACKAGE = "H4.bakeoff"


class BakeoffError(ClassifyError):
    pass


def classes(task: Task) -> tuple[str, ...]:
    """Class order for a single-label task: the labeling interface's (manual section 5)."""
    return tuple(o.value for o in SPECS[task].options)


def topics(config: ClassifyConfig) -> tuple[str, ...]:
    """The topics with their own decision, in rubric order; "other" is the remainder."""
    return tuple(d.label for d in config.rubric.for_task(Task.TOPIC) if d.label is not None)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise BakeoffError(f"{path.name} is missing: score that model first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- 1. export -----------------------------------------------------------------------------

ITEMS_SQL = """
SELECT s.item_id, i.source, i.uri, s.split, i.stratum, b.strata, p.text, p.text_sha256,
       p.received_at
FROM labeling.splits s
JOIN labeling.items i USING (item_id)
JOIN labeling.batches b ON b.batch_id = i.batch_id
JOIN clean.posts p ON p.source = i.source AND p.uri = i.uri
WHERE s.split = ANY(%s)
ORDER BY p.received_at, s.item_id
"""

TRUTH_SQL = """
SELECT l.source, l.uri, l.task, l.label_values
FROM clean.labels l
WHERE l.model_version = %s AND l.task = ANY(%s)
"""


@dataclass(frozen=True)
class Item:
    item_id: int
    source: str
    uri: str
    split: str
    weight: float
    text: str
    text_sha256: str
    received_at: datetime
    truth: dict[str, tuple[str, ...]]


def load_items(conn: psycopg.Connection, splits: Sequence[str]) -> list[Item]:
    rows = conn.execute(ITEMS_SQL, (list(splits),)).fetchall()
    if not rows:
        raise BakeoffError("no labeled posts in those splits: freeze the labeled set first")
    truth: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {}
    for source, uri, task, values in conn.execute(
        TRUTH_SQL, (HUMAN, [t.value for t in EVAL_TASKS])
    ).fetchall():
        truth.setdefault((source, uri), {})[task] = tuple(values)
    items = []
    for item_id, source, uri, split, stratum, strata, text, text_sha, received in rows:
        labels = truth.get((source, uri), {})
        if set(labels) != {t.value for t in EVAL_TASKS}:
            raise BakeoffError(f"item {item_id} lacks human labels in clean.labels")
        cell = strata[stratum]
        items.append(
            Item(
                int(item_id),
                source,
                uri,
                split,
                cell["N"] / cell["n"],
                text,
                text_sha,
                received,
                labels,
            )
        )
    return items


def h4_registration(conn: psycopg.Connection, config: ClassifyConfig) -> dict[str, Any] | None:
    """The newest H4 bake-off package in the ledger, if it registers this exact SemIf spec."""
    row = conn.execute(
        "SELECT seq, hash, prompt_sha256, config, recorded_at FROM ledger.experiment_packages "
        "WHERE package_key LIKE %s ORDER BY seq DESC LIMIT 1",
        (H4_PACKAGE + "%",),
    ).fetchone()
    if row is None:
        return None
    seq, digest, prompt, cfg, recorded = row
    if prompt != config.prompt_sha256 or cfg.get("semif_spec") != config.semif_spec:
        return None
    return {"seq": seq, "hash": digest, "recorded_at": recorded.isoformat()}


def export(
    conn: psycopg.Connection,
    config: ClassifyConfig,
    out: Path,
    *,
    splits: Sequence[str],
    run_id: str,
) -> dict[str, Any]:
    if not set(splits) <= set(SPLITS):
        raise BakeoffError(f"splits are {SPLITS}")
    registration = None
    if "test" in splits:
        registration = h4_registration(conn, config)
        if registration is None:
            raise BakeoffError(
                "the test split is scored only after H4 is registered with this exact rubric, "
                "prompt, revision and temperatures (`cryptonlp classify register-h4`)"
            )
    items = load_items(conn, splits)
    out.mkdir(parents=True, exist_ok=False)
    table = pa.table(
        {
            "item_id": [i.item_id for i in items],
            "source": [i.source for i in items],
            "uri": [i.uri for i in items],
            "split": [i.split for i in items],
            "weight": [i.weight for i in items],
            "text_sha256": [i.text_sha256 for i in items],
            "received_at": [i.received_at for i in items],
            **{t.value: [list(i.truth[t.value]) for i in items] for t in EVAL_TASKS},
        }
    )
    pq.write_table(table, out / "items.parquet")
    _write_jsonl(out / "inputs.jsonl", ({"item_id": i.item_id, "text": i.text} for i in items))
    (out / "rubric.json").write_text(
        json.dumps(config.rubric.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    )
    labeled_set = hashlib.sha256(
        canonical_json(
            [[i.item_id, i.source, i.uri, i.split, i.text_sha256] for i in items]
        ).encode()
    ).hexdigest()
    manifest = {
        "run_id": run_id,
        "exported_at": datetime.now(UTC).isoformat(),
        "splits": list(splits),
        "counts": {s: sum(1 for i in items if i.split == s) for s in splits},
        "by_platform": {p: sum(1 for i in items if i.source == p) for p in PLATFORMS[1:]},
        "labeled_set_sha256": labeled_set,
        "items_sha256": _sha256_file(out / "items.parquet"),
        "prompt_sha256": config.prompt_sha256,
        "semif_spec": config.semif_spec,
        "semif_version": config.semif_version,
        "h4_registration": registration,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def read_manifest(run: Path) -> dict[str, Any]:
    path = run / "manifest.json"
    if not path.exists():
        raise BakeoffError(f"{run} is not a bake-off run (no manifest.json)")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def read_items(run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = pq.read_table(run / "items.parquet").to_pylist()
    return rows


# --- 2. scoring here: SemIf and VADER ------------------------------------------------------


def _inputs(run: Path) -> list[dict[str, Any]]:
    return _read_jsonl(run / "inputs.jsonl")


def score_semif(run: Path, classifier: SemIfClassifier, *, repeat: int) -> dict[str, Any]:
    """Every post, every rubric decision: raw letter logits and prompt hashes. Then the first
    ``repeat`` posts of the target split again, in reverse order (other batches, other prefix
    cache state), for the reproducibility measure."""
    manifest = read_manifest(run)
    if manifest["semif_version"] != classifier.model_version:
        raise BakeoffError(
            f"the run was exported for {manifest['semif_version']}, the config is now "
            f"{classifier.model_version}: export a new run"
        )
    check = classifier.check()
    inputs = _inputs(run)
    split_of = {r["item_id"]: r["split"] for r in read_items(run)}
    posts = [PostText("x", str(r["item_id"]), r["text"], datetime.now(UTC)) for r in inputs]

    def rows(scored: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "item_id": int(uri),
                "decision": d,
                "option_ids": list(s.option_ids),
                "logits": list(s.logits),
                "prompt_sha256": s.prompt_sha256,
            }
            for uri, by_decision in scored.items()
            for d, s in by_decision.items()
        ]

    started = time.perf_counter()
    scored = classifier.score_posts(posts)
    seconds = time.perf_counter() - started
    _write_jsonl(run / "semif.jsonl", rows(scored))

    target = "test" if "test" in manifest["splits"] else "calibration"
    again = [p for p in posts if split_of[int(p.uri)] == target][:repeat]
    repeat_seconds = None
    if again:
        t0 = time.perf_counter()
        rescored = classifier.score_posts(list(reversed(again)))
        repeat_seconds = time.perf_counter() - t0
        _write_jsonl(run / "semif-repeat.jsonl", rows(rescored))
    meta = {
        **check,
        "posts": len(posts),
        "decisions": sum(len(v) for v in scored.values()),
        "score_seconds": seconds,
        "seconds_per_1000_posts": 1000 * seconds / len(posts),
        "repeat_posts": len(again),
        "repeat_seconds": repeat_seconds,
        "concurrency": classifier.client.concurrency,
    }
    (run / "semif-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def score_vader(run: Path) -> dict[str, Any]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    inputs = _inputs(run)
    started = time.perf_counter()
    out = []
    for r in inputs:
        s = analyzer.polarity_scores(r["text"])
        out.append(
            {"item_id": r["item_id"], **{k: float(s[k]) for k in ("neg", "neu", "pos", "compound")}}
        )
    seconds = time.perf_counter() - started
    _write_jsonl(run / "vader.jsonl", out)
    meta = {"model": "vader", "posts": len(inputs), "score_seconds": seconds}
    (run / "vader-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# --- 3. evaluation -------------------------------------------------------------------------

# VADER's usual rule (compound >= 0.05 positive, <= -0.05 negative) as three scores whose argmax
# is that rule, so it can be temperature-scaled like every other model. Neutral sits a hair
# below zero so the boundaries go to positive and negative, as the rule says.
VADER_CUTOFF = 0.05
VADER_NEUTRAL = -1e-9


@dataclass
class Scores:
    """One model's raw scores on one task, rows in item order. Single-label: logits [n, k] in
    ``classes(task)`` order. Topic: logits [n, topics, 2] for (no, yes)."""

    logits: np.ndarray


def _by_item(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(int(r["item_id"]), {})[r["decision"]] = r
    return out


def decision_scores(
    config: ClassifyConfig, item_ids: Sequence[int], rows: list[dict[str, Any]], *, log_p: bool
) -> dict[Task, Scores]:
    """SemIf (logits) or OpenJev (entailment, normalized over the options, then logs) per task."""
    by = _by_item(rows)
    out: dict[Task, Scores] = {}
    for task in SINGLE_TASKS:
        (d,) = config.rubric.for_task(task)
        order = classes(task)
        mat = np.empty((len(item_ids), len(order)))
        for i, item in enumerate(item_ids):
            r = by[item][d.id]
            vals = dict(zip(r["option_ids"], _scores(r, log_p), strict=True))
            mat[i] = [vals[c] for c in order]
        out[task] = Scores(mat)
    tops = config.rubric.for_task(Task.TOPIC)
    mat = np.empty((len(item_ids), len(tops), 2))
    for i, item in enumerate(item_ids):
        for j, d in enumerate(tops):
            r = by[item][d.id]
            vals = dict(zip(r["option_ids"], _scores(r, log_p), strict=True))
            mat[i, j] = [vals[NO], vals[YES]]
    out[Task.TOPIC] = Scores(mat)
    return out


def _scores(row: dict[str, Any], log_p: bool) -> list[float]:
    if not log_p:
        return [float(x) for x in row["logits"]]
    ent = np.asarray(row["entailment"], dtype=np.float64)
    p = ent / max(float(ent.sum()), 1e-9)  # openjev_decide.py's normalization
    return [math.log(max(float(x), 1e-12)) for x in p]


def classifier_scores(
    item_ids: Sequence[int], rows: list[dict[str, Any]], mapping: dict[str, str]
) -> dict[Task, Scores]:
    by = {int(r["item_id"]): r for r in rows}
    order = classes(Task.SENTIMENT)
    mat = np.empty((len(item_ids), len(order)))
    for i, item in enumerate(item_ids):
        r = by[item]
        vals = {mapping[lab]: float(x) for lab, x in zip(r["labels"], r["logits"], strict=True)}
        mat[i] = [vals[c] for c in order]
    return {Task.SENTIMENT: Scores(mat)}


def vader_scores(item_ids: Sequence[int], rows: list[dict[str, Any]]) -> dict[Task, Scores]:
    by = {int(r["item_id"]): r for r in rows}
    assert classes(Task.SENTIMENT) == ("negative", "neutral", "positive")
    mat = np.empty((len(item_ids), 3))
    for i, item in enumerate(item_ids):
        c = by[item]["compound"]
        mat[i] = [-c - VADER_CUTOFF, VADER_NEUTRAL, c - VADER_CUTOFF]
    return {Task.SENTIMENT: Scores(mat)}


def load_scores(run: Path, config: ClassifyConfig, item_ids: Sequence[int]) -> dict[str, Any]:
    """Every model scored in the run (missing ones are skipped and reported)."""
    out: dict[str, dict[Task, Scores]] = {}
    present = {name for name in MODELS if (run / f"{name}.jsonl").exists()}
    if "semif" in present:
        out["semif"] = decision_scores(
            config, item_ids, _read_jsonl(run / "semif.jsonl"), log_p=False
        )
    if "openjev" in present:
        out["openjev"] = decision_scores(
            config, item_ids, _read_jsonl(run / "openjev.jsonl"), log_p=True
        )
    for name in ("cryptobert", "finbert"):
        if name in present:
            mapping = getattr(config.bakeoff, name).labels
            out[name] = classifier_scores(item_ids, _read_jsonl(run / f"{name}.jsonl"), mapping)
    if "vader" in present:
        out["vader"] = vader_scores(item_ids, _read_jsonl(run / "vader.jsonl"))
    return out


# truths ---


def single_truth(items: Sequence[dict[str, Any]], task: Task) -> np.ndarray:
    order = classes(task)
    return np.array([order.index(i[task.value][0]) for i in items], dtype=np.int64)


def topic_truth(items: Sequence[dict[str, Any]], tops: Sequence[str]) -> np.ndarray:
    """[n, topics + 1]: one column per topic, then "other"."""
    cols = [*tops, TOPIC_OTHER]
    return np.array([[float(c in i[Task.TOPIC.value]) for c in cols] for i in items])


def fit_temperatures(
    scores: dict[str, dict[Task, Scores]],
    items: Sequence[dict[str, Any]],
    tops: Sequence[str],
    rows: np.ndarray,
) -> dict[str, dict[str, float]]:
    """One temperature per model and task, on the given rows (the calibration slice)."""
    sub = [items[i] for i in rows]
    out: dict[str, dict[str, float]] = {}
    for name, by_task in scores.items():
        out[name] = {}
        for task, s in by_task.items():
            if task is Task.TOPIC:
                y = topic_truth(sub, tops)[:, : len(tops)].astype(np.int64).ravel()
                out[name][task.value] = m.fit_temperature(s.logits[rows].reshape(-1, 2), y)
            else:
                out[name][task.value] = m.fit_temperature(s.logits[rows], single_truth(sub, task))
    return out


def topic_probs(logits: np.ndarray, t: float) -> np.ndarray:
    p_yes = m.softmax(logits, t)[..., 1]
    other = np.prod(1 - p_yes, axis=1, keepdims=True)
    out: np.ndarray = np.concatenate([p_yes, other], axis=1)
    return out


@dataclass
class Evaluation:
    run_id: str
    split: str
    models: list[str]
    temperatures: dict[str, dict[str, float]]
    metrics: list[dict[str, Any]] = field(default_factory=list)
    per_class: list[dict[str, Any]] = field(default_factory=list)
    h4: dict[str, Any] | None = None
    reproducibility: dict[str, Any] | None = None
    latency: dict[str, Any] = field(default_factory=dict)
    fallback: dict[str, Any] = field(default_factory=dict)
    semif_registered_temperatures: dict[str, float] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


def evaluate(run: Path, config: ClassifyConfig, *, eval_run_id: str) -> Evaluation:
    manifest = read_manifest(run)
    items = read_items(run)
    item_ids = [int(i["item_id"]) for i in items]
    tops = topics(config)
    scores = load_scores(run, config, item_ids)
    if not scores:
        raise BakeoffError("no model scored in this run yet")
    split = np.array([i["split"] for i in items])
    cal = np.flatnonzero(split == "calibration")
    if len(cal) == 0:
        raise BakeoffError("the run has no calibration rows to fit temperatures on")
    target = "test" if (split == "test").any() else "calibration"
    temps = fit_temperatures(scores, items, tops, cal)
    ev = Evaluation(
        run_id=eval_run_id,
        split=target,
        models=sorted(scores),
        temperatures=temps,
        semif_registered_temperatures=config.semif.temperatures.model_dump(mode="json"),
        manifest=manifest,
    )
    rows_target = np.flatnonzero(split == target)
    source = np.array([i["source"] for i in items])
    weight = np.array([float(i["weight"]) for i in items])
    for name, by_task in sorted(scores.items()):
        for task, s in by_task.items():
            t = temps[name][task.value]
            for platform in PLATFORMS:
                rows = (
                    rows_target
                    if platform == "all"
                    else rows_target[source[rows_target] == platform]
                )
                if len(rows) == 0:
                    continue
                sub = [items[i] for i in rows]
                for weighting in ("unweighted", "reweighted"):
                    w = None if weighting == "unweighted" else weight[rows]
                    if task is Task.TOPIC:
                        tm = m.multi_label_metrics(
                            [*tops, TOPIC_OTHER],
                            topic_truth(sub, tops),
                            topic_probs(s.logits[rows], t),
                            decision_cols=range(len(tops)),
                            bins=config.bakeoff.ece_bins,
                            weights=w,
                        )
                    else:
                        tm = m.single_label_metrics(
                            classes(task),
                            single_truth(sub, task),
                            m.softmax(s.logits[rows], t),
                            bins=config.bakeoff.ece_bins,
                            weights=w,
                        )
                    key = {
                        "model": name,
                        "task": task.value,
                        "split": target,
                        "platform": platform,
                        "weighting": weighting,
                    }
                    ev.metrics.append(
                        {
                            **key,
                            "n": tm.n,
                            "temperature": t,
                            "macro_f1": tm.macro_f1,
                            "balanced_accuracy": tm.balanced_accuracy,
                            "accuracy": tm.accuracy,
                            "brier": tm.brier,
                            "ece": tm.ece,
                        }
                    )
                    for cls, cs in tm.per_class.items():
                        ev.per_class.append({**key, "class": cls, **asdict(cs)})
    if target == "test" and {"semif", "cryptobert"} <= set(scores):
        ev.h4 = h4_test(config, items, scores, rows_target)
    ev.reproducibility = reproducibility(run, config, item_ids)
    ev.latency = latency(run)
    ev.fallback = fallback(config, ev, scores, items, rows_target, tops)
    return ev


def h4_test(
    config: ClassifyConfig,
    items: Sequence[dict[str, Any]],
    scores: dict[str, Any],
    rows: np.ndarray,
) -> dict[str, Any]:
    """SemIf vs CryptoBERT, sentiment macro F1 on the test posts (both platforms, unweighted)."""
    sub = [items[i] for i in rows]
    y = single_truth(sub, Task.SENTIMENT)
    a = scores["semif"][Task.SENTIMENT].logits[rows].argmax(axis=1)
    b = scores["cryptobert"][Task.SENTIMENT].logits[rows].argmax(axis=1)
    h = config.h4
    res = m.noninferiority_bootstrap(
        y,
        a,
        b,
        len(classes(Task.SENTIMENT)),
        margin=h.margin,
        alpha=h.alpha,
        replicates=h.bootstrap,
        seed=h.seed,
    )
    return {**asdict(res), "decided": h.decided}


def reproducibility(run: Path, config: ClassifyConfig, item_ids: Sequence[int]) -> dict[str, Any]:
    """SemIf scored twice: how often each decision's chosen option, and each task's label,
    stays the same; and the largest logit difference."""
    path = run / "semif-repeat.jsonl"
    if not path.exists() or not (run / "semif.jsonl").exists():
        return {}
    first = _by_item(_read_jsonl(run / "semif.jsonl"))
    again = _by_item(_read_jsonl(path))
    ids = sorted(again)
    same_decision = total = 0
    max_diff = 0.0
    for item in ids:
        for d, r2 in again[item].items():
            r1 = first[item][d]
            total += 1
            same_decision += int(np.argmax(r1["logits"]) == np.argmax(r2["logits"]))
            max_diff = max(max_diff, float(np.max(np.abs(np.subtract(r1["logits"], r2["logits"])))))
    s1 = decision_scores(config, ids, [r for i in ids for r in first[i].values()], log_p=False)
    s2 = decision_scores(config, ids, [r for i in ids for r in again[i].values()], log_p=False)
    per_task = {}
    for task in EVAL_TASKS:
        if task is Task.TOPIC:
            a = s1[task].logits.argmax(axis=2)
            b = s2[task].logits.argmax(axis=2)
            same = (a == b).all(axis=1)
        else:
            same = s1[task].logits.argmax(axis=1) == s2[task].logits.argmax(axis=1)
        per_task[task.value] = float(same.mean())
    all_same = float(
        np.mean(
            [
                all(
                    np.argmax(first[i][d]["logits"]) == np.argmax(again[i][d]["logits"])
                    for d in again[i]
                )
                for i in ids
            ]
        )
    )
    return {
        "posts": len(ids),
        "decisions": total,
        "same_decision": same_decision / total,
        "same_label_per_task": per_task,
        "same_all_labels": all_same,
        "max_abs_logit_diff": max_diff,
    }


def latency(run: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in MODELS:
        path = run / f"{name}-meta.json"
        if path.exists():
            meta = json.loads(path.read_text())
            secs = meta.get("score_seconds")
            if secs is not None and meta.get("posts"):
                out[name] = {
                    "posts": meta["posts"],
                    "seconds": secs,
                    "seconds_per_1000_posts": 1000 * secs / meta["posts"],
                }
    return out


def predictions(
    task: Task,
    s: Scores,
    items: Sequence[dict[str, Any]],
    rows: np.ndarray,
    tops: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """(truth, prediction) on ``rows``: class indices, or for topic boolean label columns (the
    topics, then "other" exactly when no topic is predicted). Temperatures never change these."""
    sub = [items[i] for i in rows]
    if task is not Task.TOPIC:
        return single_truth(sub, task), s.logits[rows].argmax(axis=1)
    yes = s.logits[rows][..., 1] > s.logits[rows][..., 0]
    pred = np.concatenate([yes, ~yes.any(axis=1, keepdims=True)], axis=1)
    return topic_truth(sub, tops) > 0.5, pred


def macro_f1_of(task: Task, truth: np.ndarray, pred: np.ndarray) -> Callable[[np.ndarray], float]:
    if task is Task.TOPIC:
        return lambda idx: m.multilabel_macro_f1(truth[idx], pred[idx])
    k = len(classes(task))
    return lambda idx: m.macro_f1(truth[idx], pred[idx], k)


def decide_task(
    task: Task,
    semif_f1: float,
    others: dict[str, dict[str, float | bool]],
    *,
    min_macro_f1: float,
    scam_recall: float | None,
    min_scam_recall: float,
) -> dict[str, Any]:
    """The fallback rule for one task. ``others[model]`` has its macro F1 and whether it clearly
    beat SemIf (``better``). Returns the reasons, the decision, and the model the task uses."""
    reasons = []
    beaten_by = sorted(
        (mod for mod, o in others.items() if o["better"]),
        key=lambda mod: -float(others[mod]["macro_f1"]),
    )
    if beaten_by:
        reasons.append(f"clearly beaten by {', '.join(beaten_by)}")
    if semif_f1 < min_macro_f1:
        reasons.append(f"macro F1 {semif_f1:.3f} below the floor {min_macro_f1}")
    if task is Task.SCAM and scam_recall is not None and scam_recall < min_scam_recall:
        reasons.append(f"likely-scam recall {scam_recall:.3f} below {min_scam_recall}")
    if not reasons:
        return {"decision": "keep_semif", "reasons": [], "use": "semif"}
    if beaten_by:
        use = beaten_by[0]
    elif others:
        use = max(others, key=lambda mod: float(others[mod]["macro_f1"]))
    else:
        use = "semif"  # nothing else was scored on this task
    return {"decision": "fallback", "reasons": reasons, "use": use}


def fallback(
    config: ClassifyConfig,
    ev: Evaluation,
    scores: dict[str, dict[Task, Scores]],
    items: Sequence[dict[str, Any]],
    rows: np.ndarray,
    tops: Sequence[str],
) -> dict[str, Any]:
    """The pre-registered fallback rule per task, on the target split, both platforms,
    unweighted (config/classify.toml [fallback]). Until a model fine-tuned on the Curator's labels
    exists, a task that leaves SemIf uses the model ``decide_task`` names (article Section 4.7)."""
    fb, h = config.fallback, config.h4
    out: dict[str, Any] = {
        "rule": "fallback if another model clearly beats SemIf (bootstrap lower bound of the "
        "macro F1 difference above 0), or SemIf's macro F1 is below the floor, or (scam) its "
        "likely-scam recall is below the minimum",
        "alpha": fb.alpha,
        "min_macro_f1": fb.min_macro_f1,
        "min_scam_recall": fb.min_scam_recall,
        "replicates": h.bootstrap,
        "seed": h.seed,
        "split": ev.split,
        "decided": h.decided,
        "tasks": {},
    }
    recall = {
        (r["model"], r["task"], r["class"]): (r["recall"], r["support"])
        for r in ev.per_class
        if r["platform"] == "all" and r["weighting"] == "unweighted"
    }
    for task in EVAL_TASKS:
        if task not in scores.get("semif", {}):
            out["tasks"][task.value] = {"decision": "unscored", "reasons": [], "use": None}
            continue
        truth, pred_semif = predictions(task, scores["semif"][task], items, rows, tops)
        f_semif = macro_f1_of(task, truth, pred_semif)
        others: dict[str, dict[str, float | bool]] = {}
        for name, by_task in sorted(scores.items()):
            if name == "semif" or task not in by_task:
                continue
            _, pred = predictions(task, by_task[task], items, rows, tops)
            res = m.superiority_bootstrap(
                len(rows),
                macro_f1_of(task, truth, pred),
                f_semif,
                alpha=fb.alpha,
                replicates=h.bootstrap,
                seed=h.seed,
            )
            others[name] = {
                "macro_f1": macro_f1_of(task, truth, pred)(np.arange(len(rows))),
                "difference": res.estimate,
                "lower": res.lower,
                "better": res.better,
            }
        semif_f1 = f_semif(np.arange(len(rows)))
        scam_recall, support = recall.get(("semif", task.value, "likely_scam"), (None, 0))
        if task is not Task.SCAM or not support:
            scam_recall = None
        decision = decide_task(
            task,
            semif_f1,
            others,
            min_macro_f1=fb.min_macro_f1,
            scam_recall=scam_recall,
            min_scam_recall=fb.min_scam_recall,
        )
        out["tasks"][task.value] = {
            "semif_macro_f1": semif_f1,
            "likely_scam_recall": scam_recall,
            "likely_scam_support": support if task is Task.SCAM else None,
            "others": others,
            **decision,
        }
    return out


# --- reports -------------------------------------------------------------------------------


def write_report(run: Path, ev: Evaluation) -> Path:
    out = run / f"report-{ev.run_id}"
    out.mkdir(parents=True, exist_ok=False)
    data = asdict(ev)
    (out / "report.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
    pq.write_table(pa.Table.from_pylist(ev.metrics), out / "metrics.parquet")
    pq.write_table(pa.Table.from_pylist(ev.per_class), out / "per_class.parquet")
    _csv(out / "metrics.csv", ev.metrics)
    _csv(out / "per_class.csv", ev.per_class)
    (out / "report.md").write_text(markdown(ev))
    return out


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _f(x: Any) -> str:
    return "n/a" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def markdown(ev: Evaluation) -> str:
    lines = [
        f"# Classifier bake-off, run {ev.run_id}",
        "",
        f"Split: **{ev.split}**. Temperatures fitted on the calibration rows. Unweighted unless "
        "said otherwise; Bluesky / Mastodon as in the article's Tabela 2.",
        "",
        "| Model | Task | n | Macro F1 (all · bsky / mast) | Balanced acc. | Brier | ECE | T |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = {
        (r["model"], r["task"], r["platform"]): r
        for r in ev.metrics
        if r["weighting"] == "unweighted"
    }
    for model in ev.models:
        for task in EVAL_TASKS:
            r = rows.get((model, task.value, "all"))
            if r is None:
                continue
            bs = rows.get((model, task.value, "bluesky"), {})
            ma = rows.get((model, task.value, "mastodon"), {})

            def pair(
                k: str, r: dict[str, Any] = r, bs: dict[str, Any] = bs, ma: dict[str, Any] = ma
            ) -> str:
                return f"{_f(r[k])} · {_f(bs.get(k))} / {_f(ma.get(k))}"

            lines.append(
                f"| {model} | {task.value} | {r['n']} | {pair('macro_f1')} | "
                f"{pair('balanced_accuracy')} | {pair('brier')} | {pair('ece')} | "
                f"{r['temperature']:.2f} |"
            )
    if ev.h4:
        h = ev.h4
        lines += [
            "",
            "## H4",
            "",
            f"SemIf {h['f1_a']:.3f} vs CryptoBERT {h['f1_b']:.3f}: difference {h['estimate']:.3f}, "
            f"one-sided {1 - h['alpha']:.0%} lower bound {h['lower']:.3f} against -margin "
            f"{-h['margin']:.3f}; p = {h['p_value']:.4f} ({h['replicates']} paired bootstrap "
            f"replicates, seed {h['seed']}, n = {h['n']}). "
            f"{'Supported' if h['supported'] else 'Not supported'}."
            + ("" if h["decided"] else " **The margin is still a proposal (h4.decided = false).**"),
        ]
    if ev.reproducibility:
        r = ev.reproducibility
        per_task = {k: round(v, 3) for k, v in r["same_label_per_task"].items()}
        lines += [
            "",
            "## SemIf reproducibility",
            "",
            f"{r['posts']} posts scored twice: {r['same_all_labels']:.3f} kept every label; "
            f"per task {json.dumps(per_task)}; "
            f"largest logit difference {r['max_abs_logit_diff']:.4f}.",
        ]
    if ev.latency:
        lines += ["", "## Latency (seconds per 1,000 posts)", ""]
        lines += [f"- {k}: {v['seconds_per_1000_posts']:.1f}" for k, v in ev.latency.items()]
    lines += [
        "",
        "## Fallback rule",
        "",
        "```json",
        json.dumps(ev.fallback, indent=2, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


def log_to_mlflow(
    run: Path,
    report: Path,
    ev: Evaluation,
    config: ClassifyConfig,
    *,
    tracking_uri: str,
    register: bool,
) -> str:
    """One MLflow run: parameters, every metric, the report and raw predictions as artifacts
    (no post text). With ``register``, each task's chosen classifier becomes a version of the
    registered model ``cryptonlp-classifier-<task>``, aliased ``live`` (move the alias back to
    roll back)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.bakeoff.mlflow_experiment)
    with mlflow.start_run(run_name=f"bakeoff-{ev.run_id}") as active:
        mlflow.set_tags(
            {
                "cryptonlp.run_id": ev.run_id,
                "cryptonlp.bakeoff_run": ev.manifest.get("run_id", ""),
                "split": ev.split,
                "semif_version": config.semif_version,
                "prompt_sha256": config.prompt_sha256,
                "labeled_set_sha256": ev.manifest.get("labeled_set_sha256", ""),
                "h4_registration": json.dumps(ev.manifest.get("h4_registration")),
            }
        )
        b = config.bakeoff
        mlflow.log_params(
            {
                "semif.model": config.semif.model,
                "semif.revision": config.semif.revision,
                "semif.rubric_version": config.rubric.version,
                "cryptobert.revision": b.cryptobert.revision,
                "finbert.revision": b.finbert.revision,
                "openjev.revision": b.openjev.revision,
                "openjev.subfolder": b.openjev.subfolder,
                "h4.margin": config.h4.margin,
                "h4.alpha": config.h4.alpha,
                "h4.decided": config.h4.decided,
                "fallback.alpha": config.fallback.alpha,
                "fallback.min_macro_f1": config.fallback.min_macro_f1,
                "fallback.min_scam_recall": config.fallback.min_scam_recall,
            }
        )
        metrics: dict[str, float] = {}
        for r in ev.metrics:
            for k in ("macro_f1", "balanced_accuracy", "brier", "ece", "temperature"):
                v = r[k]
                if isinstance(v, float) and math.isfinite(v):
                    metrics[f"{r['model']}.{r['task']}.{r['platform']}.{r['weighting']}.{k}"] = v
        if ev.h4:
            for k in ("estimate", "lower", "p_value", "f1_a", "f1_b"):
                metrics[f"h4.{k}"] = float(ev.h4[k])
        for name, lat in ev.latency.items():
            metrics[f"{name}.seconds_per_1000_posts"] = float(lat["seconds_per_1000_posts"])
        if ev.reproducibility:
            metrics["semif.repeat.same_all_labels"] = float(ev.reproducibility["same_all_labels"])
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(report), "report")
        for name in (*MODELS, "semif-repeat"):
            for suffix in (".jsonl", "-meta.json"):
                path = run / f"{name}{suffix}"
                if path.exists():
                    mlflow.log_artifact(str(path), "raw")
        mlflow.log_artifact(str(run / "manifest.json"), "raw")
        mlflow.log_artifact(str(run / "items.parquet"), "raw")
        if register:
            _register(MlflowClient(), active.info.run_id, ev, config)
        return str(active.info.run_id)


def _register(client: Any, mlflow_run: str, ev: Evaluation, config: ClassifyConfig) -> None:
    import mlflow

    revisions = {
        "semif": f"{config.semif.model}@{config.semif.revision}",
        "openjev": f"{config.bakeoff.openjev.repo}/{config.bakeoff.openjev.subfolder}"
        f"@{config.bakeoff.openjev.revision}",
        "cryptobert": f"{config.bakeoff.cryptobert.repo}@{config.bakeoff.cryptobert.revision}",
        "finbert": f"{config.bakeoff.finbert.repo}@{config.bakeoff.finbert.revision}",
        "vader": "vaderSentiment",
    }
    for task, choice in ev.fallback["tasks"].items():
        chosen = choice["use"]
        if chosen is None:
            continue
        spec = {
            "task": task,
            "classifier": chosen,
            "model": revisions[chosen],
            "temperature": ev.temperatures[chosen][task],
            "semif_version": config.semif_version if chosen == "semif" else None,
            "prompt_sha256": config.prompt_sha256 if chosen in ALL_TASKS else None,
            "decision": choice,
        }
        mlflow.log_dict(spec, f"classifier/{task}/spec.json")
        name = f"cryptonlp-classifier-{task}"
        with contextlib.suppress(Exception):
            client.create_registered_model(name)
        version = client.create_model_version(
            name,
            source=f"{mlflow.get_artifact_uri()}/classifier/{task}",
            run_id=mlflow_run,
            tags={"classifier": chosen, "model": revisions[chosen], "split": ev.split},
        )
        client.set_registered_model_alias(name, "live", version.version)
