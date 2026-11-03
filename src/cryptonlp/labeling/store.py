"""Moving batches to Label Studio and annotations back, and freezing the labeled set.

- ``push``: create one Label Studio project per batch and import its posts (text only).
- ``pull``: copy every annotation version into ``labeling.annotations``, checked against the
  tasks. Run it after each session; the scheduler also runs it hourly when a token is set, so
  the labels never live only in Label Studio.
- ``sync_deletions``: delete in Label Studio the tasks of posts deleted at the source.
- ``freeze``: fix the time split and write the human labels into ``clean.labels``, once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.types.json import Jsonb

from cryptonlp.labeling.config import LabelingConfig
from cryptonlp.labeling.sample import PASS1, RELABEL
from cryptonlp.labeling.studio import Studio
from cryptonlp.labeling.tasks import label_config, read_result
from cryptonlp.schemas.labels import Annotator, Label, Task

DELETED_IN_STUDIO = "deleted in Label Studio"
MODEL_VERSION = {PASS1: "human:pass1", RELABEL: "human:relabel"}


class FreezeError(RuntimeError):
    pass


# --- push ------------------------------------------------------------------------------------


def push(conn: psycopg.Connection, studio: Studio, batch_id: int, *, prefix: str = "") -> int:
    """Create the batch's project and import its posts in labeling order. Returns the project
    ID. A batch is pushed once; a failed push leaves no project ID and can be retried after the
    half-made project is deleted in Label Studio."""
    row = conn.execute(
        "SELECT pass, stratum_day, studio_project_id FROM labeling.batches WHERE batch_id = %s",
        (batch_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"no batch {batch_id}")
    pass_, day, project = row
    if project is not None:
        raise ValueError(f"batch {batch_id} is already in Label Studio (project {project})")
    items = conn.execute(
        "SELECT i.item_id, p.text FROM labeling.items i "
        "JOIN clean.posts p USING (source, uri) WHERE i.batch_id = %s ORDER BY i.position",
        (batch_id,),
    ).fetchall()
    if not items:
        raise ValueError(f"batch {batch_id} has no posts left to label")
    title = f"{prefix}{pass_} {day or ''} batch {batch_id}".replace("  ", " ")
    description = (
        "Second labeling, blind: do not open the first-pass projects."
        if pass_ == RELABEL
        else "First pass. Follow the labeling manual."
    )
    config = label_config()
    studio.validate_config(config)
    project_id = studio.create_project(title, config, description)
    studio.import_tasks(
        project_id, [{"data": {"item_id": item_id, "text": text}} for item_id, text in items]
    )
    ids = studio.task_ids(project_id)
    missing = {item_id for item_id, _ in items} - set(ids)
    if missing:
        raise RuntimeError(f"Label Studio project {project_id} lacks {len(missing)} posts")
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "UPDATE labeling.items SET studio_task_id = %s WHERE item_id = %s",
            [(task_id, item_id) for item_id, task_id in ids.items()],
        )
        cur.execute(
            "UPDATE labeling.batches SET studio_project_id = %s WHERE batch_id = %s",
            (project_id, batch_id),
        )
    return project_id


# --- pull ------------------------------------------------------------------------------------


INSERT_ANNOTATION = """
INSERT INTO labeling.annotations (
    item_id, studio_annotation_id, created_at, updated_at, lead_time, cancelled, result,
    labels, problem, uncertain, error, run_id
) VALUES (
    %(item_id)s, %(annotation_id)s, %(created_at)s, %(updated_at)s, %(lead_time)s,
    %(cancelled)s, %(result)s, %(labels)s, %(problem)s, %(uncertain)s, %(error)s, %(run_id)s
)
ON CONFLICT DO NOTHING
"""


def annotation_rows(
    tasks: Sequence[dict[str, Any]], known_items: set[int], run_id: str
) -> list[dict[str, Any]]:
    """Rows for every annotation version in an export, for items still in the study."""
    rows = []
    for task in tasks:
        item_id = int(task["data"]["item_id"])
        if item_id not in known_items:
            continue  # deleted at the source; sync_deletions removes the task
        for ann in task.get("annotations", []):
            read = read_result(ann.get("result") or [])
            cancelled = bool(ann.get("was_cancelled"))
            rows.append(
                {
                    "item_id": item_id,
                    "annotation_id": int(ann["id"]),
                    "created_at": ann["created_at"],
                    "updated_at": ann["updated_at"],
                    "lead_time": ann.get("lead_time"),
                    "cancelled": cancelled,
                    "result": Jsonb(ann.get("result") or []),
                    "labels": None if cancelled or read.labels is None else Jsonb(read.labels),
                    "problem": None if cancelled else read.problem,
                    "uncertain": read.uncertain,
                    "error": None if cancelled else read.error,
                    "run_id": run_id,
                }
            )
    return rows


@dataclass
class PullStats:
    projects: int = 0
    new_versions: int = 0
    withdrawn: int = 0


def pull(conn: psycopg.Connection, studio: Studio, *, now: datetime, run_id: str) -> PullStats:
    stats = PullStats()
    projects = conn.execute(
        "SELECT batch_id, studio_project_id FROM labeling.batches "
        "WHERE studio_project_id IS NOT NULL ORDER BY batch_id"
    ).fetchall()
    for batch_id, project_id in projects:
        tasks = studio.export(project_id)
        known = {
            r[0]
            for r in conn.execute(
                "SELECT item_id FROM labeling.items WHERE batch_id = %s", (batch_id,)
            )
        }
        rows = annotation_rows(tasks, known, run_id)
        # Items whose annotations were all deleted in Label Studio get a withdrawn version.
        annotated = {r["item_id"] for r in rows}
        live = conn.execute(
            "SELECT c.item_id, c.studio_annotation_id FROM labeling.current c "
            "JOIN labeling.items i USING (item_id) "
            "WHERE i.batch_id = %s AND c.error IS DISTINCT FROM %s",
            (batch_id, DELETED_IN_STUDIO),
        ).fetchall()
        withdrawn = [
            {
                "item_id": item_id,
                "annotation_id": annotation_id,
                "created_at": now,
                "updated_at": now,
                "lead_time": None,
                "cancelled": True,
                "result": Jsonb([]),
                "labels": None,
                "problem": None,
                "uncertain": False,
                "error": DELETED_IN_STUDIO,
                "run_id": run_id,
            }
            for item_id, annotation_id in live
            if item_id not in annotated
        ]
        with conn.transaction(), conn.cursor() as cur:
            for r in rows + withdrawn:
                cur.execute(INSERT_ANNOTATION, r)
                stats.new_versions += cur.rowcount
        stats.withdrawn += len(withdrawn)
        stats.projects += 1
    return stats


def sync_deletions(conn: psycopg.Connection, studio: Studio) -> int:
    """Delete in Label Studio the tasks of posts deleted at the source. Returns how many."""
    queued = [r[0] for r in conn.execute("SELECT studio_task_id FROM labeling.studio_deletions")]
    for task_id in queued:
        studio.delete_task(task_id)
        conn.execute("DELETE FROM labeling.studio_deletions WHERE studio_task_id = %s", (task_id,))
    return len(queued)


# --- status ----------------------------------------------------------------------------------


def status(conn: psycopg.Connection, config: LabelingConfig, now: datetime) -> dict[str, Any]:
    cols = ["batch_id", "pass", "day", "drawn", "items", "annotated", "usable", "problems",
            "invalid", "seconds_labeling"]  # fmt: skip
    batches = [
        dict(zip(cols, r, strict=True))
        for r in conn.execute("SELECT * FROM monitoring.labeling_progress ORDER BY batch_id")
    ]
    for b in batches:
        b["day"] = b["day"].isoformat() if b["day"] else None
        secs = b.pop("seconds_labeling") or 0.0
        b["hours_labeling"] = round(secs / 3600, 2)
        b["posts_per_hour"] = round(b["annotated"] / (secs / 3600), 1) if secs else None
    usable = sum(b["usable"] for b in batches if b["pass"] == PASS1)
    # The day the re-labeling batch can be drawn: when the size-th usable label turns old enough.
    row = conn.execute(
        "SELECT c.updated_at FROM labeling.items i JOIN labeling.current c USING (item_id) "
        "WHERE i.pass = %s AND c.labels IS NOT NULL ORDER BY c.updated_at OFFSET %s LIMIT 1",
        (PASS1, config.relabel.size - 1),
    ).fetchone()
    relabel_from = None if row is None else (row[0] + config.relabel.min_gap).isoformat()
    frozen = conn.execute("SELECT count(*) FROM labeling.splits").fetchone()
    return {
        "first_pass_usable": usable,
        "target": config.sample.target,
        "relabel_drawable_from": relabel_from,
        "split_frozen": bool(frozen and frozen[0]),
        "batches": batches,
    }


# --- freeze ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Frozen:
    labeled: int
    calibration: int
    test: int
    problems: int
    manifest: Path
    manifest_sha256: str


def _current(conn: psycopg.Connection, pass_: str) -> list[tuple[Any, ...]]:
    return conn.execute(
        """
        SELECT i.item_id, i.source, i.uri, i.batch_id, i.stratum, r.received_at,
               c.labels, c.problem, c.error, c.cancelled, c.uncertain, c.updated_at, c.lead_time
        FROM labeling.items i
        JOIN raw.posts r USING (source, uri)
        LEFT JOIN labeling.current c USING (item_id)
        WHERE i.pass = %s
        ORDER BY r.received_at, i.source, i.uri
        """,
        (pass_,),
    ).fetchall()


def _labels_for(rows: Sequence[tuple[Any, ...]], pass_: str, run_id: str) -> list[Label]:
    labels = []
    for _item, _source, uri, _b, _s, _r, values, *_rest, updated_at, _lt in rows:
        for task in Task:
            labels.append(
                Label(
                    post_uri=uri,
                    task=task,
                    values=tuple(values[task.value]),
                    annotator=Annotator.HUMAN,
                    model_version=MODEL_VERSION[pass_],
                    labeled_at=updated_at,
                    run_id=run_id,
                )
            )
    return labels


INSERT_LABEL = """
INSERT INTO clean.labels (
    source, uri, task, model_version, annotator, label_values, probabilities, confidence,
    labeled_at, run_id
) VALUES (%s, %s, %s, %s, 'human', %s, '{}', NULL, %s, %s)
"""


def _write_labels(
    conn: psycopg.Connection, rows: Sequence[tuple[Any, ...]], labels: Sequence[Label]
) -> None:
    source_of = {r[2]: r[1] for r in rows}
    with conn.cursor() as cur:
        cur.executemany(
            INSERT_LABEL,
            [
                (
                    source_of[lb.post_uri],
                    lb.post_uri,
                    lb.task.value,
                    lb.model_version,
                    list(lb.values),
                    lb.labeled_at,
                    lb.run_id,
                )
                for lb in labels
            ],
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_complete(rows: Sequence[tuple[Any, ...]], what: str) -> None:
    # No usable label, no problem flag, no error: never annotated, skipped, or withdrawn.
    invalid = [r[0] for r in rows if r[8] is not None and r[8] != DELETED_IN_STUDIO]
    unannotated = sum(1 for r in rows if r[6] is None and r[7] is None) - len(invalid)
    if invalid:
        raise FreezeError(f"{what}: items {invalid[:10]} have invalid annotations; fix and pull")
    if unannotated:
        raise FreezeError(f"{what}: {unannotated} posts are not labeled yet (pull first?)")


def freeze(
    conn: psycopg.Connection,
    config: LabelingConfig,
    out_dir: Path,
    *,
    run_id: str,
    allow_short: bool = False,
) -> Frozen:
    """Fix the split (oldest ``split.calibration`` usable posts by receive time: calibration;
    the rest: test), write the first-pass labels to ``clean.labels`` and a manifest. Once."""
    done = conn.execute("SELECT count(*) FROM labeling.splits").fetchone()
    if done and done[0]:
        raise FreezeError("the split is frozen already")
    rows = _current(conn, PASS1)
    _check_complete(rows, "first pass")
    usable = [r for r in rows if r[6] is not None]
    if len(usable) < config.sample.target and not allow_short:
        raise FreezeError(
            f"{len(usable)} usable posts, below the target of {config.sample.target}: draw a "
            "top-up batch, or freeze with allow_short"
        )
    if len(usable) <= config.split.calibration:
        raise FreezeError("not enough posts for a test set after the calibration slice")
    n_cal = config.split.calibration
    split = {r[0]: ("calibration" if i < n_cal else "test") for i, r in enumerate(usable)}
    labels = _labels_for(usable, PASS1, run_id)

    text_sha: dict[str, str] = dict(
        conn.execute(
            "SELECT uri, text_sha256 FROM clean.posts WHERE (source, uri) IN "
            "(SELECT source, uri FROM labeling.items WHERE pass = %s)",
            (PASS1,),
        ).fetchall()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"labeled-set-{run_id}.parquet"
    table = pa.table(
        {
            "item_id": [r[0] for r in usable],
            "source": [r[1] for r in usable],
            "uri": [r[2] for r in usable],
            "batch_id": [r[3] for r in usable],
            "stratum": [r[4] for r in usable],
            "received_at": [r[5] for r in usable],
            "split": [split[r[0]] for r in usable],
            "uncertain": [bool(r[10]) for r in usable],
            "labeled_at": [r[11] for r in usable],
            "text_sha256": [text_sha.get(r[2]) for r in usable],
            **{t.value: [list(r[6][t.value]) for r in usable] for t in Task},
        }
    )

    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO labeling.splits (item_id, split, run_id) VALUES (%s, %s, %s)",
                [(item_id, s, run_id) for item_id, s in split.items()],
            )
        _write_labels(conn, usable, labels)
        pq.write_table(table, manifest)
    return Frozen(
        labeled=len(usable),
        calibration=n_cal,
        test=len(usable) - n_cal,
        problems=sum(1 for r in rows if r[7] is not None),
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest),
    )


def freeze_relabel(conn: psycopg.Connection, *, run_id: str) -> int:
    """Write the re-labeling pass to ``clean.labels`` (model_version ``human:relabel``)."""
    rows = _current(conn, RELABEL)
    if not rows:
        raise FreezeError("no re-labeling batch")
    _check_complete(rows, "re-labeling")
    usable = [r for r in rows if r[6] is not None]
    written = conn.execute(
        "SELECT count(*) FROM clean.labels WHERE model_version = %s", (MODEL_VERSION[RELABEL],)
    ).fetchone()
    if written and written[0]:
        raise FreezeError("the re-labeling pass is in clean.labels already")
    with conn.transaction():
        _write_labels(conn, usable, _labels_for(usable, RELABEL, run_id))
    return len(usable)
