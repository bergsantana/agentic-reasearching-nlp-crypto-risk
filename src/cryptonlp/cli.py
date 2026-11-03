"""``cryptonlp``: run collectors and maintenance jobs by hand, or serve them on a schedule.

Every job prints its run ID and records itself in ``ops.job_runs``, which the dashboard reads.
Live jobs use the main database; ``replay`` uses only the test database, and the ``ledger``
commands that write take ``--test`` to try them there first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg

from cryptonlp import db, ledger, ops
from cryptonlp.collectors import binance, bluesky, deletions, mastodon, replay, trends
from cryptonlp.collectors.common import Recorder, WatchFilter, http_client
from cryptonlp.config import CollectorsConfig, load_config
from cryptonlp.forecast.config import load_loop_config
from cryptonlp.labeling import agreement as labeling_agreement
from cryptonlp.labeling import sample as labeling_sample
from cryptonlp.labeling import store as labeling_store
from cryptonlp.labeling.config import load_labeling_config
from cryptonlp.labeling.studio import Studio, StudioError
from cryptonlp.labeling.tasks import label_config
from cryptonlp.ledger import check as ledger_check
from cryptonlp.normalize import review
from cryptonlp.normalize.config import load_normalize_config
from cryptonlp.normalize.pipeline import Normalizer
from cryptonlp.normalize.pseudonym import PseudonymKeyError, generate_key, load_key
from cryptonlp.rawstore import RawStore
from cryptonlp.run_id import new_run_id
from cryptonlp.schemas.common import Asset, Phase
from cryptonlp.schemas.ledger import (
    DataFile,
    Document,
    DocumentKind,
    Intervention,
    InterventionType,
)
from cryptonlp.settings import REPO_ROOT, Settings
from cryptonlp.stubs import STUB_VERSION, StubClassifier

if TYPE_CHECKING:
    from cryptonlp.forecast.loop import HourlyLoop, LoopStats

log = logging.getLogger("cryptonlp")

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "replay"


def _connect_main() -> psycopg.Connection:
    conn = db.connect_main()
    conn.autocommit = True
    return conn


def _connect_test() -> psycopg.Connection:
    conn = db.connect_test()
    conn.autocommit = True
    return conn


class Context:
    def __init__(self, args: argparse.Namespace) -> None:
        self.settings = Settings()
        self.config: CollectorsConfig = load_config(self.settings.collectors_config_path)
        self.store = RawStore(self.settings.raw_path)
        record = getattr(args, "record", False) or self.settings.record_replay
        self.recorder = Recorder(self.settings.replay_path) if record else None
        self.run_id = new_run_id()


def run_job(
    name: str,
    fn: Callable[[psycopg.Connection, Context], dict[str, Any] | None],
    args: argparse.Namespace,
    *,
    exclusive: bool = False,
) -> dict[str, Any]:
    ctx = Context(args)
    print(f"run ID {ctx.run_id}", flush=True)
    tracked = ops.track_job(_connect_main, name, ctx.run_id, exclusive=exclusive)
    with tracked as job, _connect_main() as conn:
        try:
            job.stats.update(fn(conn, ctx) or {})
        finally:
            if ctx.recorder:
                ctx.recorder.close()
    print(json.dumps(job.stats, default=str), flush=True)
    return job.stats


# --- jobs ----------------------------------------------------------------------------------


def job_bluesky(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
    collector = bluesky.BlueskyCollector(
        conn,
        ctx.store,
        ctx.config.bluesky,
        WatchFilter(ctx.config.watch),
        ctx.run_id,
        recorder=ctx.recorder,
    )
    cursor = collector.saved_cursor()
    collector.last_time_us = cursor  # so a jump past the 36-hour lookback is logged as a gap
    messages = bluesky.jetstream_messages(
        ctx.config.bluesky.jetstream_hosts, cursor, ctx.config.bluesky.rewind_seconds
    )
    # `docker compose stop` sends SIGTERM: stop like Ctrl-C, so the buffer is flushed first.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        asyncio.run(collector.run(messages))
    except KeyboardInterrupt:
        log.info("stopped")
    return collector.stats


def job_mastodon(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
    collector = mastodon.MastodonCollector(
        conn,
        ctx.store,
        ctx.config.mastodon,
        WatchFilter(ctx.config.watch),
        ctx.run_id,
        recorder=ctx.recorder,
    )
    with http_client() as client:
        return collector.poll(client)


def job_trends(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
    with http_client() as client:
        return trends.snapshot(conn, ctx.store, client, ctx.config, ctx.run_id, ctx.recorder)


def job_binance_tail(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
    with http_client() as client:
        return binance.tail(conn, ctx.store, client, ctx.config.binance, ctx.run_id, ctx.recorder)


def binance_backfill_job(
    max_archives: int | None,
) -> Callable[[psycopg.Connection, Context], dict[str, Any]]:
    def job(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
        with http_client(timeout=120) as client:
            return binance.backfill(
                conn, ctx.store, client, ctx.config.binance, ctx.run_id, max_archives=max_archives
            )

    return job


def job_deletion_sync(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
    with http_client() as client:
        return deletions.sync_deletions(conn, ctx.store, client, ctx.config, ctx.run_id)


def job_rawstore_verify(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
    report = ctx.store.verify(conn)
    stats = {
        "checked": report.checked,
        "mismatched": report.mismatched[:50],
        "missing": report.missing[:50],
        "unregistered": report.unregistered[:50],
    }
    if not report.ok:
        raise RuntimeError(f"raw store check failed: {stats}")
    return stats


def job_rawstore_reindex(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
    return {"registered": ctx.store.reindex(conn)}


def ledger_check_job(
    anchor_files: Sequence[Path] = (),
) -> Callable[[psycopg.Connection, Context], dict[str, Any]]:
    """Recompute the whole chain, against every stored anchor and any anchor files given."""

    def job(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
        anchors = [
            *ledger_check.stored_anchors(conn),
            *(ledger_check.Anchor.load(p) for p in anchor_files),
        ]
        report = ledger_check.check_chain(conn, anchors)
        ledger_check.record_check(conn, report, ctx.run_id, ctx.settings.phase)
        stats = report.summary()
        if not report.ok:
            raise RuntimeError(f"ledger chain check failed: {json.dumps(stats['problems'][:5])}")
        return stats

    return job


def job_ledger_anchor(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
    if ledger.head(conn) is None:
        print("the ledger is empty: nothing to anchor yet", flush=True)
        return {"skipped": "ledger is empty"}
    anchor, path = ledger_check.make_anchor(conn, ctx.settings.ledger_path / "anchors", ctx.run_id)
    print(ledger_check.anchor_message(anchor), flush=True)
    print(f"written to {path}: send it to the advisor", flush=True)
    return {"head_seq": anchor.head_seq, "head_hash": anchor.head_hash, "path": str(path)}


def normalize_job(
    *, rebuild: bool = False, max_batches: int | None = None
) -> Callable[[psycopg.Connection, Context], dict[str, Any]]:
    """Normalize every pending raw post; with ``rebuild``, start again from an empty store."""

    def job(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
        key = load_key(ctx.settings.pseudonym_key_file)
        normalizer = Normalizer(load_normalize_config(), key, ctx.store, ctx.run_id)
        if rebuild:
            # Stub labels are thrown away with the posts. Any other labels (a real classifier,
            # the human passes) make this refuse; so will vectors, once they refer to these rows.
            row = conn.execute(
                "SELECT count(*) FROM clean.labels WHERE model_version <> %s", (STUB_VERSION,)
            ).fetchone()
            if row and row[0]:
                raise RuntimeError(
                    f"clean.labels holds {row[0]} labels besides the stub's: not rebuilding"
                )
            conn.execute("TRUNCATE clean.labels, clean.minhash, clean.posts")
        return asdict(normalizer.run(conn, max_batches=max_batches))

    return job


# The forecast modules need the statistics stack (pandas, arch), imported only by the jobs and
# commands that forecast, so every other scheduled job keeps running on an older image.
def _loop(ctx: Context) -> HourlyLoop:
    from cryptonlp.forecast.loop import HourlyLoop

    return HourlyLoop.default(
        load_loop_config(ctx.settings.loop_config_path),
        StubClassifier(),
        ctx.store,
        phase=ctx.settings.phase,
        run_id=ctx.run_id,
    )


def _loop_result(stats: LoopStats) -> dict[str, Any]:
    from cryptonlp.forecast.loop import LoopError

    if stats.errors:
        print(json.dumps(stats.as_dict(), default=str), flush=True)
        raise LoopError(stats)
    return stats.as_dict()


def job_forecast_loop(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
    return _loop_result(_loop(ctx).run(conn))


def _studio(settings: Settings) -> Studio:
    url = (
        settings.label_studio_url or load_labeling_config(settings.labeling_config_path).studio.url
    )
    return Studio(url, settings.label_studio_api_key)


def job_labeling_sync(conn: psycopg.Connection, ctx: Context) -> dict[str, Any]:
    """Copy annotations out of Label Studio, and delete there the posts deleted at the source."""
    with _studio(ctx.settings) as studio:
        deleted = labeling_store.sync_deletions(conn, studio)
        stats = labeling_store.pull(
            conn, studio, now=ledger.database_clock(conn), run_id=ctx.run_id
        )
    return {**asdict(stats), "studio_tasks_deleted": deleted}


JOBS: dict[str, Callable[[psycopg.Connection, Context], dict[str, Any]]] = {
    "collect_bluesky": job_bluesky,
    "collect_mastodon": job_mastodon,
    "collect_trends": job_trends,
    "collect_binance_tail": job_binance_tail,
    "deletion_sync": job_deletion_sync,
    "rawstore_verify": job_rawstore_verify,
    "rawstore_reindex": job_rawstore_reindex,
    "normalize": normalize_job(),
    "ledger_check": ledger_check_job(),
    "ledger_anchor": job_ledger_anchor,
    "forecast_loop": job_forecast_loop,
    "labeling_sync": job_labeling_sync,
}


# --- commands ------------------------------------------------------------------------------


def cmd_collect(args: argparse.Namespace) -> None:
    name = f"collect_{args.source.replace('-', '_')}"
    if args.source == "binance-backfill":
        run_job(name, binance_backfill_job(args.max_archives), args)
    else:
        run_job(name, JOBS[name], args, exclusive=args.source == "bluesky")


def cmd_db_init(args: argparse.Namespace) -> None:
    settings = Settings()
    conn = db.connect_test(settings) if args.test else db.connect_main(settings)
    with conn:
        applied = db.apply_init_sql(conn)
        db.set_grafana_password(conn, settings.grafana_db_password)
        if not args.test:
            created = db.ensure_label_studio_database(conn, settings.label_studio_db_password)
            if created:
                print(f"created database {db.LABEL_STUDIO_DB} for Label Studio")
    print(f"applied {', '.join(applied)} to {'test' if args.test else 'main'} database")


def cmd_freeze(args: argparse.Namespace) -> None:
    settings = Settings()
    day = date.fromisoformat(args.day)
    dest = Path(args.to) if args.to else FIXTURES / day.isoformat()
    manifest = replay.freeze(settings.replay_path, day, dest, require_full_day=not args.partial)
    print(json.dumps(manifest, indent=2))


def cmd_replay(args: argparse.Namespace) -> None:
    settings = Settings()
    config = load_config(settings.collectors_config_path)
    run_id = new_run_id()
    print(f"run ID {run_id}", flush=True)
    store = RawStore(Path(args.raw_dir))
    with _connect_test() as conn:
        db.apply_init_sql(conn)
        stats = asyncio.run(replay.replay(conn, store, config, Path(args.directory), run_id))
    print(json.dumps(stats, indent=2, default=str))


def _ledger_conn(args: argparse.Namespace) -> psycopg.Connection:
    return _connect_test() if getattr(args, "test", False) else _connect_main()


def _print_entry(entry: ledger.Entry) -> None:
    print(
        f"ledger entry {entry.seq} in {entry.table}, recorded {entry.recorded_at.isoformat()}, "
        f"hash {entry.hash}"
    )


def cmd_ledger_check(args: argparse.Namespace) -> None:
    run_job("ledger_check", ledger_check_job([Path(p) for p in args.anchor]), args)


def cmd_ledger_head(args: argparse.Namespace) -> None:
    with _ledger_conn(args) as conn:
        entry = ledger.head(conn)
    if entry is None:
        print("the ledger is empty")
    else:
        _print_entry(entry)


def cmd_ledger_export(args: argparse.Namespace) -> None:
    settings = Settings()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    default = settings.ledger_path / "exports" / f"ledger-{stamp}.jsonl"
    path = Path(args.path) if args.path else default
    with _ledger_conn(args) as conn:
        report = ledger_check.export(conn, path)
    print(json.dumps({"path": str(path), **report.summary()}, indent=2))


def cmd_ledger_verify_export(args: argparse.Namespace) -> None:
    anchors = [ledger_check.Anchor.load(Path(p)) for p in args.anchor]
    report = ledger_check.verify_export(Path(args.path), anchors)
    print(json.dumps(report.summary(), indent=2))
    if not report.ok:
        sys.exit(1)


def cmd_ledger_intervene(args: argparse.Namespace) -> None:
    run_id = new_run_id()
    print(f"run ID {run_id}", flush=True)
    record = Intervention(
        stage=args.stage,
        type=InterventionType(args.type),
        at_checkpoint=args.checkpoint,
        reason=args.reason,
        actor=args.actor,
        phase=Phase(args.phase or Settings().phase),
        run_id=run_id,
    )
    with _ledger_conn(args) as conn:
        _print_entry(ledger.append(conn, record))


def cmd_ledger_register_document(args: argparse.Namespace) -> None:
    run_id = new_run_id()
    print(f"run ID {run_id}", flush=True)
    record = Document.from_file(
        Path(args.path),
        repo_root=REPO_ROOT,
        document_key=args.key,
        kind=DocumentKind(args.kind),
        title=args.title,
        version=args.version,
        supersedes=args.supersedes,
        change_reason=args.reason,
        phase=Phase(args.phase or Settings().phase),
        run_id=run_id,
    )
    with _ledger_conn(args) as conn:
        entry = ledger.append(conn, record)
    _print_entry(entry)
    print(f"content SHA-256 {record.content_sha256}")


def cmd_keygen(args: argparse.Namespace) -> None:
    path = Path(args.path) if args.path else Settings().pseudonym_key_file
    key = generate_key(path)
    print(f"wrote a new pseudonymization key to {path} (key ID {key.key_id})")
    print("Back it up outside the repository: a lost key gives every author a new hash.")


def cmd_normalize_run(args: argparse.Namespace) -> None:
    run_job("normalize", normalize_job(max_batches=args.max_batches), args)


def cmd_normalize_rebuild(args: argparse.Namespace) -> None:
    if not args.yes:
        sys.exit("rebuild empties clean.posts and normalizes everything again: pass --yes")
    run_job("normalize_rebuild", normalize_job(rebuild=True), args, exclusive=True)


def cmd_normalize_sample(args: argparse.Namespace) -> None:
    settings = Settings()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    default = settings.ledger_path.parent / "review" / f"normalize-sample-{stamp}.csv"
    path = Path(args.out) if args.out else default
    with _ledger_conn(args) as conn:
        rows = review.sample(conn, per_group=args.per_group, seed=args.seed)
    review.write_csv(rows, path)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    print(json.dumps({"path": str(path), "rows": len(rows), "per_group": counts}, indent=2))
    print("The file holds post text: keep it under data/, never commit it.")


def cmd_loop_run(args: argparse.Namespace) -> None:
    if args.test:
        ctx = Context(args)
        print(f"run ID {ctx.run_id}", flush=True)
        with _connect_test() as conn:
            db.apply_init_sql(conn)
            stats = _loop_result(_loop(ctx).run(conn))
        print(json.dumps(stats, default=str), flush=True)
        return
    if not args.yes:
        sys.exit(
            "this writes forecasts and scores to the main ledger, where they stay for good "
            f"(phase {Settings().phase.value}): pass --yes, or --test for the test database"
        )
    run_job("forecast_loop", job_forecast_loop, args)


def cmd_loop_simulate(args: argparse.Namespace) -> None:
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    if start.tzinfo is None or end.tzinfo is None:
        sys.exit("give times with a UTC offset, e.g. 2026-11-05T00:00Z")
    ctx = Context(args)
    ctx.store = RawStore(Path(args.raw_dir))
    print(f"run ID {ctx.run_id}", flush=True)
    with _connect_test() as conn:
        db.apply_init_sql(conn)
        from cryptonlp.forecast.loop import simulate

        runs = simulate(conn, _loop(ctx), start, end, minute=args.minute)
    for stats in runs:
        print(json.dumps(stats.as_dict(), default=str), flush=True)
    failed = [s for s in runs if s.errors]
    if failed:
        sys.exit(f"{len(failed)} of {len(runs)} hourly runs had failing steps")


def cmd_arm1_walkforward(args: argparse.Namespace) -> None:
    """Walk-forward test of arm 1 and the baseline on development data (read-only)."""
    import pandas as pd

    from cryptonlp.features import bars
    from cryptonlp.forecast import walkforward

    settings = Settings()
    config = load_loop_config(settings.loop_config_path)
    if args.end:
        end = datetime.fromisoformat(args.end)
        if end.tzinfo is None:
            sys.exit("give --end with a UTC offset, e.g. 2026-11-02T00:00Z")
        config = config.model_copy(
            update={"walkforward": config.walkforward.model_copy(update={"end": end})}
        )
    store = RawStore(settings.raw_path)
    run_id = new_run_id()
    print(f"run ID {run_id}", flush=True)
    out = settings.walkforward_path / run_id
    results: list[walkforward.WalkForwardResult] = []
    files: list[DataFile] = []
    with _connect_test() if args.test else _connect_main() as conn:
        now = ledger.database_clock(conn)
        last = config.walkforward.end or now.replace(minute=0, second=0, microsecond=0)
        for name in args.asset:
            asset = Asset(name)
            hourly = bars.load_hourly(conn, store, asset, config.arm1.train_start, last, now=now)
            have = int(hourly.frame["ret"].notna().sum())
            print(
                f"{asset.value}: {have} hourly bars with data, {len(hourly.files)} files",
                flush=True,
            )
            result = walkforward.run(hourly.frame, asset, config)
            results.append(result)
            files += hourly.files
            out.mkdir(parents=True, exist_ok=True)
            result.metrics.to_csv(out / f"{asset.value}-metrics.csv", index=False)
            result.comparisons.to_csv(out / f"{asset.value}-comparisons.csv", index=False)
            result.forecasts.to_parquet(out / f"{asset.value}-forecasts.parquet", index=False)
            with pd.option_context("display.width", 160, "display.max_columns", 20):
                print(walkforward.summary(result).to_string(index=False), flush=True)
    print(f"tables in {out}", flush=True)
    if args.no_mlflow:
        return
    uri = args.tracking_uri or settings.mlflow_tracking_uri
    local = None if uri.startswith(("http://", "https://")) else settings.walkforward_path
    mlflow_run = walkforward.log_to_mlflow(
        results,
        config,
        run_id=run_id,
        data_files=files,
        tracking_uri=uri,
        artifact_location=(local / "mlflow-artifacts").as_uri() if local else None,
    )
    print(f"MLflow run {mlflow_run} at {uri}", flush=True)


def _draw_and_push(args: argparse.Namespace, draw: Callable[..., labeling_sample.Batch]) -> None:
    settings = Settings()
    config = load_labeling_config(settings.labeling_config_path)
    run_id = new_run_id()
    print(f"run ID {run_id}", flush=True)
    with _ledger_conn(args) as conn:
        batch = draw(conn, config, now=ledger.database_clock(conn), run_id=run_id)
        print(labeling_sample.describe(batch), flush=True)
        if args.no_push:
            print(f"not pushed: run `cryptonlp labeling push {batch.batch_id}`")
            return
        with _studio(settings) as studio:
            project = labeling_store.push(
                conn, studio, batch.batch_id, prefix="TEST " if args.test else ""
            )
    print(f"Label Studio project {project}: {config.studio.url}/projects/{project}")


def cmd_labeling_draw(args: argparse.Namespace) -> None:
    day = date.fromisoformat(args.day) if args.day else None

    def draw(conn: psycopg.Connection, config: Any, *, now: datetime, run_id: str) -> Any:
        the_day = day or (now - timedelta(days=1)).date()
        return labeling_sample.draw_pass1(
            conn, config, the_day, now=now, run_id=run_id, size=args.size, seed=args.seed
        )

    _draw_and_push(args, draw)


def cmd_labeling_relabel(args: argparse.Namespace) -> None:
    def draw(conn: psycopg.Connection, config: Any, *, now: datetime, run_id: str) -> Any:
        return labeling_sample.draw_relabel(conn, config, now=now, run_id=run_id, seed=args.seed)

    _draw_and_push(args, draw)


def cmd_labeling_push(args: argparse.Namespace) -> None:
    settings = Settings()
    with _ledger_conn(args) as conn, _studio(settings) as studio:
        project = labeling_store.push(conn, studio, args.batch, prefix="TEST " if args.test else "")
    print(f"Label Studio project {project}")


def cmd_labeling_status(args: argparse.Namespace) -> None:
    config = load_labeling_config(Settings().labeling_config_path)
    with _ledger_conn(args) as conn:
        report = labeling_store.status(conn, config, ledger.database_clock(conn))
        invalid = conn.execute(
            "SELECT c.item_id, i.batch_id, i.position, c.error FROM labeling.current c "
            "JOIN labeling.items i USING (item_id) WHERE c.error IS NOT NULL "
            "AND c.error <> %s ORDER BY 2, 3",
            (labeling_store.DELETED_IN_STUDIO,),
        ).fetchall()
    report["invalid_annotations"] = [
        {"item_id": r[0], "batch_id": r[1], "position": r[2], "error": r[3]} for r in invalid
    ]
    print(json.dumps(report, indent=2, default=str))


def cmd_labeling_agreement(args: argparse.Namespace) -> None:
    settings = Settings()
    config = load_labeling_config(settings.labeling_config_path)
    run_id = new_run_id()
    with _ledger_conn(args) as conn:
        report = labeling_agreement.agreement(conn, config.relabel.min_gap)
    report["run_id"] = run_id
    out = settings.labeling_path / f"agreement-{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    print(f"written to {out}")


def cmd_labeling_freeze(args: argparse.Namespace) -> None:
    settings = Settings()
    config = load_labeling_config(settings.labeling_config_path)
    run_id = new_run_id()
    print(f"run ID {run_id}", flush=True)
    with _ledger_conn(args) as conn:
        if args.relabel:
            n = labeling_store.freeze_relabel(conn, run_id=run_id)
            print(f"wrote the re-labeling of {n} posts to clean.labels (human:relabel)")
            return
        frozen = labeling_store.freeze(
            conn, config, settings.labeling_path, run_id=run_id, allow_short=args.allow_short
        )
    print(json.dumps(asdict(frozen), indent=2, default=str))
    print(
        "The manifest holds post URIs: keep it under data/. Cite its SHA-256 in the H4 "
        "experiment package, and log the stage 3 approval with `cryptonlp ledger intervene "
        "--stage 3 --type approval --checkpoint --reason ...`."
    )


def cmd_serve(args: argparse.Namespace) -> None:
    from cryptonlp.flows.collect import serve

    serve()


def _add_ledger_parser(sub: Any) -> None:
    p = sub.add_parser("ledger", help="the pre-registration ledger")
    l_sub = p.add_subparsers(dest="action", required=True)
    phases = [ph.value for ph in Phase]

    def with_test(q: argparse.ArgumentParser) -> argparse.ArgumentParser:
        q.add_argument("--test", action="store_true", help="the test database instead")
        return q

    q = l_sub.add_parser("check", help="recompute the whole hash chain (the daily job)")
    q.add_argument(
        "--anchor", action="append", default=[], help="an anchor file to check against too"
    )
    q.set_defaults(func=cmd_ledger_check)

    q = l_sub.add_parser("anchor", help="record the head hash to send to the advisor (weekly)")
    q.set_defaults(func=lambda a: run_job("ledger_anchor", job_ledger_anchor, a))

    q = with_test(l_sub.add_parser("head", help="show the newest entry"))
    q.set_defaults(func=cmd_ledger_head)

    q = with_test(l_sub.add_parser("export", help="write the chain as JSON lines"))
    q.add_argument("path", nargs="?", help="default data/ledger/exports/ledger-<time>.jsonl")
    q.set_defaults(func=cmd_ledger_export)

    q = l_sub.add_parser("verify-export", help="check an export file with SHA-256 alone (no DB)")
    q.add_argument("path")
    q.add_argument("--anchor", action="append", default=[], help="an anchor file to check")
    q.set_defaults(func=cmd_ledger_verify_export)

    q = with_test(l_sub.add_parser("intervene", help="log a human intervention or approval"))
    q.add_argument("--stage", type=int, required=True, choices=range(1, 10), metavar="1-9")
    q.add_argument("--type", required=True, choices=[t.value for t in InterventionType])
    q.add_argument("--reason", required=True)
    q.add_argument("--checkpoint", action="store_true", help="a planned stage approval (Quadro 5)")
    q.add_argument("--actor", default="researcher")
    q.add_argument("--phase", choices=phases, help="default CRYPTONLP_PHASE")
    q.set_defaults(func=cmd_ledger_intervene)

    q = with_test(
        l_sub.add_parser("register-document", help="register a protocol document before use")
    )
    q.add_argument("path")
    q.add_argument("--key", required=True, help="stable name, e.g. labeling-manual")
    q.add_argument("--kind", required=True, choices=[k.value for k in DocumentKind])
    q.add_argument("--title", required=True)
    q.add_argument("--version", required=True)
    q.add_argument("--supersedes", type=int, help="ledger entry of the version this replaces")
    q.add_argument("--reason", help="why it changed (required with --supersedes)")
    q.add_argument("--phase", choices=phases, help="default CRYPTONLP_PHASE")
    q.set_defaults(func=cmd_ledger_register_document)


def _add_labeling_parser(sub: Any) -> None:
    p = sub.add_parser("labeling", help="the human-labeled set (step 4)")
    l_sub = p.add_subparsers(dest="action", required=True)

    def with_test(q: argparse.ArgumentParser) -> argparse.ArgumentParser:
        q.add_argument("--test", action="store_true", help="the test database instead")
        return q

    q = with_test(l_sub.add_parser("draw", help="draw a first-pass batch and push it"))
    q.add_argument("--day", help="UTC day to draw from, YYYY-MM-DD (default yesterday)")
    q.add_argument("--size", type=int, help="batch size (required for a top-up of a drawn day)")
    q.add_argument("--seed", type=int, help="default: a fresh random seed, recorded")
    q.add_argument("--no-push", action="store_true", help="draw only")
    q.set_defaults(func=cmd_labeling_draw)

    q = with_test(l_sub.add_parser("relabel", help="draw the blind re-labeling batch and push it"))
    q.add_argument("--seed", type=int, help="default: a fresh random seed, recorded")
    q.add_argument("--no-push", action="store_true", help="draw only")
    q.set_defaults(func=cmd_labeling_relabel)

    q = with_test(l_sub.add_parser("push", help="push a drawn batch to Label Studio"))
    q.add_argument("batch", type=int)
    q.set_defaults(func=cmd_labeling_push)

    q = l_sub.add_parser("pull", help="copy annotations from Label Studio (also hourly)")
    q.set_defaults(func=lambda a: run_job("labeling_sync", job_labeling_sync, a))

    q = with_test(l_sub.add_parser("status", help="progress, labeling rate, invalid annotations"))
    q.set_defaults(func=cmd_labeling_status)

    q = with_test(l_sub.add_parser("agreement", help="intra-annotator kappa (re-labeling)"))
    q.set_defaults(func=cmd_labeling_agreement)

    q = with_test(l_sub.add_parser("freeze", help="fix the split and write the human labels"))
    q.add_argument("--relabel", action="store_true", help="write the re-labeling pass instead")
    q.add_argument("--allow-short", action="store_true", help="freeze below the target")
    q.set_defaults(func=cmd_labeling_freeze)

    q = l_sub.add_parser("interface", help="print the Label Studio labeling interface (XML)")
    q.set_defaults(func=lambda a: print(label_config(), end=""))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request otherwise
    parser = argparse.ArgumentParser(prog="cryptonlp")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("db", help="database setup")
    p_sub = p.add_subparsers(dest="action", required=True)
    p_init = p_sub.add_parser("init", help="apply db/init/*.sql and set the Grafana password")
    p_init.add_argument("--test", action="store_true", help="the test database instead")
    p_init.set_defaults(func=cmd_db_init)

    p = sub.add_parser("collect", help="run a collector once (bluesky runs until stopped)")
    p.add_argument(
        "source", choices=["bluesky", "mastodon", "trends", "binance-tail", "binance-backfill"]
    )
    p.add_argument("--record", action="store_true", help="also record messages for replay")
    p.add_argument("--max-archives", type=int, help="binance-backfill: stop after N archives")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("deletions", help="find posts deleted at the source and purge them")
    p.set_defaults(func=lambda a: run_job("deletion_sync", job_deletion_sync, a))

    p = sub.add_parser("rawstore", help="raw-store integrity")
    p.add_argument("action", choices=["verify", "reindex"])
    p.set_defaults(func=lambda a: run_job(f"rawstore_{a.action}", JOBS[f"rawstore_{a.action}"], a))

    p = sub.add_parser("replay", help="the frozen replay day")
    p_sub = p.add_subparsers(dest="action", required=True)
    p_freeze = p_sub.add_parser("freeze", help="copy one recorded UTC day to the test fixtures")
    p_freeze.add_argument("day", help="YYYY-MM-DD")
    p_freeze.add_argument("--to", help="destination (default tests/fixtures/replay/<day>)")
    p_freeze.add_argument("--partial", action="store_true", help="allow an incomplete day")
    p_freeze.set_defaults(func=cmd_freeze)
    p_run = p_sub.add_parser("run", help="feed a recorded day through the collectors (test DB)")
    p_run.add_argument("directory")
    p_run.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "replay-store"))
    p_run.set_defaults(func=cmd_replay)

    _add_ledger_parser(sub)

    p = sub.add_parser("keygen", help="create the pseudonymization key (once, outside the repo)")
    p.add_argument("--path", help="default CRYPTONLP_PSEUDONYM_KEY_FILE")
    p.set_defaults(func=cmd_keygen)

    p = sub.add_parser("normalize", help="language, dedupe, entities, spam, pseudonymization")
    n_sub = p.add_subparsers(dest="action", required=True)
    q = n_sub.add_parser("run", help="normalize every pending raw post (also every 10 minutes)")
    q.add_argument("--max-batches", type=int, help="stop after N batches of 5,000 posts")
    q.set_defaults(func=cmd_normalize_run)
    q = n_sub.add_parser("rebuild", help="empty clean.posts and normalize everything again")
    q.add_argument("--yes", action="store_true")
    q.set_defaults(func=cmd_normalize_rebuild)
    q = n_sub.add_parser("sample", help="CSV of decisions for the hand review")
    q.add_argument("--per-group", type=int, default=30)
    q.add_argument("--seed", default="1")
    q.add_argument("--out", help="default data/review/normalize-sample-<time>.csv")
    q.add_argument("--test", action="store_true", help="the test database instead")
    q.set_defaults(func=cmd_normalize_sample)

    p = sub.add_parser("loop", help="the hourly forecast loop (step 3b)")
    lp_sub = p.add_subparsers(dest="action", required=True)
    q = lp_sub.add_parser("run", help="one cycle: classify, features, forecast, score")
    q.add_argument("--test", action="store_true", help="the test database instead")
    q.add_argument("--yes", action="store_true", help="write to the main ledger")
    q.set_defaults(func=cmd_loop_run)
    q = lp_sub.add_parser(
        "simulate", help="run the loop hour by hour on a simulated clock (test database)"
    )
    q.add_argument("start", help="ISO time with offset, e.g. 2026-11-05T00:00Z")
    q.add_argument("end", help="ISO time with offset")
    q.add_argument("--minute", type=int, default=25, help="minute past the hour (default 25)")
    q.add_argument(
        "--raw-dir",
        default=str(REPO_ROOT / "data" / "replay-store"),
        help="raw store the replay wrote to (default data/replay-store)",
    )
    q.set_defaults(func=cmd_loop_simulate)

    p = sub.add_parser("arm1", help="arm 1, statistics only (step 5)")
    a_sub = p.add_subparsers(dest="action", required=True)
    q = a_sub.add_parser(
        "walkforward", help="walk-forward test of arm 1 and the baseline (development data)"
    )
    q.add_argument("--asset", nargs="+", default=["BTC", "ETH"], choices=["BTC", "ETH"])
    q.add_argument("--end", help="last bar to test, exclusive (ISO time with offset)")
    q.add_argument(
        "--tracking-uri", help="MLflow tracking URI (default CRYPTONLP_MLFLOW_TRACKING_URI)"
    )
    q.add_argument("--no-mlflow", action="store_true", help="only write the tables under data/")
    q.add_argument("--test", action="store_true", help="read candles from the test database")
    q.set_defaults(func=cmd_arm1_walkforward)

    _add_labeling_parser(sub)

    p = sub.add_parser("serve", help="run the scheduled jobs with Prefect")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (
        db.DatabaseGuardError,
        ledger.LedgerError,
        PseudonymKeyError,
        labeling_sample.DrawError,
        labeling_store.FreezeError,
        StudioError,
    ) as exc:
        sys.exit(f"error: {exc}")
