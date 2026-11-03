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
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import psycopg

from cryptonlp import db, ledger, ops
from cryptonlp.collectors import binance, bluesky, deletions, mastodon, replay, trends
from cryptonlp.collectors.common import Recorder, WatchFilter, http_client
from cryptonlp.config import CollectorsConfig, load_config
from cryptonlp.ledger import check as ledger_check
from cryptonlp.rawstore import RawStore
from cryptonlp.run_id import new_run_id
from cryptonlp.schemas.common import Phase
from cryptonlp.schemas.ledger import Document, DocumentKind, Intervention, InterventionType
from cryptonlp.settings import REPO_ROOT, Settings

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


JOBS: dict[str, Callable[[psycopg.Connection, Context], dict[str, Any]]] = {
    "collect_bluesky": job_bluesky,
    "collect_mastodon": job_mastodon,
    "collect_trends": job_trends,
    "collect_binance_tail": job_binance_tail,
    "deletion_sync": job_deletion_sync,
    "rawstore_verify": job_rawstore_verify,
    "rawstore_reindex": job_rawstore_reindex,
    "ledger_check": ledger_check_job(),
    "ledger_anchor": job_ledger_anchor,
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

    p = sub.add_parser("serve", help="run the scheduled jobs with Prefect")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (db.DatabaseGuardError, ledger.LedgerError) as exc:
        sys.exit(f"error: {exc}")
