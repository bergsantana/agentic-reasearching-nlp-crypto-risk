"""Prefect flows and schedules: the step 1 jobs and the step 2 ledger checks.

The Bluesky stream is not here: it runs for days, so it is its own Compose service
(``collector-bluesky``), restarted automatically. Everything else runs on a schedule.

Each flow calls the same job as the CLI, so it also writes ``ops.job_runs`` (what Grafana
reads); Prefect adds retries and run history. All schedules are in UTC.
"""

from __future__ import annotations

import argparse
from typing import Any, cast

from prefect import flow
from prefect import serve as prefect_serve
from prefect.deployments.runner import RunnerDeployment
from prefect.schedules import Cron

from cryptonlp import cli


def _run(job: str) -> dict[str, Any]:
    return cli.run_job(job, cli.JOBS[job], argparse.Namespace(record=False))


@flow(name="collect-mastodon", retries=1, retry_delay_seconds=60)
def collect_mastodon() -> dict[str, Any]:
    return _run("collect_mastodon")


@flow(name="collect-trends", retries=2, retry_delay_seconds=120)
def collect_trends() -> dict[str, Any]:
    return _run("collect_trends")


@flow(name="collect-binance-tail", retries=2, retry_delay_seconds=60)
def collect_binance_tail() -> dict[str, Any]:
    return _run("collect_binance_tail")


@flow(name="collect-binance-backfill", retries=1, retry_delay_seconds=600)
def collect_binance_backfill() -> dict[str, Any]:
    return cli.run_job(
        "collect_binance_backfill",
        cli.binance_backfill_job(None),
        argparse.Namespace(record=False),
    )


@flow(name="deletion-sync", retries=1, retry_delay_seconds=600)
def deletion_sync() -> dict[str, Any]:
    return _run("deletion_sync")


@flow(name="rawstore-verify")
def rawstore_verify() -> dict[str, Any]:
    return _run("rawstore_verify")


# No retries: a failed chain check is a finding, not a glitch to retry around.
@flow(name="ledger-check")
def ledger_check() -> dict[str, Any]:
    return _run("ledger_check")


@flow(name="ledger-anchor", retries=1, retry_delay_seconds=600)
def ledger_anchor() -> dict[str, Any]:
    return _run("ledger_anchor")


# (flow, cron in UTC). Development needs less (tech/build-plan.md, "Run mode by phase"), but
# these are cheap, and post data only builds up with time.
SCHEDULES = (
    (collect_mastodon, "*/5 * * * *"),
    (collect_trends, "1 * * * *"),  # hourly snapshot
    (collect_binance_tail, "2 * * * *"),  # just after each hourly bar closes
    (collect_binance_backfill, "30 3 * * *"),  # picks up yesterday's Vision files
    (deletion_sync, "0 4 * * *"),
    (rawstore_verify, "30 5 * * *"),
    (ledger_check, "45 5 * * *"),
    (ledger_anchor, "0 6 * * 1"),  # Mondays: the head hash to send to the advisor
)


def serve() -> None:
    deployments = [
        # to_deployment returns a coroutine only when called from async code.
        cast(
            RunnerDeployment,
            f.to_deployment(
                name="scheduled", schedule=Cron(cron, timezone="UTC"), concurrency_limit=1
            ),
        )
        for f, cron in SCHEDULES
    ]
    prefect_serve(*deployments)
