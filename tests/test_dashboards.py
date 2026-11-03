"""Every query on the dashboard and in the alert rules runs as Grafana's read-only role."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml

DASHBOARDS = Path(__file__).parents[1] / "dashboards"
DATASOURCE_UID = "cryptonlp-pg"


def _dashboard_queries() -> Iterator[tuple[str, str]]:
    for path in sorted((DASHBOARDS / "json").glob("*.json")):
        dash = json.loads(path.read_text())
        for panel in dash["panels"]:
            for t in panel.get("targets", []):
                yield f"{path.stem}: {panel['title']}", t["rawSql"]


def _alert_rules() -> list[dict[str, Any]]:
    doc = yaml.safe_load((DASHBOARDS / "provisioning" / "alerting" / "alerts.yaml").read_text())
    return [rule for group in doc["groups"] for rule in group["rules"]]


def _alert_queries() -> Iterator[tuple[str, str]]:
    for rule in _alert_rules():
        for q in rule["data"]:
            if q["datasourceUid"] == DATASOURCE_UID:
                yield f"alert: {rule['title']}", q["model"]["rawSql"]


QUERIES = [*_dashboard_queries(), *_alert_queries()]


def _expand_macros(sql: str) -> str:
    return re.sub(r"\$__timeFilter\((\w+)\)", r"\1 > now() - interval '2 days'", sql)


def test_dashboard_is_well_formed() -> None:
    dash = json.loads((DASHBOARDS / "json" / "system-health.json").read_text())
    ids = [p["id"] for p in dash["panels"]]
    assert len(ids) == len(set(ids))
    for panel in dash["panels"]:
        if panel["type"] != "row":
            assert panel["datasource"]["uid"] == DATASOURCE_UID, panel["title"]
    provisioned = yaml.safe_load(
        (DASHBOARDS / "provisioning" / "datasources" / "postgres.yaml").read_text()
    )
    assert provisioned["datasources"][0]["uid"] == DATASOURCE_UID
    assert provisioned["datasources"][0]["user"] == "grafana_reader"


def test_required_alerts_exist() -> None:
    rules = {r["uid"]: r for r in _alert_rules()}
    assert {"bluesky-lag", "job-failed", "source-silent", "ledger-check"} <= set(rules)
    lag = rules["bluesky-lag"]
    threshold = next(q for q in lag["data"] if q["refId"] == lag["condition"])
    assert threshold["model"]["conditions"][0]["evaluator"] == {"type": "gt", "params": [1]}
    assert lag["noDataState"] == "Alerting"  # a collector that never reported is an alert


@pytest.mark.db
@pytest.mark.parametrize(("name", "sql"), QUERIES, ids=[q[0] for q in QUERIES])
def test_query_runs_as_grafana_reader(db: psycopg.Connection, name: str, sql: str) -> None:
    with db.transaction(force_rollback=True):
        db.execute("SET LOCAL ROLE grafana_reader")
        db.execute(_expand_macros(sql)).fetchall()


@pytest.mark.db
def test_grafana_reader_cannot_see_raw_data_or_write(db: psycopg.Connection) -> None:
    for sql in (
        "SELECT * FROM raw.posts",
        "SELECT * FROM ops.job_runs",
        "SELECT * FROM ledger.interventions",
        "SELECT * FROM ledger.entries",
        "DELETE FROM ops.collector_state",
    ):
        with (
            pytest.raises(psycopg.errors.InsufficientPrivilege),
            db.transaction(force_rollback=True),
        ):
            db.execute("SET LOCAL ROLE grafana_reader")
            db.execute(sql)
