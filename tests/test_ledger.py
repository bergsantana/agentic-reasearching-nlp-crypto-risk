"""The pre-registration ledger: one hash chain, append-only, deadline on the database clock.

The tamper tests are the step 2 check (tech/build-plan.md): editing an old row breaks the
chain, and the daily check catches it.
"""

from __future__ import annotations

import itertools
import json
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from pydantic import ValidationError

from cryptonlp import cli, ledger
from cryptonlp.db import DatabaseGuardError, apply_init_sql
from cryptonlp.ledger.check import (
    GENESIS,
    Anchor,
    check_chain,
    entry_hash,
    export,
    load_chain,
    make_anchor,
    record_check,
    stored_anchors,
    verify_export,
)
from cryptonlp.ledger.writer import (
    AppendOnlyError,
    EarlyScoreError,
    InvalidEntryError,
    IsolationError,
    LatePredictionError,
    PhaseMismatchError,
    append,
    record_prediction,
    set_simulated_clock,
)
from cryptonlp.schemas import Arm, Asset, Forecast, Phase, Target
from cryptonlp.schemas.ledger import (
    DataFile,
    Document,
    DocumentKind,
    ExperimentPackage,
    Hypothesis,
    HypothesisKind,
    Intervention,
    InterventionType,
    Proposer,
    Score,
    Verdict,
    VerdictKind,
    Violation,
    ViolationKind,
    sha256_text,
)

pytestmark = pytest.mark.db

RUN = "20261103T120000Z-0123456789ab"
BAR = datetime(2026, 11, 3, 14, 0, tzinfo=UTC)
BEFORE = BAR - timedelta(minutes=5)
AFTER_CLOSE = BAR + timedelta(hours=1, minutes=1)
DEV = Phase.DEVELOPMENT
SHA = "ab" * 32


def clear_ledger(conn: psycopg.Connection) -> None:
    # Replica mode skips the triggers that make the ledger append-only: tests only.
    with conn.transaction():
        conn.execute("SET LOCAL session_replication_role = replica")
        conn.execute(
            "TRUNCATE ledger.entries, ops.ledger_checks, ops.ledger_anchors "
            "RESTART IDENTITY CASCADE"
        )


def tamper(conn: psycopg.Connection, statement: str, params: Any = None) -> None:
    """What an operator with full access could do: bypass the triggers and edit rows."""
    with conn.transaction():
        conn.execute("SET LOCAL session_replication_role = replica")
        conn.execute(statement, params)


@pytest.fixture
def lconn(connect: Callable[[], psycopg.Connection]) -> Iterator[psycopg.Connection]:
    """Autocommit test-database connection with an empty ledger, the clock set before BAR."""
    conn = connect()
    clear_ledger(conn)
    set_simulated_clock(conn, BEFORE)
    yield conn
    conn.close()


def forecast(
    bar_open: datetime = BAR,
    *,
    target: Target = Target.RV_1H,
    value: float = 0.012,
    phase: Phase = DEV,
    model: str = "yesterday",
    asset: Asset = Asset.BTC,
) -> Forecast:
    return Forecast(
        asset=asset,
        arm=Arm.BASELINE,
        model=model,
        model_version="1",
        target=target,
        bar_open=bar_open,
        value=value,
        var_level=0.01 if target is Target.VAR_1H else None,
        issued_at=bar_open - timedelta(minutes=10),
        phase=phase,
        run_id=RUN,
    )


def intervention(reason: str = "restarted the Bluesky collector") -> Intervention:
    return Intervention(
        stage=2,
        type=InterventionType.CORRECTION,
        at_checkpoint=False,
        reason=reason,
        phase=DEV,
        run_id=RUN,
    )


def package(key: str = "pkg-1", phase: Phase = DEV) -> ExperimentPackage:
    return ExperimentPackage(
        package_key=key,
        description="baseline forecaster on the replay day",
        code_commit="0123456789ab",
        data=(DataFile(path="binance_klines/date=2026-11-03/a.parquet", sha256=SHA),),
        config={"window_hours": 24},
        phase=phase,
        run_id=RUN,
    )


def one_of_each(conn: psycopg.Connection) -> list[ledger.Entry]:
    entries = [
        append(
            conn,
            Hypothesis(
                hypothesis_key="H4",
                kind=HypothesisKind.CONFIRMATORY,
                statement="Após calibração, o SemIf atinge F1 macro em sentimento igual ou "
                "superior ao do CryptoBERT no conjunto de teste.",
                analysis_plan={"metric": "macro_f1", "test": "non-inferiority"},
                proposed_by=Proposer.RESEARCHER,
                phase=DEV,
                run_id=RUN,
            ),
        ),
        append(
            conn,
            Document(
                document_key="labeling-manual",
                kind=DocumentKind.LABELING_MANUAL,
                title="Manual de rotulagem",
                version="1.0",
                content="# Manual\n\nSentimento: negativo, neutro, positivo.",
                content_sha256=sha256_text("# Manual\n\nSentimento: negativo, neutro, positivo."),
                phase=DEV,
                run_id=RUN,
            ),
        ),
    ]
    pkg = append(conn, package())
    entries.append(pkg)
    pred = record_prediction(conn, forecast(), package_seq=pkg.seq)
    entries.append(pred)
    set_simulated_clock(conn, AFTER_CLOSE)
    entries += [
        append(
            conn,
            Score(
                prediction_seq=pred.seq,
                realized=0.015,
                losses={"qlike": 0.03},
                phase=DEV,
                run_id=RUN,
            ),
        ),
        append(
            conn,
            Verdict(
                kind=VerdictKind.REPLICATION,
                agent="validator",
                package_seq=pkg.seq,
                verdict="reproduced",
                phase=DEV,
                run_id=RUN,
            ),
        ),
        append(conn, intervention("café ✓, ação")),
        append(
            conn,
            Violation(
                kind=ViolationKind.PERMISSION_DENIED,
                reported_by="analyst",
                detail={"tool": "ledger_writer"},
                phase=DEV,
                run_id=RUN,
            ),
        ),
    ]
    return entries


# --- the chain -----------------------------------------------------------------------------


def test_every_table_joins_one_chain(lconn: psycopg.Connection) -> None:
    entries = one_of_each(lconn)
    assert [e.seq for e in entries] == list(range(1, 9))
    assert {e.table for e in entries} == {
        "hypotheses",
        "documents",
        "experiment_packages",
        "predictions",
        "scores",
        "verdicts",
        "interventions",
        "violations",
    }
    chain = load_chain(lconn)
    assert chain[0].prev_hash == GENESIS
    for before, after in itertools.pairwise(chain):
        assert after.prev_hash == before.hash
    for e in chain:
        assert e.row_matches is True
        assert entry_hash(e.prev_hash, e.canonical) == e.hash  # Python agrees with Postgres
        assert json.loads(e.canonical)["seq"] == e.seq
    report = check_chain(lconn)
    assert report.ok, report.problems
    assert (report.entries, report.head_seq) == (8, 8)


def test_database_sets_time_role_and_hash_not_the_client(lconn: psycopg.Connection) -> None:
    lconn.execute(
        "INSERT INTO ledger.interventions (stage, type, at_checkpoint, reason, phase, run_id, "
        "seq, recorded_at, recorded_by, prev_hash, hash, canonical) "
        "VALUES (2, 'approval', true, 'stage 2 approved', 'development', %s, "
        "999, '2020-01-01', 'someone else', %s, %s, '{}')",
        (RUN, SHA, SHA),
    )
    row = lconn.execute(
        "SELECT seq, recorded_at, recorded_by, prev_hash, current_user FROM ledger.interventions"
    ).fetchone()
    assert row is not None
    assert row[0] == 1
    assert row[1] == BEFORE
    assert row[2] == row[4]
    assert row[3] == GENESIS
    assert check_chain(lconn).ok


def test_simulated_clock_drives_the_ledger_in_the_test_database(lconn: psycopg.Connection) -> None:
    entry = append(lconn, intervention())
    assert entry.recorded_at == BEFORE
    set_simulated_clock(lconn, None)
    real = append(lconn, intervention("second"))
    assert real.recorded_at > datetime(2026, 1, 1, tzinfo=UTC)


def test_simulated_clock_is_refused_outside_the_test_database(
    connect: Callable[[], psycopg.Connection],
) -> None:
    """In any database not named *_test, setting a simulated time is an error."""
    admin = connect()
    name = "cryptonlp_clockguard"
    admin.execute(f"DROP DATABASE IF EXISTS {name}")
    admin.execute(f"CREATE DATABASE {name}")
    try:
        info = admin.info
        other = psycopg.connect(
            host=info.host, port=info.port, user=info.user, password=info.password, dbname=name
        )
        with other:
            apply_init_sql(other)
            other.execute("SET cryptonlp.sim_now = '2026-11-03T13:55:00Z'")
            with pytest.raises(psycopg.Error) as exc:
                other.execute("SELECT ledger.clock()")
            assert exc.value.sqlstate == "LG007"
            other.rollback()
            with pytest.raises(DatabaseGuardError, match="test database"):
                set_simulated_clock(other, BEFORE)
    finally:
        admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        admin.close()


# --- append-only ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE ledger.interventions SET reason = 'nicer reason'",
        "DELETE FROM ledger.interventions",
        "TRUNCATE ledger.interventions",
        "TRUNCATE ledger.entries CASCADE",
        "UPDATE ledger.entries SET phase = 'confirmatory'",
        "DELETE FROM ledger.entries",
    ],
)
def test_nothing_can_be_changed_or_removed(lconn: psycopg.Connection, statement: str) -> None:
    append(lconn, intervention())
    with pytest.raises(psycopg.Error) as exc:
        lconn.execute(statement)
    assert exc.value.sqlstate == AppendOnlyError.sqlstate
    assert check_chain(lconn).entries == 1


def test_ledger_writer_role_may_only_read_and_append(lconn: psycopg.Connection) -> None:
    with lconn.transaction(force_rollback=True):
        lconn.execute("SET LOCAL ROLE ledger_writer")
        entry = append(lconn, intervention())
        assert entry.seq == 1
        row = lconn.execute("SELECT recorded_by FROM ledger.interventions").fetchone()
        assert row == ("ledger_writer",)
        for statement in (
            "UPDATE ledger.interventions SET reason = 'x'",
            "DELETE FROM ledger.interventions",
            "TRUNCATE ledger.interventions",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege), lconn.transaction():
                lconn.execute(statement)


def test_rows_go_into_a_ledger_table_not_the_parent(lconn: psycopg.Connection) -> None:
    with pytest.raises(psycopg.Error) as exc:
        lconn.execute(
            "INSERT INTO ledger.entries (seq, recorded_at, recorded_by, phase, run_id, "
            "prev_hash, hash, canonical) VALUES (1, now(), 'x', 'development', %s, %s, %s, '{}')",
            (RUN, SHA, SHA),
        )
    assert exc.value.sqlstate == InvalidEntryError.sqlstate


def test_writes_must_use_read_committed(lconn: psycopg.Connection) -> None:
    lconn.autocommit = False
    lconn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    with pytest.raises(IsolationError):
        append(lconn, intervention())
    lconn.rollback()


def test_concurrent_writers_keep_a_single_chain(connect: Callable[[], psycopg.Connection]) -> None:
    with connect() as conn:
        clear_ledger(conn)

    def writer(n: int) -> None:
        with connect() as conn:
            for i in range(15):
                append(conn, intervention(f"writer {n}, intervention {i}"))

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with connect() as conn:
        report = check_chain(conn)
    assert report.ok, report.problems
    assert (report.entries, report.head_seq) == (45, 45)


# --- the deadline --------------------------------------------------------------------------


@pytest.mark.parametrize("clock", [BAR, BAR + timedelta(seconds=1), BAR + timedelta(hours=3)])
def test_prediction_at_or_after_the_bar_open_is_rejected_and_logged(
    lconn: psycopg.Connection, clock: datetime
) -> None:
    # The client says it issued the forecast in time; only the database clock counts.
    late = forecast()
    assert late.issued_at < late.bar_open
    set_simulated_clock(lconn, clock)
    with pytest.raises(LatePredictionError, match="late prediction"):
        record_prediction(lconn, late)
    assert lconn.execute("SELECT count(*) FROM ledger.predictions").fetchone() == (0,)
    rows = lconn.execute("SELECT kind, detail FROM ledger.violations").fetchall()
    assert len(rows) == 1
    kind, detail = rows[0]
    assert kind == "late_prediction"
    assert detail["forecast"]["bar_open"] == "2026-11-03T14:00:00Z"
    assert check_chain(lconn).ok


def test_deadline_is_checked_again_at_commit(lconn: psycopg.Connection) -> None:
    """Inserting in time and committing once the outcome is known does not get through."""
    lconn.autocommit = False
    lconn.execute("SELECT 1")  # open the transaction the forecast is inserted in
    append(lconn, forecast())  # in time
    set_simulated_clock(lconn, BAR + timedelta(minutes=2))
    with pytest.raises(psycopg.Error) as exc:
        lconn.commit()
    assert exc.value.sqlstate == LatePredictionError.sqlstate
    lconn.autocommit = True
    assert lconn.execute("SELECT count(*) FROM ledger.predictions").fetchone() == (0,)


def test_one_prediction_per_model_target_and_bar(lconn: psycopg.Connection) -> None:
    record_prediction(lconn, forecast())
    record_prediction(lconn, forecast(target=Target.VAR_1H, value=-0.02))
    record_prediction(lconn, forecast(asset=Asset.ETH))
    with pytest.raises(psycopg.errors.UniqueViolation):
        record_prediction(lconn, forecast(value=0.5))
    # The same forecast in another phase is a different row.
    record_prediction(lconn, forecast(phase=Phase.PILOT))


@pytest.mark.parametrize("target", list(Target))
def test_database_accepts_every_target(lconn: psycopg.Connection, target: Target) -> None:
    value = -0.02 if target is Target.VAR_1H else 0.2
    entry = record_prediction(lconn, forecast(target=target, value=value))
    assert entry.table == "predictions"


# --- scores, phases, versions ------------------------------------------------------------


def test_score_waits_for_the_bar_to_close(lconn: psycopg.Connection) -> None:
    hourly = record_prediction(lconn, forecast())
    daily = record_prediction(lconn, forecast(target=Target.RV_1D))

    def score(seq: int) -> ledger.Entry:
        return append(lconn, Score(prediction_seq=seq, realized=0.01, phase=DEV, run_id=RUN))

    set_simulated_clock(lconn, BAR + timedelta(minutes=59))
    with pytest.raises(EarlyScoreError):
        score(hourly.seq)
    set_simulated_clock(lconn, AFTER_CLOSE)
    score(hourly.seq)
    with pytest.raises(EarlyScoreError):
        score(daily.seq)
    set_simulated_clock(lconn, BAR + timedelta(days=1))
    score(daily.seq)
    with pytest.raises(psycopg.errors.UniqueViolation):
        score(daily.seq)


def test_phases_never_mix(lconn: psycopg.Connection) -> None:
    dev_pkg = append(lconn, package())
    with pytest.raises(PhaseMismatchError):
        record_prediction(lconn, forecast(phase=Phase.PILOT), package_seq=dev_pkg.seq)
    pred = record_prediction(lconn, forecast(), package_seq=dev_pkg.seq)
    set_simulated_clock(lconn, AFTER_CLOSE)
    with pytest.raises(PhaseMismatchError):
        append(
            lconn,
            Score(prediction_seq=pred.seq, realized=0.01, phase=Phase.CONFIRMATORY, run_id=RUN),
        )
    with pytest.raises(PhaseMismatchError):
        append(
            lconn,
            Verdict(
                kind=VerdictKind.REPLICATION,
                agent="validator",
                package_seq=dev_pkg.seq,
                verdict="divergent",
                cause="different seed",
                phase=Phase.PILOT,
                run_id=RUN,
            ),
        )


def test_hypothesis_changes_are_new_versions_with_a_reason(lconn: psycopg.Connection) -> None:
    def h1(**kw: Any) -> Hypothesis:
        fields: dict[str, Any] = {
            "hypothesis_key": "H1",
            "kind": HypothesisKind.CONFIRMATORY,
            "statement": "A abordagem combinada reduz a perda QLIKE.",
            "proposed_by": Proposer.RESEARCHER,
            "phase": DEV,
            "run_id": RUN,
        }
        return Hypothesis(**(fields | kw))

    first = append(lconn, h1())
    with pytest.raises(psycopg.errors.UniqueViolation):
        append(lconn, h1(statement="a second first version"))
    revised = append(lconn, h1(supersedes=first.seq, deviation_reason="typo in the target"))
    with pytest.raises(psycopg.errors.UniqueViolation):  # no forks: one successor per version
        append(lconn, h1(supersedes=first.seq, deviation_reason="another branch"))
    with pytest.raises(InvalidEntryError):
        append(lconn, h1(hypothesis_key="H2", supersedes=revised.seq, deviation_reason="x"))
    with pytest.raises(ValidationError):
        h1(supersedes=revised.seq)  # a revision without its reason


def test_agents_cannot_register_confirmatory_hypotheses(lconn: psycopg.Connection) -> None:
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_key="H9",
            kind=HypothesisKind.CONFIRMATORY,
            statement="x",
            proposed_by=Proposer.ANALYST,
            phase=DEV,
            run_id=RUN,
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        lconn.execute(
            "INSERT INTO ledger.hypotheses (hypothesis_key, kind, statement, proposed_by, "
            "phase, run_id) VALUES ('H9', 'confirmatory', 'x', 'analyst', 'development', %s)",
            (RUN,),
        )
    append(
        lconn,
        Hypothesis(
            hypothesis_key="X-2026-11-01",
            kind=HypothesisKind.EXPLORATORY,
            statement="Mentions of exchange outages precede volatility spikes.",
            proposed_by=Proposer.ANALYST,
            phase=DEV,
            run_id=RUN,
        ),
    )


def test_document_content_must_match_its_hash(lconn: psycopg.Connection) -> None:
    with pytest.raises(ValidationError):
        Document(
            document_key="rubric",
            kind=DocumentKind.RUBRIC,
            title="t",
            version="1",
            content="text",
            content_sha256=SHA,
            phase=DEV,
            run_id=RUN,
        )
    with pytest.raises(psycopg.Error) as exc:
        lconn.execute(
            "INSERT INTO ledger.documents (document_key, kind, title, version, content_sha256, "
            "content, phase, run_id) VALUES ('rubric', 'rubric', 't', '1', %s, 'text', "
            "'development', %s)",
            (SHA, RUN),
        )
    assert exc.value.sqlstate == InvalidEntryError.sqlstate


def test_register_a_file(lconn: psycopg.Connection, tmp_path: Path) -> None:
    path = tmp_path / "manual.md"
    path.write_text("Rótulos: negativo, neutro, positivo.\n", encoding="utf-8")
    doc = Document.from_file(
        path,
        document_key="labeling-manual",
        kind=DocumentKind.LABELING_MANUAL,
        title="Manual",
        version="1.0",
        phase=DEV,
        run_id=RUN,
    )
    append(lconn, doc)
    row = lconn.execute("SELECT content_sha256, content FROM ledger.documents").fetchone()
    assert row == (sha256_text(path.read_text(encoding="utf-8")), path.read_text())


# --- tampering (the step 2 check) --------------------------------------------------------


def three_interventions(conn: psycopg.Connection) -> None:
    for reason in ("first", "second", "third"):
        append(conn, intervention(reason))


def test_edited_column_breaks_the_chain_and_the_check_records_it(
    lconn: psycopg.Connection,
) -> None:
    three_interventions(lconn)
    tamper(lconn, "UPDATE ledger.interventions SET reason = 'rewritten' WHERE seq = 2")
    report = check_chain(lconn)
    assert not report.ok
    assert report.first_problem is not None
    assert report.first_problem.seq == 2
    assert "edited column" in report.first_problem.issue

    record_check(lconn, report, RUN, DEV)
    record_check(lconn, check_chain(lconn), RUN, DEV)  # still broken the next day
    checks = lconn.execute("SELECT ok FROM ops.ledger_checks ORDER BY check_id").fetchall()
    assert checks == [(False,), (False,)]
    breaks = lconn.execute(
        "SELECT detail FROM ledger.violations WHERE kind = 'chain_break'"
    ).fetchall()
    assert len(breaks) == 1  # counted once, not once a day
    assert breaks[0][0]["problems"][0]["seq"] == 2


def test_edited_row_with_its_own_hash_recomputed_breaks_the_next_link(
    lconn: psycopg.Connection,
) -> None:
    three_interventions(lconn)
    tamper(
        lconn,
        """
        UPDATE ledger.interventions t
        SET reason = 'rewritten',
            canonical = ledger.canonical('interventions',
                                         to_jsonb(t) || '{"reason": "rewritten"}')::text
        WHERE seq = 2
        """,
    )
    tamper(
        lconn,
        "UPDATE ledger.interventions SET hash = ledger.entry_hash(prev_hash, canonical) "
        "WHERE seq = 2",
    )
    problems = check_chain(lconn).problems
    assert [(p.seq, p.issue) for p in problems] == [
        (3, "prev_hash is not the hash of the entry before it")
    ]


def test_deleted_row_is_caught(lconn: psycopg.Connection) -> None:
    three_interventions(lconn)
    tamper(lconn, "DELETE FROM ledger.interventions WHERE seq = 2")
    issues = {p.issue for p in check_chain(lconn).problems}
    assert "entries 2 to 2 are missing" in issues
    assert "prev_hash is not the hash of the entry before it" in issues


def rechain_from(conn: psycopg.Connection, seq: int) -> None:
    """Recompute every canonical text and hash from ``seq`` on, as a careful forger would."""
    rows = conn.execute(
        "SELECT seq FROM ledger.interventions WHERE seq >= %s ORDER BY seq", (seq,)
    ).fetchall()
    for (s,) in rows:
        tamper(
            conn,
            """
            UPDATE ledger.interventions t SET
                prev_hash = coalesce((SELECT hash FROM ledger.entries WHERE seq = t.seq - 1),
                                     repeat('0', 64)),
                canonical = ledger.canonical('interventions', to_jsonb(t))::text
            WHERE seq = %s
            """,
            (s,),
        )
        tamper(
            conn,
            "UPDATE ledger.interventions SET hash = ledger.entry_hash(prev_hash, canonical) "
            "WHERE seq = %s",
            (s,),
        )


def test_consistent_rewrite_of_the_tail_is_caught_only_by_the_anchor(
    lconn: psycopg.Connection, tmp_path: Path
) -> None:
    three_interventions(lconn)
    anchor, path = make_anchor(lconn, tmp_path, RUN)
    assert (anchor.head_seq, anchor.entries) == (3, 3)
    assert Anchor.load(path) == anchor
    assert stored_anchors(lconn)[0].head_hash == anchor.head_hash
    assert check_chain(lconn, [anchor]).ok

    tamper(lconn, "UPDATE ledger.interventions SET reason = 'rewritten' WHERE seq = 2")
    rechain_from(lconn, 2)
    assert check_chain(lconn).ok  # internally consistent again
    problems = check_chain(lconn, [anchor]).problems
    assert len(problems) == 1
    assert "history rewritten" in problems[0].issue


def test_cutting_the_tail_is_caught_by_the_anchor(
    lconn: psycopg.Connection, tmp_path: Path
) -> None:
    three_interventions(lconn)
    anchor, _ = make_anchor(lconn, tmp_path, RUN)
    tamper(lconn, "DELETE FROM ledger.interventions WHERE seq = 3")
    assert check_chain(lconn).ok
    problems = check_chain(lconn, [anchor]).problems
    assert [p.issue for p in problems] == [
        f"entry anchored on {anchor.created_at} is missing (chain cut back)"
    ]


# --- export --------------------------------------------------------------------------------


def test_export_verifies_with_sha256_alone(lconn: psycopg.Connection, tmp_path: Path) -> None:
    one_of_each(lconn)
    path = tmp_path / "ledger.jsonl"
    assert export(lconn, path).ok
    assert verify_export(path).ok

    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[6])
    assert entry["table"] == "interventions"
    entry["canonical"] = entry["canonical"].replace("café", "cafe")
    lines[6] = json.dumps(entry, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = verify_export(path).problems
    assert [(p.seq, p.issue) for p in problems] == [
        (7, "hash is not SHA-256(prev_hash || canonical)")
    ]


# --- CLI -----------------------------------------------------------------------------------


def test_cli_logs_an_intervention_in_the_test_database(
    lconn: psycopg.Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    set_simulated_clock(lconn, None)
    cli.main(
        [
            "ledger",
            "intervene",
            "--test",
            "--stage",
            "2",
            "--type",
            "approval",
            "--checkpoint",
            "--reason",
            "continuous run with deletions synced",
        ]
    )
    assert "ledger entry 1 in interventions" in capsys.readouterr().out
    row = lconn.execute("SELECT stage, type, at_checkpoint, phase FROM ledger.interventions")
    assert row.fetchone() == (2, "approval", True, "development")
    cli.main(["ledger", "head", "--test"])
    assert "ledger entry 1" in capsys.readouterr().out


def test_cli_verify_export_fails_on_a_broken_file(
    lconn: psycopg.Connection, tmp_path: Path
) -> None:
    three_interventions(lconn)
    path = tmp_path / "ledger.jsonl"
    export(lconn, path)
    cli.main(["ledger", "verify-export", str(path)])
    path.write_text(path.read_text().replace("second", "secund"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["ledger", "verify-export", str(path)])
    assert exc.value.code == 1


def test_finding_verdicts_name_their_finding() -> None:
    with pytest.raises(ValidationError):
        Verdict(
            kind=VerdictKind.VERIFICATION,
            agent="verifier",
            verdict="reproduced",
            finding_id=uuid4(),
            phase=DEV,
            run_id=RUN,
        )
    Verdict(
        kind=VerdictKind.VERIFICATION,
        agent="verifier",
        verdict="needs_review",
        finding_id=uuid4(),
        confidence=0.4,
        phase=DEV,
        run_id=RUN,
    )
