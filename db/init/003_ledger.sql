-- Step 2: the pre-registration ledger (paper, Section 4.9; Apêndice C).
-- Idempotent: Compose runs it on first start, and `cryptonlp db init` re-applies it.
--
-- Every ledger table inherits ledger.entries, and all of them form ONE hash chain:
--   seq        1, 2, 3, ... across all tables, assigned under a lock, with no gaps
--   canonical  jsonb text of the row (every column but prev_hash, hash and canonical, plus
--              "table"), written once at insert in UTC
--   hash       SHA-256 of prev_hash || canonical (UTF-8), in hex; the first prev_hash is 64 zeros
-- Anyone holding an export can recompute the chain with SHA-256 alone (cryptonlp ledger
-- verify-export). The daily check also confirms each row still matches its canonical text, so
-- editing a column, the canonical text or a hash breaks the chain at that row. Rewriting the
-- whole tail consistently is caught by the weekly anchor sent to the advisor.
--
-- Nothing in the ledger can be updated, deleted or truncated (triggers, and ledger_writer has
-- no such privileges). Times come from the database clock, never from the client: a prediction
-- for bar t is rejected if it arrives, or commits, at or after bar t opens.
--
-- The operator (database owner) can still disable triggers, so the guarantee is detection, not
-- impossibility (paper, Section 4.9).

CREATE SCHEMA IF NOT EXISTS ledger;

-- Error codes raised here (psycopg exposes them as exc.sqlstate):
--   LG001 late prediction    LG002 append-only        LG003 phase mismatch
--   LG004 score too early    LG005 wrong isolation     LG006 invalid reference or content
--   LG007 simulated clock outside the test database

-- --- clock ---------------------------------------------------------------------------------

-- The database clock. In a database whose name ends in _test (the only one tests, CI and the
-- simulated clock may use), a session may set cryptonlp.sim_now to replay a day in minutes.
-- Anywhere else, setting it is an error, so the real ledger can never run on a fake clock.
CREATE OR REPLACE FUNCTION ledger.clock() RETURNS timestamptz
LANGUAGE plpgsql VOLATILE AS $$
DECLARE
    sim text := current_setting('cryptonlp.sim_now', true);
BEGIN
    IF sim IS NULL OR sim = '' THEN
        RETURN clock_timestamp();
    END IF;
    IF current_database() NOT LIKE '%\_test' THEN
        RAISE EXCEPTION USING ERRCODE = 'LG007',
            MESSAGE = 'cryptonlp.sim_now is set, but the simulated clock is only allowed in the test database';
    END IF;
    RETURN sim::timestamptz;
END
$$;

-- --- the chain -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ledger.entries (
    seq          bigint      NOT NULL CHECK (seq > 0),
    recorded_at  timestamptz NOT NULL,  -- ledger.clock() at insert
    recorded_by  text        NOT NULL,  -- database role that wrote the row
    phase        text        NOT NULL CHECK (phase IN ('development', 'pilot', 'confirmatory')),
    run_id       text        NOT NULL
        CHECK (run_id ~ '^\d{8}T\d{6}Z-(?:[0-9a-f]{12}(?:-dirty)?|nogit)$'),
    prev_hash    text        NOT NULL CHECK (prev_hash ~ '^[0-9a-f]{64}$'),
    hash         text        NOT NULL CHECK (hash ~ '^[0-9a-f]{64}$'),
    canonical    text        NOT NULL
);

CREATE OR REPLACE FUNCTION ledger.canonical(tbl text, r jsonb) RETURNS jsonb
LANGUAGE sql IMMUTABLE AS $$
    SELECT jsonb_build_object('table', tbl) || (r - 'prev_hash' - 'hash' - 'canonical')
$$;

CREATE OR REPLACE FUNCTION ledger.entry_hash(prev_hash text, canonical text) RETURNS text
LANGUAGE sql IMMUTABLE AS $$
    SELECT encode(sha256(convert_to(prev_hash || canonical, 'UTF8')), 'hex')
$$;

-- BEFORE INSERT on every ledger table, after its own checks (triggers fire in name order).
CREATE OR REPLACE FUNCTION ledger.chain_row() RETURNS trigger
LANGUAGE plpgsql
SET timezone = 'UTC'
SET extra_float_digits = 1
AS $$
DECLARE
    head record;
BEGIN
    -- Under REPEATABLE READ the head lookup below would read a stale snapshot.
    IF current_setting('transaction_isolation') <> 'read committed' THEN
        RAISE EXCEPTION USING ERRCODE = 'LG005',
            MESSAGE = 'ledger writes must run at READ COMMITTED isolation';
    END IF;
    -- One writer at a time until commit, so the chain stays a single line in commit order.
    PERFORM pg_advisory_xact_lock(7352460150482437101);
    SELECT e.seq, e.hash INTO head FROM ledger.entries e ORDER BY e.seq DESC LIMIT 1;
    NEW.seq := coalesce(head.seq, 0) + 1;
    NEW.prev_hash := coalesce(head.hash, repeat('0', 64));
    NEW.recorded_at := ledger.clock();
    NEW.recorded_by := current_user;
    NEW.canonical := ledger.canonical(TG_TABLE_NAME, to_jsonb(NEW))::text;
    NEW.hash := ledger.entry_hash(NEW.prev_hash, NEW.canonical);
    RETURN NEW;
END
$$;

CREATE OR REPLACE FUNCTION ledger.reject_change() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION USING ERRCODE = 'LG002',
        MESSAGE = format('ledger.%s is append-only: %s is not allowed', TG_TABLE_NAME, TG_OP);
END
$$;

CREATE OR REPLACE FUNCTION ledger.reject_parent_insert() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION USING ERRCODE = 'LG006',
        MESSAGE = 'insert into one of the ledger tables, not into ledger.entries';
END
$$;

-- --- ledger tables -------------------------------------------------------------------------

-- Hypotheses (stage 1 confirmatory, by the researcher; stage 4 exploratory, by agents too).
-- A change after registration is a new row that supersedes the old one, with the reason: it
-- feeds the table of deviations (paper, Section 5.6).
CREATE TABLE IF NOT EXISTS ledger.hypotheses (
    hypothesis_key    text  NOT NULL CHECK (hypothesis_key ~ '^[A-Za-z0-9][A-Za-z0-9_.-]*$'),
    kind              text  NOT NULL CHECK (kind IN ('confirmatory', 'exploratory')),
    statement         text  NOT NULL CHECK (statement <> ''),  -- verbatim wording
    analysis_plan     jsonb NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(analysis_plan) = 'object'),
    proposed_by       text  NOT NULL CHECK (proposed_by IN ('researcher', 'orchestrator', 'analyst')),
    supersedes        bigint UNIQUE,
    deviation_reason  text,
    UNIQUE (seq),
    FOREIGN KEY (supersedes) REFERENCES ledger.hypotheses (seq),
    CHECK ((supersedes IS NULL) = (deviation_reason IS NULL)),
    CHECK (deviation_reason IS NULL OR deviation_reason <> ''),
    -- Confirmatory hypotheses are the researcher's alone (Quadro 5, stage 1).
    CHECK (kind = 'exploratory' OR proposed_by = 'researcher')
) INHERITS (ledger.entries);
CREATE UNIQUE INDEX IF NOT EXISTS hypotheses_first_version
    ON ledger.hypotheses (hypothesis_key) WHERE supersedes IS NULL;

-- Protocol documents registered before they are used: the labeling manual, the rubric, the
-- analysis plan, the power simulation, the Section 5 table shells, the planned interpretation.
CREATE TABLE IF NOT EXISTS ledger.documents (
    document_key    text NOT NULL CHECK (document_key ~ '^[A-Za-z0-9][A-Za-z0-9_.-]*$'),
    kind            text NOT NULL CHECK (kind IN (
                        'labeling_manual', 'rubric', 'analysis_plan', 'power_simulation',
                        'result_tables', 'interpretation_plan', 'other')),
    title           text NOT NULL CHECK (title <> ''),
    version         text NOT NULL CHECK (version <> ''),
    path            text,          -- where it lives in the repository, if it does
    content_sha256  text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    content         text,          -- the full text, so the ledger stands on its own
    supersedes      bigint UNIQUE,
    change_reason   text,
    UNIQUE (seq),
    FOREIGN KEY (supersedes) REFERENCES ledger.documents (seq),
    CHECK ((supersedes IS NULL) = (change_reason IS NULL)),
    CHECK (change_reason IS NULL OR change_reason <> '')
) INHERITS (ledger.entries);
CREATE UNIQUE INDEX IF NOT EXISTS documents_first_version
    ON ledger.documents (document_key) WHERE supersedes IS NULL;

-- Everything needed to re-run an experiment: data hashes, code version, model revision and
-- SemIf prompt hash. The validator agent receives only this (paper, Section 4.9).
CREATE TABLE IF NOT EXISTS ledger.experiment_packages (
    package_key     text   NOT NULL UNIQUE CHECK (package_key ~ '^[A-Za-z0-9][A-Za-z0-9_.-]*$'),
    hypothesis_seq  bigint REFERENCES ledger.hypotheses (seq),
    description     text   NOT NULL CHECK (description <> ''),
    code_commit     text   NOT NULL CHECK (code_commit ~ '^[0-9a-f]{12,40}$'),
    data            jsonb  NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(data) = 'array'),
    model_revision  text,
    prompt_sha256   text   CHECK (prompt_sha256 ~ '^[0-9a-f]{64}$'),
    config          jsonb  NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(config) = 'object'),
    UNIQUE (seq)
) INHERITS (ledger.entries);

-- One forecast per model, target and bar, in before the bar opens.
CREATE TABLE IF NOT EXISTS ledger.predictions (
    asset          text   NOT NULL CHECK (asset IN ('BTC', 'ETH')),
    arm            text   NOT NULL CHECK (arm IN ('baseline', 'arm1_stats', 'arm2_nlp', 'arm3_combined')),
    model          text   NOT NULL CHECK (model <> ''),
    model_version  text   NOT NULL CHECK (model_version <> ''),
    target         text   NOT NULL CHECK (target IN ('rv_1h', 'p_large_move_1h', 'var_1h', 'rv_1d')),
    bar_open       timestamptz NOT NULL CHECK (extract(epoch FROM bar_open) % 3600 = 0),
    value          double precision NOT NULL
        CHECK (value NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)),
    var_level      double precision CHECK (var_level > 0 AND var_level < 1),
    issued_at      timestamptz NOT NULL,  -- the client's clock: kept, never trusted
    package_seq    bigint REFERENCES ledger.experiment_packages (seq),
    UNIQUE (seq),
    CHECK ((target = 'var_1h') = (var_level IS NOT NULL)),
    CHECK (target <> 'p_large_move_1h' OR value BETWEEN 0 AND 1),
    CHECK (target NOT IN ('rv_1h', 'rv_1d') OR value >= 0)
) INHERITS (ledger.entries);
CREATE UNIQUE INDEX IF NOT EXISTS predictions_one_per_bar
    ON ledger.predictions (phase, asset, arm, model, target, var_level, bar_open) NULLS NOT DISTINCT;
CREATE INDEX IF NOT EXISTS predictions_bar_open ON ledger.predictions (bar_open);

-- The score of one prediction, once its bar has closed.
CREATE TABLE IF NOT EXISTS ledger.scores (
    prediction_seq  bigint NOT NULL UNIQUE REFERENCES ledger.predictions (seq),
    -- realized volatility, the 0/1 large-move indicator, or the realized return for VaR
    realized        double precision NOT NULL
        CHECK (realized NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)),
    losses          jsonb  NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(losses) = 'object'),
    data            jsonb  NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(data) = 'array'),
    UNIQUE (seq)
) INHERITS (ledger.entries);

-- Verifier verdicts on findings (stage 6) and validator verdicts on replications (stage 7).
CREATE TABLE IF NOT EXISTS ledger.verdicts (
    kind         text   NOT NULL CHECK (kind IN ('verification', 'replication')),
    agent        text   NOT NULL CHECK (agent IN ('verifier', 'validator', 'researcher')),
    package_seq  bigint REFERENCES ledger.experiment_packages (seq),
    finding_id   uuid,
    verdict      text   NOT NULL,
    confidence   double precision CHECK (confidence BETWEEN 0 AND 1),
    cause        text,  -- required for a divergent replication
    details      jsonb  NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(details) = 'object'),
    UNIQUE (seq),
    CHECK (kind <> 'replication'
           OR (package_seq IS NOT NULL AND verdict IN ('reproduced', 'divergent'))),
    CHECK (kind <> 'verification'
           OR (finding_id IS NOT NULL AND verdict IN ('supported', 'rejected', 'needs_review'))),
    CHECK (verdict <> 'divergent' OR coalesce(cause, '') <> '')
) INHERITS (ledger.entries);

-- Human interventions (paper, Section 4.2): every one outside the checkpoints of Quadro 5, and
-- the stage approvals at them, written when they happen.
CREATE TABLE IF NOT EXISTS ledger.interventions (
    stage          smallint NOT NULL CHECK (stage BETWEEN 1 AND 9),
    type           text     NOT NULL CHECK (type IN ('correction', 'interruption', 'approval')),
    at_checkpoint  boolean  NOT NULL,
    reason         text     NOT NULL CHECK (reason <> ''),
    actor          text     NOT NULL DEFAULT 'researcher' CHECK (actor <> ''),
    UNIQUE (seq),
    CHECK (NOT at_checkpoint OR type = 'approval')
) INHERITS (ledger.entries);

-- Protocol violations detected (Quadro 9): late predictions rejected, hash-chain breaks found
-- by the daily check, and actions denied for lack of permission.
CREATE TABLE IF NOT EXISTS ledger.violations (
    kind         text  NOT NULL
        CHECK (kind IN ('late_prediction', 'chain_break', 'permission_denied', 'other')),
    reported_by  text  NOT NULL CHECK (reported_by <> ''),
    detail       jsonb NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(detail) = 'object'),
    UNIQUE (seq)
) INHERITS (ledger.entries);

-- --- per-table checks ----------------------------------------------------------------------

CREATE OR REPLACE FUNCTION ledger.check_versioned() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    old_key text;
BEGIN
    IF NEW.supersedes IS NULL THEN
        RETURN NEW;
    END IF;
    IF TG_TABLE_NAME = 'hypotheses' THEN
        SELECT hypothesis_key INTO old_key FROM ledger.hypotheses WHERE seq = NEW.supersedes;
        IF old_key IS DISTINCT FROM NEW.hypothesis_key THEN
            RAISE EXCEPTION USING ERRCODE = 'LG006', MESSAGE = format(
                'entry %s is not a version of hypothesis %s', NEW.supersedes, NEW.hypothesis_key);
        END IF;
    ELSE
        SELECT document_key INTO old_key FROM ledger.documents WHERE seq = NEW.supersedes;
        IF old_key IS DISTINCT FROM NEW.document_key THEN
            RAISE EXCEPTION USING ERRCODE = 'LG006', MESSAGE = format(
                'entry %s is not a version of document %s', NEW.supersedes, NEW.document_key);
        END IF;
    END IF;
    RETURN NEW;
END
$$;

CREATE OR REPLACE FUNCTION ledger.check_document() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.content IS NOT NULL
       AND encode(sha256(convert_to(NEW.content, 'UTF8')), 'hex') <> NEW.content_sha256 THEN
        RAISE EXCEPTION USING ERRCODE = 'LG006',
            MESSAGE = 'content_sha256 is not the SHA-256 of the content (UTF-8)';
    END IF;
    RETURN NEW;
END
$$;

CREATE OR REPLACE FUNCTION ledger.require_same_phase(ref_table text, ref_seq bigint, phase text)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE
    ref_phase text;
BEGIN
    IF ref_seq IS NULL THEN
        RETURN;
    END IF;
    EXECUTE format('SELECT phase FROM ledger.%I WHERE seq = $1', ref_table)
        INTO ref_phase USING ref_seq;
    IF ref_phase IS NULL THEN
        RAISE EXCEPTION USING ERRCODE = 'LG006',
            MESSAGE = format('no entry %s in ledger.%s', ref_seq, ref_table);
    END IF;
    IF ref_phase <> phase THEN
        RAISE EXCEPTION USING ERRCODE = 'LG003', MESSAGE = format(
            'phase %s cannot refer to entry %s of phase %s', phase, ref_seq, ref_phase);
    END IF;
END
$$;

CREATE OR REPLACE FUNCTION ledger.check_prediction() RETURNS trigger
LANGUAGE plpgsql SET timezone = 'UTC' AS $$
DECLARE
    now_ timestamptz := ledger.clock();
BEGIN
    IF now_ >= NEW.bar_open THEN
        RAISE EXCEPTION USING ERRCODE = 'LG001', MESSAGE = format(
            'late prediction: the database clock reads %s, at or after the open of bar %s',
            now_, NEW.bar_open);
    END IF;
    PERFORM ledger.require_same_phase('experiment_packages', NEW.package_seq, NEW.phase);
    RETURN NEW;
END
$$;

-- Checked again at commit, so a transaction can't insert in time, wait for the outcome, and
-- only then decide to commit.
CREATE OR REPLACE FUNCTION ledger.check_prediction_at_commit() RETURNS trigger
LANGUAGE plpgsql SET timezone = 'UTC' AS $$
DECLARE
    now_ timestamptz := ledger.clock();
BEGIN
    IF now_ >= NEW.bar_open THEN
        RAISE EXCEPTION USING ERRCODE = 'LG001', MESSAGE = format(
            'late prediction: committed at %s, at or after the open of bar %s', now_, NEW.bar_open);
    END IF;
    RETURN NULL;
END
$$;

CREATE OR REPLACE FUNCTION ledger.horizon(target text) RETURNS interval
LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE WHEN target LIKE '%\_1d' THEN interval '1 day' ELSE interval '1 hour' END
$$;

CREATE OR REPLACE FUNCTION ledger.check_score() RETURNS trigger
LANGUAGE plpgsql SET timezone = 'UTC' AS $$
DECLARE
    p record;
    now_ timestamptz := ledger.clock();
BEGIN
    SELECT phase, bar_open, target INTO p FROM ledger.predictions WHERE seq = NEW.prediction_seq;
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = 'LG006',
            MESSAGE = format('no prediction %s', NEW.prediction_seq);
    END IF;
    PERFORM ledger.require_same_phase('predictions', NEW.prediction_seq, NEW.phase);
    IF now_ < p.bar_open + ledger.horizon(p.target) THEN
        RAISE EXCEPTION USING ERRCODE = 'LG004', MESSAGE = format(
            'prediction %s cannot be scored before %s (database clock: %s)',
            NEW.prediction_seq, p.bar_open + ledger.horizon(p.target), now_);
    END IF;
    RETURN NEW;
END
$$;

CREATE OR REPLACE FUNCTION ledger.check_verdict() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM ledger.require_same_phase('experiment_packages', NEW.package_seq, NEW.phase);
    RETURN NEW;
END
$$;

-- --- triggers ------------------------------------------------------------------------------

CREATE OR REPLACE TRIGGER entries_no_insert BEFORE INSERT ON ledger.entries
    FOR EACH ROW EXECUTE FUNCTION ledger.reject_parent_insert();
CREATE OR REPLACE TRIGGER entries_no_change BEFORE UPDATE OR DELETE ON ledger.entries
    FOR EACH ROW EXECUTE FUNCTION ledger.reject_change();
CREATE OR REPLACE TRIGGER entries_no_truncate BEFORE TRUNCATE ON ledger.entries
    FOR EACH STATEMENT EXECUTE FUNCTION ledger.reject_change();

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['hypotheses', 'documents', 'experiment_packages', 'predictions',
                             'scores', 'verdicts', 'interventions', 'violations'] LOOP
        EXECUTE format('CREATE OR REPLACE TRIGGER %I BEFORE INSERT ON ledger.%I '
                       'FOR EACH ROW EXECUTE FUNCTION ledger.chain_row()', t || '_90_chain', t);
        EXECUTE format('CREATE OR REPLACE TRIGGER %I BEFORE UPDATE OR DELETE ON ledger.%I '
                       'FOR EACH ROW EXECUTE FUNCTION ledger.reject_change()', t || '_no_change', t);
        EXECUTE format('CREATE OR REPLACE TRIGGER %I BEFORE TRUNCATE ON ledger.%I '
                       'FOR EACH STATEMENT EXECUTE FUNCTION ledger.reject_change()',
                       t || '_no_truncate', t);
    END LOOP;
END
$$;

CREATE OR REPLACE TRIGGER hypotheses_10_check BEFORE INSERT ON ledger.hypotheses
    FOR EACH ROW EXECUTE FUNCTION ledger.check_versioned();
CREATE OR REPLACE TRIGGER documents_10_check BEFORE INSERT ON ledger.documents
    FOR EACH ROW EXECUTE FUNCTION ledger.check_versioned();
CREATE OR REPLACE TRIGGER documents_11_check BEFORE INSERT ON ledger.documents
    FOR EACH ROW EXECUTE FUNCTION ledger.check_document();
CREATE OR REPLACE TRIGGER predictions_10_check BEFORE INSERT ON ledger.predictions
    FOR EACH ROW EXECUTE FUNCTION ledger.check_prediction();
CREATE OR REPLACE TRIGGER scores_10_check BEFORE INSERT ON ledger.scores
    FOR EACH ROW EXECUTE FUNCTION ledger.check_score();
CREATE OR REPLACE TRIGGER verdicts_10_check BEFORE INSERT ON ledger.verdicts
    FOR EACH ROW EXECUTE FUNCTION ledger.check_verdict();

-- Constraint triggers can't be replaced in place.
DROP TRIGGER IF EXISTS predictions_deadline_at_commit ON ledger.predictions;
CREATE CONSTRAINT TRIGGER predictions_deadline_at_commit AFTER INSERT ON ledger.predictions
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION ledger.check_prediction_at_commit();

-- --- verification --------------------------------------------------------------------------

-- Every entry with its chain fields and whether the row still matches its canonical text.
-- cryptonlp.ledger.check recomputes the hashes and the links from this.
CREATE OR REPLACE FUNCTION ledger.verify_rows()
RETURNS TABLE (seq bigint, table_name text, canonical text, prev_hash text, hash text,
               row_matches boolean)
LANGUAGE plpgsql STABLE
SET timezone = 'UTC'
SET extra_float_digits = 1
AS $$
#variable_conflict use_column
DECLARE
    child record;
BEGIN
    FOR child IN
        SELECT c.oid::regclass AS rel, c.relname::text AS name
        FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid
        WHERE i.inhparent = 'ledger.entries'::regclass
    LOOP
        RETURN QUERY EXECUTE format(
            'SELECT t.seq, %L::text, t.canonical, t.prev_hash, t.hash, '
            'CASE WHEN pg_input_is_valid(t.canonical, ''jsonb'') '
            'THEN ledger.canonical(%L, to_jsonb(t)) = t.canonical::jsonb ELSE false END '
            'FROM ONLY %s t', child.name, child.name, child.rel);
    END LOOP;
    -- The parent never holds rows of its own.
    RETURN QUERY SELECT t.seq, 'entries'::text, t.canonical, t.prev_hash, t.hash, false
                 FROM ONLY ledger.entries t;
END
$$;

-- --- operations (not part of the ledger) ---------------------------------------------------

-- One row per daily chain check, for the dashboard.
CREATE TABLE IF NOT EXISTS ops.ledger_checks (
    check_id         bigserial   PRIMARY KEY,
    checked_at       timestamptz NOT NULL DEFAULT now(),
    run_id           text        NOT NULL,
    ok               boolean     NOT NULL,
    entries          bigint      NOT NULL,
    head_seq         bigint,
    head_hash        text,
    anchors_checked  integer     NOT NULL DEFAULT 0,
    problems         jsonb       NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS ledger_checks_checked_at ON ops.ledger_checks (checked_at DESC);

-- The weekly head hash sent to the advisor. The copy the advisor holds is what counts; this
-- one lets the daily check confirm every past anchor is still in the chain.
CREATE TABLE IF NOT EXISTS ops.ledger_anchors (
    anchor_id   bigserial   PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    run_id      text        NOT NULL,
    head_seq    bigint      NOT NULL,
    head_hash   text        NOT NULL,
    entries     bigint      NOT NULL,
    path        text        NOT NULL
);

-- --- monitoring (Grafana) ------------------------------------------------------------------

CREATE OR REPLACE VIEW monitoring.ledger_checks AS
SELECT checked_at,
       ok,
       entries,
       head_seq,
       anchors_checked,
       jsonb_array_length(problems) AS problems,
       extract(epoch FROM now() - checked_at)::double precision / 3600 AS age_hours
FROM ops.ledger_checks;

-- Counts only: no ledger content reaches Grafana.
CREATE OR REPLACE VIEW monitoring.ledger_entries AS
SELECT c.relname::text     AS table_name,
       e.phase,
       count(*)            AS entries,
       max(e.recorded_at)  AS last_recorded_at
FROM ledger.entries e
JOIN pg_class c ON c.oid = e.tableoid
GROUP BY 1, 2;

CREATE OR REPLACE VIEW monitoring.ledger_violations AS
SELECT date_trunc('day', recorded_at) AS day, phase, kind, count(*) AS violations
FROM ledger.violations
GROUP BY 1, 2, 3;

GRANT SELECT ON monitoring.ledger_checks, monitoring.ledger_entries, monitoring.ledger_violations
    TO grafana_reader;

-- --- roles ---------------------------------------------------------------------------------

-- The role agents' ledger tools will use (step 7): read and append, nothing else.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ledger_writer') THEN
        CREATE ROLE ledger_writer NOLOGIN;
    END IF;
END
$$;
REVOKE ALL ON ALL TABLES IN SCHEMA ledger FROM PUBLIC;
GRANT USAGE ON SCHEMA ledger TO ledger_writer;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA ledger TO ledger_writer;
REVOKE UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA ledger FROM ledger_writer;
