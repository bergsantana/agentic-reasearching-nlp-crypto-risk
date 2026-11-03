-- Step 1: raw-store manifest, post index, deletions, collector state and job runs.
-- Idempotent: Compose runs it on first start, and `cryptonlp db init` re-applies it.
--
-- The raw data itself lives in Parquet files (data/raw), never in Postgres. These tables only
-- say which files exist, with their hashes, and which posts they hold, so deletion sync and
-- the dashboard work without opening the files.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- One row per file in the raw store. Files are never edited. Honoring a deletion at the
-- source writes a successor file without the deleted posts; the old row keeps its hash and
-- gets removed_at, so the chain of file versions stays auditable.
CREATE TABLE IF NOT EXISTS raw.files (
    file_id          bigserial PRIMARY KEY,
    dataset          text        NOT NULL,
    partition_date   date        NOT NULL,
    path             text        NOT NULL UNIQUE,  -- relative to the raw-store root
    sha256           char(64)    NOT NULL,
    bytes            bigint      NOT NULL CHECK (bytes >= 0),
    rows             integer     NOT NULL CHECK (rows >= 0),
    min_received_at  timestamptz,
    max_received_at  timestamptz,
    source_url       text,                          -- set for downloaded archives
    run_id           text        NOT NULL,
    written_at       timestamptz NOT NULL DEFAULT now(),
    supersedes       bigint      REFERENCES raw.files (file_id),
    removed_at       timestamptz,
    removed_reason   text
);
CREATE INDEX IF NOT EXISTS files_dataset_date ON raw.files (dataset, partition_date);
CREATE INDEX IF NOT EXISTS files_source_url ON raw.files (source_url) WHERE source_url IS NOT NULL;

-- One row per post in the raw store, without text or author.
CREATE TABLE IF NOT EXISTS raw.posts (
    source           text        NOT NULL CHECK (source IN ('bluesky', 'mastodon')),
    uri              text        NOT NULL,
    instance         text,        -- Mastodon instance the post was read from
    native_id        text,        -- that instance's status id, for the deletion sweep
    created_at       timestamptz NOT NULL,
    received_at      timestamptz NOT NULL,
    origin           text        NOT NULL CHECK (origin IN ('live', 'backfill', 'replay')),
    file_id          bigint      NOT NULL REFERENCES raw.files (file_id),
    last_checked_at  timestamptz, -- last deletion sweep that found the post still online
    PRIMARY KEY (source, uri)
);
CREATE INDEX IF NOT EXISTS posts_received_at ON raw.posts (received_at);
CREATE INDEX IF NOT EXISTS posts_file_id ON raw.posts (file_id);
CREATE INDEX IF NOT EXISTS posts_sweep ON raw.posts (source, last_checked_at NULLS FIRST);

-- Posts deleted at the source. Every reader excludes these at once; the purge job then
-- rewrites the raw files without them and sets purged_at.
CREATE TABLE IF NOT EXISTS raw.deletions (
    source       text        NOT NULL,
    uri          text        NOT NULL,
    detected_at  timestamptz NOT NULL DEFAULT now(),
    method       text        NOT NULL CHECK (method IN ('stream', 'sweep')),
    run_id       text        NOT NULL,
    purged_at    timestamptz,
    PRIMARY KEY (source, uri),
    FOREIGN KEY (source, uri) REFERENCES raw.posts (source, uri)
);
CREATE INDEX IF NOT EXISTS deletions_unpurged ON raw.deletions (source) WHERE purged_at IS NULL;

-- Where each collector is. last_event_at means "caught up to": Jetstream's own time for the
-- stream, and the time of the last successful poll for pollers. Lag is now() minus it.
CREATE TABLE IF NOT EXISTS ops.collector_state (
    collector      text        PRIMARY KEY,  -- 'bluesky', 'mastodon:<instance>:tag:<tag>', ...
    source         text        NOT NULL,     -- 'bluesky', 'mastodon', 'binance', 'trends'
    cursor         text,
    last_event_at  timestamptz,
    heartbeat_at   timestamptz NOT NULL DEFAULT now(),
    run_id         text        NOT NULL
);

-- Spans the stream could not deliver (a Jetstream cursor past the 36-hour lookback is
-- clamped silently, so the collector detects and logs the jump itself).
CREATE TABLE IF NOT EXISTS ops.collector_gaps (
    gap_id       bigserial   PRIMARY KEY,
    collector    text        NOT NULL,
    gap_start    timestamptz NOT NULL,
    gap_end      timestamptz NOT NULL CHECK (gap_end > gap_start),
    detected_at  timestamptz NOT NULL DEFAULT now(),
    run_id       text        NOT NULL
);

-- Every job run, written by cryptonlp.ops.track_job whether Prefect or the CLI started it.
CREATE TABLE IF NOT EXISTS ops.job_runs (
    job_run_id   bigserial   PRIMARY KEY,
    job          text        NOT NULL,
    run_id       text        NOT NULL,
    started_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    status       text        NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    error        text,
    stats        jsonb       NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS job_runs_job_started ON ops.job_runs (job, started_at DESC);

-- Monitoring views: the only thing Grafana's read-only role can see. No text, no authors.
CREATE OR REPLACE VIEW monitoring.collector_lag AS
SELECT collector,
       source,
       last_event_at,
       heartbeat_at,
       extract(epoch FROM now() - coalesce(last_event_at, heartbeat_at))::double precision
           AS lag_seconds,
       extract(epoch FROM now() - heartbeat_at)::double precision AS heartbeat_age_seconds
FROM ops.collector_state;

CREATE OR REPLACE VIEW monitoring.posts_per_hour AS
SELECT date_trunc('hour', p.received_at) AS hour,
       p.source,
       count(*)                          AS posts,
       count(d.uri)                      AS deleted
FROM raw.posts p
LEFT JOIN raw.deletions d USING (source, uri)
WHERE p.origin = 'live'
GROUP BY 1, 2;

CREATE OR REPLACE VIEW monitoring.job_runs AS
SELECT job,
       run_id,
       started_at,
       finished_at,
       status,
       left(error, 500) AS error,
       extract(epoch FROM coalesce(finished_at, now()) - started_at)::double precision
           AS duration_seconds
FROM ops.job_runs;

CREATE OR REPLACE VIEW monitoring.collector_gaps AS
SELECT collector, gap_start, gap_end,
       extract(epoch FROM gap_end - gap_start)::double precision AS gap_seconds,
       detected_at
FROM ops.collector_gaps;

CREATE OR REPLACE VIEW monitoring.raw_store AS
SELECT dataset,
       partition_date,
       count(*) FILTER (WHERE removed_at IS NULL)    AS files,
       coalesce(sum(rows) FILTER (WHERE removed_at IS NULL), 0)  AS rows,
       coalesce(sum(bytes) FILTER (WHERE removed_at IS NULL), 0) AS bytes,
       max(written_at)                               AS last_written_at
FROM raw.files
GROUP BY 1, 2;

CREATE OR REPLACE VIEW monitoring.deletions_per_day AS
SELECT date_trunc('day', detected_at) AS day, source, method,
       count(*) AS deleted, count(purged_at) AS purged
FROM raw.deletions
GROUP BY 1, 2, 3;

-- Read-only role for Grafana. `cryptonlp db init` sets its password from the environment.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'grafana_reader') THEN
        CREATE ROLE grafana_reader LOGIN;
    END IF;
END
$$;
ALTER ROLE grafana_reader SET default_transaction_read_only = on;
GRANT USAGE ON SCHEMA monitoring TO grafana_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO grafana_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA monitoring GRANT SELECT ON TABLES TO grafana_reader;
