-- Step 6: the classifier. What each classifier version is, how long each batch took, the
-- embeddings for the Analyst's semantic search, and the views behind the classification panels.
-- Idempotent: Compose runs it on first start, and `cryptonlp db init` re-applies it.
--
-- Like labels, all of this is derived data: rebuildable, not part of the ledger. What H4 rests
-- on (the rubric, the prompt hash, the model revision) is registered in the ledger.

-- --- classifier versions ---------------------------------------------------------------------

-- The full spec behind each model_version in clean.labels: model, revision, readout, prompt hash,
-- rubric version, temperatures. Written the first time a version labels anything; never changed
-- (a new spec is a new version).
CREATE TABLE IF NOT EXISTS clean.classifier_versions (
    model_version  text        PRIMARY KEY CHECK (model_version <> ''),
    spec           jsonb       NOT NULL CHECK (jsonb_typeof(spec) = 'object'),
    spec_sha256    text        NOT NULL CHECK (spec_sha256 ~ '^[0-9a-f]{64}$'),
    prompt_sha256  text        CHECK (prompt_sha256 ~ '^[0-9a-f]{64}$'),
    created_at     timestamptz NOT NULL DEFAULT now(),
    run_id         text        NOT NULL
);

CREATE OR REPLACE FUNCTION clean.classifier_versions_frozen() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'clean.classifier_versions rows are never changed: a new spec is a new version';
END $$;

CREATE OR REPLACE TRIGGER classifier_versions_frozen
    BEFORE UPDATE ON clean.classifier_versions
    FOR EACH ROW EXECUTE FUNCTION clean.classifier_versions_frozen();

-- One row per classifier call over a batch of posts, wall-clock timed (not the database clock:
-- under the simulated clock the duration would mean nothing). Feeds "latency per 1,000 posts".
CREATE TABLE IF NOT EXISTS clean.classify_batches (
    batch_id       bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_version  text        NOT NULL,
    posts          integer     NOT NULL CHECK (posts > 0),
    started_at     timestamptz NOT NULL,
    finished_at    timestamptz NOT NULL CHECK (finished_at >= started_at),
    run_id         text        NOT NULL
);
CREATE INDEX IF NOT EXISTS classify_batches_finished ON clean.classify_batches (finished_at);

-- --- embeddings (pgvector) -------------------------------------------------------------------

-- One vector per in-study post and embedding model. A deletion at the source removes it with
-- the post. The dimension is the model's (Qwen3-Embedding-0.6B: 1,024); a model with another
-- dimension needs its own column or table.
CREATE TABLE IF NOT EXISTS clean.embeddings (
    source         text         NOT NULL,
    uri            text         NOT NULL,
    model_version  text         NOT NULL CHECK (model_version <> ''),
    embedding      vector(1024) NOT NULL,
    embedded_at    timestamptz  NOT NULL DEFAULT now(),
    run_id         text         NOT NULL,
    PRIMARY KEY (source, uri, model_version),
    FOREIGN KEY (source, uri) REFERENCES clean.posts (source, uri) ON DELETE CASCADE
);
-- Cosine distance (the vectors are normalized by the model); HNSW keeps queries fast as the
-- store grows, with no training step.
CREATE INDEX IF NOT EXISTS clean_embeddings_hnsw
    ON clean.embeddings USING hnsw (embedding vector_cosine_ops);

-- --- monitoring (Grafana): counts and times only, no text ------------------------------------

-- Per hourly bar of posts (by receive time) and model classifier version active in the last 7
-- days: in-study posts, how many that version has labeled, and how long after the bar closed its
-- last post was labeled. `newest` marks the version recorded last (the one in use). The hourly loop needs this well under 35 minutes (the forecast goes in
-- at minute 25 of the next hour; claims M14). Every post gets all its labels in one
-- transaction, so the sentiment label stands for the post.
CREATE OR REPLACE VIEW monitoring.classify_per_bar AS
WITH bars AS (
    SELECT date_trunc('hour', received_at) AS bar_open, count(*) AS posts
    FROM clean.study_posts
    WHERE received_at > now() - interval '7 days'
    GROUP BY 1
), versions AS (
    SELECT DISTINCT model_version
    FROM clean.labels
    WHERE annotator = 'model' AND labeled_at > now() - interval '7 days'
), labeled AS (
    SELECT date_trunc('hour', p.received_at) AS bar_open,
           l.model_version,
           count(*)          AS labeled,
           max(l.labeled_at) AS last_labeled_at
    FROM clean.study_posts p
    JOIN clean.labels l USING (source, uri)
    WHERE l.task = 'sentiment' AND l.annotator = 'model'
      AND p.received_at > now() - interval '7 days'
    GROUP BY 1, 2
)
SELECT b.bar_open,
       v.model_version,
       v.model_version = (SELECT model_version FROM clean.classifier_versions
                          ORDER BY created_at DESC, model_version LIMIT 1) AS newest,
       b.posts,
       coalesce(l.labeled, 0)            AS labeled,
       b.posts - coalesce(l.labeled, 0)  AS waiting,
       extract(epoch FROM l.last_labeled_at - (b.bar_open + interval '1 hour')) / 60
                                         AS minutes_after_close
FROM bars b
CROSS JOIN versions v
LEFT JOIN labeled l ON l.bar_open = b.bar_open AND l.model_version = v.model_version;

-- Classifier throughput per hour of work: posts, seconds spent, seconds per 1,000 posts.
CREATE OR REPLACE VIEW monitoring.classify_throughput AS
SELECT date_trunc('hour', finished_at)                                   AS hour,
       model_version,
       sum(posts)                                                        AS posts,
       sum(extract(epoch FROM finished_at - started_at))                 AS seconds,
       1000 * sum(extract(epoch FROM finished_at - started_at)) / sum(posts)
                                                                         AS seconds_per_1000
FROM clean.classify_batches
GROUP BY 1, 2;

-- In-study posts without a vector, per embedding model in use.
CREATE OR REPLACE VIEW monitoring.embedding_backlog AS
SELECT m.model_version,
       count(*) FILTER (WHERE e.uri IS NULL) AS waiting,
       count(e.uri)                          AS embedded
FROM (SELECT DISTINCT model_version FROM clean.embeddings) m
CROSS JOIN clean.study_posts p
LEFT JOIN clean.embeddings e
       ON e.source = p.source AND e.uri = p.uri AND e.model_version = m.model_version
GROUP BY 1;

GRANT SELECT ON monitoring.classify_per_bar, monitoring.classify_throughput,
    monitoring.embedding_backlog TO grafana_reader;
