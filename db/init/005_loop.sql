-- Step 3b: the thin end-to-end loop. Labels, hourly NLP features, and the views behind the
-- forecast panels. Idempotent: Compose runs it on first start, and `cryptonlp db init`
-- re-applies it.
--
-- Labels and features are derived data, like clean.posts: rebuildable, not part of the ledger.
-- What the study rests on is the forecasts and scores, which go into the ledger.

-- --- labels --------------------------------------------------------------------------------

-- One label per post, task and classifier version (or human pass). Appended, never
-- overwritten: a new classifier version writes new rows next to the old ones. A deletion at
-- the source removes the post's labels with it.
CREATE TABLE IF NOT EXISTS clean.labels (
    source         text        NOT NULL,
    uri            text        NOT NULL,
    task           text        NOT NULL CHECK (task IN ('sentiment', 'topic', 'scam', 'entity', 'event')),
    model_version  text        NOT NULL CHECK (model_version <> ''),  -- "stub", a revision, "human:pass1"
    annotator      text        NOT NULL CHECK (annotator IN ('model', 'human')),
    label_values   text[]      NOT NULL CHECK (cardinality(label_values) > 0),
    probabilities  jsonb       NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(probabilities) = 'object'),
    confidence     real        CHECK (confidence BETWEEN 0 AND 1),
    labeled_at     timestamptz NOT NULL,  -- the database clock (simulated in the test database)
    run_id         text        NOT NULL,
    PRIMARY KEY (source, uri, task, model_version),
    FOREIGN KEY (source, uri) REFERENCES clean.posts (source, uri) ON DELETE CASCADE,
    CHECK (annotator = 'human' OR confidence IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS clean_labels_version ON clean.labels (model_version, task);

-- --- features ------------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS features;

-- NLP features per asset and hourly bar, from the in-study posts received during the bar that
-- mention the asset. Written once the bar has closed and settled (every post on disk,
-- normalized and labeled), so a row never changes afterwards. The aggregates hold no text and
-- no authors, so they stay when a post is deleted later.
CREATE TABLE IF NOT EXISTS features.hourly (
    asset               text        NOT NULL CHECK (asset IN ('BTC', 'ETH')),
    bar_open            timestamptz NOT NULL CHECK (extract(epoch FROM bar_open) % 3600 = 0),
    bot_filtered        boolean     NOT NULL,  -- the Verifier's filter (step 8); false until then
    classifier_version  text        NOT NULL CHECK (classifier_version <> ''),
    post_count          integer     NOT NULL CHECK (post_count >= 0),
    source_cutoff       timestamptz,  -- latest received_at among the posts used
    sentiment_shares    jsonb       NOT NULL DEFAULT '{}',
    topic_shares        jsonb       NOT NULL DEFAULT '{}',
    scam_shares         jsonb       NOT NULL DEFAULT '{}',
    event_counts        jsonb       NOT NULL DEFAULT '{}',
    run_id              text        NOT NULL,
    computed_at         timestamptz NOT NULL,  -- the database clock (simulated in the test database)
    PRIMARY KEY (asset, bar_open, bot_filtered, classifier_version),
    -- The leakage rule, again in the database: no post received after the bar closed.
    CHECK (source_cutoff IS NULL OR source_cutoff < bar_open + interval '1 hour'),
    CHECK ((post_count = 0) = (source_cutoff IS NULL)),
    -- Written only after the bar closed.
    CHECK (computed_at >= bar_open + interval '1 hour')
);
CREATE INDEX IF NOT EXISTS features_hourly_bar_open ON features.hourly (bar_open);

-- --- monitoring (Grafana): counts only, no forecast values ---------------------------------

-- Per bar and phase: forecasts in on time, forecasts the database clock refused (late), and of
-- the on-time ones, how many are scored and how many are due but not scored yet.
CREATE OR REPLACE VIEW monitoring.forecast_bars AS
WITH issued AS (
    SELECT p.bar_open,
           p.phase,
           count(*)                                    AS on_time,
           count(s.seq)                                AS scored,
           count(*) FILTER (WHERE s.seq IS NULL
                            AND p.bar_open + ledger.horizon(p.target) <= now()) AS pending
    FROM ledger.predictions p
    LEFT JOIN ledger.scores s ON s.prediction_seq = p.seq
    GROUP BY 1, 2
), late AS (
    SELECT (detail -> 'forecast' ->> 'bar_open')::timestamptz AS bar_open,
           phase,
           count(*) AS late
    FROM ledger.violations
    WHERE kind = 'late_prediction'
    GROUP BY 1, 2
)
SELECT bar_open,
       phase,
       coalesce(i.on_time, 0) AS on_time,
       coalesce(l.late, 0)    AS late,
       coalesce(i.scored, 0)  AS scored,
       coalesce(i.pending, 0) AS pending
FROM issued i
FULL JOIN late l USING (bar_open, phase);

CREATE OR REPLACE VIEW monitoring.features_per_hour AS
SELECT bar_open, asset, classifier_version, bot_filtered, post_count
FROM features.hourly;

GRANT SELECT ON monitoring.forecast_bars, monitoring.features_per_hour TO grafana_reader;
