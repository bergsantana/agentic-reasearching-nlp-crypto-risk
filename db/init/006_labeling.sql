-- Step 4: the human-labeled set. Batches drawn for labeling, their items, every version of
-- every annotation pulled from Label Studio, and the time split once it is frozen.
-- Idempotent: Compose runs it on first start, and `cryptonlp db init` re-applies it.
--
-- Items reference raw.posts, not clean.posts: `normalize rebuild` empties clean.posts, and
-- days of labeling must survive it. A deletion at the source still removes the item and its
-- annotations (trigger below), and queues the Label Studio task, which holds the text, for
-- deletion there too (`cryptonlp labeling sync`).
--
-- The labels themselves reach clean.labels (model_version 'human:pass1' or 'human:relabel')
-- only when a pass is frozen, because clean.labels is append-only and the annotator may still
-- correct a label until then. Until then, labeling.annotations keeps every version.

CREATE SCHEMA IF NOT EXISTS labeling;

CREATE TABLE IF NOT EXISTS labeling.batches (
    batch_id           bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pass               text        NOT NULL CHECK (pass IN ('pass1', 'relabel')),
    stratum_day        date,       -- pass1: the UTC day the batch was drawn from
    size               integer     NOT NULL CHECK (size > 0),
    seed               bigint      NOT NULL,
    -- Per stratum ('bluesky:other', 'mastodon:candidate', ...): population N and sample n.
    strata             jsonb       NOT NULL CHECK (jsonb_typeof(strata) = 'object'),
    config_sha256      text        NOT NULL CHECK (config_sha256 ~ '^[0-9a-f]{64}$'),
    drawn_at           timestamptz NOT NULL DEFAULT now(),
    run_id             text        NOT NULL,
    studio_project_id  integer     UNIQUE,  -- set once the batch is pushed to Label Studio
    CHECK ((pass = 'pass1') = (stratum_day IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS labeling.items (
    item_id         bigint  GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id        bigint  NOT NULL REFERENCES labeling.batches (batch_id),
    pass            text    NOT NULL CHECK (pass IN ('pass1', 'relabel')),
    source          text    NOT NULL,
    uri             text    NOT NULL,
    stratum         text    NOT NULL,
    position        integer NOT NULL CHECK (position >= 0),  -- labeling order (shuffled)
    studio_task_id  integer UNIQUE,
    FOREIGN KEY (source, uri) REFERENCES raw.posts (source, uri),
    UNIQUE (batch_id, position),
    -- A post is labeled at most once per pass.
    UNIQUE (pass, source, uri)
);
CREATE INDEX IF NOT EXISTS labeling_items_post ON labeling.items (source, uri);

-- Every version of an annotation, as pulled. The current one is the latest by updated_at. An
-- annotation deleted in Label Studio gets a cancelled row with error 'deleted in Label Studio'.
CREATE TABLE IF NOT EXISTS labeling.annotations (
    item_id               bigint      NOT NULL REFERENCES labeling.items (item_id) ON DELETE CASCADE,
    studio_annotation_id  integer     NOT NULL,
    created_at            timestamptz NOT NULL,  -- Label Studio's clock
    updated_at            timestamptz NOT NULL,
    lead_time             real,                  -- seconds spent on the post
    cancelled             boolean     NOT NULL,
    result                jsonb       NOT NULL,  -- exactly as Label Studio returned it
    labels                jsonb,                 -- {task: [values]}, when valid and usable
    problem               text        CHECK (problem IN ('not_english', 'unreadable')),
    uncertain             boolean     NOT NULL DEFAULT false,
    error                 text,                  -- why the result is not a valid annotation
    pulled_at             timestamptz NOT NULL DEFAULT now(),
    run_id                text        NOT NULL,
    PRIMARY KEY (item_id, studio_annotation_id, updated_at),
    CHECK (labels IS NULL OR (error IS NULL AND problem IS NULL AND NOT cancelled))
);

-- The annotation each item has now.
CREATE OR REPLACE VIEW labeling.current AS
SELECT DISTINCT ON (a.item_id) a.*
FROM labeling.annotations a
ORDER BY a.item_id, a.updated_at DESC, a.studio_annotation_id DESC;

-- The time split, written once by `cryptonlp labeling freeze`.
CREATE TABLE IF NOT EXISTS labeling.splits (
    item_id        bigint      PRIMARY KEY REFERENCES labeling.items (item_id) ON DELETE CASCADE,
    split          text        NOT NULL CHECK (split IN ('calibration', 'test')),
    frozen_at      timestamptz NOT NULL DEFAULT now(),
    run_id         text        NOT NULL
);

-- Label Studio tasks of posts deleted at the source, still to be deleted there.
CREATE TABLE IF NOT EXISTS labeling.studio_deletions (
    studio_task_id  integer     PRIMARY KEY,
    queued_at       timestamptz NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION labeling.drop_deleted() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO labeling.studio_deletions (studio_task_id)
    SELECT studio_task_id FROM labeling.items
    WHERE source = NEW.source AND uri = NEW.uri AND studio_task_id IS NOT NULL
    ON CONFLICT DO NOTHING;
    DELETE FROM labeling.items WHERE source = NEW.source AND uri = NEW.uri;
    RETURN NULL;
END
$$;
CREATE OR REPLACE TRIGGER deletions_drop_labeling AFTER INSERT ON raw.deletions
    FOR EACH ROW EXECUTE FUNCTION labeling.drop_deleted();

-- --- monitoring (Grafana): counts only, no text ----------------------------------------------

CREATE OR REPLACE VIEW monitoring.labeling_progress AS
SELECT b.batch_id,
       b.pass,
       b.stratum_day,
       b.size                                                   AS drawn,
       count(i.item_id)                                         AS items,
       count(c.item_id) FILTER (WHERE NOT c.cancelled)          AS annotated,
       count(c.item_id) FILTER (WHERE c.labels IS NOT NULL)     AS usable,
       count(c.item_id) FILTER (WHERE c.problem IS NOT NULL)    AS problems,
       count(c.item_id) FILTER (WHERE c.error IS NOT NULL)      AS invalid,
       sum(c.lead_time)                                         AS seconds_labeling
FROM labeling.batches b
LEFT JOIN labeling.items i USING (batch_id)
LEFT JOIN labeling.current c USING (item_id)
GROUP BY b.batch_id;

GRANT SELECT ON monitoring.labeling_progress TO grafana_reader;
