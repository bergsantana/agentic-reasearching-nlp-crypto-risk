-- Step 3: normalized, deduplicated, pseudonymized posts.
-- Idempotent: Compose runs it on first start, and `cryptonlp db init` re-applies it.
--
-- Derived from the raw store and rebuildable from it (`cryptonlp normalize rebuild`, with the
-- same pseudonymization key). Every raw post gets one row, whatever the filters decide, so the
-- filter rates can be audited; `in_study` says whether the study may use it. A row depends only
-- on posts received before it, so no later post can change it (no leakage).
--
-- Unlike the raw store and the ledger, this schema is mutable: a deletion at the source removes
-- the row at once, and later labels and vectors go with it (ON DELETE CASCADE).

CREATE SCHEMA IF NOT EXISTS clean;

CREATE TABLE IF NOT EXISTS clean.posts (
    source            text        NOT NULL,
    uri               text        NOT NULL,
    author_hash       text        NOT NULL CHECK (author_hash ~ '^[0-9a-f]{64}$'),
    text              text        NOT NULL,
    created_at        timestamptz NOT NULL,
    received_at       timestamptz NOT NULL,
    origin            text        NOT NULL,
    -- Language: detected from the text with URLs, handles, tags and numbers removed; the
    -- client's declared tag only when too little text is left ('declared'), else 'und'.
    lang              text        NOT NULL,
    lang_method       text        NOT NULL CHECK (lang_method IN ('detected', 'declared', 'none')),
    lang_confidence   real        CHECK (lang_confidence BETWEEN 0 AND 1),
    langs_declared    text[]      NOT NULL DEFAULT '{}',
    -- Entities
    assets            text[]      NOT NULL DEFAULT '{}',  -- 'BTC', 'ETH' (lexicon in config)
    cashtags          text[]      NOT NULL DEFAULT '{}',  -- '$BTC', upper case
    hashtags          text[]      NOT NULL DEFAULT '{}',  -- lower case, without '#'
    addresses         text[]      NOT NULL DEFAULT '{}',  -- 'evm:0x..', 'btc:..', 'sol:..'
    has_url           boolean     NOT NULL,
    -- Evident spam (pattern rules); the scam classifier and the Verifier come later.
    spam_rules        text[]      NOT NULL DEFAULT '{}',
    is_spam           boolean     NOT NULL,
    -- Duplicates: of the earliest-received post seen with the same text, or a near copy.
    text_sha256       text        NOT NULL CHECK (text_sha256 ~ '^[0-9a-f]{64}$'),
    duplicate_of      text,
    duplicate_kind    text        CHECK (duplicate_kind IN ('exact', 'near')),
    near_duplicate_score real     CHECK (near_duplicate_score BETWEEN 0 AND 1),
    -- Provenance
    raw_file_id       bigint      NOT NULL REFERENCES raw.files (file_id),
    config_sha256     text        NOT NULL,  -- config/normalize.toml as loaded
    key_id            text        NOT NULL,  -- fingerprint of the pseudonymization key
    normalize_run_id  text        NOT NULL,
    normalized_at     timestamptz NOT NULL DEFAULT now(),
    in_study          boolean     GENERATED ALWAYS AS (
        lang = 'en' AND NOT is_spam AND duplicate_of IS NULL AND origin = 'live'
    ) STORED,
    PRIMARY KEY (source, uri),
    FOREIGN KEY (source, uri) REFERENCES raw.posts (source, uri),
    CHECK ((duplicate_of IS NULL) = (duplicate_kind IS NULL)),
    CHECK (duplicate_kind IS DISTINCT FROM 'near' OR near_duplicate_score IS NOT NULL),
    CHECK (duplicate_of IS DISTINCT FROM uri)
);
CREATE INDEX IF NOT EXISTS clean_posts_received_at ON clean.posts (received_at);
CREATE INDEX IF NOT EXISTS clean_posts_text_sha256 ON clean.posts (text_sha256);
CREATE INDEX IF NOT EXISTS clean_posts_study ON clean.posts (received_at) WHERE in_study;

-- MinHash signatures of recent English posts, for near-duplicate search. Only the last
-- `window_days` are needed; older rows are pruned.
CREATE TABLE IF NOT EXISTS clean.minhash (
    source       text        NOT NULL,
    uri          text        NOT NULL,
    received_at  timestamptz NOT NULL,
    signature    bytea       NOT NULL,
    PRIMARY KEY (source, uri),
    FOREIGN KEY (source, uri) REFERENCES clean.posts (source, uri) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS clean_minhash_received_at ON clean.minhash (received_at);

-- A deletion at the source removes the normalized post at once (and, by cascade, its
-- signature and later its labels and vectors). The raw files are purged by deletion sync.
CREATE OR REPLACE FUNCTION clean.drop_deleted() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM clean.posts WHERE source = NEW.source AND uri = NEW.uri;
    RETURN NULL;
END
$$;
CREATE OR REPLACE TRIGGER deletions_drop_clean AFTER INSERT ON raw.deletions
    FOR EACH ROW EXECUTE FUNCTION clean.drop_deleted();

-- What the study reads.
CREATE OR REPLACE VIEW clean.study_posts AS
SELECT source, uri, author_hash, text, created_at, received_at, lang, assets, cashtags,
       hashtags, addresses, has_url
FROM clean.posts
WHERE in_study;

-- --- monitoring (Grafana): counts only, no text, no authors --------------------------------

CREATE OR REPLACE VIEW monitoring.normalized_per_hour AS
SELECT date_trunc('hour', received_at)                         AS hour,
       source,
       count(*)                                                AS posts,
       count(*) FILTER (WHERE lang = 'en')                     AS english,
       count(*) FILTER (WHERE is_spam)                         AS spam,
       count(*) FILTER (WHERE duplicate_kind = 'exact')        AS exact_duplicates,
       count(*) FILTER (WHERE duplicate_kind = 'near')         AS near_duplicates,
       count(*) FILTER (WHERE in_study)                        AS in_study
FROM clean.posts
GROUP BY 1, 2;

CREATE OR REPLACE VIEW monitoring.normalize_backlog AS
SELECT count(*) AS pending, min(p.received_at) AS oldest_pending
FROM raw.posts p
LEFT JOIN clean.posts c USING (source, uri)
LEFT JOIN raw.deletions d USING (source, uri)
WHERE c.uri IS NULL AND d.uri IS NULL;

CREATE OR REPLACE VIEW monitoring.spam_rules_per_day AS
SELECT date_trunc('day', received_at) AS day, source, rule, count(*) AS posts
FROM clean.posts, unnest(spam_rules) AS rule
GROUP BY 1, 2, 3;

GRANT SELECT ON monitoring.normalized_per_hour, monitoring.normalize_backlog,
    monitoring.spam_rules_per_day TO grafana_reader;
