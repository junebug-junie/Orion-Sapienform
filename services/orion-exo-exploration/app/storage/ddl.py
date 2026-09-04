"""DDL for orion-exo-exploration. Every table is prefixed exo_exploration_.

Mirrors services/orion-topic-foundry/app/storage/ddl.py's shape: one
CREATE TABLE IF NOT EXISTS string per table, applied idempotently by
app/storage/repository.py::ensure_tables().
"""

LISTINGS_OBSERVED_DDL = """
CREATE TABLE IF NOT EXISTS exo_exploration_listings_observed (
    observed_id           UUID PRIMARY KEY,
    source                VARCHAR NOT NULL,
    source_category       VARCHAR NOT NULL,
    external_listing_id   VARCHAR NOT NULL,
    url                    TEXT NOT NULL,
    title                  TEXT NOT NULL,
    price                  NUMERIC,
    price_raw              VARCHAR,
    description             TEXT,
    posted_or_renewed_at   TIMESTAMPTZ,
    raw_content_hash       VARCHAR NOT NULL,
    crawl_id               UUID NOT NULL,
    observed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_exo_exploration_listings_observed_external_id
    ON exo_exploration_listings_observed (external_listing_id);
CREATE INDEX IF NOT EXISTS ix_exo_exploration_listings_observed_crawl_id
    ON exo_exploration_listings_observed (crawl_id);
CREATE INDEX IF NOT EXISTS ix_exo_exploration_listings_observed_observed_at
    ON exo_exploration_listings_observed (observed_at);
"""

LISTINGS_CURRENT_DDL = """
CREATE TABLE IF NOT EXISTS exo_exploration_listings_current (
    external_listing_id   VARCHAR PRIMARY KEY,
    source                VARCHAR NOT NULL,
    source_category       VARCHAR NOT NULL,
    url                    TEXT NOT NULL,
    title                  TEXT NOT NULL,
    price                  NUMERIC,
    price_raw              VARCHAR,
    description             TEXT,
    posted_or_renewed_at   TIMESTAMPTZ,
    first_seen_at          TIMESTAMPTZ NOT NULL,
    last_seen_at            TIMESTAMPTZ NOT NULL,
    times_seen              INTEGER NOT NULL DEFAULT 1,
    is_currently_listed     BOOLEAN NOT NULL DEFAULT TRUE,
    interest_score           DOUBLE PRECISION NOT NULL DEFAULT 0,
    interest_reasons         JSONB NOT NULL DEFAULT '[]'::jsonb,
    possible_duplicate_of    VARCHAR,
    expires_at               TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_exo_exploration_listings_current_category
    ON exo_exploration_listings_current (source_category);
CREATE INDEX IF NOT EXISTS ix_exo_exploration_listings_current_interest_score
    ON exo_exploration_listings_current (interest_score DESC);
CREATE INDEX IF NOT EXISTS ix_exo_exploration_listings_current_expires_at
    ON exo_exploration_listings_current (expires_at);
CREATE INDEX IF NOT EXISTS ix_exo_exploration_listings_current_is_currently_listed
    ON exo_exploration_listings_current (is_currently_listed);
"""

CRAWL_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS exo_exploration_crawl_runs (
    crawl_id            UUID PRIMARY KEY,
    started_at           TIMESTAMPTZ NOT NULL,
    finished_at           TIMESTAMPTZ,
    categories_crawled    JSONB NOT NULL DEFAULT '[]'::jsonb,
    listings_seen          INTEGER NOT NULL DEFAULT 0,
    new_listings            INTEGER NOT NULL DEFAULT 0,
    errors                  INTEGER NOT NULL DEFAULT 0,
    status                  VARCHAR NOT NULL DEFAULT 'running'
);

CREATE INDEX IF NOT EXISTS ix_exo_exploration_crawl_runs_started_at
    ON exo_exploration_crawl_runs (started_at DESC);
"""

INTEREST_RULES_DDL = """
CREATE TABLE IF NOT EXISTS exo_exploration_interest_rules (
    rule_id       UUID PRIMARY KEY,
    category_url   TEXT,
    keyword         VARCHAR,
    min_price       NUMERIC,
    max_price       NUMERIC,
    weight          DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    added_by        VARCHAR NOT NULL DEFAULT 'seed',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_exo_exploration_interest_rules_keyword
    ON exo_exploration_interest_rules (keyword);
"""
