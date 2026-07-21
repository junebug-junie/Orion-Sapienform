-- Memory cards schema (Orion Memory Cards v1). Idempotent DDL for apply_memory_cards_schema.
-- Hub / shared DAL also ship a copy at orion/core/storage/sql/memory_cards.sql — keep them in sync when editing DDL.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS memory_cards (
    card_id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slug             text UNIQUE NOT NULL,
    types            text[] NOT NULL,
    anchor_class     text,
    status           text NOT NULL DEFAULT 'pending_review',
    confidence       text NOT NULL DEFAULT 'likely',
    sensitivity      text NOT NULL DEFAULT 'private',
    priority         text NOT NULL DEFAULT 'episodic_detail',
    visibility_scope text[] NOT NULL DEFAULT '{chat}',
    time_horizon     jsonb,
    provenance       text NOT NULL,
    trust_source     text,
    project          text,
    title            text NOT NULL,
    summary          text NOT NULL,
    still_true       text[],
    anchors          text[],
    tags             text[],
    evidence         jsonb NOT NULL DEFAULT '[]'::jsonb,
    subschema        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_card_edges (
    edge_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    from_card_id uuid NOT NULL REFERENCES memory_cards(card_id) ON DELETE CASCADE,
    to_card_id   uuid NOT NULL REFERENCES memory_cards(card_id) ON DELETE CASCADE,
    edge_type    text NOT NULL,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (from_card_id, to_card_id, edge_type)
);

CREATE TABLE IF NOT EXISTS memory_card_history (
    history_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    card_id    uuid REFERENCES memory_cards(card_id) ON DELETE SET NULL,
    edge_id    uuid REFERENCES memory_card_edges(edge_id) ON DELETE SET NULL,
    op         text NOT NULL,
    actor      text NOT NULL,
    "before"   jsonb,
    "after"    jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mc_anchors  ON memory_cards USING GIN (anchors);
CREATE INDEX IF NOT EXISTS idx_mc_tags     ON memory_cards USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_mc_types    ON memory_cards USING GIN (types);
CREATE INDEX IF NOT EXISTS idx_mc_status   ON memory_cards (status);
CREATE INDEX IF NOT EXISTS idx_mc_priority ON memory_cards (priority);
CREATE INDEX IF NOT EXISTS idx_mc_prov     ON memory_cards (provenance);
CREATE INDEX IF NOT EXISTS idx_mce_from    ON memory_card_edges (from_card_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_mce_to      ON memory_card_edges (to_card_id, edge_type);

-- Item 1c (2026-07-21 memory-cards substrate spec): structured, weighted
-- full-text scoring replacing the live-embedding cosine path
-- (cards_embedding.py, deleted). Weights map the original 2026-05-01
-- design's per-field ratios (anchor +2.0, title +1.0, summary +0.5,
-- tag +0.3) onto ts_rank_cd's four weight labels A/B/C/D -- see
-- services/orion-recall/app/cards_adapter.py's _TS_RANK_WEIGHTS comment for
-- the exact D,C,B,A ordering ts_rank_cd expects.
--
-- array_to_string() is catalogued STABLE (not IMMUTABLE) in this Postgres
-- build (confirmed live: `SELECT provolatile FROM pg_proc WHERE proname =
-- 'array_to_string'` returns 's'), so Postgres refuses it directly inside a
-- GENERATED column expression ("generation expression is not immutable").
-- It IS deterministic for a fixed text[] input and delimiter -- this local
-- wrapper simply asserts that via an explicit IMMUTABLE declaration, the
-- standard workaround for this exact class of over-conservative catalog
-- volatility marking.
CREATE OR REPLACE FUNCTION memory_cards_array_to_text(arr text[]) RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
        SELECT array_to_string(coalesce(arr, '{}'::text[]), ' ')
    $$;

ALTER TABLE memory_cards ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english'::regconfig, coalesce(anchor_class, '') || ' ' || memory_cards_array_to_text(anchors)), 'A') ||
        setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'B') ||
        setweight(to_tsvector('english'::regconfig, coalesce(summary, '')), 'C') ||
        setweight(to_tsvector('english'::regconfig, memory_cards_array_to_text(tags)), 'D')
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_mc_search_vector ON memory_cards USING GIN (search_vector);
