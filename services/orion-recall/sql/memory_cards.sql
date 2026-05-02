-- Memory Cards v1 — idempotent DDL for conjourney Postgres
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
    before     jsonb,
    after      jsonb,
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
