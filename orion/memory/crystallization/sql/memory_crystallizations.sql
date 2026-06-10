-- Memory crystallization canonical storage (idempotent, applied at service boot).
-- Postgres preserves canonical crystallizations; Chroma/cards/Graphiti are
-- rebuildable projections.
--
-- crystallization_id is text ("crys_<hex>") to match the
-- MemoryCrystallizationV1 contract. The full serialized artifact is kept in
-- `doc` for lossless round-trips; child tables are queryable projections of
-- the same artifact and are rebuilt on every write.

CREATE TABLE IF NOT EXISTS memory_crystallizations (
    crystallization_id text PRIMARY KEY,
    kind text NOT NULL,
    subject text NOT NULL,
    summary text NOT NULL,
    status text NOT NULL DEFAULT 'proposed',
    confidence text NOT NULL DEFAULT 'likely',
    salience numeric NOT NULL DEFAULT 0.5,
    scope text[] NOT NULL DEFAULT '{}',
    tags text[] NOT NULL DEFAULT '{}',
    grammar_envelope jsonb NOT NULL DEFAULT '{}'::jsonb,
    planning_effects text[] NOT NULL DEFAULT '{}',
    retrieval_affordances text[] NOT NULL DEFAULT '{}',
    governance jsonb NOT NULL DEFAULT '{}'::jsonb,
    projection_refs jsonb NOT NULL DEFAULT '{}'::jsonb,
    doc jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_crystallization_claims (
    claim_id text NOT NULL,
    crystallization_id text NOT NULL REFERENCES memory_crystallizations(crystallization_id) ON DELETE CASCADE,
    claim text NOT NULL,
    status text NOT NULL DEFAULT 'active',
    confidence text NOT NULL DEFAULT 'likely',
    evidence_ref_ids text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (crystallization_id, claim_id)
);

CREATE TABLE IF NOT EXISTS memory_crystallization_sources (
    source_ref_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    crystallization_id text NOT NULL REFERENCES memory_crystallizations(crystallization_id) ON DELETE CASCADE,
    source_kind text NOT NULL,
    source_id text NOT NULL,
    excerpt text,
    strength numeric NOT NULL DEFAULT 0.5,
    note text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_crystallization_links (
    link_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    from_crystallization_id text NOT NULL REFERENCES memory_crystallizations(crystallization_id) ON DELETE CASCADE,
    to_crystallization_id text NOT NULL,
    relation text NOT NULL,
    confidence numeric NOT NULL DEFAULT 0.5,
    note text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (from_crystallization_id, to_crystallization_id, relation)
);

CREATE TABLE IF NOT EXISTS memory_crystallization_history (
    history_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    crystallization_id text REFERENCES memory_crystallizations(crystallization_id) ON DELETE SET NULL,
    op text NOT NULL,
    actor text NOT NULL,
    before jsonb,
    after jsonb,
    reason text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_crystallization_retrieval_events (
    retrieval_event_id text PRIMARY KEY,
    query text NOT NULL,
    packet jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mcr_status ON memory_crystallizations (status);
CREATE INDEX IF NOT EXISTS idx_mcr_kind ON memory_crystallizations (kind);
CREATE INDEX IF NOT EXISTS idx_mcr_salience ON memory_crystallizations (salience DESC);
CREATE INDEX IF NOT EXISTS idx_mcr_scope ON memory_crystallizations USING GIN (scope);
CREATE INDEX IF NOT EXISTS idx_mcr_tags ON memory_crystallizations USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_mcr_grammar ON memory_crystallizations USING GIN (grammar_envelope);
CREATE INDEX IF NOT EXISTS idx_mcr_sources ON memory_crystallization_sources (source_kind, source_id);
CREATE INDEX IF NOT EXISTS idx_mcr_links_from ON memory_crystallization_links (from_crystallization_id, relation);
CREATE INDEX IF NOT EXISTS idx_mcr_links_to ON memory_crystallization_links (to_crystallization_id, relation);
CREATE INDEX IF NOT EXISTS idx_mcr_history_cid ON memory_crystallization_history (crystallization_id, created_at DESC);
