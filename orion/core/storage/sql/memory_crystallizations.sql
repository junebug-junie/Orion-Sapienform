-- Memory crystallization canonical store (Postgres preserves canonical crystallizations)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS memory_crystallizations (
    crystallization_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind text NOT NULL,
    subject text NOT NULL,
    summary text NOT NULL,
    status text NOT NULL DEFAULT 'proposed',
    confidence text NOT NULL DEFAULT 'likely',
    salience numeric NOT NULL DEFAULT 0.5,
    dynamics jsonb NOT NULL DEFAULT '{}'::jsonb,
    scope text[] NOT NULL DEFAULT '{}',
    tags text[] NOT NULL DEFAULT '{}',
    grammar_envelope jsonb NOT NULL DEFAULT '{}'::jsonb,
    planning_effects text[] NOT NULL DEFAULT '{}',
    retrieval_affordances text[] NOT NULL DEFAULT '{}',
    governance jsonb NOT NULL DEFAULT '{}'::jsonb,
    projection_refs jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_card_ids text[] NOT NULL DEFAULT '{}',
    source_grammar_event_ids text[] NOT NULL DEFAULT '{}',
    source_atom_ids text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE memory_crystallizations ADD COLUMN IF NOT EXISTS dynamics jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS memory_crystallization_claims (
    claim_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    crystallization_id uuid NOT NULL REFERENCES memory_crystallizations(crystallization_id) ON DELETE CASCADE,
    claim text NOT NULL,
    status text NOT NULL DEFAULT 'active',
    confidence text NOT NULL DEFAULT 'likely',
    evidence_ref_ids text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_crystallization_sources (
    source_ref_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    crystallization_id uuid NOT NULL REFERENCES memory_crystallizations(crystallization_id) ON DELETE CASCADE,
    source_kind text NOT NULL,
    source_id text NOT NULL,
    excerpt text,
    strength numeric NOT NULL DEFAULT 0.5,
    note text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_crystallization_links (
    link_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    from_crystallization_id uuid NOT NULL REFERENCES memory_crystallizations(crystallization_id) ON DELETE CASCADE,
    to_crystallization_id uuid NOT NULL REFERENCES memory_crystallizations(crystallization_id) ON DELETE CASCADE,
    relation text NOT NULL,
    confidence numeric NOT NULL DEFAULT 0.5,
    note text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (from_crystallization_id, to_crystallization_id, relation)
);

CREATE TABLE IF NOT EXISTS memory_crystallization_history (
    history_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    crystallization_id uuid REFERENCES memory_crystallizations(crystallization_id) ON DELETE SET NULL,
    op text NOT NULL,
    actor text NOT NULL,
    before jsonb,
    after jsonb,
    reason text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_crystallization_projection_refs (
    projection_ref_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    crystallization_id uuid NOT NULL REFERENCES memory_crystallizations(crystallization_id) ON DELETE CASCADE,
    projection_kind text NOT NULL,
    external_id text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    synced_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (crystallization_id, projection_kind, external_id)
);

CREATE TABLE IF NOT EXISTS memory_crystallization_quarantine (
    quarantine_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    crystallization_id uuid REFERENCES memory_crystallizations(crystallization_id) ON DELETE SET NULL,
    reason text,
    errors jsonb NOT NULL DEFAULT '[]'::jsonb,
    actor text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_crystallization_retrieval_events (
    retrieval_event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query text NOT NULL,
    task_type text,
    project_id text,
    session_id text,
    crystallization_ids text[] NOT NULL DEFAULT '{}',
    card_refs text[] NOT NULL DEFAULT '{}',
    trace jsonb NOT NULL DEFAULT '{}'::jsonb,
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
