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
ALTER TABLE memory_crystallizations ADD COLUMN IF NOT EXISTS provenance jsonb NOT NULL DEFAULT '{}'::jsonb;

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

-- Decision log for maybe_resolve_concept_relation() (orion/memory/crystallization/
-- concept_relation.py). One row per real LLM classification call, regardless of which
-- way the confidence-floor / relation branch below it goes -- previously only the
-- decisive outcome reached a log line, so every "unrelated" decision and every
-- sub-floor "contradicts"/"refines" decision vanished silently. `digested` is a simple
-- watermark for scripts/concept_relation_digest.py so repeated runs don't reprocess
-- rows (no separate cursor-state table).
CREATE TABLE IF NOT EXISTS memory_concept_relation_decisions (
    decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_crystallization_id text NOT NULL,
    target_crystallization_id text,
    relation text NOT NULL,
    confidence numeric NOT NULL,
    floor_cleared boolean NOT NULL,
    decided_at timestamptz NOT NULL DEFAULT now(),
    digested boolean NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_mcrd_digested ON memory_concept_relation_decisions (digested, decided_at);

CREATE INDEX IF NOT EXISTS idx_mcr_status ON memory_crystallizations (status);
CREATE INDEX IF NOT EXISTS idx_mcr_kind ON memory_crystallizations (kind);
CREATE INDEX IF NOT EXISTS idx_mcr_salience ON memory_crystallizations (salience DESC);
CREATE INDEX IF NOT EXISTS idx_mcr_scope ON memory_crystallizations USING GIN (scope);
CREATE INDEX IF NOT EXISTS idx_mcr_tags ON memory_crystallizations USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_mcr_grammar ON memory_crystallizations USING GIN (grammar_envelope);
CREATE INDEX IF NOT EXISTS idx_mcr_sources ON memory_crystallization_sources (source_kind, source_id);
CREATE INDEX IF NOT EXISTS idx_mcr_links_from ON memory_crystallization_links (from_crystallization_id, relation);
CREATE INDEX IF NOT EXISTS idx_mcr_links_to ON memory_crystallization_links (to_crystallization_id, relation);

-- One-time cleanup for rows written before the dedup fix (WindowStore.append_turn
-- appending a second turn_correlation_ids entry for a reclassified turn, which
-- fanned out into duplicate chat_turn AND grammar_event evidence rows -- 571
-- duplicate groups confirmed live 2026-08-20). apply_memory_crystallizations_schema()
-- executes this whole file, autocommit, on EVERY orion-hub / orion-memory-crystallizer
-- boot -- so the DELETE self-join is gated on the target index not existing yet,
-- rather than running unconditionally forever. Once the index below exists, no
-- duplicate can ever be written again (ON CONFLICT DO NOTHING in
-- insert_crystallization()), so the cleanup can never have new work after its
-- first successful run; without this guard, every future restart would still
-- pay a full self-join over the table plus a RowExclusiveLock against the same
-- table insert_crystallization() writes to concurrently, for a result
-- guaranteed empty.
--
-- Tie-break note: this keeps the earliest (created_at, source_ref_id) row of
-- each group. Duplicate evidence rows for one crystallization are inserted
-- inside a single transaction, so created_at is normally identical between
-- them and the real tie-break is source_ref_id -- a random UUID, uncorrelated
-- with which turn snapshot was newer. For grammar_event duplicates (516 of the
-- 571 groups) the rows are byte-identical so this is moot. For chat_turn
-- duplicates (55 groups) the only field that can differ is the display-only
-- `note` (e.g. "shift=TOPIC" vs "shift=STANCE"), so this is a one-time,
-- cosmetic, arbitrary pick among historical rows -- not a systematic bias, and
-- not worth a real ordering column for data this doesn't recur going forward.
DO $$
BEGIN
    IF to_regclass('idx_mcr_sources_unique') IS NULL THEN
        DELETE FROM memory_crystallization_sources newer
        USING memory_crystallization_sources older
        WHERE newer.crystallization_id = older.crystallization_id
          AND newer.source_kind = older.source_kind
          AND newer.source_id = older.source_id
          AND (newer.created_at, newer.source_ref_id) > (older.created_at, older.source_ref_id);
    END IF;
END $$;

-- Enforces at most one evidence row per (crystallization, source) going forward.
-- A plain unique index (not a named constraint) so ON CONFLICT in
-- insert_crystallization() can target it without needing psycopg2-side
-- constraint introspection. Must run after the cleanup DELETE above, or the
-- CREATE would fail on the very duplicates it exists to prevent.
CREATE UNIQUE INDEX IF NOT EXISTS idx_mcr_sources_unique ON memory_crystallization_sources (crystallization_id, source_kind, source_id);
