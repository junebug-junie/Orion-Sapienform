-- Memory Stage 2 PR B (2026-10-06): referents as substrate nodes + alias lifecycle.
-- Spec: docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md, sections 1.2-1.3, 2.3.
--
-- referent_alias is the source of truth for which substrate node a name means. A node exists
-- because it has a 'key' row (its writer key, e.g. project:hecate); its promotion state is that
-- row's state. Writer: orion/memory/referents/store.py (orion-durable-runs persist, and the
-- checkpoint backfill). Readers: the same resolution on the next persist, and the referent
-- projector in orion-memory-consolidation (builds the Falkor nodes).
-- referent_projection is the projector's own ledger: what it last wrote for each node/memory
-- (a fingerprint), so it rewrites only what changed. Truncating it forces a full, idempotent rebuild.
-- Apply AFTER manual_migration_episode_memory_v1.sql and BEFORE deploying orion-durable-runs or
-- orion-memory-consolidation from this branch. Additive only.
-- Rollback: manual_migration_referent_alias_v1_rollback.sql.

CREATE TABLE IF NOT EXISTS referent_alias (
    node_id TEXT NOT NULL,                 -- substrate node id (referent-<uuid5>)
    alias_norm TEXT NOT NULL,              -- lowercased, whitespace-collapsed, edge punctuation trimmed
    alias_text TEXT NOT NULL,
    alias_class TEXT NOT NULL,             -- key | name | descriptor
    referent_kind TEXT NOT NULL,           -- the writer kind of the node (person, place, ...)
    promotion_state TEXT NOT NULL,         -- proposed | provisional | canonical | rejected | deprecated
    admitted_by TEXT NOT NULL,             -- the rule that set the state (alias_grounding_v1, collision, ...)
    proposed_by TEXT NOT NULL,             -- episode_writer | checkpoint_backfill
    grounded_in TEXT,                      -- chat_prompt:<chat_history_log.correlation_id>
    valid_until TIMESTAMPTZ,               -- descriptors only: last use + 90 days
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (node_id, alias_norm),
    CONSTRAINT referent_alias_class_chk CHECK (alias_class IN ('key', 'name', 'descriptor')),
    CONSTRAINT referent_alias_state_chk CHECK (
        promotion_state IN ('proposed', 'provisional', 'canonical', 'rejected', 'deprecated'))
);
CREATE INDEX IF NOT EXISTS idx_referent_alias_live ON referent_alias (alias_norm)
    WHERE promotion_state IN ('provisional', 'canonical');

ALTER TABLE episode_memory_referent ADD COLUMN IF NOT EXISTS node_id TEXT;
CREATE INDEX IF NOT EXISTS idx_episode_memory_referent_node ON episode_memory_referent (node_id);

CREATE TABLE IF NOT EXISTS referent_projection (
    subject_id TEXT PRIMARY KEY,           -- a referent node id or an episode_memory id
    subject_kind TEXT NOT NULL,            -- node | memory
    fingerprint TEXT NOT NULL,
    projected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
