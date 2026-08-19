-- Movement III (docs/superpowers/specs/2026-08-12-perception-frontier-design.md):
-- a reverie thought's optional falsifiable expectation about the room, plus its
-- eventual scoring verdict. Additive backfill on top of
-- manual_migration_substrate_reverie_thought.sql -- ALTER TABLE ADD COLUMN IF NOT
-- EXISTS is a no-op on a deployment where these already exist, same pattern as
-- manual_migration_attention_salience_trace.sql's `description` backfill.
-- Apply before enabling ORION_REVERIE_EXPECTATION_SCORING_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_reverie_thought_expectation.sql

alter table substrate_reverie_thought add column if not exists expectation text;
alter table substrate_reverie_thought add column if not exists expectation_checkable_by timestamptz;
alter table substrate_reverie_thought add column if not exists expectation_verdict text;
alter table substrate_reverie_thought add column if not exists expectation_scored_at timestamptz;

-- Backs load_pending_expectations()'s "most-overdue-first, window closed,
-- unresolved" scan (app/store.py).
create index if not exists idx_substrate_reverie_thought_expectation_pending
    on substrate_reverie_thought (expectation_checkable_by)
    where expectation is not null and expectation_verdict is null;
