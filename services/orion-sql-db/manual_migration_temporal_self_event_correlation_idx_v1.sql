-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Act consequence bets, Patch 1a (docs/superpowers/specs/2026-11-25-act-consequence-bets-design.md, PR #3953).
-- The act resolver joins each open output_used act to the experiences it made by correlation_id
-- (orion/grammar/output_used.py DUE_OUTPUT_USED_SQL). Without this index that join scans the table
-- (~81k rows, 93 MB on 2026-11-25) once per open act per 60 s sweep.
-- CONCURRENTLY: an index on an existing table must not take a write lock (post-pull auto-apply allowlist).
-- Applied live 2026-11-26 by hand in this form (valid=t).
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_temporal_self_event_correlation
    ON temporal_self_event (correlation_id)
 WHERE correlation_id IS NOT NULL;
