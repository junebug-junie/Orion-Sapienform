-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs
-- The day dream's grounding gate (2026-11-18 hardening, docs/superpowers/pr-reports/2026-11-18-dream-day-hardening-pr.md).
-- Contract: orion/schemas/dream_day.py DreamDayV1 (memories_unsupported, memories_unjudged, grounding).
--
-- Before a dream memory is written, the local judge checks it against the full text of the records it cites.
-- A memory the judge says is not shown by them (unsupported), or that it could not judge (failed call,
-- unparseable verdict), is never written. These columns keep the counts and every verdict (statement, refs,
-- status, unsupported claims, raw answer) on the night's row.
-- Writer: services/orion-durable-runs/app/dream_day_store.py (persist_dream_day). Readers:
-- services/orion-durable-runs/evals/run_dream_day_eval.py.
--
-- Additive and idempotent (post-pull auto-apply allowlist: ALTER TABLE ADD COLUMN; the applier sets lock_timeout). Must be applied BEFORE
-- orion-durable-runs restarts on this code: its INSERT names these columns.
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_dream_day_grounding.sql
-- Rollback: ALTER TABLE dream_day DROP COLUMN grounding, DROP COLUMN memories_unjudged, DROP COLUMN memories_unsupported;

ALTER TABLE dream_day ADD COLUMN IF NOT EXISTS memories_unsupported integer NOT NULL DEFAULT 0;
ALTER TABLE dream_day ADD COLUMN IF NOT EXISTS memories_unjudged integer NOT NULL DEFAULT 0;
ALTER TABLE dream_day ADD COLUMN IF NOT EXISTS grounding jsonb;
