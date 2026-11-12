-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Consult outcomes recorded on Orion's beliefs (2026-11-12, design "Patch 1d").
-- Design: docs/superpowers/specs/2026-11-11-consult-claude-before-juniper-design.md.
--
-- A curiosity run is shown a consult outcome and asked to write one :Finding with a fixed id
-- (consult:<id>:answer | consult:<id>:decision) joined to a :Prior. At turn end Hub reads the
-- graph (read only) and stamps what it found here.
--   belief_offers: how many runs were shown this outcome (at most 3, then Hub stops asking).
--   belief_finding_id / belief_prior_ids: the finding found, and the priors joined to it.
--   belief_recorded_run_id / belief_recorded_at: the run that recorded it, and when Hub saw it.
--
-- Additive only. Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_orion_consult_belief_v1.sql
-- Rollback (additive): DROP INDEX orion_consult_told_run_idx; ALTER TABLE orion_consult DROP COLUMN belief_offers, DROP COLUMN belief_finding_id, DROP COLUMN belief_prior_ids, DROP COLUMN belief_recorded_run_id, DROP COLUMN belief_recorded_at;

ALTER TABLE orion_consult ADD COLUMN IF NOT EXISTS belief_offers INTEGER NOT NULL DEFAULT 0;
ALTER TABLE orion_consult ADD COLUMN IF NOT EXISTS belief_finding_id TEXT;
ALTER TABLE orion_consult ADD COLUMN IF NOT EXISTS belief_prior_ids TEXT[];
ALTER TABLE orion_consult ADD COLUMN IF NOT EXISTS belief_recorded_run_id TEXT;
ALTER TABLE orion_consult ADD COLUMN IF NOT EXISTS belief_recorded_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS orion_consult_told_run_idx ON orion_consult (orion_told_run_id);
