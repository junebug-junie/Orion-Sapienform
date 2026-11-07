-- ORION-MIGRATION-REQUIRED-BY: none drops the tables of the retired feedback (L10) and consolidation (L11) rungs
-- DESTRUCTIVE
--
-- Retire the feedback and consolidation rungs, step 2 of 2: drop their tables.
-- Spec: docs/superpowers/specs/2026-11-04-retire-feedback-and-consolidation-rungs.md (Juniper, 2026-11-04:
-- "drop 2 days after", with a backup first). Applied BY HAND only; the post-pull applier never runs a
-- DESTRUCTIVE file.
--
-- NOT BEFORE: 2 days after the retirement PR is deployed (both services stopped and removed, every
-- reader rebuilt). Check first that nothing has written since the deploy:
--   docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select max(generated_at) from substrate_feedback_frames; select max(generated_at) from substrate_consolidation_frames"
--   docker ps --format '{{.Names}}' | grep -E 'feedback-runtime|consolidation-runtime'   (must print nothing)
--
-- BACKUP FIRST (about 10 GB before compression), then apply:
--   mkdir -p /mnt/storage-warm/orion-backups
--   docker exec orion-athena-sql-db pg_dump -U postgres -d conjourney -Fc -t substrate_feedback_frames -t substrate_feedback_control_reductions -t substrate_consolidation_frames -t substrate_expectations -t substrate_schema_candidates -t substrate_tensor_slices > /mnt/storage-warm/orion-backups/retired-feedback-consolidation-rungs.dump
--   docker exec orion-athena-sql-db pg_restore --list /dev/stdin < /mnt/storage-warm/orion-backups/retired-feedback-consolidation-rungs.dump | grep -c 'TABLE DATA'   (must print 6)
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_retire_feedback_consolidation_rungs_drop.sql
--
-- Kept on purpose: substrate_signal_control_cells, substrate_action_outcomes and
-- substrate_action_effect_posterior (the world-action settle scorer in orion-sql-writer and the act
-- ledger still write them). No foreign key or view depends on the dropped tables (checked live
-- 2026-11-04).

-- The dispatch frames' feedback marker: a column DEFAULT TRUE that only the feedback runtime cleared.
-- Its partial index first, concurrently (cannot run inside a transaction).
DROP INDEX CONCURRENTLY IF EXISTS idx_substrate_execution_dispatch_frames_feedback_pending;

BEGIN;
SET LOCAL lock_timeout = '10s';
DROP TABLE IF EXISTS substrate_feedback_frames;
DROP TABLE IF EXISTS substrate_feedback_control_reductions;
DROP TABLE IF EXISTS substrate_consolidation_frames;
DROP TABLE IF EXISTS substrate_expectations;
DROP TABLE IF EXISTS substrate_schema_candidates;
DROP TABLE IF EXISTS substrate_tensor_slices;
ALTER TABLE substrate_execution_dispatch_frames DROP COLUMN IF EXISTS feedback_pending;
COMMIT;
