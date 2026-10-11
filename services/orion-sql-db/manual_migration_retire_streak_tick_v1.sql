-- DESTRUCTIVE: apply only with operator approval, AFTER both attention-runtime
-- and sql-writer have been rebuilt/restarted from the focus-run patch and the
-- new run path is verified. Export old history first if it must be retained.
-- No CASCADE: unexpected dependents must block retirement, not disappear.
-- ORION-MIGRATION-REQUIRED-BY: none -- destructive retire of superseded streak ticks; operator-approved only, never a deploy dependency
BEGIN;
SET LOCAL lock_timeout = '5s';
DROP TABLE IF EXISTS goal_provenance_streak_ticks;
COMMIT;
