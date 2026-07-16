-- Drive audit slim table v3: composite (subject, window) index.
-- Producer: orion-spark-concept-induction (memory.drives.audit.v1 on orion:memory:drives:audit)
-- New consumer: services/orion-cortex-orch/app/mind_runtime.py's
-- fetch_drive_state_facet_for_mind() -- a bounded (MIND_DRIVE_STATE_FETCH_TIMEOUT_SEC,
-- default 0.4s) read of the single latest row for subject='orion', feeding Mind's
-- drive_state_compact evidence facet.
--
-- v1's idx_drive_audits_window indexes COALESCE(observed_at, created_at) DESC alone,
-- which serves the autonomy measurement gate's unfiltered windowed range scans. Mind's
-- read additionally filters WHERE subject = 'orion' before ordering/limiting -- under a
-- tight timeout budget that filter should be able to use an index directly rather than
-- falling back to a scan of idx_drive_audits_window filtered in-memory. This adds a
-- leading-subject composite covering that exact query shape
-- (WHERE subject = ? ORDER BY COALESCE(observed_at, created_at) DESC LIMIT 1).
--
-- idx_drive_audits_window (v1) is NOT superseded and is left in place: it is still the
-- correct index for the gate's subject-unfiltered scans.
--
-- On boot, sql-writer applies this same index creation via app/main.py lifespan
-- (CREATE INDEX IF NOT EXISTS), so pre-existing deployments are upgraded automatically.
-- This file is the standalone equivalent for manual application against a running
-- Postgres; it is a harmless no-op once applied.
--
-- Indexes are owned here (not via model index=True) to keep a single source of truth,
-- per the convention established in manual_migration_drive_audits_v1.sql /
-- services/orion-sql-writer/app/models/drive_audit.py.

CREATE INDEX IF NOT EXISTS idx_drive_audits_subject_window
    ON drive_audits (subject, (COALESCE(observed_at, created_at)) DESC);
