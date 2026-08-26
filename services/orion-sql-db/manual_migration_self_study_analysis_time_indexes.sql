-- Time-column indexes for skills.self_study.analyze.v1's window reads.
-- 2026-08-25. ADDITIVE ONLY: three CREATE INDEX statements, no data touched,
-- no column or table altered. Safe to re-run (IF NOT EXISTS).
--
-- WHY. The action reads two adjacent windows per run, and EXPLAIN (ANALYZE,
-- BUFFERS) against the live database on 2026-08-25 showed three of its four
-- sources have no index on the column those windows are cut on, so each read
-- is a full sequential scan:
--
--   memory_crystallizations      created_at    Seq Scan  455 buffers   6.5 ms
--   vision_events                created_at    Seq Scan  410 buffers   9.3 ms
--   juniper_affective_state_log  observed_at   Seq Scan   81 buffers   1.0 ms
--   substrate_codebase_delta_log observed_at   Index Scan (already indexed)
--
-- Measured live the same day, the top dispatch templates run ~180x/hour. At
-- that rate this action alone would add on the order of 40,000 buffer reads
-- per hour of pure sequential scanning to a host whose Postgres I/O ceiling is
-- already a known constraint (see docs/superpowers/pr-reports/ for the
-- 2026-08-20 pg_stat_statements work).
--
-- SCOPE, after the review round. The action's OTHER two statements are already
-- fine and need nothing here: the cooldown lookup plans as
-- `Index Scan using idx_journal_entries_source_ref` (3 buffers, 0.161 ms), and
-- the source selector no longer touches Postgres at all -- an earlier version
-- of it scanned `journal_entries` (4,306 buffers, 21.9 ms, zero rows returned,
-- because there is no index on `source_kind`) on every single dispatch. That
-- query was removed outright rather than indexed around, since rotation state
-- belongs in a short-lived Redis mark, not in a 35k-row append-only table.
--
-- NOT REQUIRED for correctness. The action works without these -- 9 ms is not
-- slow in absolute terms, and its own per-connection statement_timeout is
-- 4,000 ms. This is a cost reduction, and it benefits every other reader of
-- these three tables, not only this action.
--
-- CONCURRENTLY so none of these takes a write lock. That means each statement
-- must run OUTSIDE a transaction block -- run this file with psql's default
-- autocommit, NOT wrapped in BEGIN/COMMIT.
--
-- Apply:
--   psql -h localhost -p 55432 -U postgres -d conjourney \
--     -f services/orion-sql-db/manual_migration_self_study_analysis_time_indexes.sql
--
-- Rollback (also additive-safe, no data loss):
--   DROP INDEX CONCURRENTLY IF EXISTS idx_mcr_created_at;
--   DROP INDEX CONCURRENTLY IF EXISTS ix_vision_events_created_at;
--   DROP INDEX CONCURRENTLY IF EXISTS ix_juniper_affective_state_log_observed_at;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mcr_created_at
    ON public.memory_crystallizations USING btree (created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_vision_events_created_at
    ON public.vision_events USING btree (created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_juniper_affective_state_log_observed_at
    ON public.juniper_affective_state_log USING btree (observed_at);
