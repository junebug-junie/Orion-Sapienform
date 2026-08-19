-- ROADMAP D2 follow-through, 2026-08-19. Four missing indexes that were athena's I/O ceiling.
--
-- Every "latest row" lookup on these tables orders by created_at:
--
--     SELECT dispatch_frame_json ->> 'staleness_discard_count_ewma' ...
--     FROM substrate_execution_dispatch_frames ORDER BY created_at DESC LIMIT 1
--     (services/orion-execution-dispatch-runtime/app/store.py)
--
-- None of them had an index on created_at -- only on generated_at. So each of those lookups
-- became a parallel sequential scan plus a sort of the whole table, per tick.
--
-- Measured live with EXPLAIN (ANALYZE, BUFFERS) on the query above, 419,526 rows:
--
--     Buffers: shared hit=794,829 read=127,372   -- 7.2 GB touched, ~1 GB off disk
--     ... to return ONE row.
--
-- The natural control group is already in this database: substrate_reduction_receipts is the
-- only one of the five hot frame tables that HAS a created_at index, and it was reading
-- 13,342 tuples/sec against 335,532/sec for substrate_execution_dispatch_frames -- 25x quieter.
--
-- CONCURRENTLY because these are live tables under continuous write; it does not take an
-- ACCESS EXCLUSIVE lock. It cannot run inside a transaction block, so run this file with
-- psql directly (single-statement autocommit), not wrapped in BEGIN/COMMIT.
--
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_substrate_frames_created_at_index.sql
--
-- NOTE the `-i` and the SHELL redirect, not `-f`. The repo is not mounted into the sql-db
-- container (its only bind is the data directory), so `-f /path/...` resolves inside the
-- container and fails with "could not open file". Caught in review after the file shipped
-- with the unusable form.
--
-- IF NOT EXISTS so re-running is a no-op. If a CONCURRENTLY build is interrupted it leaves an
-- INVALID index behind -- check with:
--   select indexrelid::regclass from pg_index where not indisvalid;
-- and DROP INDEX + re-run if so.

\set ON_ERROR_STOP on

create index concurrently if not exists idx_substrate_execution_dispatch_frames_created_at
    on substrate_execution_dispatch_frames (created_at desc);

create index concurrently if not exists idx_substrate_proposal_frames_created_at
    on substrate_proposal_frames (created_at desc);

create index concurrently if not exists idx_substrate_policy_decision_frames_created_at
    on substrate_policy_decision_frames (created_at desc);

create index concurrently if not exists idx_substrate_feedback_frames_created_at
    on substrate_feedback_frames (created_at desc);
