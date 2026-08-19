-- ROADMAP D2 follow-through, 2026-08-19. The remaining half of athena's I/O ceiling.
--
-- WHAT WAS FOUND
-- PR #1745 replaced the unbounded anti-joins with pending markers and cut whole-database
-- sequential reads from 1,021,558 to ~157,000 tuples/sec. What was left was almost entirely
-- ONE query, and it was not an anti-join at all -- it was a point lookup on an unindexed
-- column, which is why 14 rounds of pg_stat_activity sampling never named it.
--
--   SELECT proposal_frame_json FROM substrate_proposal_frames
--    WHERE source_field_tick_id = '<a different literal every tick>'
--    ORDER BY generated_at DESC LIMIT 1
--
-- This is the proposal stage's "have I already produced a frame for this tick?" dedup guard
-- (services/orion-proposal-runtime/app/store.py::load_proposal_frame_for_field_tick). Measured
-- with EXPLAIN (ANALYZE, BUFFERS) on the live table:
--
--   Parallel Seq Scan, 18,515 blocks read (144 MB), 148,138 rows discarded, 52 ms
--   -- at ~0.9 executions/sec, ~130 MB/sec of disk reads from this one guard.
--
-- Three sibling tables (substrate_attention_frames, substrate_execution_dispatch_frames,
-- substrate_policy_decision_frames) already index source_field_tick_id. substrate_proposal_frames
-- was simply missed, and nothing failed -- it just read the whole table forever instead.
--
-- WHY (source_field_tick_id, generated_at DESC) AND NOT source_field_tick_id ALONE
-- The query orders by generated_at DESC and takes 1. With the composite the planner gets the
-- row directly from the index; with the bare column it must still fetch and sort every matching
-- row. In steady state that is one row, so the difference is small -- but the guard exists
-- precisely to handle the case where a tick produced MORE than one frame, which is exactly when
-- the sort would not be free. The column is the cheap half of the key either way.
--
-- ALSO HERE: memory_crystallization_{sources,claims} are read by crystallization_id
-- (orion/memory/crystallization/repository.py::load) and have no index on it. Live counters:
-- 848,215 sequential scans against 6 index scans on sources. The tables are small (4,772 rows,
-- 1.3 MB) so this is not the ceiling, but the sibling table memory_crystallization_links IS
-- indexed on the same access path and shows 848,214 index scans -- so this is the same
-- omission, in the same loader, caught by the same sweep. Indexing it costs almost nothing and
-- removes 848k sequential scans.
--
-- HOW TO RUN. Note `-i` and a shell redirect, not `-f`: the repo is not mounted into the
-- sql-db container, so `-f /path/...` resolves inside the container and fails to open.
--
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_field_tick_lookup_indexes.sql
--
-- Every statement is IF NOT EXISTS, so re-running is a no-op. CONCURRENTLY means no write lock
-- is taken on these live tables; it also means an interrupted build leaves an INVALID index.
-- Check with:  select indexrelid::regclass from pg_index where not indisvalid;
-- and DROP INDEX + re-run if so.
--
-- NOTE: CREATE INDEX CONCURRENTLY cannot run inside a transaction block, so this file
-- deliberately has no BEGIN/COMMIT and no DO blocks.

\set ON_ERROR_STOP on

create index concurrently if not exists idx_substrate_proposal_frames_source_field_tick_id
    on substrate_proposal_frames (source_field_tick_id, generated_at desc);

create index concurrently if not exists idx_mcr_sources_crystallization_id
    on memory_crystallization_sources (crystallization_id);

create index concurrently if not exists idx_mcr_claims_crystallization_id
    on memory_crystallization_claims (crystallization_id);

-- VERIFY. A partially-applied migration is not self-announcing; read this output.
-- All three must be present with indisvalid = t.
select indexrelid::regclass as index, indisvalid
  from pg_index
 where indexrelid::regclass::text in (
        'idx_substrate_proposal_frames_source_field_tick_id',
        'idx_mcr_sources_crystallization_id',
        'idx_mcr_claims_crystallization_id')
 order by 1;
