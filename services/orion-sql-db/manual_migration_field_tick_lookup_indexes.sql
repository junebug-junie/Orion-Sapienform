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
-- substrate_policy_decision_frames) already index source_field_tick_id, so this was an omission
-- rather than a decision -- nothing failed, it just read the whole table forever instead.
--
-- Treat that consistency as context, not as the argument. Lifetime scan counts on those three
-- sibling indexes are 674,758 / 1 / 0: two of the three cited precedents are themselves barely
-- used, so "the siblings have it" proves convention, not value. The evidence is the 18,515-block
-- seq scan above.
--
-- WHY (source_field_tick_id, generated_at DESC) AND NOT source_field_tick_id ALONE
-- The query orders by generated_at DESC and takes 1. With the composite the planner gets the row
-- straight from the index; with the bare column it must fetch every match and add a Sort node.
-- Confirmed live: the proposal plan on the composite has NO Sort, while the attention plan on
-- its bare (source_field_tick_id) index does.
--
-- Note what this reasoning is NOT. An earlier draft of this file justified the composite by "the
-- guard exists to handle a tick that produced more than one frame." Live data flatly contradicts
-- that: max frames per tick is 1 on both tables across 544,401 ticks (avg 1.000), which makes
-- sense -- preventing a second frame is the guard's entire job. The composite is worth ~8 bytes
-- per entry to make the ORDER BY structurally free, not because the fanout case occurs.
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

-- substrate_reduction_receipts: the SAME defect as the attention lookup above, found by code
-- review of this patch rather than by the metrics sweep that found the other two.
-- services/orion-attention-runtime/app/store.py filtered
-- `receipt_json -> 'state_deltas' -> 0 ->> 'reducer_id'` while the table has carried a real
-- `reducer_name` column all along. Live equivalence check across all 9,345 rows: 0 disagreeing,
-- the 4,138 NULLs coincide exactly, and max(jsonb_array_length(state_deltas)) = 1 so the
-- `[0]` subscript in the JSON form cannot be hiding a second delta.
--
-- This one is NOT part of the I/O ceiling -- measured 3,248 of 3,254 buffers served from cache,
-- so it costs CPU rather than disk. It is fixed here anyway because the fix is the same one-line
-- swap plus this index, and because the table has no pruner bounding its growth: a seq scan that
-- is cache-resident at 9k rows is not cache-resident at 500k.
--
-- Both call sites filter reducer_name and range/order on created_at (one DESC, one ASC with a
-- cursor); a btree serves both directions from one index.
create index concurrently if not exists idx_substrate_reduction_receipts_reducer_name
    on substrate_reduction_receipts (reducer_name, created_at);

-- VERIFY. A partially-applied migration is not self-announcing; read this output.
-- All three must be present with indisvalid = t.
select indexrelid::regclass as index, indisvalid
  from pg_index
 where indexrelid::regclass::text in (
        'idx_substrate_proposal_frames_source_field_tick_id',
        'idx_mcr_sources_crystallization_id',
        'idx_mcr_claims_crystallization_id',
        'idx_substrate_reduction_receipts_reducer_name')
 order by 1;
