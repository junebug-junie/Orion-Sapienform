-- ROADMAP D2 follow-through, 2026-08-19. The real fix for athena's I/O ceiling.
--
-- WHAT THIS REPLACES
-- Each substrate pipeline stage finds its next unit of work by asking "which row upstream has
-- no row downstream yet", written as an unbounded anti-join over both full tables. Measured
-- with EXPLAIN (ANALYZE, BUFFERS) on 419,526 rows: 106,052 blocks read (829 MB) plus 465 MB
-- spilled to temp, PER EXECUTION, every 2 seconds, forever.
--
-- WHY NOT JUST BOUND THE SCAN BY TIME (attempted first, reverted -- do not retry)
-- Because this pipeline legitimately runs hours to days behind. On 2026-08-14 the
-- dispatch->feedback stage produced 29,264 feedback frames for dispatch rows that had waited
-- ~34 HOURS, while 26,148 new rows arrived the same day; 8 of the last 30 days were entirely in
-- that regime. Over 7 days the lag is p50 124,613s, max 975,770s. Any "recent rows only" window
-- strands the backlog, and strands it permanently once fresh work keeps the fast path busy.
--
-- WHAT THIS DOES INSTEAD
-- A pending marker per stage, plus a PARTIAL index over only the pending rows. "Oldest
-- unprocessed" becomes O(pending) instead of O(history), which is correct no matter how far
-- behind the stage is -- the property the time bound could never have.
--
-- The partial indexes stay tiny by construction: they contain only rows still awaiting the next
-- stage, which is single digits in steady state and at most one backlog's worth otherwise.
--
-- SAFETY PROPERTIES
--  * DEFAULT TRUE, so a row that is new, or that predates this migration and has not been
--    backfilled yet, is treated as PENDING. The default fails toward doing the work twice,
--    never toward skipping it. Downstream writes are idempotent (ON CONFLICT (frame_id) DO
--    UPDATE), so a repeat is a no-op rather than a duplicate.
--  * The flag is cleared in the SAME TRANSACTION as the downstream insert, so a crash between
--    them cannot leave the flag cleared with no downstream row.
--  * A periodic reconciler (see the runtime services) re-sets the flag to true for any row that
--    lost it without a downstream row existing. It can only ADD work back, never remove it.
--
-- HOW TO RUN. Note `-i` and a shell redirect, not `-f`: the repo is not mounted into the sql-db
-- container, so `-f /path/...` resolves inside the container and fails to open.
--
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_substrate_pending_markers.sql
--
-- Re-running is a no-op: every statement is IF NOT EXISTS, and the backfill only ever clears
-- flags for rows that provably have a downstream row.
--
-- If a CONCURRENTLY build is interrupted it leaves an INVALID index behind. Check with:
--   select indexrelid::regclass from pg_index where not indisvalid;
-- and DROP INDEX + re-run.

\set ON_ERROR_STOP on

-- Instant on PG11+: a non-volatile DEFAULT does not rewrite the table.
alter table substrate_execution_dispatch_frames
    add column if not exists feedback_pending boolean not null default true;

alter table substrate_proposal_frames
    add column if not exists policy_pending boolean not null default true;

-- Backfill. Batched deliberately: a single UPDATE over 419,526 rows creates that many dead
-- tuples at once and holds one long transaction. This clears at most 20,000 rows per pass and
-- is safe to run repeatedly until it reports 0 -- each pass is its own transaction.
--
-- Run the two DO blocks below as many times as needed; they self-terminate when nothing is
-- left to clear.

do $$
declare
    touched integer := 1;
    total integer := 0;
begin
    while touched > 0 loop
        update substrate_execution_dispatch_frames d
           set feedback_pending = false
         where d.frame_id in (
               select d2.frame_id
                 from substrate_execution_dispatch_frames d2
                 join substrate_feedback_frames f
                   on f.source_execution_dispatch_frame_id = d2.frame_id
                where d2.feedback_pending
                limit 20000
         );
        get diagnostics touched = row_count;
        total := total + touched;
        commit;
    end loop;
    raise notice 'dispatch feedback_pending backfill cleared % row(s)', total;
end $$;

do $$
declare
    touched integer := 1;
    total integer := 0;
begin
    while touched > 0 loop
        update substrate_proposal_frames p
           set policy_pending = false
         where p.frame_id in (
               select p2.frame_id
                 from substrate_proposal_frames p2
                 join substrate_policy_decision_frames d
                   on d.source_proposal_frame_id = p2.frame_id
                where p2.policy_pending
                limit 20000
         );
        get diagnostics touched = row_count;
        total := total + touched;
        commit;
    end loop;
    raise notice 'proposal policy_pending backfill cleared % row(s)', total;
end $$;

-- The partial indexes. `generated_at` is the ordering column both stages use for oldest-first.
create index concurrently if not exists idx_substrate_execution_dispatch_frames_feedback_pending
    on substrate_execution_dispatch_frames (generated_at)
    where feedback_pending;

create index concurrently if not exists idx_substrate_proposal_frames_policy_pending
    on substrate_proposal_frames (generated_at)
    where policy_pending;
