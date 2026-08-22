-- ROADMAP D2 follow-through, 2026-08-19. The THIRD and last pipeline stage.
--
-- Sibling of manual_migration_substrate_pending_markers.sql, which did proposal->policy and
-- dispatch->feedback. This does policy->dispatch (orion-execution-dispatch-runtime). After the
-- other two shipped, this stage's anti-join was the largest remaining sequential-scan source in
-- the database -- ~146,000 tuples/sec against each of the three frame tables.
--
-- THIS DOES NOT REVERSE THE 2026-07-30 DECISION, IT REMOVES ITS PREMISE.
-- That work (docs/superpowers/specs/2026-07-30-execution-dispatch-staleness-discard-design.md,
-- and the long docstring on load_oldest_policy_frames_without_dispatch) measured that for the
-- ASCENDING direction a `NOT EXISTS` rewrite is SLOWER than the hash anti-join -- 6+ seconds --
-- because "a huge prefix of already-processed ancient history sits before the real backlog
-- start, and a nested loop would have to walk hundreds of thousands of already-matched rows
-- before finding the first true miss". That analysis is correct, and it is an argument about
-- how best to scan.
--
-- A partial index on the marker has no already-processed prefix to walk: it contains ONLY
-- unprocessed rows. The batching fix from that patch stays exactly as it is and remains the
-- right shape for reading a backlog in chunks; what changes is that finding the chunk no longer
-- costs a full-table join.
--
-- HOW TO RUN (note `-i` and a shell redirect, not `-f`: the repo is not mounted into the
-- sql-db container):
--
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_policy_dispatch_pending_marker.sql
--
-- Re-running is a no-op. If a CONCURRENTLY build is interrupted it leaves an INVALID index:
--   select indexrelid::regclass from pg_index where not indisvalid;

\set ON_ERROR_STOP on

-- Instant on PG11+: a non-volatile DEFAULT does not rewrite the table.
alter table substrate_policy_decision_frames
    add column if not exists dispatch_pending boolean not null default true;

-- Index BEFORE the backfill, deliberately: if the backfill aborts, a half-applied migration is
-- then merely slow to drain rather than STRICTLY WORSE than the anti-join it replaces (which is
-- what happens when the column exists, every row is `true`, and no index does).
--
-- Both directions this stage reads -- oldest-first batches and the freshest single row -- are
-- served by this one index, forward and backward.
create index concurrently if not exists idx_substrate_policy_decision_frames_dispatch_pending
    on substrate_policy_decision_frames (generated_at)
    where dispatch_pending;

-- Backfill, batched: a single UPDATE over 423k rows creates that many dead tuples in one
-- transaction. Safe to re-run until it clears 0.
do $$
declare
    touched integer := 1;
    total integer := 0;
begin
    while touched > 0 loop
        update substrate_policy_decision_frames p
           set dispatch_pending = false
         where p.frame_id in (
               select p2.frame_id
                 from substrate_policy_decision_frames p2
                 join substrate_execution_dispatch_frames d
                   on d.source_policy_frame_id = p2.frame_id
                where p2.dispatch_pending
                limit 20000
         );
        get diagnostics touched = row_count;
        total := total + touched;
        commit;
    end loop;
    raise notice 'policy dispatch_pending backfill cleared % row(s)', total;
end $$;

-- VERIFY. Read this -- a partially-applied migration does not announce itself. `pending` should
-- be a handful of genuinely undispatched frames, NOT ~423k.
select 'policy dispatch pending' as what,
       count(*) filter (where dispatch_pending) as pending,
       count(*) as total
  from substrate_policy_decision_frames;

select indexrelid::regclass as index, indisvalid
  from pg_index
 where indexrelid::regclass::text = 'idx_substrate_policy_decision_frames_dispatch_pending';
