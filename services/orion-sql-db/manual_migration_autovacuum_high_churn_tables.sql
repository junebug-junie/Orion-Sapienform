-- Per-table autovacuum tuning for the high-churn retention tables, 2026-08-20.
--
-- THE PROBLEM
-- The cluster runs the stock global settings:
--
--   autovacuum_vacuum_scale_factor  = 0.2   (20% of the table must be dead first)
--   autovacuum_vacuum_threshold     = 50
--   autovacuum_analyze_scale_factor = 0.1
--   autovacuum_naptime              = 1min
--   autovacuum_max_workers          = 3
--   autovacuum_vacuum_cost_limit    = -1    (inherits vacuum_cost_limit = 200)
--   autovacuum_vacuum_cost_delay    = 2ms
--   shared_buffers                  = 128MB
--
-- A 20% PROPORTIONAL threshold is the wrong instrument for a table that is large AND
-- churning, because the trigger point scales with the table. Measured live 2026-08-20:
--
--   table                                  live rows      dead   last autovacuum
--   grammar_events                         5,903,115   572,758   2026-08-20 13:40 (~3h ago)
--   grammar_edges                            737,852   100,000   2026-08-20 16:27
--   grammar_atoms                            712,078   125,744   2026-08-20 15:29
--   substrate_organ_emissions                552,595    33,689   2026-08-20 12:03
--   substrate_execution_dispatch_frames      472,512    51,822   2026-08-19 06:04 (~34h ago)
--   substrate_policy_decision_frames         471,850    50,854   2026-08-19 06:35 (~34h ago)
--   substrate_proposal_frames                430,744    51,417   2026-08-19 06:04 (~34h ago)
--   grammar_traces                            81,971     1,559   2026-08-20 16:33
--
-- grammar_events must accumulate ~1.18 MILLION dead tuples before autovacuum will look at
-- it. An earlier sample (2026-08-17) caught it sitting on 1,564,122. Those dead tuples are
-- not free: every sequential scan and every index range scan walks them, and the visibility
-- map goes stale, which is what turns an Index Only Scan back into a heap-fetching one.
--
-- This matters more now than it did last week. Bounded retention (see
-- services/orion-sql-writer/app/grammar_truth.py) deletes from these tables every 60
-- seconds, so DELETE traffic is continuous rather than occasional.
--
-- WHAT THIS DOES
--   autovacuum_vacuum_scale_factor  0.05  + threshold 3000
--   autovacuum_analyze_scale_factor 0.05  + threshold 3000
--
-- Computed against live row counts, that is a 2.3x to 4.0x increase in vacuum frequency:
--
--   table                                  old trigger   new trigger   ratio
--   grammar_events                           1,180,673       298,156   3.96x
--   grammar_edges                              147,620        39,893   3.70x
--   grammar_atoms                              142,466        38,604   3.69x
--   substrate_organ_emissions                  110,569        30,630   3.61x
--   substrate_execution_dispatch_frames         94,552        26,626   3.55x
--   substrate_policy_decision_frames            94,420        26,593   3.55x
--   substrate_proposal_frames                   86,199        24,537   3.51x
--   grammar_traces                              16,444         7,099   2.32x
--
-- WHY 0.05 AND NOT SOMETHING MORE AGGRESSIVE. An earlier draft of this file used
-- scale_factor 0.01 + threshold 10000 (a ~17x increase on grammar_events) and justified it
-- as "same total work, spread out". THAT REASONING WAS WRONG and code review caught it.
-- Heap work scales with dead tuples, but INDEX vacuum cost does not -- each pass scans the
-- indexes at a cost proportional to INDEX SIZE, not to how many dead tuples prompted it.
-- Live index sizes: grammar_edges 3442 MB, grammar_events 2756 MB, grammar_atoms 1184 MB.
-- Multiplying pass count by 17 multiplies that index-scan I/O by 17 as well -- on the order
-- of +45 GB/day on grammar_events alone. Postgres' INDEX_CLEANUP bypass only applies below a
-- small dead-item fraction, which continuous 60s DELETE traffic will not hold.
--
-- athena is I/O-stalled roughly 22% of wall time (see
-- docs/superpowers/pr-reports/2026-08-19-*). ~3.5x is a deliberate compromise: it cuts the
-- dead-tuple ceiling by the same factor while keeping the added index-scan I/O bounded. It
-- is NOT a measured optimum. If dead tuples still climb, measure the I/O cost before
-- lowering scale_factor further.
--
-- TOAST GETS ITS OWN SETTINGS, BECAUSE MOST OF THE BYTES ARE THERE. Live:
--
--   table                                 indexes    TOAST
--   substrate_proposal_frames              144 MB   1471 MB   (84% of the relation)
--   substrate_organ_emissions              257 MB   1083 MB
--   substrate_policy_decision_frames       261 MB    762 MB
--   substrate_execution_dispatch_frames    464 MB    632 MB
--   grammar_events / grammar_edges / grammar_traces        ~0
--
-- TOAST relations have their own autovacuum settings (`toast.autovacuum_*`) and inherit
-- NOTHING from the main table's. All four were sitting on reloptions = NULL. Pruning 240k
-- proposal rows produces on the order of a gigabyte of dead TOAST chunks, so leaving TOAST
-- on the stock 0.2 trigger would have skipped the very relation this file exists to fix.
-- TOAST relations are never ANALYZEd, so only the vacuum parameters apply.
--
-- WHAT THIS DELIBERATELY DOES NOT DO
-- It does not touch autovacuum_vacuum_cost_limit or autovacuum_vacuum_cost_delay. See the
-- I/O ceiling note above -- un-throttling vacuum is the wrong move on this host.
--
-- It leaves substrate_reduction_receipts alone, which carries an explicit
-- autovacuum_enabled=false. That is somebody's deliberate decision and not this file's
-- business to reverse.
--
-- grammar_traces is included for consistency with its siblings, but note it is the weakest
-- case in the set: at 81,971 rows the absolute threshold dominates and the change is only
-- 2.32x. Harmless, but do not cite it as evidence this file did anything.
--
-- SAFETY
-- ALTER TABLE ... SET (storage parameters) takes SHARE UPDATE EXCLUSIVE. It does NOT rewrite
-- the table and does not block readers or writers. It takes effect on the next autovacuum
-- cycle, and is fully reversible with ALTER TABLE ... RESET (...) -- see the rollback block
-- at the bottom.
--
-- HOW TO RUN. Note `-i` and a shell redirect, not `-f`: the repo is not mounted into the
-- sql-db container, so `-f /path/...` resolves inside the container and fails to open.
--
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_autovacuum_high_churn_tables.sql
--
-- Re-running is a no-op: SET on an already-set parameter just writes the same value.

\set ON_ERROR_STOP on

-- The three grammar tables with no meaningful TOAST.
alter table grammar_events set (
    autovacuum_vacuum_scale_factor = 0.05, autovacuum_vacuum_threshold = 3000,
    autovacuum_analyze_scale_factor = 0.05, autovacuum_analyze_threshold = 3000
);
alter table grammar_atoms set (
    autovacuum_vacuum_scale_factor = 0.05, autovacuum_vacuum_threshold = 3000,
    autovacuum_analyze_scale_factor = 0.05, autovacuum_analyze_threshold = 3000
);
alter table grammar_edges set (
    autovacuum_vacuum_scale_factor = 0.05, autovacuum_vacuum_threshold = 3000,
    autovacuum_analyze_scale_factor = 0.05, autovacuum_analyze_threshold = 3000
);
alter table grammar_traces set (
    autovacuum_vacuum_scale_factor = 0.05, autovacuum_vacuum_threshold = 3000,
    autovacuum_analyze_scale_factor = 0.05, autovacuum_analyze_threshold = 3000
);

-- The four substrate tables, where most of the bytes live in TOAST.
alter table substrate_proposal_frames set (
    autovacuum_vacuum_scale_factor = 0.05, autovacuum_vacuum_threshold = 3000,
    autovacuum_analyze_scale_factor = 0.05, autovacuum_analyze_threshold = 3000,
    toast.autovacuum_vacuum_scale_factor = 0.05, toast.autovacuum_vacuum_threshold = 3000
);
alter table substrate_policy_decision_frames set (
    autovacuum_vacuum_scale_factor = 0.05, autovacuum_vacuum_threshold = 3000,
    autovacuum_analyze_scale_factor = 0.05, autovacuum_analyze_threshold = 3000,
    toast.autovacuum_vacuum_scale_factor = 0.05, toast.autovacuum_vacuum_threshold = 3000
);
alter table substrate_execution_dispatch_frames set (
    autovacuum_vacuum_scale_factor = 0.05, autovacuum_vacuum_threshold = 3000,
    autovacuum_analyze_scale_factor = 0.05, autovacuum_analyze_threshold = 3000,
    toast.autovacuum_vacuum_scale_factor = 0.05, toast.autovacuum_vacuum_threshold = 3000
);
alter table substrate_organ_emissions set (
    autovacuum_vacuum_scale_factor = 0.05, autovacuum_vacuum_threshold = 3000,
    autovacuum_analyze_scale_factor = 0.05, autovacuum_analyze_threshold = 3000,
    toast.autovacuum_vacuum_scale_factor = 0.05, toast.autovacuum_vacuum_threshold = 3000
);

-- Verify. NOTE the join to the TOAST relation: `toast.autovacuum_*` does NOT appear in the
-- main table's own reloptions -- it is stored on the TOAST relation's reloptions. Checking
-- only pg_class.reloptions for the main table would report the TOAST settings as missing
-- when they applied fine.
select t.relname,
       t.reloptions                      as table_reloptions,
       coalesce(tt.relname, '(none)')    as toast_rel,
       tt.reloptions                     as toast_reloptions
  from pg_class t
  left join pg_class tt on tt.oid = t.reltoastrelid
 where t.relname in (
       'grammar_events', 'grammar_atoms', 'grammar_edges', 'grammar_traces',
       'substrate_proposal_frames', 'substrate_policy_decision_frames',
       'substrate_execution_dispatch_frames', 'substrate_organ_emissions'
 )
 order by t.relname;

-- ROLLBACK (run by hand if these settings turn out wrong; restores the global defaults):
--
--   alter table <t> reset (
--       autovacuum_vacuum_scale_factor, autovacuum_vacuum_threshold,
--       autovacuum_analyze_scale_factor, autovacuum_analyze_threshold,
--       toast.autovacuum_vacuum_scale_factor, toast.autovacuum_vacuum_threshold);
--
-- ... for each of the eight tables above (the two toast.* names are simply ignored on the
-- four tables that never had them set).
