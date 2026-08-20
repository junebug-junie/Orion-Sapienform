-- Per-table autovacuum tuning for the high-churn retention tables, 2026-08-20.
--
-- THE PROBLEM
-- The cluster runs the stock global settings:
--
--   autovacuum_vacuum_scale_factor = 0.2      (20% of the table must be dead first)
--   autovacuum_analyze_scale_factor = 0.1
--   autovacuum_naptime = 1min
--   autovacuum_max_workers = 3
--
-- A 20% PROPORTIONAL threshold is the wrong instrument for a table that is large AND
-- churning, because the trigger point scales with the table. Measured live 2026-08-20:
--
--   table                              live       dead    %dead   last autovacuum
--   grammar_events                  6,068,177   572,758     8.6   2026-08-20 13:40 (~3h ago)
--   grammar_atoms                     697,952   125,744    15.3   2026-08-20 15:29
--   grammar_edges                     839,131   100,000    10.6   2026-08-20 16:27
--   substrate_execution_dispatch_..   471,892    51,822     9.9   2026-08-19 06:04 (~34h ago)
--   substrate_proposal_frames         474,319    51,417     9.8   2026-08-19 06:04 (~34h ago)
--   substrate_policy_decision_frames  474,414    50,854     9.7   2026-08-19 06:35 (~34h ago)
--   substrate_organ_emissions         530,994    33,689     6.0   2026-08-20 12:03
--
-- grammar_events must accumulate ~1.21 MILLION dead tuples before autovacuum will look at
-- it. An earlier sample (2026-08-17) caught it sitting on 1,564,122. Those dead tuples are
-- not free: every sequential scan and every index range scan walks them, and the visibility
-- map goes stale, which is what turns an Index Only Scan back into a heap-fetching one.
--
-- This matters more now than it did last week. Bounded retention (see
-- services/orion-sql-writer/app/grammar_truth.py) deletes from these tables every 60
-- seconds, so DELETE traffic is continuous rather than occasional. Retention that creates
-- dead tuples faster than autovacuum reclaims them just moves the problem.
--
-- WHAT THIS DOES
-- Replaces the proportional trigger with a mostly-ABSOLUTE one on these seven tables:
--
--   scale_factor 0.01 + threshold 10000  ->  vacuum at ~10k + 1% dead
--
-- For grammar_events that is ~70k dead instead of ~1.21M: roughly 17x more often, each pass
-- doing roughly 17x less work. Same total work, spread out. Analyze gets the same treatment
-- so the planner keeps current row counts on a table that is about to lose 60% of its rows
-- to the new substrate_proposal_frames window -- stale stats after a large delete is its own
-- class of bad plan.
--
-- WHAT THIS DELIBERATELY DOES NOT DO
-- It does not touch autovacuum_vacuum_cost_limit or autovacuum_vacuum_cost_delay. athena has
-- a known I/O ceiling (see docs/superpowers/pr-reports/2026-08-19-*), and un-throttling
-- vacuum is exactly the wrong move there. Smaller, more frequent passes are the cheaper half
-- of this trade; making each pass faster is not requested here.
--
-- It also leaves substrate_reduction_receipts alone, which carries an explicit
-- autovacuum_enabled=false. That is somebody's deliberate decision and is not this file's
-- business to reverse.
--
-- SAFETY
-- ALTER TABLE ... SET (storage parameters) is a catalog update. It does NOT rewrite the
-- table, does not lock it for readers, and takes effect on the next autovacuum cycle. It is
-- fully reversible with ALTER TABLE ... RESET (...), which restores the global defaults --
-- see the rollback block at the bottom of this file.
--
-- HOW TO RUN. Note `-i` and a shell redirect, not `-f`: the repo is not mounted into the
-- sql-db container, so `-f /path/...` resolves inside the container and fails to open.
--
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_autovacuum_high_churn_tables.sql
--
-- Re-running is a no-op: SET on an already-set parameter just writes the same value.

\set ON_ERROR_STOP on

alter table grammar_events set (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_vacuum_threshold = 10000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 10000
);

alter table grammar_atoms set (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_vacuum_threshold = 10000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 10000
);

alter table grammar_edges set (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_vacuum_threshold = 10000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 10000
);

alter table grammar_traces set (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_vacuum_threshold = 10000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 10000
);

alter table substrate_proposal_frames set (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_vacuum_threshold = 10000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 10000
);

alter table substrate_policy_decision_frames set (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_vacuum_threshold = 10000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 10000
);

alter table substrate_execution_dispatch_frames set (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_vacuum_threshold = 10000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 10000
);

alter table substrate_organ_emissions set (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_vacuum_threshold = 10000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 10000
);

-- Verify: every tuned table should list its reloptions here.
select relname, reloptions
  from pg_class
 where relname in (
       'grammar_events', 'grammar_atoms', 'grammar_edges', 'grammar_traces',
       'substrate_proposal_frames', 'substrate_policy_decision_frames',
       'substrate_execution_dispatch_frames', 'substrate_organ_emissions'
 )
 order by relname;

-- ROLLBACK (run by hand if these settings turn out wrong; restores the global defaults):
--
--   alter table grammar_events                      reset (autovacuum_vacuum_scale_factor,
--       autovacuum_vacuum_threshold, autovacuum_analyze_scale_factor, autovacuum_analyze_threshold);
--   ... and the same for the other seven tables.
