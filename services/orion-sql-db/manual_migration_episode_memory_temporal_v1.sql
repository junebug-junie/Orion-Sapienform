-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs
-- Memory corrections, rev 2 (issue #3659, #3660): bi-temporal validity, the precision eval's
-- verdict, and one append-only audit row per supersede run.
--
-- 1. episode_memory gets when a row stopped being current and why (Zep/Graphiti-style valid time):
--      invalid_at      when the fact stopped being true. update: the newer memory's occurred_at
--                      (the old fact WAS true until then). correction: the old row's own time
--                      (it was never true).
--      invalidated_by  the newer memory_id that ended it.
--      invalid_reason  'update' | 'correction'.
--    Written by orion/memory/episode/supersede.py apply_link together with status 'superseded', so
--    every reader that serves status = 'active' keeps serving only what is currently true.
--    "What was true then" reads orion/memory/lived_memory.EPISODE_VALID_AT_SQL. Undo clears all three.
-- 2. memory_supersede_eval_run: one row per precision eval (scripts/memory_supersede_eval.py
--    run --record-db). Write-time apply is open only while the latest finished row for the current
--    check version and vote count has passed = true and is under 30 days old (supersede.eval_gate).
-- 3. memory_supersede_run: one row per supersede run (the nightly distill's check and each pass
--    dry-run/apply), never updated: ids, verdicts, votes, which candidate legs fired, and the
--    model's one-sentence reasons (which may paraphrase a statement; never the statements themselves).
--
-- Additive only. Readable by orion_readonly after
-- scripts/sql/2026-11-18_grant_orion_readonly_supersede_runs.sql (applied by hand).
-- Rollback: manual_migration_episode_memory_temporal_v1_rollback.sql.

ALTER TABLE episode_memory ADD COLUMN IF NOT EXISTS invalid_at TIMESTAMPTZ;
ALTER TABLE episode_memory ADD COLUMN IF NOT EXISTS invalidated_by UUID;
ALTER TABLE episode_memory ADD COLUMN IF NOT EXISTS invalid_reason TEXT;

CREATE TABLE IF NOT EXISTS memory_supersede_eval_run (
    run_id TEXT PRIMARY KEY,
    check_version TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    passed BOOLEAN NOT NULL,
    n_items INTEGER NOT NULL,
    n_real INTEGER NOT NULL,
    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    false_acts INTEGER NOT NULL,
    hard_negative_false_acts INTEGER NOT NULL,
    errors INTEGER NOT NULL,
    route TEXT,
    votes INTEGER,
    model TEXT,                           -- the model id the gateway reported for the route
    summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- (Also for a copy hand-applied before the column was added.)
ALTER TABLE memory_supersede_eval_run ADD COLUMN IF NOT EXISTS model TEXT;
CREATE INDEX IF NOT EXISTS memory_supersede_eval_run_version_idx
    ON memory_supersede_eval_run (check_version, finished_at DESC);

CREATE TABLE IF NOT EXISTS memory_supersede_run (
    run_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,                 -- distill | pass_dry_run | pass_apply
    episode_id TEXT,
    check_version TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dense TEXT,                           -- on | off | unavailable
    apply TEXT,                           -- on | refused | proposal_only
    refused_because TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    decisions JSONB NOT NULL DEFAULT '[]'::jsonb,   -- per asked pair: ids, legs, votes, verdict
    applied JSONB NOT NULL DEFAULT '[]'::jsonb      -- pairs actually written
);
CREATE INDEX IF NOT EXISTS memory_supersede_run_finished_idx ON memory_supersede_run (finished_at DESC);
