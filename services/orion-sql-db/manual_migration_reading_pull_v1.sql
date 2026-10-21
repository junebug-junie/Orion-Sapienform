-- ORION-MIGRATION-REQUIRED-BY: orion-hub, orion-execution-dispatch-runtime
-- Thinking rung 3 follow-up (2026-10-21): the motor allocator's reading pull.
-- world_pulse_read_seed: which allocator decision pulled a seed, its arm, and when (Hub writes,
--   orion/world_pulse_read/pull.py; Hub also adds these lazily with a short lock_timeout).
-- substrate_act_episodes: the object a decision chose (the act ledger key, read_seed:<seed_id>),
--   why the doer refused it, and when its outcome was resolved (orion/autonomy/act_episodes.py;
--   the dispatch runtime also adds these lazily).
-- Additive only: nullable columns, no default, no constraint. Rollback: drop the six columns.
ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS pull_dispatch_id text,
    ADD COLUMN IF NOT EXISTS pull_arm text,
    ADD COLUMN IF NOT EXISTS pulled_at timestamptz;
ALTER TABLE substrate_act_episodes
    ADD COLUMN IF NOT EXISTS object_ref text,
    ADD COLUMN IF NOT EXISTS void_reason text,
    ADD COLUMN IF NOT EXISTS resolved_at timestamptz;
