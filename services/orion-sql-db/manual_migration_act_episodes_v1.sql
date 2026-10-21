-- ORION-MIGRATION-REQUIRED-BY: orion-execution-dispatch-runtime
-- Thinking rung 3, stage 3 option B (2026-10-21): one row per allocator decision to run or hold back
-- a proven-effect act (orion/autonomy/act_episodes.py). The dispatch runtime also creates it lazily
-- (CREATE TABLE IF NOT EXISTS, short lock_timeout); applying it here first is the operator path.
-- Additive only: one new table, two indexes. Rollback: DROP TABLE substrate_act_episodes (data loss of
-- the control/treated arms only).
CREATE TABLE IF NOT EXISTS substrate_act_episodes (
    episode_id text PRIMARY KEY,
    act_label text NOT NULL,
    verb text NOT NULL,
    actor text NOT NULL,
    signal_id text NOT NULL,
    dispatch_kind text NOT NULL,
    target_id text NOT NULL,
    arm text NOT NULL,
    path text NOT NULL,
    decided_at timestamptz NOT NULL,
    horizon_at timestamptz NOT NULL,
    dispatch_frame_id text,
    expected jsonb NOT NULL DEFAULT '{}'::jsonb,
    outcome_value double precision,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS substrate_act_episodes_act_idx ON substrate_act_episodes (act_label, arm, decided_at DESC);
CREATE INDEX IF NOT EXISTS substrate_act_episodes_horizon_idx ON substrate_act_episodes (actor, horizon_at DESC);
