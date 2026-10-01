-- Attend-to-act loop (2026-10-01): one row per world-action DECISION, treated or held-back control.
-- The precommit (expected effect + eligibility snapshot written BEFORE the shed RPC) and the join key
-- for the whole chain (broadcast -> proposal -> decision -> dispatch -> pool shed -> outcome -> loop
-- verdict). Spec: docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md (D4).
-- orion-execution-dispatch-runtime also creates it lazily (CREATE TABLE IF NOT EXISTS, short
-- lock_timeout) on its first world decision; applying it here first is the operator path. Additive.
CREATE TABLE IF NOT EXISTS substrate_world_action_episodes (
    episode_id text PRIMARY KEY,
    template text NOT NULL,
    dispatch_kind text NOT NULL,
    target_id text NOT NULL,
    arm text NOT NULL,
    decided_at timestamptz NOT NULL,
    open_loop_id text,
    broadcast_log_id text,
    node_id text,
    proposal_id text,
    decision_id text,
    dispatch_frame_id text,
    eligibility jsonb NOT NULL DEFAULT '{}'::jsonb,
    expected_effect jsonb,
    shed_id text,
    settlement_state text,
    settlement jsonb NOT NULL DEFAULT '{}'::jsonb,
    scoring_due_at timestamptz NOT NULL,
    scored_at timestamptz,
    outcome jsonb,
    loop_outcome_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS substrate_world_action_episodes_unscored_idx ON substrate_world_action_episodes (scoring_due_at) WHERE scored_at IS NULL;
CREATE INDEX IF NOT EXISTS substrate_world_action_episodes_decided_idx ON substrate_world_action_episodes (decided_at DESC);
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'orion_readonly') THEN
    GRANT SELECT ON substrate_world_action_episodes TO orion_readonly;
  END IF;
END $$;
