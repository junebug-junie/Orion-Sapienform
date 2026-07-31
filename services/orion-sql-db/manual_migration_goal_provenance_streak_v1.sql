-- Goal-provenance node-target dominance streak v1 (2026-07-31 fix)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_goal_provenance_streak_v1.sql
--
-- Persists orion-attention-runtime's node-target goal-provenance dominance
-- streak (orion/attention/field_attention/goal_provenance.py::DominanceStreak)
-- across process restarts. Previously in-memory only, resetting to a cold
-- streak (count=0) on every restart -- named as a disclosed gap in PR #1529
-- (Sentience Striving Program README section 12) and revisited after PR #1543
-- (docs-only design doc) proposed surfacing this same streak count directly
-- into FieldGoalProvenanceV1.goal_text, which reaches the real LLM-facing
-- chat-stance prompt: a restart-truncated streak would then read as a small,
-- plausible-looking-but-wrong number in that prompt rather than just a brief
-- internal gating delay. Single singleton row (fixed streak_id), UPSERTed by
-- AttentionRuntimeStore.save_node_dominance_streak on every real tick.

create table if not exists substrate_goal_provenance_streak (
    streak_id text primary key,
    target_id text,
    count integer not null default 0,
    updated_at timestamptz not null default now()
);
