-- Consumer-first rollout: install before enabling visual baseline policy.
-- Durable attempts are immutable replay identities. An ambiguous active/unknown
-- attempt blocks new GPU work until its production receipt can be reconciled.
CREATE TABLE IF NOT EXISTS reverie_visual_attempt (
    dispatch_id text PRIMARY KEY,
    need_id text,
    attempt_id text NOT NULL UNIQUE,
    started_at timestamptz NOT NULL,
    retry_after timestamptz NOT NULL,
    outcome text NOT NULL,
    request_json jsonb NOT NULL,
    result_json jsonb
);
CREATE INDEX IF NOT EXISTS idx_reverie_visual_attempt_need
    ON reverie_visual_attempt (need_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_reverie_visual_attempt_active
    ON reverie_visual_attempt (started_at DESC) WHERE outcome IN ('active', 'unknown');
