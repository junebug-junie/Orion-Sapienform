-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs
-- Personality rev 4, move 4 ("Juniper feels it"): Juniper's yes / no on a tendency, from the Hub
-- Personality tab ("Does this look like Orion?").
--
-- personality_juniper_verdict: one row per answer. Insert-only, so a changed mind is a new row and
-- the latest row per disposition is the current answer. Single writer: orion-durable-runs
-- (POST /personality/tendencies/{disposition_id}/verdict, proxied by the Hub). The same
-- transaction retires the tendency on a "no" (personality_disposition.status = retired), so it
-- leaves the frame, the Prime tending line and the market's tendency read.
-- label_at_verdict, weight_at_verdict, n_at_verdict, m_at_verdict: what Juniper was shown.
--
-- Apply BEFORE deploying orion-durable-runs (without it the verdict route answers 503 and the
-- tendencies list shows no answers; nothing else is affected):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < services/orion-sql-db/manual_migration_personality_juniper_verdict_v1.sql
-- Additive and idempotent. Rollback: manual_migration_personality_juniper_verdict_v1_rollback.sql.

CREATE TABLE IF NOT EXISTS personality_juniper_verdict (
    verdict_id         text PRIMARY KEY,
    disposition_id     text NOT NULL,
    -- yes | no
    verdict            text NOT NULL,
    label_at_verdict   text NOT NULL,
    weight_at_verdict  double precision,
    n_at_verdict       integer,
    m_at_verdict       integer,
    status_before      text,
    -- true when this answer moved the tendency to retired
    retired            boolean NOT NULL DEFAULT false,
    created_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS personality_juniper_verdict_disposition_idx
    ON personality_juniper_verdict (disposition_id, created_at);
