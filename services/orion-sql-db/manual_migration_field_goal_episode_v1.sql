-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Goal episodes, issue #3924.
--
-- field_goal_episode: one row per goal episode, a run of Orion wanting the same target.
-- Writer: orion-sql-writer off bus orion:memory:goals:proposed (FieldGoalProvenanceV1, kind
-- memory.field_goals.proposed.v1; producer: orion-attention-runtime app/worker.py _publish_goal,
-- which re-publishes the held goal every qualifying tick). Fold rule (app/field_goal_episode.py):
-- a goal joins the newest episode when it is for the same target and last_ts is within 10 min;
-- otherwise it opens a new episode. repeat_count = goal messages folded in (sum = goals
-- published). first_payload_json = the first goal's validated payload (schema fields only).
--
-- field_goal_episode_artifact: seen artifact_ids, so a re-delivered goal does not bump
-- repeat_count. The writer prunes rows older than 48 h on write.
--
-- sql-writer's create_all makes the same tables at boot (index names match the models'
-- __table_args__); this file is the operator contract. Additive only.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_field_goal_episode_v1.sql
-- Rollback: DROP TABLE field_goal_episode_artifact; DROP TABLE field_goal_episode;  (loses the stored goal history)

CREATE TABLE IF NOT EXISTS field_goal_episode (
    episode_id          text PRIMARY KEY,
    field_target_id     text NOT NULL,
    target_kind         text NOT NULL,
    first_artifact_id   text NOT NULL,
    first_ts            timestamptz NOT NULL,
    last_ts             timestamptz NOT NULL,
    repeat_count        integer NOT NULL,
    max_salience        double precision NOT NULL,
    last_salience       double precision NOT NULL,
    first_payload_json  jsonb NOT NULL,
    first_envelope_id   text,
    received_at_first   timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_field_goal_episode_first_ts ON field_goal_episode (first_ts);
CREATE INDEX IF NOT EXISTS ix_field_goal_episode_target_last_ts ON field_goal_episode (field_target_id, last_ts);

CREATE TABLE IF NOT EXISTS field_goal_episode_artifact (
    artifact_id  text PRIMARY KEY,
    episode_id   text NOT NULL,
    seen_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_field_goal_episode_artifact_seen_at ON field_goal_episode_artifact (seen_at);
