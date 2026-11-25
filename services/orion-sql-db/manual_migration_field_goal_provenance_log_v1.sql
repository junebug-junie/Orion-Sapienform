-- ORION-MIGRATION-REQUIRED-BY: none retired 2026-11-25 (#3924 rework); orion-sql-writer writes field_goal_episode instead and no service reads or writes this table
-- Goal proposal log, issue #3924.
-- RETIRED 2026-11-25: no writer. Table and rows kept until Juniper drops them (DROP TABLE field_goal_provenance_log;).
--
-- field_goal_provenance_log: one row per goal Orion proposes. Writer: orion-sql-writer off bus
-- orion:memory:goals:proposed (FieldGoalProvenanceV1, kind memory.field_goals.proposed.v1;
-- producer: orion-attention-runtime app/worker.py _publish_goal). Insert-only, PK artifact_id
-- (a uuid the producer sets once per goal, so a re-delivered envelope is skipped). ts = the
-- goal's own timestamp; received_at = when sql-writer stored it; envelope_id = bus envelope id;
-- payload_json = the validated payload (schema fields only). The producer's streak count is not
-- on the schema, so it is not stored here.
-- sql-writer's create_all makes the same table at boot (index names match the model's
-- __table_args__); this file is the operator contract. Additive only.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_field_goal_provenance_log_v1.sql
-- Rollback: DROP TABLE field_goal_provenance_log;  (loses the stored goal history)

CREATE TABLE IF NOT EXISTS field_goal_provenance_log (
    artifact_id                text PRIMARY KEY,
    field_target_id            text NOT NULL,
    target_kind                text NOT NULL,
    salience_score             double precision NOT NULL,
    priority                   double precision NOT NULL,
    proposal_status            text NOT NULL,
    source_field_tick_id       text NOT NULL,
    source_attention_frame_id  text NOT NULL,
    confidence                 double precision,
    correlation_id             text,
    ts                         timestamptz NOT NULL,
    envelope_id                text,
    received_at                timestamptz NOT NULL DEFAULT now(),
    payload_json               jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_field_goal_provenance_log_received_at ON field_goal_provenance_log (received_at);
CREATE INDEX IF NOT EXISTS ix_field_goal_provenance_log_ts ON field_goal_provenance_log (ts);
CREATE INDEX IF NOT EXISTS ix_field_goal_provenance_log_target_ts ON field_goal_provenance_log (field_target_id, ts);
