-- Embodied presence v1: "is someone at the camera, and for how long" as a
-- self-state observable. Mirrors manual_migration_hub_presence_v1.sql's shape
-- exactly (single-row upsert, JSONB blob) -- same pattern, camera-shaped
-- presence_id instead of the fixed 'hub'.
--
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_embodied_presence_v1.sql

create table if not exists substrate_embodied_presence (
    presence_id text primary key,       -- stream_id, e.g. 'cam0', 'carbon'
    generated_at timestamptz not null,
    presence_json jsonb not null,       -- {state, since_sec, last_seen_sec, subject}
    updated_at timestamptz not null default now()
);
