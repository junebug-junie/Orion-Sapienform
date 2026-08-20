-- Append-only store for PerceptionEmbeddingBaseline (P2, perceptual
-- prediction error -- orion/substrate/prediction_error.py's
-- perception_prediction_error()), one row per real embedding-bearing vision
-- artifact actually consumed, per camera stream.
--
-- Same shape as manual_migration_codebase_mass_baseline_v1.sql, and for the
-- identical reason: this domain's EWMA baseline has no persisted reducer
-- projection to live on (unlike the grammar-event-driven domains in
-- orion/substrate/prediction_error.py, each of which stores its own
-- baseline directly on a projection object) -- see that migration's own
-- comment for why the alternative (piggyback on node:substrate.perception's
-- own ConceptNodeV1.metadata) is wrong for the identical reason there.
--
-- Keyed by stream_id (unlike substrate_codebase_mass_baseline, which is
-- global): each camera stream's own running EWMA embedding vector must stay
-- independent, since mixing two streams' visual content into one baseline
-- would produce a meaningless average vector.
--
-- Deliberately a brand-new table with exactly one writer
-- (services/orion-substrate-runtime/app/worker.py's
-- _handle_perception_prediction_error_message(), gated behind
-- SUBSTRATE_PERCEPTION_PREDICTION_ERROR_TICK_ENABLED, default false) -- same
-- "a table nothing else has ever written to makes the cross-lane-stomp
-- failure class structurally impossible, not just avoided by convention"
-- reasoning as the codebase-mass baseline table.
--
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_perception_embedding_baseline_v1.sql

create table if not exists substrate_perception_embedding_baseline (
    baseline_id text primary key,
    stream_id text not null,
    generated_at timestamptz not null,
    baseline_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_perception_embedding_baseline_stream_generated
    on substrate_perception_embedding_baseline(stream_id, generated_at desc);
