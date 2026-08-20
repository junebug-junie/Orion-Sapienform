-- Reverie visual chain store (Patch 1 of docs/superpowers/specs/
-- 2026-08-20-reverie-visual-chain-design.md). A second, parallel reverie
-- chain: generate -> observe -> interpret, not narration alone. Deliberately
-- NOT under the `substrate_` prefix -- that namespace is scoped to the live
-- attention-coalition rung; this chain pulls from broader context (chats,
-- dreams), so it doesn't belong there (design doc §5).
--
-- `prior_description` is the enforced continuity column: the context-builder
-- for step N+1 reads it directly, not via `chain_json`. This is a deliberate
-- departure from `substrate_reverie_chain`'s dead `next_focus`/`drift` JSON
-- fields (design doc §2) -- nothing ever read those, so this chain gets a
-- real column a real consumer reads instead.
--
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_reverie_visual_chain.sql

create table if not exists reverie_visual_chain (
    chain_id text primary key,
    created_at timestamptz not null,
    theme_key text,
    terminal_reason text not null,
    ema_salience double precision not null default 0.0,
    prior_description text,
    chain_json jsonb not null,
    stored_at timestamptz not null default now()
);

create index if not exists idx_reverie_visual_chain_created_at
    on reverie_visual_chain (created_at desc);

-- One row per generated image, content-addressed by sha256 of the bytes on
-- disk (orion/reverie/visual_storage.py). Image bytes never go in this row or
-- in chain_json -- only the pointer (`path`) does (design doc §6, following
-- the incident that motivated services/orion-hub/scripts/chat_attachments.py:
-- per-event blobs into Postgres already took the host down once, TOAST OOM
-- crash loop, 2026-07-23).
create table if not exists reverie_visual_artifact (
    sha256 text primary key,
    chain_id text not null references reverie_visual_chain(chain_id),
    step_index integer not null,
    mime text not null,
    bytes bigint not null,
    width integer,
    height integer,
    path text not null,
    description text,
    created_at timestamptz not null default now()
);

create index if not exists idx_reverie_visual_artifact_chain_id
    on reverie_visual_artifact (chain_id);
