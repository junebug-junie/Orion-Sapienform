CREATE TABLE IF NOT EXISTS graphiti_episodes (
    episode_id text PRIMARY KEY,
    crystallization_id text NOT NULL,
    kind text NOT NULL,
    subject text NOT NULL,
    summary text NOT NULL,
    status text NOT NULL DEFAULT 'active',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    synced_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS graphiti_entities (
    entity_id text PRIMARY KEY,
    crystallization_id text NOT NULL,
    name text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS graphiti_edges (
    edge_id text PRIMARY KEY,
    from_id text NOT NULL,
    to_id text NOT NULL,
    relation text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_graphiti_ep_crystallization ON graphiti_episodes (crystallization_id);
CREATE INDEX IF NOT EXISTS idx_graphiti_edges_from ON graphiti_edges (from_id, relation);
CREATE INDEX IF NOT EXISTS idx_graphiti_edges_to ON graphiti_edges (to_id, relation);
