DATASETS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_datasets (
    dataset_id    UUID PRIMARY KEY,
    name          VARCHAR NOT NULL,
    source_table  VARCHAR NOT NULL,
    id_column     VARCHAR NOT NULL,
    time_column   VARCHAR NOT NULL,
    text_columns  JSONB NOT NULL,
    timezone      VARCHAR NOT NULL DEFAULT 'UTC',
    boundary_column VARCHAR,
    boundary_strategy VARCHAR,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_datasets_name ON topic_foundry_datasets (name);
"""

MODELS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_models (
    model_id       UUID PRIMARY KEY,
    name           VARCHAR NOT NULL,
    version        VARCHAR NOT NULL,
    stage          VARCHAR,
    dataset_id     UUID NOT NULL,
    model_spec     JSONB NOT NULL,
    windowing_spec JSONB NOT NULL,
    enrichment_spec JSONB,
    model_meta     JSONB,
    metadata       JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_models_name ON topic_foundry_models (name);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_models_dataset_id ON topic_foundry_models (dataset_id);
"""

RUNS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_runs (
    run_id         UUID PRIMARY KEY,
    model_id       UUID NOT NULL,
    dataset_id     UUID NOT NULL,
    specs          JSONB NOT NULL,
    spec_hash      VARCHAR,
    status         VARCHAR NOT NULL,
    stage          VARCHAR,
    run_scope      VARCHAR,
    stats          JSONB NOT NULL,
    artifact_paths JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at     TIMESTAMPTZ,
    completed_at   TIMESTAMPTZ,
    error          TEXT
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_runs_model_id ON topic_foundry_runs (model_id);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_runs_dataset_id ON topic_foundry_runs (dataset_id);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_runs_status ON topic_foundry_runs (status);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_runs_spec_hash ON topic_foundry_runs (spec_hash);
"""

SEGMENTS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_segments (
    segment_id UUID PRIMARY KEY,
    run_id     UUID NOT NULL,
    size       INTEGER NOT NULL,
    provenance JSONB NOT NULL,
    label      TEXT,
    topic_id   INTEGER,
    topic_prob FLOAT,
    is_outlier BOOLEAN,
    title      TEXT,
    aspects    JSONB,
    sentiment  JSONB,
    meaning    JSONB,
    enrichment JSONB,
    enriched_at TIMESTAMPTZ,
    enrichment_version TEXT,
    snippet   TEXT,
    chars     INTEGER,
    row_ids_count INTEGER,
    start_at TIMESTAMPTZ,
    end_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_segments_run_id ON topic_foundry_segments (run_id);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_segments_run_id_created_at ON topic_foundry_segments (run_id, created_at);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_segments_run_id_start_at ON topic_foundry_segments (run_id, start_at);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_segments_run_id_end_at ON topic_foundry_segments (run_id, end_at);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_segments_run_id_topic_id ON topic_foundry_segments (run_id, topic_id);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_segments_aspects ON topic_foundry_segments USING GIN (aspects);
"""

TOPICS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_topics (
    run_id UUID NOT NULL,
    topic_id INTEGER NOT NULL,
    scope VARCHAR NOT NULL,
    parent_topic_id INTEGER,
    centroid JSONB,
    count INTEGER,
    label TEXT,
    title TEXT,
    aspects JSONB,
    sentiment JSONB,
    meaning JSONB,
    enrichment JSONB,
    enriched_at TIMESTAMPTZ,
    enrichment_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, topic_id, scope)
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_topics_run_id ON topic_foundry_topics (run_id);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_topics_scope ON topic_foundry_topics (scope);
"""

WINDOW_FILTERS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_window_filters (
    filter_id UUID PRIMARY KEY,
    run_id UUID,
    segment_id UUID,
    policy VARCHAR NOT NULL,
    decision JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_window_filters_run_id ON topic_foundry_window_filters (run_id);
"""

CONVERSATION_ROLLUPS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_conversation_rollups (
    conversation_id UUID PRIMARY KEY,
    run_id UUID NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_conversation_rollups_run_id ON topic_foundry_conversation_rollups (run_id);
"""

BOUNDARY_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_boundary_cache (
    cache_key TEXT PRIMARY KEY,
    run_id UUID,
    spec_hash TEXT,
    dataset_id UUID,
    model_id UUID,
    boundary_index INT,
    context_hash TEXT,
    decision JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_boundary_cache_run_id ON topic_foundry_boundary_cache (run_id);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_boundary_cache_spec_hash ON topic_foundry_boundary_cache (spec_hash);
"""

MODEL_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_model_events (
    event_id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    name VARCHAR NOT NULL,
    version VARCHAR NOT NULL,
    from_stage VARCHAR,
    to_stage VARCHAR NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_model_events_model_id ON topic_foundry_model_events (model_id);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_model_events_name ON topic_foundry_model_events (name);
"""

DRIFT_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_drift (
    drift_id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    js_divergence DOUBLE PRECISION NOT NULL,
    outlier_pct DOUBLE PRECISION NOT NULL,
    threshold_js DOUBLE PRECISION,
    threshold_outlier DOUBLE PRECISION,
    topic_shares JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_drift_model_id ON topic_foundry_drift (model_id);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_drift_created_at ON topic_foundry_drift (created_at);
"""

EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_events (
    event_id UUID PRIMARY KEY,
    kind TEXT NOT NULL,
    run_id UUID,
    model_id UUID,
    drift_id UUID,
    payload JSONB,
    bus_status TEXT,
    bus_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_events_kind ON topic_foundry_events (kind);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_events_created_at ON topic_foundry_events (created_at);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_events_run_id ON topic_foundry_events (run_id);
"""

EDGES_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_edges (
    edge_id UUID PRIMARY KEY,
    segment_id UUID NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_edges_segment_id ON topic_foundry_edges (segment_id);
CREATE INDEX IF NOT EXISTS ix_topic_foundry_edges_subject ON topic_foundry_edges (subject);
"""

CONVERSATIONS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_conversations (
    conversation_id UUID PRIMARY KEY,
    dataset_id UUID NOT NULL,
    observed_start_at TIMESTAMPTZ,
    observed_end_at TIMESTAMPTZ,
    block_count INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_conversations_dataset_id ON topic_foundry_conversations (dataset_id);
"""

CONVERSATION_BLOCKS_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_conversation_blocks (
    conversation_id UUID NOT NULL,
    block_index INT NOT NULL,
    row_ids JSONB NOT NULL,
    timestamps JSONB NOT NULL,
    role_summary TEXT,
    text_snippet TEXT,
    PRIMARY KEY (conversation_id, block_index)
);
"""

CONVERSATION_OVERRIDES_DDL = """
CREATE TABLE IF NOT EXISTS topic_foundry_conversation_overrides (
    override_id UUID PRIMARY KEY,
    dataset_id UUID NOT NULL,
    kind TEXT NOT NULL,
    payload JSONB NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_foundry_conversation_overrides_dataset_id ON topic_foundry_conversation_overrides (dataset_id);
"""
