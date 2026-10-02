-- Memory episode redesign, Stage 1 PR 2 (2026-10-02): the shadow distiller's tables.
-- Spec: docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md, section 2.
--
-- SHADOW: written only by orion-durable-runs' memory.episode_distill graph. No consumer reads
-- these tables to change behavior during Stages 1-3 (the old-vs-new report reads them).
-- Table names are the spec's own, so Stage 4 cutover renames nothing.
-- Apply BEFORE deploying orion-durable-runs from this branch; additive only.
-- Rollback: MEMORY_EPISODE_WRITER_ENABLED=false, then these tables can be dropped.

CREATE TABLE IF NOT EXISTS episode_memory (
    memory_id UUID PRIMARY KEY,
    episode_id TEXT,                       -- memory_episode_shadow.episode_id; NULL for migrated legacy
    purpose TEXT NOT NULL,                 -- happened | about_juniper | orion_view | follow_up
    voice TEXT NOT NULL,                   -- juniper_said | worked_out_together | orion_thought | orion_read | orion_self_knowledge
    channel TEXT NOT NULL,                 -- chat | reverie | curiosity | dream | reading | journal | topic_model | graphify | legacy_crystallization
    statement TEXT NOT NULL,               -- Orion's words, first person, one claim
    occurred_at TIMESTAMPTZ,
    stakes TEXT NOT NULL,                  -- low | high
    stakes_reason TEXT,
    confirmation_state TEXT NOT NULL,      -- auto | pending_confirmation | confirmed | corrected | rejected
    confirmation_loop_id TEXT,
    strength REAL NOT NULL,
    half_life_days REAL,
    last_reinforced_at TIMESTAMPTZ NOT NULL,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    due_after TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'active', -- active | faded | done | expired | superseded | retired
    supersedes_memory_id UUID,
    model_route TEXT,
    prompt_version TEXT,
    run_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT episode_memory_purpose_chk CHECK (purpose IN ('happened', 'about_juniper', 'orion_view', 'follow_up')),
    CONSTRAINT episode_memory_voice_chk CHECK (
        voice IN ('juniper_said', 'worked_out_together', 'orion_thought', 'orion_read', 'orion_self_knowledge')),
    CONSTRAINT episode_memory_stakes_chk CHECK (stakes IN ('low', 'high'))
);
CREATE INDEX IF NOT EXISTS idx_episode_memory_episode ON episode_memory (episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_memory_created ON episode_memory (created_at);

CREATE TABLE IF NOT EXISTS episode_memory_evidence (
    memory_id UUID NOT NULL,
    source_kind TEXT NOT NULL,             -- chat_prompt | chat_response
    source_id TEXT NOT NULL,               -- chat_history_log.correlation_id
    quote TEXT NOT NULL,
    verified BOOLEAN NOT NULL,
    PRIMARY KEY (memory_id, source_kind, source_id, quote)
);

CREATE TABLE IF NOT EXISTS episode_memory_referent (
    memory_id UUID NOT NULL,
    referent_key TEXT NOT NULL,            -- kind:slug
    role TEXT NOT NULL,
    PRIMARY KEY (memory_id, referent_key, role)
);
CREATE INDEX IF NOT EXISTS idx_episode_memory_referent_key ON episode_memory_referent (referent_key);

-- Every state change, and every rejected candidate (memory_id NULL, op='rejected_invalid',
-- the candidate in evidence) -- invalid memories are logged here, never stored above.
CREATE TABLE IF NOT EXISTS episode_memory_event (
    event_id UUID PRIMARY KEY,
    memory_id UUID,
    op TEXT NOT NULL,
    actor TEXT NOT NULL,
    episode_id TEXT,
    outcome_id TEXT,
    evidence JSONB,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_episode_memory_event_episode ON episode_memory_event (episode_id);

-- Open questions/tensions during Stages 1-2 (spec section 8): the curiosity_self_questions
-- columns plus text. Migrated into curiosity_self_questions at Stage 3.
CREATE TABLE IF NOT EXISTS memory_tension_shadow (
    question_id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'question',
    scope TEXT NOT NULL DEFAULT 'juniper',
    answer_via TEXT NOT NULL DEFAULT 'conversation',
    source_episode_id TEXT,
    source_refs JSONB,
    referent_keys TEXT[],
    status TEXT NOT NULL DEFAULT 'open',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per distilled episode: cost and outcome, for the report and the Stage 1 cost check.
CREATE TABLE IF NOT EXISTS episode_distill_run (
    episode_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    model_route TEXT,
    model TEXT,
    prompt_version TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    llm_latency_ms INTEGER,
    hold_wait_ms INTEGER,
    memories_kept INTEGER NOT NULL,
    memories_rejected INTEGER NOT NULL,
    questions_kept INTEGER NOT NULL,
    downgrades INTEGER NOT NULL,
    coverage REAL,
    finished_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
