-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub
-- Claude's read-only tools in Hub chat, gated by Orion's consent (Patch 1b of
-- docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md).
--
-- claude_tool_call_log: one row per phase (ask, answer, result) of each Claude tool call. Writer:
-- orion-sql-writer off bus orion:claude:tool:audit (ClaudeToolCallV1; producer: the broker in
-- orion-harness-governor). Reader: Hub GET /api/claude-tools/audit. Insert-only, PK event_id
-- (<call_id>:<phase>). sql-writer's create_all makes the same table at boot (same index names); this
-- file is the jsonb-typed operator contract. Additive only: a new table and its indexes.

CREATE TABLE IF NOT EXISTS claude_tool_call_log (
    event_id            text PRIMARY KEY,
    phase               text NOT NULL,
    call_id             text NOT NULL,
    turn_request_id     text NOT NULL,
    room_id             text NOT NULL,
    claude_session_id   text NOT NULL,
    hub_session_id      text,
    trigger             text,
    tool                text NOT NULL,
    category            text,
    arguments           jsonb,
    reason              text,
    setting             text,
    decision_path       text,
    shared              boolean,
    orion_answer        text,
    orion_scope         text,
    orion_reason        text,
    consent_path        text,
    consent_latency_ms  integer,
    act_id              text,
    result_chars        integer,
    error               text,
    created_at          timestamptz NOT NULL,
    written_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_claude_tool_call_log_created ON claude_tool_call_log (created_at);
CREATE INDEX IF NOT EXISTS idx_claude_tool_call_log_call ON claude_tool_call_log (call_id);
CREATE INDEX IF NOT EXISTS idx_claude_tool_call_log_turn ON claude_tool_call_log (turn_request_id);
CREATE INDEX IF NOT EXISTS idx_claude_tool_call_log_category_created ON claude_tool_call_log (category, created_at);
