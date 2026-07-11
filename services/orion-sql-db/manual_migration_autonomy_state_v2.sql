-- AutonomyStateV2 persistence (closes the reducer's own fold loop).
-- Producer/consumer: orion/autonomy/state_store.py (load_autonomy_state_v2 /
-- save_autonomy_state_v2), called from services/orion-cortex-exec/app/chat_stance.py
-- _run_autonomy_reducer().
--
-- Single-row-per-subject latest-state table: the reducer upserts its own output
-- here every turn instead of only reading the V1/graph baseline, so
-- previous_state carries forward turn-to-turn.
--
-- Deliberately separate from the graph/Fuseki-backed homeostatic drives system --
-- do not merge or reference that store from here.

CREATE TABLE IF NOT EXISTS autonomy_state_v2 (
    subject TEXT PRIMARY KEY,
    state JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
