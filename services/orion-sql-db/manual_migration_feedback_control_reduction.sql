-- ORION-MIGRATION-REQUIRED-BY: orion-feedback-runtime, orion-execution-dispatch-runtime, orion-consolidation-runtime
-- Compact completion receipts for structurally empty execution-dispatch frames.
--
-- A receipt is the durable child proving feedback-runtime folded this dispatch's untreated
-- field delta directly into substrate_signal_control_cells. It deliberately stores no source
-- payload, field snapshot, prompt, response, or reasoning. The worker inserts it, advances the
-- aggregate cells, and clears feedback_pending in one transaction.

CREATE TABLE IF NOT EXISTS substrate_feedback_control_reductions (
    source_execution_dispatch_frame_id TEXT PRIMARY KEY,
    source_dispatch_generated_at       TIMESTAMPTZ NOT NULL,
    control_observation_count          INTEGER NOT NULL,
    control_cell_count                 INTEGER NOT NULL,
    created_at                         TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT substrate_feedback_control_reductions_observation_count_chk
        CHECK (control_observation_count > 0),
    CONSTRAINT substrate_feedback_control_reductions_cell_count_chk
        CHECK (control_cell_count > 0)
);

CREATE INDEX IF NOT EXISTS idx_substrate_feedback_control_reductions_created_at
    ON substrate_feedback_control_reductions (created_at DESC);

COMMENT ON TABLE substrate_feedback_control_reductions IS
    'Payload-free receipts for empty dispatches reduced directly into untreated control cells.';

-- Producer-visibility seam for Layer 11. Execution dispatch inserts this row BEFORE assigning
-- the frame's generated_at and starting any RPC. save_dispatch_frame atomically inserts the real
-- frame and deletes this row, so consolidation can distinguish "no frame" from "frame still
-- materializing" without guessing from a timeout.
CREATE TABLE IF NOT EXISTS substrate_execution_dispatch_inflight (
    frame_id       TEXT PRIMARY KEY,
    generated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_substrate_execution_dispatch_inflight_generated_at
    ON substrate_execution_dispatch_inflight (generated_at);

COMMENT ON TABLE substrate_execution_dispatch_inflight IS
    'Transient producer receipts replaced atomically by execution dispatch frames.';
