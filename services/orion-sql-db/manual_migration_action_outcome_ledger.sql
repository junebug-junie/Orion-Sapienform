-- Action-outcome ledger (2026-08-21).
--
-- Phase 1 of the autonomy/decision-budget rebuild. Records, for every
-- dispatched action that declared a falsifiable claim, what it predicted,
-- what actually happened, and how much that observation moved the belief
-- about what the action does (Bayesian surprise, in nats -- see
-- orion/autonomy/prediction.py).
--
-- Phase 1 is MEASUREMENT ONLY. Nothing in this migration changes what gets
-- dispatched or how the daily risk budget is computed. It exists so the
-- question "did that action do anything" has an answer at all.

CREATE TABLE IF NOT EXISTS substrate_action_outcomes (
    id                  BIGSERIAL PRIMARY KEY,
    dispatch_id         TEXT        NOT NULL,
    dispatch_frame_id   TEXT        NOT NULL,
    feedback_frame_id   TEXT        NOT NULL,

    dispatch_kind       TEXT        NOT NULL,
    target_id           TEXT        NOT NULL,
    signal_id           TEXT        NOT NULL,
    direction           TEXT        NOT NULL,

    observed_at         TIMESTAMPTZ NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    baseline            DOUBLE PRECISION NOT NULL,
    observed_after      DOUBLE PRECISION NOT NULL,
    observed_delta      DOUBLE PRECISION NOT NULL,

    predicted_delta     DOUBLE PRECISION NOT NULL,
    prediction_error    DOUBLE PRECISION NOT NULL,
    surprise_nats       DOUBLE PRECISION NOT NULL,

    posterior_mean      DOUBLE PRECISION NOT NULL,
    posterior_variance  DOUBLE PRECISION NOT NULL,
    posterior_n         INTEGER     NOT NULL,

    co_predictors       INTEGER     NOT NULL DEFAULT 0,
    latency_ms          DOUBLE PRECISION,

    -- Did the action do what it declared? NULL only for a directional claim
    -- whose delta landed inside the 1e-6 dead band (undecidable), never as a
    -- shrug. Added in the same migration file rather than a new one because
    -- this table was created the same day and holds no rows anywhere yet.
    claim_upheld        BOOLEAN
);

-- Idempotent, for an install that applied the CREATE TABLE above before
-- claim_upheld existed.
ALTER TABLE substrate_action_outcomes
    ADD COLUMN IF NOT EXISTS claim_upheld BOOLEAN;

-- One row per (dispatch, signal). A dispatch_id is stable per proposal per
-- policy, so a retried feedback pass must not double-count an observation
-- into the posterior.
CREATE UNIQUE INDEX IF NOT EXISTS substrate_action_outcomes_dispatch_signal_uidx
    ON substrate_action_outcomes (dispatch_id, signal_id);

-- The analysis access pattern: "what has this action done to this signal",
-- newest first.
CREATE INDEX IF NOT EXISTS substrate_action_outcomes_key_idx
    ON substrate_action_outcomes (dispatch_kind, target_id, signal_id, observed_at DESC);

-- Retention/scan support. This table grows at roughly the real dispatch
-- rate (~5,400/day as of 2026-08-21), so it gets a time index from day one
-- rather than after it becomes a problem.
CREATE INDEX IF NOT EXISTS substrate_action_outcomes_observed_at_idx
    ON substrate_action_outcomes (observed_at DESC);


-- Current belief per action/signal. Deliberately a separate, tiny,
-- upserted table rather than "read the newest ledger row": the daily risk
-- baseline already re-derives its state by scanning a 2 GB frame table on
-- every check, and that single pattern is 49.8% of this database's entire
-- buffer traffic (measured via pg_stat_statements, 2026-08-20). One row per
-- (kind, target, signal) -- tens of rows, primary-key lookup, O(1).
CREATE TABLE IF NOT EXISTS substrate_action_effect_posterior (
    dispatch_kind       TEXT        NOT NULL,
    target_id           TEXT        NOT NULL,
    signal_id           TEXT        NOT NULL,

    posterior_mean      DOUBLE PRECISION NOT NULL,
    posterior_variance  DOUBLE PRECISION NOT NULL,
    posterior_n         INTEGER     NOT NULL,

    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (dispatch_kind, target_id, signal_id)
);
