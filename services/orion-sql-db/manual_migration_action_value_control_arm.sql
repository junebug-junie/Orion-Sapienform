-- Action value needs a control arm (2026-08-21, phase 2).
--
-- Supersedes the VALUE DEFINITION shipped in
-- manual_migration_action_outcome_ledger.sql (phase 1, same day, PR #1798).
-- Phase 1 recorded an action's value as the unconditional field delta across
-- its window. Actions fire because a pressure is high and high pressures
-- fall on their own, so that number measures mean reversion, not effect:
-- live over 3 days the raw prune gap reads as a 5.8x effect (-0.148 vs
-- -0.026) and INVERTS in 6 of 8 baseline deciles.
--
-- This migration adds what the estimator needs to be honest:
--   * which arm an observation belongs to,
--   * where the signal started (the matching key),
--   * how contaminated the tick was,
--   * and a place to keep the untreated arm.
--
-- Still measurement-only. Nothing here changes what Orion dispatches.

-- ---------------------------------------------------------------------------
-- 1. Ledger: arm, baseline bin, tick contamination
-- ---------------------------------------------------------------------------

ALTER TABLE substrate_action_outcomes
    ADD COLUMN IF NOT EXISTS arm TEXT NOT NULL DEFAULT 'dispatched';

ALTER TABLE substrate_action_outcomes
    ADD COLUMN IF NOT EXISTS baseline_bin SMALLINT;

-- The field delta is measured frame-wide. A record drawn from a tick where
-- other actions also ran carries their effect too, and 5 of 16 live
-- templates declare no signal at all while accounting for 72% of dispatch
-- volume -- so "no candidate claimed this signal" is NOT the same as
-- "nothing acted on it". Stored so an analysis can restrict to clean ticks
-- rather than the contamination being unrecoverable after the fact.
ALTER TABLE substrate_action_outcomes
    ADD COLUMN IF NOT EXISTS frame_dispatch_count INTEGER NOT NULL DEFAULT 0;

-- Backfill is exact, not approximate: baseline_bin is a pure function of a
-- column already stored on the row (floor(baseline * 10), clamped to 0..9 --
-- orion.autonomy.contrast.baseline_bin). Rows written by phase 1 therefore
-- lose nothing.
UPDATE substrate_action_outcomes
   SET baseline_bin = LEAST(9, GREATEST(0, FLOOR(baseline * 10)::int))
 WHERE baseline_bin IS NULL;

ALTER TABLE substrate_action_outcomes
    ALTER COLUMN baseline_bin SET NOT NULL;

-- The old unique key was (dispatch_id, signal_id), which is wrong in two
-- ways that only became visible once a second arm existed:
--   * `dispatch_id` is stable per (proposal, policy), and starvation aging
--     means the SAME proposal is capacity-blocked on many consecutive ticks
--     and may finally be dispatched later. Under the old key the first
--     blocked observation won and every later one -- including the real
--     dispatch -- was silently discarded by ON CONFLICT DO NOTHING.
--   * Each tick is a separate field window, so each is a genuinely separate
--     observation, not a duplicate.
-- Adding dispatch_frame_id keeps the property that actually matters (a
-- reprocessed feedback pass over the same dispatch frame inserts nothing)
-- while letting distinct ticks through.
DROP INDEX IF EXISTS substrate_action_outcomes_dispatch_signal_uidx;

CREATE UNIQUE INDEX IF NOT EXISTS substrate_action_outcomes_dispatch_frame_signal_uidx
    ON substrate_action_outcomes (dispatch_id, signal_id, dispatch_frame_id);

-- The contrast's access pattern: treated cells for one action, by bin.
CREATE INDEX IF NOT EXISTS substrate_action_outcomes_arm_bin_idx
    ON substrate_action_outcomes (dispatch_kind, target_id, signal_id, arm, baseline_bin);

-- ---------------------------------------------------------------------------
-- 2. Treated cells are per baseline bin
-- ---------------------------------------------------------------------------
--
-- A posterior pooled across baseline bins is the confounded quantity itself:
-- it averages "what happened when the signal started at 0.9" together with
-- "what happened when it started at 0.2", and the action's own volume
-- decides the mix. Matching requires the cell.
--
-- The 7 rows phase 1 wrote today are pooled and cannot be split, so they are
-- MOVED to a backup table rather than reinterpreted -- a silently redefined
-- column is exactly the defect class this arc exists to stop. Nothing is
-- destroyed; the ledger rows they were derived from are all still present
-- and carry an exact baseline_bin after the backfill above.

CREATE TABLE IF NOT EXISTS substrate_action_effect_posterior_phase1_backup
    (LIKE substrate_action_effect_posterior INCLUDING ALL);

INSERT INTO substrate_action_effect_posterior_phase1_backup
SELECT * FROM substrate_action_effect_posterior
ON CONFLICT DO NOTHING;

DELETE FROM substrate_action_effect_posterior;

ALTER TABLE substrate_action_effect_posterior
    ADD COLUMN IF NOT EXISTS baseline_bin SMALLINT NOT NULL DEFAULT 0;

ALTER TABLE substrate_action_effect_posterior
    DROP CONSTRAINT IF EXISTS substrate_action_effect_posterior_pkey;

ALTER TABLE substrate_action_effect_posterior
    ADD PRIMARY KEY (dispatch_kind, target_id, signal_id, baseline_bin);

-- ---------------------------------------------------------------------------
-- 3. The control arm
-- ---------------------------------------------------------------------------
--
-- One row per (signal, arm, baseline bin), holding a running Normal-Normal
-- posterior over the signal's delta on UNTREATED ticks.
--
-- Not a per-observation ledger on purpose: the untreated population is every
-- tick where nothing was dispatched (~94% of ~32,000 ticks/day), so rows
-- would arrive at ~128k/day for a quantity only ever read in aggregate.
-- Folding them into the cell keeps writes O(1) and keeps the ledger growing
-- at the real action rate.
--
-- `arm` is a column rather than a separate table because the vocabulary is
-- open at exactly one point: `randomized_holdback` (step 3 of the design,
-- off by default) is a SECOND control arm with a strictly better claim, and
-- the two must be reportable side by side and never pooled -- pooling an
-- experimental arm with a quasi-experimental one yields a number that is
-- neither and gets described as the better of the two.

CREATE TABLE IF NOT EXISTS substrate_signal_control_cells (
    signal_id           TEXT        NOT NULL,
    arm                 TEXT        NOT NULL,
    baseline_bin        SMALLINT    NOT NULL,

    posterior_mean      DOUBLE PRECISION NOT NULL,
    posterior_variance  DOUBLE PRECISION NOT NULL,
    posterior_n         INTEGER     NOT NULL,

    -- Observations that actually LEFT the dead band. A Normal-Normal
    -- posterior with a fixed observation variance shrinks as 1/n whether
    -- the data varies or is one constant repeated, so without this column a
    -- frozen instrument produces the most confident-looking cell in the
    -- table. Live proof it is needed: resource_pressure sat at exactly 0.85,
    -- stddev exactly 0.0, across ~12,000 consecutive frames on 2026-08-21
    -- (a vision channel saturated at 1.0 times a 0.85 topology edge weight),
    -- against 2,600 distinct values/day on every prior day.
    moved_n             INTEGER     NOT NULL DEFAULT 0,

    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (signal_id, arm, baseline_bin)
);
