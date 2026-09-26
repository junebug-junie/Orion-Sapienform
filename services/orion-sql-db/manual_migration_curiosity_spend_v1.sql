-- The curiosity spend log: what each investigation run was offered, and what
-- it bought. P1 phase 1 of
-- docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md.
--
-- WHY THIS EXISTS: a curiosity investigation is Orion's most expensive
-- self-directed act (a ~20-minute turn on the pipeline Juniper's chat shares,
-- capped at HUB_CURIOSITY_INVESTIGATION_DAILY_CAP a day). Before this, nothing
-- recorded which priors a run was shown, and nothing scored what it changed:
-- Orion writes a :PriorRevision only when a confidence MOVES, so a test that
-- moved nothing left no trace (21 revisions across 94 journaled runs by
-- 2026-09-14). These two tables are the choice set and the outcome, so the
-- offer order can be compared against what it actually bought.
--
-- Single writer: services/orion-hub/scripts/curiosity_offer_decisions.py,
-- called from scripts/curiosity_investigation.py:
--   * curiosity_offer_decisions -- one row at dispatch (`_investigate`), then
--     `turn_started_at` + `turn_snapshot` when the turn STARTS (`_run_turn`):
--     the first snapshot taken wins, and the first attempt's start time is
--     kept. A retry may fill in a snapshot the first attempt could not take;
--     a start that already carries the run's own stamps is scored unknown, so
--     a run is scored from where it really began -- or not at all.
--   * curiosity_run_outcomes -- one row when the turn ends, upserted on retry.
-- Readers: the same module (learning-yield history for the next offer) and
-- scripts/analysis/replay_curiosity_realized_nats.py.
--
-- run_id is Hub's own run id (uuid4 hex[:12]) and the primary key of both:
-- one offer, one outcome, per run. A decision row with no outcome row means
-- the run never finished a turn (cancelled, refunded, or still queued).
--
-- realized_nats NULL means UNKNOWN, and unknown_reason says why:
--   no_start_snapshot          unreadable, over the Atlas row cap, or expired
--   no_end_snapshot            unreadable or over the row cap at the end
--   start_stamped_by_this_run  an earlier attempt wrote before the start
--                              snapshot was taken
--   no_scorable_test           every prior it tested had an unusable
--                              confidence, before or after the test
-- 0.0 means no measurable belief change among the tests that could be
-- scored: they moved nothing, or the run tested nothing
-- (n_invalid_confidence counts the tests that could not be scored). Never
-- conflate NULL and 0.0. A turn that failed (no text) is still scored, since
-- it may have written before failing, but carries turn_ok = false: read its
-- 0.0 as a failure, not a result.
--
-- Retention: turn_snapshot (every prior, every run -- the largest column) is
-- set back to NULL after 14 days by the writer itself
-- (SNAPSHOT_RETENTION_DAYS). Nothing needs it once the run's last attempt
-- has ended: the outcome row keeps before/after for every prior it changed.
--
-- Deliberately no GRANT to orion_readonly: the FCC sandbox should not read
-- its own scores (see the design's Goodhart section). Apply as the same role
-- Hub's memory pool uses, like every other manual migration here.
--
-- Apply (idempotent):
--   psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_curiosity_spend_v1.sql
-- Check:
--   python3 scripts/check_sql_migrations_applied.py --file manual_migration_curiosity_spend_v1.sql

CREATE TABLE IF NOT EXISTS curiosity_offer_decisions (
    run_id text PRIMARY KEY,
    decided_at timestamptz NOT NULL DEFAULT now(),
    -- 'value_order' (entropy x measured learning yield) or
    -- 'uncertainty_order' (today's most-uncertain-first).
    arm text NOT NULL,
    -- P(value arm) for this run: 0.0 when the switch is off.
    value_arm_propensity double precision NOT NULL,
    -- [{rank, prior_id, confidence, times_tested, entropy_nats, yield,
    --   expected_nats}] in the order Orion was shown them.
    offered jsonb NOT NULL DEFAULT '[]'::jsonb,
    stale_offered jsonb NOT NULL DEFAULT '[]'::jsonb,
    -- 'crystallization:<id>' / 'relation:<decision_id>' -- ids only, never
    -- the content, which stays in the stores it was sampled from.
    material_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    -- window, pseudo_tests, pool_yield, history_tests, clamp, sample sizes.
    constants jsonb NOT NULL DEFAULT '{}'::jsonb,
    -- Every prior's {prior_id, confidence, times_tested, status,
    -- last_run_id, run_id} when the turn started. NULL until then; stays
    -- NULL if the graph could not be read at that moment; set back to NULL
    -- after 14 days.
    turn_snapshot jsonb,
    -- When the FIRST attempt's turn started, kept across retries. Set even
    -- when the snapshot could not be taken.
    turn_started_at timestamptz
);

CREATE INDEX IF NOT EXISTS curiosity_offer_decisions_decided_at_idx
    ON curiosity_offer_decisions (decided_at);

CREATE TABLE IF NOT EXISTS curiosity_run_outcomes (
    run_id text PRIMARY KEY,
    completed_at timestamptz NOT NULL DEFAULT now(),
    -- Whether the last attempt's turn produced text. False rows are failures,
    -- not zero-value runs; the replay leaves them out of the distribution.
    turn_ok boolean NOT NULL,
    -- Sum of KL(after || before) over the priors this run tested (stamped
    -- last_run_id = run_id and times_tested went up). NULL = unknown.
    realized_nats double precision,
    -- Why realized_nats is NULL, one of the reasons in the header. NULL when
    -- the number is known.
    unknown_reason text,
    n_tested integer NOT NULL DEFAULT 0,
    n_moved integer NOT NULL DEFAULT 0,
    n_formed integer NOT NULL DEFAULT 0,
    -- Confidence moved without times_tested moving: protocol drift, reported
    -- with its nats in per_prior but not summed.
    n_moved_untested integer NOT NULL DEFAULT 0,
    -- Changes between the two snapshots stamped by another run, or unstamped.
    n_unattributed integer NOT NULL DEFAULT 0,
    n_invalid_confidence integer NOT NULL DEFAULT 0,
    -- [{prior_id, kind, before, after, tested_delta, nats}]
    per_prior jsonb NOT NULL DEFAULT '[]'::jsonb,
    -- Share of the moves Hub measured that Orion also wrote as a matching
    -- :PriorRevision. NULL when the run moved nothing.
    revision_agreement double precision
);

CREATE INDEX IF NOT EXISTS curiosity_run_outcomes_completed_at_idx
    ON curiosity_run_outcomes (completed_at);

-- Columns added while this migration was still on its branch. A no-op on a
-- table created above; re-applying the file brings an earlier copy up to
-- date (`CREATE TABLE IF NOT EXISTS` never adds columns, and the
-- applied-check only looks for the table).
ALTER TABLE curiosity_run_outcomes ADD COLUMN IF NOT EXISTS turn_ok boolean;
ALTER TABLE curiosity_run_outcomes ADD COLUMN IF NOT EXISTS unknown_reason text;
