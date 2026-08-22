-- Artifact ratings: the first outcome Orion cannot produce itself (2026-08-21).
--
-- Every outcome an Orion action could previously claim was one of six field
-- pressures, all derived from its own CPU/disk/GPU telemetry -- the action,
-- the outcome and the grader all inside Orion. Measured over three days, every
-- action in the vocabulary scores inside its own error bar, which is what
-- marking your own homework looks like.
--
-- A human rating is decided by someone else. This migration is what lets one
-- be attributed to the action that earned it.

-- ---------------------------------------------------------------------------
-- 1. The existing feedback channel learns to point at an artifact
-- ---------------------------------------------------------------------------
--
-- The channel already existed end-to-end (Hub UI -> bus -> sql-writer) with a
-- registered schema, discrete value, categories and freetext. What it could
-- not do was target anything other than a chat turn -- the schema validator
-- required turn/message/correlation -- so the only gradeable thing in the
-- system was conversation.
--
-- NOTE: orion-sql-writer's _write_row() filters incoming payload keys against
-- the ORM's columns, so before this column existed a target_artifact_ref would
-- have been SILENTLY DROPPED rather than rejected. The sql-writer must be
-- deployed before anything starts producing artifact-targeted feedback, or the
-- ratings land with nothing to attribute them to.
ALTER TABLE chat_response_feedback
    ADD COLUMN IF NOT EXISTS target_artifact_ref TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_artifact_ref
    ON chat_response_feedback (target_artifact_ref);

-- ---------------------------------------------------------------------------
-- 2. Scored ratings
-- ---------------------------------------------------------------------------
--
-- Deliberately NOT substrate_action_outcomes. That table measures
-- `after - before` on a mean-reverting field pressure and needs a baseline bin
-- and a control arm for exactly that reason. A rating has no "before" and does
-- not mean-revert; there is no confound to subtract and the value IS the
-- rating. Reusing that shape because it exists would be the mirror image of
-- the defect it was built to remove.
--
-- surprise_nats here is the same Bayesian surprise, in the same UNIT as the
-- pressure ledger's, and NOT on the same scale. Measured: a pressure
-- observation with delta exactly 0.0 -- no effect at all -- scores 0.5595
-- nats, while a maximally informative human rating scores 0.2216. KL is not
-- scale-free across differently-parameterised priors. Anything ranking across
-- both ledgers must normalise (orion.autonomy.rating.cold_start_surprise_nats)
-- or it will systematically down-weight the human grader in favour of the
-- self-graded telemetry this table exists to provide an alternative to.

CREATE TABLE IF NOT EXISTS substrate_action_ratings (
    id                  BIGSERIAL PRIMARY KEY,

    feedback_id         TEXT        NOT NULL,
    artifact_ref        TEXT        NOT NULL,
    dispatch_id         TEXT        NOT NULL,
    dispatch_kind       TEXT        NOT NULL,
    target_id           TEXT        NOT NULL,

    feedback_value      TEXT        NOT NULL,
    rating              DOUBLE PRECISION NOT NULL,

    -- Recorded, never scored. Five thumbs-down categories is not five times
    -- worse than one, and turning a count of labels into a magnitude is
    -- precisely the defect that made risk_score useless (five hand-typed
    -- constants, 67% of them identical). The categories say WHY, and why is
    -- for reading, not for arithmetic.
    categories          TEXT[]      NOT NULL DEFAULT ARRAY[]::TEXT[],
    -- Worth more than the score. The only part that says why, and the only
    -- part Orion can read back later as content rather than as a number.
    free_text           TEXT        NULL,

    predicted_rating    DOUBLE PRECISION NOT NULL,
    prediction_error    DOUBLE PRECISION NOT NULL,
    surprise_nats       DOUBLE PRECISION NOT NULL,

    posterior_mean      DOUBLE PRECISION NOT NULL,
    posterior_variance  DOUBLE PRECISION NOT NULL,
    posterior_n         INTEGER     NOT NULL,

    rated_at            TIMESTAMPTZ NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT substrate_action_ratings_value_chk
        CHECK (feedback_value IN ('up', 'down')),
    CONSTRAINT substrate_action_ratings_rating_chk
        CHECK (rating IN (-1.0, 1.0))
);

-- One scoring per feedback event. A re-read of the feedback table must not
-- absorb the same human rating twice; an observation counted twice corrupts
-- the belief permanently and silently, and there is no control arm here to
-- make the corruption visible by contrast.
CREATE UNIQUE INDEX IF NOT EXISTS substrate_action_ratings_feedback_uidx
    ON substrate_action_ratings (feedback_id);

CREATE INDEX IF NOT EXISTS substrate_action_ratings_action_idx
    ON substrate_action_ratings (dispatch_kind, target_id, rated_at DESC);

-- The dedup that actually matters, and the one the first version of this file
-- got wrong. `feedback_id` is a fresh uuid4 per CLI invocation, so keying only
-- on it stops a re-SCORING of one event and is blind to two events carrying
-- the same human opinion -- which is the likely case, because a rater who is
-- not sure the first one landed will simply run it again. `submission_
-- fingerprint` is a sha256 over (target_key, value, categories, free_text,
-- source) and is identical across those two runs.
--
-- Scope honesty: the first version of this patch asserted in its commit
-- message that submission_fingerprint "is what stops a resubmission landing
-- twice." That was false -- the column was written and never read, and its
-- only indexes were non-unique. This is the statement that makes it true, and
-- it is enforced on the ARTIFACT path only, because 2 chat rows already exist
-- and a unique constraint over historical chat data is a different decision
-- with a different blast radius.
CREATE UNIQUE INDEX IF NOT EXISTS chat_response_feedback_artifact_fingerprint_uidx
    ON chat_response_feedback (submission_fingerprint)
    WHERE target_artifact_ref IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 3. Belief about what each action produces
-- ---------------------------------------------------------------------------
--
-- Keyed by (dispatch_kind, target_id) only. No signal column -- the rating IS
-- the signal. No baseline bin and no arm, for the reasons above.

CREATE TABLE IF NOT EXISTS substrate_action_rating_posterior (
    dispatch_kind       TEXT        NOT NULL,
    target_id           TEXT        NOT NULL,

    posterior_mean      DOUBLE PRECISION NOT NULL,
    posterior_variance  DOUBLE PRECISION NOT NULL,
    posterior_n         INTEGER     NOT NULL,

    -- Artifacts produced by this action that nobody rated. Unrated is NOT
    -- bad and NOT good, so it never becomes an observation -- but it must
    -- stay countable, because "we made 200 and 3 were rated" and "we made 3
    -- and 3 were rated" are opposite situations that an unrated-blind table
    -- would render identical. Also the honest denominator for a selection
    -- effect: Juniper rates what Juniper notices.
    unrated_count       INTEGER     NOT NULL DEFAULT 0,

    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (dispatch_kind, target_id)
);
