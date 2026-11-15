-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Curiosity finding register (curiosity event wants, step 2; 2026-11-15).
-- docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md ("Which findings compete").
--
-- One row per finished curiosity turn (both lines), written once by Hub after the turn ends:
--   register  -- the judge's label: philosophical / operational / no_finding. NULL with judge_error set
--                when the judge failed or was not an evaluated model: the run is then absent, never
--                "operational by default".
--   novelty   -- 1 - max cosine between the finding and everything Orion wrote before it (earlier
--                findings and answers, prior claims, Day letters). NULL = unknown (embedder down).
-- Insert-only (ON CONFLICT DO NOTHING): a label is never re-judged.
--
-- curiosity_novelty_corpus caches one embedding per piece of Orion's writing, so each run embeds only
-- what is new. A text change gets a new row (ref + text_sha256); the old vector stays.
--
-- Single writer: services/orion-hub/scripts/curiosity_finding_register.py.
-- Readers: the curiosity event-want producer (step 3) and scripts/analysis/score_curiosity_findings.py.

CREATE TABLE IF NOT EXISTS curiosity_finding_register (
    run_id          text PRIMARY KEY,
    line            text NOT NULL,
    question_ref    text,
    register        text CHECK (register IN ('philosophical', 'operational', 'no_finding')),
    register_reason text,
    judge_model     text,
    judge_route     text,
    prompt_sha256   text,
    judge_error     text,
    judged_at       timestamptz,
    novelty         double precision,
    nearest_ref     text,
    novelty_error   text,
    embed_model     text,
    corpus_version  text,
    corpus_size     integer,
    answer_chars    integer,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curiosity_finding_register_created
    ON curiosity_finding_register (created_at);

CREATE TABLE IF NOT EXISTS curiosity_novelty_corpus (
    ref          text NOT NULL,
    text_sha256  text NOT NULL,
    kind         text NOT NULL,
    run_id       text,
    written_at   timestamptz,
    embed_model  text NOT NULL,
    embedding    real[] NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (ref, text_sha256)
);

CREATE INDEX IF NOT EXISTS idx_curiosity_novelty_corpus_run
    ON curiosity_novelty_corpus (run_id);
