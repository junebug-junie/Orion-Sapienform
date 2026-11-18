-- ORION-MIGRATION-REQUIRED-BY: orion-attention-runtime
-- Hourly counts of world-first attention frames (orion/attention/frame_rollup.py), added by
-- orion-attention-runtime in the same transaction that saves each frame. `dig attention` (#3600) reads
-- these few hundred rows instead of detoasting ~41k frame_json documents a day, which on a cold cache
-- took longer than its 25 s statement timeout. Additive only; never pruned (about 1k small rows a day).
BEGIN;
CREATE TABLE IF NOT EXISTS substrate_attention_hour_rollup (
    hour TIMESTAMPTZ NOT NULL,
    part TEXT NOT NULL,
    target TEXT NOT NULL,
    cause TEXT NOT NULL,
    a BIGINT NOT NULL DEFAULT 0,
    b BIGINT NOT NULL DEFAULT 0,
    c BIGINT NOT NULL DEFAULT 0,
    d BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (hour, part, target, cause)
);
COMMIT;
