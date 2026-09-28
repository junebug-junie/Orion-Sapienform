-- Consumer-first rollout: install after manual_migration_reverie_visual_attempt.sql and
-- before enabling the reverie.visual durable workflow
-- (docs/superpowers/specs/2026-09-28-visual-reverie-durable-graph-design.md).
-- stage_json is the per-stage checkpoint orion-thought writes for one attempt:
-- the frozen prompt plan (prepare), the recorded image (generate) and the cached
-- caption (caption). Additive and idempotent. The legacy run-once claim reads it only
-- to release attempts a dead durable run left behind, and works without it.
ALTER TABLE reverie_visual_attempt
    ADD COLUMN IF NOT EXISTS stage_json jsonb NOT NULL DEFAULT '{}'::jsonb;
