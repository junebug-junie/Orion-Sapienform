-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- OSNet shadow body vector on walkway person crops (2026-10-16, Juniper approved).
-- Design: docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md, "A better body vector".
--
-- 1. vision_crop_observation.osnet_embedding: the 512-d OSNet-AIN person
--    re-identification vector of the same crop pixels as `embedding`, written
--    by orion-sql-writer from the crop payload. SHADOW ONLY: the individuals
--    reducer never reads it. NULL for every older row, for non-person crops,
--    and whenever orion-vision-host has VISION_OSNET_SHADOW_ENABLED=false or
--    could not load OSNet.
-- 2. osnet_model_id: which OSNet weights made it ("osnet_ain_x1_0:ms_d_c:<sha12>").
-- 3. embedding_model_id: the body model that ACTUALLY made `embedding`
--    (SigLIP v1 today: the requested SigLIP2 does not load and the host falls back).
--
-- Retention follows the crop rows (the walkway reducers prune
-- vision_crop_observation after 7 days); nothing new to prune.
--
-- Safe for the post-pull auto-apply (orion/sql_migration_drift.py
-- auto_apply_refusals): nullable columns with no default or constraint on the
-- existing table. Scored by services/orion-sql-writer/evals/run_osnet_vs_siglip_eval.py.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_osnet_shadow_v1.sql
-- Rollback (additive): ALTER TABLE vision_crop_observation DROP COLUMN osnet_embedding, DROP COLUMN osnet_model_id, DROP COLUMN embedding_model_id;

ALTER TABLE vision_crop_observation ADD COLUMN IF NOT EXISTS osnet_embedding REAL[];
ALTER TABLE vision_crop_observation ADD COLUMN IF NOT EXISTS osnet_model_id TEXT;
ALTER TABLE vision_crop_observation ADD COLUMN IF NOT EXISTS embedding_model_id TEXT;
