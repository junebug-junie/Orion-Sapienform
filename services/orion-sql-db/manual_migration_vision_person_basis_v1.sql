-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Why a person-model guess named who it named (2026-11-04, office camera prior).
-- basis = 'prior': the camera's prior chose the name, because no album match on
-- that camera was confident (config/person_model/camera_priors.*.yaml).
-- basis = 'evidence': the scores chose it (every camera without a prior).
-- NULL on rows written before this column. Readers treat NULL as 'evidence'.
-- Additive. orion-sql-writer writes it only when the column exists.
ALTER TABLE vision_person_attribution ADD COLUMN IF NOT EXISTS basis text;
