-- Additive visual non-observation contract; existing outcome rows stay unchanged.
ALTER TABLE action_outcomes ADD COLUMN IF NOT EXISTS visual_outcome TEXT NULL;
