-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: world-model shadow on action bets. When a bet carries body-level transition
-- features (orion/embodiment/transition.py), the service stores the trained TransitionModel's p of the
-- skill's success here: {model, tag, p, heads}. Scored and graded (shadow_verdict), never served.
--
-- Additive only (post-pull auto-apply allowlist): one nullable column on an existing table.

ALTER TABLE prediction_action_bet ADD COLUMN IF NOT EXISTS shadow jsonb;
