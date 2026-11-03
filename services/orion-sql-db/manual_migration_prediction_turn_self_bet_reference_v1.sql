-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: the "usual size" a relative turn self-bet was graded against. Bet v2 asks
-- stance "more steps / a longer reply than usual for this kind of turn?". reference is the median
-- true size of the same caller's turns placed in the 7 days before the bet, never shown to stance.
-- It is NULL on every other faculty.
-- Spec: docs/superpowers/specs/2026-11-02-turn-self-bets-relative-size-design.md.
--
-- Additive only (post-pull auto-apply allowlist): one nullable column.

ALTER TABLE prediction_turn_self_bet ADD COLUMN IF NOT EXISTS reference real;

-- Undo: ALTER TABLE prediction_turn_self_bet DROP COLUMN reference;
