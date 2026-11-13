-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: action bets -- "this body action, in this situation, at this place, works with
-- probability p". The service computes p (action_beta_backoff) and the baseline p (action_skill_rate)
-- when the bet is placed; the caller settles it success / failure / void; open bets older than 1 h are
-- closed void with outcome 'never_settled'.
-- Spec: docs/superpowers/specs/2026-10-23-embodied-experience-learning-design.md
-- ("Prediction service: the action bet").
--
-- Additive only (post-pull auto-apply allowlist): one new table and three indexes on it. The table is
-- new, so the indexes need not be CONCURRENTLY.

CREATE TABLE IF NOT EXISTS prediction_action_bet (
  bet_id        text PRIMARY KEY,
  domain        text NOT NULL,
  body          text NOT NULL,
  skill         text NOT NULL,
  context       jsonb NOT NULL,
  place         text,
  action        jsonb NOT NULL,
  model         text NOT NULL,
  p             real NOT NULL,
  p_baseline    real NOT NULL,
  level         text NOT NULL,
  counts        jsonb NOT NULL,
  chosen_rank   smallint,
  session_id    text,
  placed_at     timestamptz NOT NULL DEFAULT now(),
  settled_at    timestamptz,
  result        text,
  outcome       text,
  detail        jsonb
);
CREATE INDEX IF NOT EXISTS prediction_action_bet_skill ON prediction_action_bet (skill, settled_at);
CREATE INDEX IF NOT EXISTS prediction_action_bet_domain ON prediction_action_bet (domain, settled_at);
CREATE INDEX IF NOT EXISTS prediction_action_bet_place ON prediction_action_bet (place) WHERE place IS NOT NULL;

-- Undo: DROP TABLE prediction_action_bet;
