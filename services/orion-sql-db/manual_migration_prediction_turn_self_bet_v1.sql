-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: turn self-bets, settled. Stance writes a private guess about its own turn
-- (thought_decision.self_prediction, A1, live 2026-10-24); orion-prediction settles each of its four
-- faculties against the turn's record (harness_turn_trace.run_artifact) and grades the calibration.
-- One row per (turn, faculty); bet_id = <correlation_id>:<faculty>. status upheld / broken / void
-- (void_reason: failed_to_execute, expired_unsettled, ungradeable_no_stated, ungradeable_no_outcome).
-- Spec: docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md (A3).
--
-- Additive only (post-pull auto-apply allowlist): one new table and two indexes on it. The table is
-- new, so the indexes need not be CONCURRENTLY.

CREATE TABLE IF NOT EXISTS prediction_turn_self_bet (
  bet_id          text PRIMARY KEY,
  correlation_id  text NOT NULL,
  faculty         text NOT NULL,
  claim_kind      text NOT NULL,
  caller          text NOT NULL,
  placed_at       timestamptz NOT NULL,
  settled_at      timestamptz NOT NULL,
  status          text NOT NULL,
  void_reason     text,
  stated_p        real,
  stated_lo       real,
  stated_hi       real,
  coverage        real NOT NULL,
  observed        real,
  hit             boolean
);
CREATE INDEX IF NOT EXISTS prediction_turn_self_bet_corr ON prediction_turn_self_bet (correlation_id);
CREATE INDEX IF NOT EXISTS prediction_turn_self_bet_placed ON prediction_turn_self_bet (placed_at);

-- Undo: DROP TABLE prediction_turn_self_bet;
