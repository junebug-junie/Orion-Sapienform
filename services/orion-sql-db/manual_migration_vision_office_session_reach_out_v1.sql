-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Office reach-out (2026-11-05, office camera design PR #3127 step C): Hub sends Juniper at most ONE
-- curious chat message per office visit (services/orion-hub/scripts/office_reach_out.py,
-- HUB_OFFICE_REACH_OUT_ENABLED). These two columns mark the visit as told, so a Hub restart never sends
-- a second message for the same visit.
--
--   reached_out_at         when Hub claimed the visit for its message (set just before delivery; cleared
--                          again when the delivery gates refused it, so a later tick retries).
--   reach_out_decision_id  the decision_id of the endogenous_outreach_decisions row for that message.
--
-- Writer: orion-hub (claim + release). orion-sql-writer's UPSERT never names these columns, so its
-- rebuild of an open visit keeps them. Without this migration the Hub loop sends nothing (it refuses
-- to message without a durable "already told" mark).
-- Additive and idempotent; re-running is safe.
ALTER TABLE vision_office_session ADD COLUMN IF NOT EXISTS reached_out_at timestamptz;
ALTER TABLE vision_office_session ADD COLUMN IF NOT EXISTS reach_out_decision_id text;
