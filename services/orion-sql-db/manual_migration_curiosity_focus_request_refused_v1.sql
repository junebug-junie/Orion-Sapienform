-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Curiosity focus refused by the organ (curiosity replica, design
-- docs/superpowers/specs/2026-11-04-curiosity-replica-design.md section 9; Juniper 2026-11-04: the
-- replica may challenge AND refuse Prime's focus questions).
--
-- A refusal takes a Prime focus out of the pending slot before any run takes it, with a reason.
-- The status CHECK on this table cannot gain a value without dropping and re-adding it (not
-- additive), so a refusal is stored as status 'cleared' (out of the slot, no run took it) plus
-- refused_by / refused_reason. orion/curiosity/focus.py reads such a row as effective status
-- 'refused_by_organ', never as "cleared by Juniper"; a row Juniper's Clear button wrote has
-- refused_by NULL. Raw SQL readers: status = 'cleared' AND refused_by IS NOT NULL is a refusal.
--
--   refused_by      the act-ledger actor that refused it ('curiosity.replica').
--   refused_reason  its one-sentence reason (1-200 characters, checked in code).
--   refused_act_id  the refusal's own act (grammar_acts.act_id), parented to Prime's question act.
--
-- Two NOT VALID checks (enforced on every new write; the existing rows all have refused_by NULL):
--   * a reason always comes with a refusal, and a refusal is always a cleared row;
--   * a focus with origin 'juniper' can never be refused. This is the database half of the rule;
--     orion/curiosity/focus.py refuse_focus also only touches pending Prime focus rows.
--
-- Every focus SELECT in orion/curiosity/focus.py names these columns, so Hub needs them before
-- it starts. Additive DDL only, so the post-pull applier applies it.
-- Apply by hand, once, in one transaction (a second run fails on the two constraints, which
-- Postgres cannot add IF NOT EXISTS without a DO block the applier refuses; the transaction then
-- rolls back and nothing changes):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney --single-transaction \
--     -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_curiosity_focus_request_refused_v1.sql

ALTER TABLE curiosity_focus_request ADD COLUMN IF NOT EXISTS refused_by TEXT;
ALTER TABLE curiosity_focus_request ADD COLUMN IF NOT EXISTS refused_reason TEXT;
ALTER TABLE curiosity_focus_request ADD COLUMN IF NOT EXISTS refused_act_id TEXT;

ALTER TABLE curiosity_focus_request ADD CONSTRAINT curiosity_focus_request_refusal_shape
    CHECK ((refused_by IS NULL) = (refused_reason IS NULL)
           AND (refused_by IS NULL OR status = 'cleared')) NOT VALID;
ALTER TABLE curiosity_focus_request ADD CONSTRAINT curiosity_focus_request_refusal_not_juniper
    CHECK (refused_by IS NULL OR origin = 'self') NOT VALID;
