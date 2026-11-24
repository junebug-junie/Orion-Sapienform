-- Let Orion's read-only sandbox role read prediction_joint_surprise, the backing table of
-- node:prediction.body_joint and node:prediction.room_joint (family A joint surprise: one row per
-- scored joint group, with its conformal p and excess). Issue #3821.
--
-- Checked live 2026-11-24: has_table_privilege('orion_readonly', 'prediction_joint_surprise', 'SELECT')
-- = f, so Orion cannot dig into why its own joint nodes moved. (prediction_turn_self_bet, the backing
-- table of node:prediction.chat_turn, already reads t.) No grant on
-- substrate_node_prediction_error_history: it is being retired.
--
-- Read-only grant; no write. Not a migration: GRANT is not on the post-pull auto-apply allowlist.
-- Idempotent: re-running is safe.
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-24_grant_orion_readonly_prediction_joint_surprise.sql

BEGIN;

GRANT SELECT ON public.prediction_joint_surprise TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.prediction_joint_surprise FROM orion_readonly;
--
-- Verify (expect t):
--   SELECT has_table_privilege('orion_readonly', 'prediction_joint_surprise', 'SELECT');
