-- ORION-MIGRATION-REQUIRED-BY: none drops the tables of the retired proposal (L7), policy (L8) and dispatch (L9) rungs
-- DESTRUCTIVE
--
-- Retire the proposal, policy and dispatch rungs, step 2 of 2: drop their tables.
-- Spec: docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md ("PR 7"; Juniper,
-- 2026-11-05: drop 2 days after, with a backup first). Applied BY HAND only; the post-pull applier
-- never runs a DESTRUCTIVE file. Merge this file in the same sitting it is applied, not before:
-- merged-but-unapplied, the migration drift check reads these tables as "should have been dropped"
-- and cards Hub red.
--
-- NOT BEFORE: 2 days after the retirement PR (PR 6, feat/retire-l7-l9) is deployed (all three
-- services stopped and removed, every reader rebuilt).
--
-- ORDER: the feedback/consolidation drop (#3116, manual_migration_retire_feedback_consolidation_rungs_drop.sql)
-- must have run first (it did, 2026-11-07, and #3116 is merged; the check below confirms). Its last
-- statement ALTERs substrate_execution_dispatch_frames, which this file drops; run after this one,
-- that ALTER fails and its whole transaction (its six DROPs) rolls back.
--   docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select to_regclass('substrate_feedback_frames')"   (must print an empty line)
--
-- Check first that nothing has written since the deploy (each must be older than the deploy):
--   docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select max(created_at) from substrate_proposal_frames; select max(created_at) from substrate_policy_decision_frames; select max(created_at) from substrate_execution_dispatch_frames; select max(created_at) from substrate_dispatch_results; select max(created_at) from substrate_act_episodes; select count(*) from substrate_execution_dispatch_inflight"
--   docker ps -a --format '{{.Names}}' | grep -E 'proposal-runtime|policy-runtime|execution-dispatch-runtime'   (must print nothing)
--
-- BACKUP FIRST (about 21 GB before compression, mostly TOAST), then apply:
--   mkdir -p /mnt/storage-warm/orion-backups
--   docker exec orion-athena-sql-db pg_dump -U postgres -d conjourney -Fc -t substrate_proposal_frames -t substrate_policy_decision_frames -t substrate_execution_dispatch_frames -t substrate_execution_dispatch_inflight -t substrate_dispatch_results -t substrate_act_episodes > /mnt/storage-warm/orion-backups/retired-proposal-policy-dispatch-rungs.dump
--   docker exec -i orion-athena-sql-db pg_restore --list /dev/stdin < /mnt/storage-warm/orion-backups/retired-proposal-policy-dispatch-rungs.dump | grep -c 'TABLE DATA'   (must print 6)
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_retire_proposal_policy_dispatch_rungs_drop.sql
--
-- What goes (sizes live 2026-11-07): substrate_execution_dispatch_frames 11 GB, substrate_policy_decision_frames
-- 7.6 GB, substrate_proposal_frames 2.3 GB, substrate_dispatch_results 514 MB, substrate_execution_dispatch_inflight
-- 880 kB, substrate_act_episodes 248 kB (writer-less: only the dispatch runtime wrote and resolved it; 140
-- rows, last 2026-11-05). Their indexes go with them.
--
-- Kept on purpose: substrate_action_outcomes, substrate_action_effect_posterior, substrate_signal_control_cells,
-- substrate_world_action_episodes and gpu_pool_orion_shed (the act ledger, the world-action scorer in
-- orion-sql-writer and the cooling act in orion-durable-runs write them); world_pulse_read_seed's
-- pull_dispatch_id / pull_arm / pulled_at columns (history of the ended reading-pull experiment; nothing
-- writes or reads them). No foreign key and no view depends on the dropped tables (checked live
-- 2026-11-07: pg_constraint contype='f' and pg_rewrite dependencies, both empty).

BEGIN;
SET LOCAL lock_timeout = '10s';
DROP TABLE IF EXISTS substrate_proposal_frames;
DROP TABLE IF EXISTS substrate_policy_decision_frames;
DROP TABLE IF EXISTS substrate_execution_dispatch_frames;
DROP TABLE IF EXISTS substrate_execution_dispatch_inflight;
DROP TABLE IF EXISTS substrate_dispatch_results;
DROP TABLE IF EXISTS substrate_act_episodes;
COMMIT;
