-- orion-gpu-pool stage 5.7: the actuation emergency stop (control verbs pause_actuation /
-- resume_actuation), persisted so a paused pool stays paused across a restart.
-- Spec: docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md (5.7)
-- Runbook: docs/runbooks/2026-09-30-gpu-pool-stage5-7-enforce.md
--
-- Additive only, safe on the live table. Since 2026-09-30 the pool also adds these columns itself
-- at boot (app/store.py BOOT_ADDITIVE_COLUMNS, kept identical by tests/test_schema_drift_gate.py),
-- so running this first is recommended, not required. The pool writes the same value to every card
-- row; it reads "paused" if any row carries a timestamp.
--
-- lock_timeout: ADD COLUMN takes an ACCESS EXCLUSIVE lock for an instant (no rewrite: both columns
-- are nullable). If the pool is mid-commit it waits at most 5s and fails instead of queueing every
-- lease RPC behind it; just re-run the file.
SET lock_timeout = '5s';

ALTER TABLE gpu_pool_cards ADD COLUMN IF NOT EXISTS actuation_paused_at timestamptz;
ALTER TABLE gpu_pool_cards ADD COLUMN IF NOT EXISTS actuation_paused_by text;
